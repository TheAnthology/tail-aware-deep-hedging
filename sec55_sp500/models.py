"""Stock-only Simpler CVaR75 / Probability experts and residual policy."""

import numpy as np
import tensorflow as tf
from scipy.special import ndtr

HEDGE_DAYS = np.array([0, 5, 10, 15, 20, 25], dtype=int)
RESIDUAL_KEYS = ["f", "changes", "hc", "hp", "liability", "payoff"]


def market_data(prices, vol, strike=100.0):
    """Current features and Black-Scholes liability marks at decision dates.

    All 31 daily states are supplied, but each decision reads only its current
    price and volatility. Volatility is a daily return standard deviation;
    the second feature is annualized variance, 252 * sigma_daily**2.
    Terminal payoff and subsequent changes are targets, not network features.
    """
    prices = np.asarray(prices, dtype="float64")
    vol = np.asarray(vol, dtype="float64")
    if prices.ndim != 2 or prices.shape[1] != 31 or vol.shape != prices.shape:
        raise ValueError("Aligned daily price and volatility arrays[N,31] required")
    if (
        not np.isfinite(prices).all()
        or np.any(prices <= 0)
        or not np.isfinite(vol).all()
        or np.any(vol < 0)
    ):
        raise ValueError("Finite positive prices and nonnegative volatility required")
    spot = prices[:, HEDGE_DAYS]
    sigma = vol[:, HEDGE_DAYS]
    total_vol = sigma * np.sqrt(30 - HEDGE_DAYS)
    d1 = np.divide(
        np.log(spot / strike) + 0.5 * total_vol**2,
        total_vol,
        out=np.zeros_like(total_vol),
        where=total_vol > 0,
    )
    liability = np.where(
        total_vol > 0,
        spot * ndtr(d1) - strike * ndtr(d1 - total_vol),
        np.maximum(spot - strike, 0.0),
    )
    # Preserve the premium's exact arithmetic at standard ATM initial states.
    standard = spot[:, 0] == strike
    liability[standard, 0] = strike * (2 * ndtr(total_vol[standard, 0] / 2) - 1)
    f = np.stack([np.log(spot), 252 * sigma * sigma], axis=-1).astype("float32")
    changes = np.diff(prices[:, np.r_[HEDGE_DAYS, 30]], axis=1)
    payoff = np.maximum(prices[:, -1] - strike, 0.0)
    return {"f": f, "changes": changes, "liability": liability, "payoff": payoff}


class CVaRExpert(tf.keras.Model):
    """Six independent 2 -> 20 BN/ReLU -> 20 BN/ReLU -> 1 date networks.

    Output is total stock holdings. C and P share this architecture but use
    different training objectives: CVaR75 and smooth loss probability.
    """

    def __init__(self, seed=8501, bn_epsilon=0.001, bn_momentum=0.99):
        super().__init__(name="sp500_simple_expert", dtype="float32")
        self.nets = []
        for k in range(6):

            def dense(width, index):
                return tf.keras.layers.Dense(
                    width,
                    dtype="float32",
                    kernel_initializer=tf.keras.initializers.GlorotUniform(
                        seed=seed + 100 * k + index
                    ),
                    bias_initializer="zeros",
                    name=f"dense_{index}",
                )

            def bn(index):
                return tf.keras.layers.BatchNormalization(
                    epsilon=bn_epsilon,
                    momentum=bn_momentum,
                    dtype="float32",
                    name=f"bn_{index}",
                )

            self.nets.append(
                tf.keras.Sequential(
                    [
                        dense(20, 0),
                        bn(0),
                        tf.keras.layers.ReLU(),
                        dense(20, 1),
                        bn(1),
                        tf.keras.layers.ReLU(),
                        dense(1, 2),
                    ],
                    name=f"date_{k}",
                )
            )
        self.eta = self.add_weight(
            name="cvar_intercept",
            shape=(),
            initializer="zeros",
            trainable=True,
            dtype="float32",
        )

    def call(self, features, training=False):
        features = tf.ensure_shape(
            tf.convert_to_tensor(features, tf.float32), [None, 6, 2]
        )
        return tf.concat(
            [net(features[:, k], training=training) for k, net in enumerate(self.nets)],
            axis=1,
        )

    def initialize(self):
        self(tf.zeros([2, 6, 2], tf.float32), training=False)

    def initialize_without_bn_updates(self):
        self.initialize()


def predict(model, features, chunk=2048):
    return np.concatenate(
        [
            model(features[i : i + chunk], training=False).numpy()
            for i in range(0, len(features), chunk)
        ]
    )


def calibrate(model, features):
    """Set per-date/layer BatchNorm moments while keeping learned weights fixed."""
    records = []
    learned = [v.numpy().copy() for v in model.trainable_variables]
    for k, net in enumerate(model.nets):
        layer_inputs = features[:, k, :]
        row = {"date": k, "day": int(HEDGE_DAYS[k])}
        for dense_index, bn_index, relu_index in [(0, 1, 2), (3, 4, 5)]:
            preactivations = np.concatenate(
                [
                    net.layers[dense_index](layer_inputs[i : i + 256]).numpy()
                    for i in range(0, len(layer_inputs), 256)
                ]
            )
            mean = preactivations.astype("float64").mean(0)
            variance = preactivations.astype("float64").var(0)
            net.layers[bn_index].moving_mean.assign(mean)
            net.layers[bn_index].moving_variance.assign(variance)
            layer_inputs = net.layers[relu_index](
                net.layers[bn_index](preactivations, training=False)
            )
            row[str(bn_index)] = {"mean": mean.tolist(), "variance": variance.tolist()}
        records.append(row)
    assert all(
        np.array_equal(v.numpy(), old)
        for v, old in zip(model.trainable_variables, learned)
    )
    return records


def tail_cvar_tf(loss, alpha=0.95):
    """Differentiable exact fractional empirical tail average."""
    values = tf.sort(loss, direction="DESCENDING")
    mass = tf.cast(tf.size(loss), loss.dtype) * (1 - alpha)
    whole = tf.cast(tf.floor(mass), tf.int32)
    fraction = mass - tf.cast(whole, loss.dtype)
    # Weight the boundary observation when the tail contains a fractional count.
    return (tf.reduce_sum(values[:whole]) + fraction * values[whole]) / mass


def position_pnl(data, holdings, q):
    return (
        q
        + np.sum(np.asarray(holdings, dtype="float64") * data["changes"], axis=1)
        - data["payoff"]
    )


class StockResidual(tf.keras.Model):
    """Add a learned stock correction to the frozen CVaR policy.

    Inputs are current log spot, variance, actual strategy surplus, time, and
    the normalized difference between probability and CVaR expert holdings.
    """

    def __init__(self, scale, budget, q, seed=9501):
        super().__init__(name="sp500_stock_residual", autocast=False)
        if not np.isfinite(scale) or scale <= 0:
            raise ValueError(
                "Strictly positive finite training-derived stock holding scale required"
            )
        if not np.isfinite(budget) or budget <= 0:
            raise ValueError(
                "Nonpositive or nonfinite CVaR95 budget; record this origin as invalid, never silently floor the budget"
            )
        if not np.isfinite(q):
            raise ValueError("Finite fixed common premium required")
        self.scale = float(scale)
        self.budget = float(budget)
        self.q = float(q)
        self.a = tf.keras.layers.Dense(
            32,
            activation="tanh",
            dtype="float32",
            kernel_initializer=tf.keras.initializers.GlorotUniform(seed=seed),
            bias_initializer="zeros",
        )
        self.b = tf.keras.layers.Dense(
            32,
            activation="tanh",
            dtype="float32",
            kernel_initializer=tf.keras.initializers.GlorotUniform(seed=seed + 1),
            bias_initializer="zeros",
        )
        # A zero output head starts the combined policy exactly at CVaR.
        self.out = tf.keras.layers.Dense(
            1, dtype="float32", kernel_initializer="zeros", bias_initializer="zeros"
        )
        self.a(tf.zeros([1, 5], tf.float32))
        self.b(tf.zeros([1, 32], tf.float32))
        self.out(tf.zeros([1, 32], tf.float32))

    def call(self, inputs):
        """Return terminal P&L, holdings, corrections, and pre-trade surplus."""
        features, changes, cvar_holdings, probability_holdings, liability, payoff = (
            inputs
        )
        batch_size = tf.shape(features)[0]
        cumulative_gain = tf.zeros([batch_size], tf.float64)
        positions = []
        corrections = []
        surplus_rows = []
        for k, day in enumerate(HEDGE_DAYS):
            # Use gains through the previous interval, before this date's move.
            surplus = self.q + cumulative_gain - liability[:, k]
            network_inputs = tf.stack(
                [
                    (features[:, k, 0] - np.log(100.0)) / 0.2,
                    features[:, k, 1] / 0.04,
                    tf.cast(surplus / self.budget, tf.float32),
                    tf.fill([batch_size], tf.cast(int(day) / 30.0, tf.float32)),
                    tf.cast(
                        (probability_holdings[:, k] - cvar_holdings[:, k]) / self.scale,
                        tf.float32,
                    ),
                ],
                axis=1,
            )
            correction = self.scale * tf.cast(
                tf.squeeze(self.out(self.b(self.a(network_inputs))), axis=1), tf.float64
            )
            holding = cvar_holdings[:, k] + correction
            positions.append(holding)
            corrections.append(correction)
            surplus_rows.append(surplus)
            # Accumulate gains from the actual combined holdings.
            cumulative_gain = cumulative_gain + holding * changes[:, k]
        return (
            self.q + cumulative_gain - payoff,
            tf.stack(positions, axis=1),
            tf.stack(corrections, axis=1),
            tf.stack(surplus_rows, axis=1),
        )

    def initialize(self):
        self(
            (
                tf.zeros([2, 6, 2], tf.float32),
                tf.zeros([2, 6], tf.float64),
                tf.zeros([2, 6], tf.float64),
                tf.zeros([2, 6], tf.float64),
                tf.fill([2, 6], tf.constant(self.q, tf.float64)),
                tf.zeros([2], tf.float64),
            )
        )


def residual_tensors(data, ix=None):
    return tuple(
        tf.convert_to_tensor(
            data[k] if ix is None else data[k][ix],
            dtype=tf.float32 if k == "f" else tf.float64,
        )
        for k in RESIDUAL_KEYS
    )


def make_res_forward(model):
    @tf.function(reduce_retracing=True)
    def forward(f, changes, hc, hp, liability, payoff):
        return model((f, changes, hc, hp, liability, payoff))

    return forward


def evaluate_res(model, data, chunk=2048, forward=None):
    if forward is None:
        forward = make_res_forward(model)
    rows = [[], [], [], []]
    for lo in range(0, len(data["payoff"]), chunk):
        values = forward(*residual_tensors(data, slice(lo, lo + chunk)))
        for target, value in zip(rows, values):
            target.append(value.numpy())
    return tuple(np.concatenate(v) for v in rows)


def make_res_step(model):
    """Build the residual's CVaR95 primal-dual update.

    Diagnostics: soft loss probability, CVaR95, relative budget excess, updated
    dual multiplier, gradient norm, and normalized correction RMS.
    """
    optimizer = tf.keras.optimizers.Adam(0.001)
    dual = tf.Variable(1.0, trainable=False, dtype=tf.float64)

    @tf.function(reduce_retracing=True)
    def step(features, changes, cvar_holdings, probability_holdings, liability, payoff):
        with tf.GradientTape() as tape:
            pnl, holdings, correction, surplus = model(
                (
                    features,
                    changes,
                    cvar_holdings,
                    probability_holdings,
                    liability,
                    payoff,
                )
            )
            soft_probability = tf.reduce_mean(tf.sigmoid(-pnl / 0.1))
            tail_risk = tail_cvar_tf(-pnl, 0.95)
            # Positive excess means CVaR95 is above the fixed risk budget.
            risk_excess = tail_risk / model.budget - 1
            objective = (
                soft_probability
                + tf.stop_gradient(dual) * risk_excess
                + 5 * tf.nn.relu(risk_excess) ** 2
            )
        gradients = tape.gradient(objective, model.trainable_variables)
        if any(gradient is None for gradient in gradients):
            raise RuntimeError("Disconnected residual parameter")
        tf.debugging.check_numerics(objective, "residual objective")
        for gradient in gradients:
            tf.debugging.check_numerics(gradient, "residual gradient")
        clipped, norm = tf.clip_by_global_norm(gradients, 10.0)
        optimizer.apply_gradients(zip(clipped, model.trainable_variables))
        # Projected ascent raises the penalty when the risk budget is exceeded.
        dual.assign(
            tf.maximum(
                tf.constant(0.0, tf.float64),
                dual + 0.05 * tf.stop_gradient(risk_excess),
            )
        )
        return (
            soft_probability,
            tail_risk,
            risk_excess,
            dual,
            tf.cast(norm, tf.float64),
            tf.sqrt(tf.reduce_mean((correction / model.scale) ** 2)),
        )

    return step, optimizer, dual
