"""Causal two-asset residual holdings and CVaR95-constrained training."""

import numpy as np
import tensorflow as tf
from common.risk import tail95_tf

KEYS = ["f", "changes", "hc", "hp", "liability", "payoff"]


class ResidualCombiner(tf.keras.Model):
    """Shared date-independent 6 -> 32 -> 32 -> 2 tanh MLP.

    Residual: h_C + D F(z, (h_P-h_C)/D).
    z uses the current actual-strategy surplus, before the current increment.
    """

    def __init__(self, scales, budget, q, seed=6501):
        super().__init__(name="residual_combiner", autocast=False)
        scales = np.asarray(scales, dtype="float64")
        if scales.shape != (2,) or not np.isfinite(scales).all() or np.any(scales <= 0):
            raise ValueError(
                "Two finite strictly positive fixed position scales required"
            )
        if not np.isfinite(budget) or budget <= 0 or not np.isfinite(q):
            raise ValueError(
                "Finite premium and strictly positive risk budget required"
            )
        self.scales = scales.tolist()
        self.scale_tensor = tf.constant(scales, dtype=tf.float64)
        self.budget = float(budget)
        self.q = float(q)
        self.seed = int(seed)
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
            2, dtype="float32", kernel_initializer="zeros", bias_initializer="zeros"
        )
        self.a(tf.zeros([1, 6], tf.float32))
        self.b(tf.zeros([1, 32], tf.float32))
        self.out(tf.zeros([1, 32], tf.float32))

    def call(self, inputs):
        """Return terminal P&L, holdings, corrections, and pre-trade surplus."""
        features, changes, cvar_holdings, probability_holdings, liability, payoff = (
            inputs
        )
        batch_size = tf.shape(features)[0]
        cumulative_gain = tf.zeros([batch_size], tf.float64)
        positions, corrections, surpluses = [], [], []
        for k in range(30):
            # Choose this date's holdings before observing its price increment.
            surplus = self.q + cumulative_gain - liability[:, k]
            state_features = [
                (features[:, k, 0] - np.log(100.0)) / 0.2,
                features[:, k, 1] / 0.04,
                tf.cast(surplus / self.budget, tf.float32),
                tf.fill([batch_size], tf.cast(k / 30.0, tf.float32)),
            ]
            network_inputs = tf.stack(state_features, axis=1)
            expert_difference = tf.cast(
                (probability_holdings[:, k, :] - cvar_holdings[:, k, :])
                / self.scale_tensor,
                tf.float32,
            )
            network_inputs = tf.concat([network_inputs, expert_difference], axis=1)
            normalized_output = self.out(self.b(self.a(network_inputs)))
            scaled_output = self.scale_tensor * tf.cast(normalized_output, tf.float64)
            holding = cvar_holdings[:, k, :] + scaled_output
            correction = holding - cvar_holdings[:, k, :]
            positions.append(holding)
            corrections.append(correction)
            surpluses.append(surplus)
            # Preserve graph through actual combined prior gains. Expert
            # positions are cached constants; no future increment enters z_k.
            cumulative_gain = cumulative_gain + tf.reduce_sum(
                holding * changes[:, k, :], axis=1
            )
        return (
            self.q + cumulative_gain - payoff,
            tf.stack(positions, axis=1),
            tf.stack(corrections, axis=1),
            tf.stack(surpluses, axis=1),
        )

    def initialize(self):
        self(
            (
                tf.zeros([2, 30, 2], tf.float32),
                tf.zeros([2, 30, 2], tf.float64),
                tf.zeros([2, 30, 2], tf.float64),
                tf.zeros([2, 30, 2], tf.float64),
                tf.fill([2, 30], tf.constant(self.q, tf.float64)),
                tf.zeros([2], tf.float64),
            )
        )


def tensors(data, ix=None):
    return tuple(
        tf.convert_to_tensor(
            data[k] if ix is None else data[k][ix],
            dtype=tf.float32 if k == "f" else tf.float64,
        )
        for k in KEYS
    )


def make_forward(model):
    @tf.function(reduce_retracing=True)
    def forward(f, changes, hc, hp, liability, payoff):
        return model((f, changes, hc, hp, liability, payoff))

    return forward


def evaluate(model, data, chunk=2048, forward=None):
    if forward is None:
        forward = make_forward(model)
    results = [[], [], [], []]
    for lo in range(0, len(data["payoff"]), chunk):
        values = forward(*tensors(data, slice(lo, lo + chunk)))
        for target, value in zip(results, values):
            target.append(value.numpy())
    return tuple(np.concatenate(v) for v in results)


def make_step(model):
    """Normalized CVaR95 primal-dual penalty and prescribed optimizer.

    Diagnostics: soft loss probability, CVaR95, relative budget excess, updated
    dual multiplier, gradient norm, and normalized correction RMS. The RMS is
    diagnostic only; it is not added to the objective.
    """
    optimizer = tf.keras.optimizers.Adam(0.001)
    dual = tf.Variable(1.0, trainable=False, dtype=tf.float64)

    @tf.function(reduce_retracing=True)
    def step(features, changes, cvar_holdings, probability_holdings, liability, payoff):
        with tf.GradientTape() as tape:
            pnl, holding, correction, surplus = model(
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
            tail_risk = tail95_tf(-pnl)
            # Positive excess means CVaR95 is above the fixed risk budget.
            risk_excess = tail_risk / model.budget - 1
            objective = (
                soft_probability
                + tf.stop_gradient(dual) * risk_excess
                + 5 * tf.nn.relu(risk_excess) ** 2
            )
        gradients = tape.gradient(objective, model.trainable_variables)
        if any(grad is None for grad in gradients):
            raise RuntimeError("Disconnected combiner parameter")
        tf.debugging.check_numerics(objective, "combiner objective")
        for grad in gradients:
            tf.debugging.check_numerics(grad, "combiner gradient")
        clipped, norm = tf.clip_by_global_norm(gradients, 10.0)
        optimizer.apply_gradients(zip(clipped, model.trainable_variables))
        # Projected ascent raises the penalty when the risk budget is exceeded.
        dual.assign(
            tf.maximum(
                tf.constant(0.0, tf.float64),
                dual + 0.05 * tf.stop_gradient(risk_excess),
            )
        )
        correction_rms = tf.sqrt(tf.reduce_mean((correction / model.scale_tensor) ** 2))
        return (
            soft_probability,
            tail_risk,
            risk_excess,
            dual,
            tf.cast(norm, tf.float64),
            correction_rms,
        )

    return step, optimizer, dual
