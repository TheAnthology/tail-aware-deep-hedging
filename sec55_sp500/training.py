import numpy as np
import tensorflow as tf
from .models import tail_cvar_tf


def make_step(model, preference, q, alpha=0.75, learning_rate=0.0001):
    """Build terminal-P&L backpropagation for an already initialized policy.

    Returns(step, optimizer). The step accepts features[N,6,2], holding-period
    stock changes[N,6], and terminal call payoff[N], all float32. Outputs are
    objective, OCE75 minus q, soft loss probability, empirical batch CVaR75,
    hard loss probability, total gradient norm, and six network gradient norms.

    Eta is stored in liability units: eta_loss=eta-q. Probability-only excludes
    the unused eta from optimization. No holding or gradient clipping is added.
    """
    if preference not in ["C", "P"]:
        raise ValueError("preference must be C or P")
    if not 0 <= alpha < 1:
        raise ValueError("alpha must lie in[0,1)")
    if not np.isfinite(q) or not np.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("Finite premium and strictly positive learning rate required")
    if not model.built:
        raise ValueError("Initialize the model before constructing its optimizer")
    variables = tuple(
        v for v in model.trainable_variables if preference == "C" or v is not model.eta
    )
    optimizer = tf.keras.optimizers.Adam(
        learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-7
    )
    optimizer.build(list(variables))
    indices = [
        [
            i
            for i, v in enumerate(variables)
            if any(v is w for w in net.trainable_variables)
        ]
        for net in model.nets
    ]
    if len(indices) != 6 or any(not ix for ix in indices):
        raise RuntimeError("Every one of the six date networks must be registered")
    q32 = tf.constant(float(q), tf.float32)
    coefficient = tf.constant(1 / (1 - alpha), tf.float32)

    @tf.function(
        input_signature=[
            tf.TensorSpec([None, 6, 2], tf.float32),
            tf.TensorSpec([None, 6], tf.float32),
            tf.TensorSpec([None], tf.float32),
        ]
    )
    def step(features, changes, payoff):
        with tf.GradientTape() as tape:
            holding = model(features, training=True)
            liability_loss = payoff - tf.reduce_sum(holding * changes, axis=1)
            loss = liability_loss - q32
            cvar_objective = (
                model.eta
                + coefficient * tf.reduce_mean(tf.nn.relu(liability_loss - model.eta))
                - q32
            )
            soft_probability = tf.reduce_mean(tf.sigmoid(loss / 0.1))
            objective = cvar_objective if preference == "C" else soft_probability
        gradients = tape.gradient(objective, variables)
        if any(g is None for g in gradients):
            raise RuntimeError("Disconnected expert parameter")
        tf.debugging.check_numerics(objective, "single-expert objective")
        for gradient in gradients:
            tf.debugging.check_numerics(gradient, "single-expert gradient")
        network_norms = tf.stack(
            [tf.linalg.global_norm([gradients[i] for i in ix]) for ix in indices]
        )
        empirical_cvar = tail_cvar_tf(loss, alpha)
        loss_probability = tf.reduce_mean(tf.cast(loss > 0, tf.float32))
        optimizer.apply_gradients(zip(gradients, variables))
        return (
            objective,
            cvar_objective,
            soft_probability,
            empirical_cvar,
            loss_probability,
            tf.linalg.global_norm(gradients),
            network_norms,
        )

    return step, optimizer
