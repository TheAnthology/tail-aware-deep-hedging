"""TensorFlow core for the 2019 paper's Section 5.5 baseline reconstruction.

Paper-specified: separate network at each hedge date; two hidden layers of 20
ReLU neurons; BatchNorm immediately before each hidden activation; inputs are
log spot and previous holdings; Adam learning rate 1e-4; CVaR75 for 30 days.

Declared conventions where the paper is incomplete: float32, Glorot-uniform
Dense kernels with distinct fixed seeds, zero Dense biases, and the 2019
TensorFlow 1.13.1 Keras BatchNorm defaults (momentum=.99, epsilon=.001).
These are explicit reconstruction conventions, not verified author settings.

This module does not generate data, select checkpoints, change the initial-date
network, freeze BatchNorm, or recalibrate its moving statistics. Every forward
call receives an explicit training flag. Model construction uses inference mode
so that dummy inputs do not update BatchNorm state.

The CVaR optimization is in original currency units and excludes the option
premium, following the paper's pi(-Z) optimization. A fixed common premium is
added only when reporting P&L. No paper-result replication claim is implied.
"""
from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import tensorflow as tf


class PaperHedge(tf.keras.Model):
    """Semi-recurrent holdings policy with independent per-date parameters.

    Input: log_spots, shape (paths, hedge_dates). Output: holdings of the stock,
    with the same shape. The previous holdings remain in the gradient graph.
    The first date is treated exactly like the remaining network dates, with
    initial previous holdings zero, rather than replaced by a scalar parameter.
    """

    def __init__(
        self,
        steps: int = 6,
        seed: int = 2701,
        bn_momentum: float = 0.99,
        bn_epsilon: float = 0.001,
        name: str = "paper_hedge",
    ) -> None:
        if steps < 1:
            raise ValueError("steps must be positive")
        super().__init__(name=name, dtype="float32")
        self.steps = int(steps)
        self.seed = int(seed)
        self.bn_momentum = float(bn_momentum)
        self.bn_epsilon = float(bn_epsilon)
        self.nets = []
        for k in range(self.steps):
            def dense(width: int, index: int) -> tf.keras.layers.Dense:
                return tf.keras.layers.Dense(
                    width,
                    activation=None,
                    kernel_initializer=tf.keras.initializers.GlorotUniform(
                        seed=self.seed + 100 * k + index
                    ),
                    bias_initializer="zeros",
                    dtype="float32",
                    name=f"dense_{index}",
                )

            def batchnorm(index: int) -> tf.keras.layers.BatchNormalization:
                return tf.keras.layers.BatchNormalization(
                    axis=-1,
                    momentum=self.bn_momentum,
                    epsilon=self.bn_epsilon,
                    center=True,
                    scale=True,
                    beta_initializer="zeros",
                    gamma_initializer="ones",
                    moving_mean_initializer="zeros",
                    moving_variance_initializer="ones",
                    dtype="float32",
                    name=f"batchnorm_{index}",
                )

            self.nets.append(
                tf.keras.Sequential(
                    [
                        dense(20, 0),
                        batchnorm(0),
                        tf.keras.layers.ReLU(name="relu_0", dtype="float32"),
                        dense(20, 1),
                        batchnorm(1),
                        tf.keras.layers.ReLU(name="relu_1", dtype="float32"),
                        dense(1, 2),
                    ],
                    name=f"hedge_date_{k}",
                )
            )
        self.eta = self.add_weight(
            name="cvar_intercept",
            shape=(),
            initializer="zeros",
            trainable=True,
            dtype="float32",
        )

    def call(self, log_spots: tf.Tensor, training: bool = False) -> tf.Tensor:
        log_spots = tf.ensure_shape(
            tf.convert_to_tensor(log_spots, dtype=tf.float32), [None, self.steps]
        )
        previous = tf.zeros_like(log_spots[:, :1])
        holdings = []
        for k, net in enumerate(self.nets):
            features = tf.concat([log_spots[:, k:k + 1], previous], axis=1)
            previous = net(features, training=training)
            holdings.append(previous)
        return tf.concat(holdings, axis=1)

    def initialize_without_bn_updates(self) -> None:
        """Create all variables without treating dummy paths as training data."""
        self(tf.zeros([2, self.steps], dtype=tf.float32), training=False)


def make_adam(learning_rate: float = 0.0001) -> tf.keras.optimizers.Adam:
    """Explicit optimizer convention; no weight decay or gradient clipping."""
    return tf.keras.optimizers.Adam(
        learning_rate=learning_rate,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-7,
        amsgrad=False,
    )


def cvar_objective(
    losses: tf.Tensor, eta: tf.Tensor, alpha: float = 0.75
) -> tf.Tensor:
    """Variational empirical CVaR in original currency units."""
    if not 0.0 <= alpha < 1.0:
        raise ValueError("alpha must satisfy 0 <= alpha < 1")
    losses = tf.reshape(tf.convert_to_tensor(losses, dtype=tf.float32), [-1])
    return eta + tf.reduce_mean(tf.nn.relu(losses - eta)) / (1.0 - alpha)


def create_train_step(
    model: PaperHedge,
    optimizer: tf.keras.optimizers.Optimizer,
    alpha: float = 0.75,
) -> Callable[[tf.Tensor, tf.Tensor, tf.Tensor], dict[str, tf.Tensor]]:
    """Build one compiled mini-batch step over the complete hedge trajectory.

    Arguments to the returned function:
      log_spots: (batch, hedge_dates), log of the *absolute* stock prices.
      price_changes: (batch, hedge_dates), each next holding-period price change
        including the final period up to maturity, in original currency units.
      payoff: (batch,), positive liability Z, in original currency units.

    Both the six networks and eta receive gradients from the single terminal
    objective. Diagnostics are pre-update gradient norms and current batch loss;
    there is no gradient clipping, stop_gradient, per-step detached loss, premium
    scaling, or direct label for the holdings.
    """
    if not 0.0 <= alpha < 1.0:
        raise ValueError("alpha must satisfy 0 <= alpha < 1")
    model.initialize_without_bn_updates()
    variables = tuple(model.trainable_variables)
    net_variable_indices = [
        [i for i, variable in enumerate(variables)
         if any(variable is net_var for net_var in net.trainable_variables)]
        for net in model.nets
    ]
    if any(not indices for indices in net_variable_indices):
        raise RuntimeError("One or more decision networks have no registered weights")
    eta_indices = [i for i, variable in enumerate(variables) if variable is model.eta]
    if len(eta_indices) != 1:
        raise RuntimeError("The CVaR intercept must be one registered trainable weight")
    eta_index = eta_indices[0]
    if hasattr(optimizer, "build"):
        optimizer.build(list(variables))

    @tf.function(
        input_signature=[
            tf.TensorSpec([None, model.steps], tf.float32),
            tf.TensorSpec([None, model.steps], tf.float32),
            tf.TensorSpec([None], tf.float32),
        ],
        reduce_retracing=True,
    )
    def train_step(
        log_spots: tf.Tensor, price_changes: tf.Tensor, payoff: tf.Tensor
    ) -> dict[str, tf.Tensor]:
        tf.debugging.assert_equal(tf.shape(log_spots)[0], tf.shape(price_changes)[0])
        tf.debugging.assert_equal(tf.shape(log_spots)[0], tf.shape(payoff)[0])
        with tf.GradientTape() as tape:
            positions = model(log_spots, training=True)
            terminal_losses = payoff - tf.reduce_sum(positions * price_changes, axis=1)
            objective = cvar_objective(terminal_losses, model.eta, alpha)
            tf.debugging.check_numerics(objective, "Nonfinite CVaR objective")
        gradients = tape.gradient(objective, variables)
        if any(gradient is None for gradient in gradients):
            raise RuntimeError("A registered weight is disconnected from terminal CVaR")
        for gradient in gradients:
            tf.debugging.check_numerics(gradient, "Nonfinite CVaR gradient")
        network_gradient_norms = tf.stack([
            tf.linalg.global_norm([gradients[i] for i in indices])
            for indices in net_variable_indices
        ])
        eta_gradient = gradients[eta_index]
        optimizer.apply_gradients(zip(gradients, variables))
        return {
            "objective": objective,
            "mean_terminal_loss": tf.reduce_mean(terminal_losses),
            "network_gradient_norms": network_gradient_norms,
            "eta_gradient": eta_gradient,
            "eta_after_update": tf.identity(model.eta),
            "max_abs_position": tf.reduce_max(tf.abs(positions)),
        }

    return train_step


def predict_positions(
    model: PaperHedge, log_spots: np.ndarray, batch_size: int = 4096
) -> np.ndarray:
    """Inference-mode holdings, without modifying BatchNorm state."""
    data = np.asarray(log_spots, dtype=np.float32)
    if data.ndim != 2 or data.shape[1] != model.steps:
        raise ValueError(f"Expected (paths, {model.steps}) log prices")
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    if not len(data):
        return np.empty((0, model.steps), dtype=np.float32)
    return np.concatenate([
        model(tf.convert_to_tensor(data[i:i + batch_size]), training=False).numpy()
        for i in range(0, len(data), batch_size)
    ])


def compare_training_and_inference(
    model: PaperHedge, log_spots: np.ndarray, batch_size: int = 256
) -> dict[str, Any]:
    """Diagnostic only: compare modes, then restore every moving-state buffer.

    Training-mode outputs depend on companion paths in a batch and must never be
    used as the deployed hedging strategy. This function applies no optimizer
    update. It restores all non-trainable variables, even if the diagnostic fails,
    so it cannot silently recalibrate validation/test statistics into the model.
    Use training or validation simulation data, not historical test data.
    """
    if batch_size < 2:
        raise ValueError("Use at least two paths per diagnostic batch")
    model.initialize_without_bn_updates()
    data = np.asarray(log_spots, dtype=np.float32)
    if data.ndim != 2 or data.shape[1] != model.steps or not len(data):
        raise ValueError(f"Expected nonempty (paths, {model.steps}) log prices")
    buffers = tuple(model.non_trainable_variables)
    saved_buffers = [variable.numpy().copy() for variable in buffers]
    learned_variables = tuple(model.trainable_variables)
    saved_learned = [variable.numpy().copy() for variable in learned_variables]
    try:
        inference = predict_positions(model, data, batch_size)
        batch_mode = np.concatenate([
            model(tf.convert_to_tensor(data[i:i + batch_size]), training=True).numpy()
            for i in range(0, len(data), batch_size)
        ])
    finally:
        for variable, value in zip(buffers, saved_buffers):
            variable.assign(value)
    if any(not np.array_equal(variable.numpy(), value)
           for variable, value in zip(learned_variables, saved_learned)):
        raise RuntimeError("Mode diagnostic unexpectedly changed a learned weight")
    differences = np.abs(inference - batch_mode)
    return {
        "inference_positions": inference,
        "training_mode_positions": batch_mode,
        "mean_abs_difference_per_date": differences.mean(axis=0),
        "max_abs_difference_per_date": differences.max(axis=0),
        "non_trainable_state_restored": all(
            np.array_equal(variable.numpy(), value)
            for variable, value in zip(buffers, saved_buffers)
        ),
    }
