"""Shared loss metrics; CVaR is computed on loss=-P&L."""

import numpy as np
import tensorflow as tf
from scipy.special import expit


def empirical_cvar(loss, alpha):
    """Average the worst (1 - alpha) loss mass, including a fractional boundary."""
    if not 0 <= alpha < 1:
        raise ValueError("alpha must lie in [0,1)")
    sorted_losses = np.sort(np.asarray(loss, dtype=np.float64))[::-1]
    if sorted_losses.ndim != 1 or not len(sorted_losses):
        raise ValueError("Expected nonempty loss vector")
    # A nonintegral tail uses only the required fraction of its last loss.
    tail_mass = len(sorted_losses) * (1 - alpha)
    whole_count = int(np.floor(tail_mass))
    boundary_weight = tail_mass - whole_count
    tail_sum = sorted_losses[:whole_count].sum() + (
        boundary_weight * sorted_losses[whole_count] if boundary_weight > 1e-10 else 0.0
    )
    return float(tail_sum / tail_mass)


def tail95_tf(loss):
    """TensorFlow version of the fractional worst-5% loss average for training."""
    sorted_losses = tf.sort(loss, direction="DESCENDING")
    tail_mass = tf.cast(tf.size(loss), loss.dtype) * 0.05
    whole_count = tf.cast(tf.floor(tail_mass), tf.int32)
    return (
        tf.reduce_sum(sorted_losses[:whole_count])
        + (tail_mass - tf.cast(whole_count, loss.dtype)) * sorted_losses[whole_count]
    ) / tail_mass


def metrics(pl):
    pl = np.asarray(pl, dtype="float64")
    if pl.ndim != 1 or not len(pl) or not np.isfinite(pl).all():
        raise ValueError("Finite nonempty P&L vector required")
    return {
        "paths": len(pl),
        "mean_pnl": float(pl.mean()),
        "mse": float(np.mean(pl**2)),
        "probability_loss": float(np.mean(pl < 0)),
        "soft_probability": float(expit(-pl / 0.1).mean()),
        **{f"cvar{int(100*a)}": empirical_cvar(-pl, a) for a in [0.5, 0.75, 0.95]},
        "pnl_quantiles": {
            str(a): float(np.quantile(pl, a))
            for a in [0, 0.01, 0.05, 0.5, 0.95, 0.99, 1]
        },
    }
