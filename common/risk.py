"""Shared loss metrics; CVaR is computed on loss=-P&L."""
import numpy as np
import tensorflow as tf
from scipy.special import expit

def empirical_cvar(loss, alpha):
    if not 0 <= alpha < 1:
        raise ValueError('alpha must lie in [0,1)')
    x = np.sort(np.asarray(loss,dtype=np.float64))[::-1]
    if x.ndim != 1 or not len(x):
        raise ValueError('Expected nonempty loss vector')
    mass = len(x)*(1-alpha)
    whole = int(np.floor(mass));fraction = mass-whole
    total = x[:whole].sum()+(fraction*x[whole] if fraction > 1e-10 else 0.)
    return float(total/mass)

def tail95_tf(loss):
    values = tf.sort(loss, direction='DESCENDING')
    mass = tf.cast(tf.size(loss), loss.dtype) * .05
    whole = tf.cast(tf.floor(mass), tf.int32)
    return (tf.reduce_sum(values[:whole]) + (mass-tf.cast(whole, loss.dtype))*values[whole])/mass

def metrics(pl):
    pl = np.asarray(pl, dtype='float64')
    if pl.ndim != 1 or not len(pl) or not np.isfinite(pl).all():
        raise ValueError('Finite nonempty P&L vector required')
    return {'paths': len(pl), 'mean_pnl': float(pl.mean()),
            'mse': float(np.mean(pl**2)), 'probability_loss': float(np.mean(pl<0)),
            'soft_probability': float(expit(-pl/.1).mean()),
            **{f'cvar{int(100*a)}': empirical_cvar(-pl,a) for a in [.5,.75,.95]},
            'pnl_quantiles': {str(a): float(np.quantile(pl,a)) for a in [0,.01,.05,.5,.95,.99,1]}}
