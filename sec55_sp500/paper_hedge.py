"""Semi-recurrent S&P 500 paper comparison in original index units."""

import numpy as np
import tensorflow as tf

from .models import CVaRExpert, HEDGE_DAYS


def paper_features(normalized_prices, original_spot):
    """Restore the absolute log index level used by this checkpoint."""
    levels = normalized_prices[:, HEDGE_DAYS] * (original_spot / 100.0)
    return np.log(levels).astype("float32")


class PaperDeepHedge(CVaRExpert):
    """Each date network reads log index level and the previous stock holding."""

    def initialize(self):
        self(tf.zeros([2, 6], tf.float32), training=False)

    def call(self, log_spots, training=False):
        log_spots = tf.ensure_shape(
            tf.convert_to_tensor(log_spots, tf.float32), [None, 6]
        )
        previous = tf.zeros_like(log_spots[:, :1])
        holdings = []
        for date, network in enumerate(self.nets):
            inputs = tf.concat([log_spots[:, date : date + 1], previous], axis=1)
            previous = network(inputs, training=training)
            holdings.append(previous)
        return tf.concat(holdings, axis=1)
