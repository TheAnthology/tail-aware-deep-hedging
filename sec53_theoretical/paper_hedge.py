"""Semi-recurrent Heston comparison from Deep Hedging, Sections 5.1 and 5.2."""

import tensorflow as tf

from .experts import CVaRExpert


class PaperDeepHedge(CVaRExpert):
    """At each date, use log spot, variance and the previous stock/swap holdings."""

    def call(self, features, training=False):
        features = tf.ensure_shape(
            tf.convert_to_tensor(features, tf.float32), [None, 30, 2]
        )
        previous = tf.zeros_like(features[:, 0, :])
        holdings = []
        for date, network in enumerate(self.nets):
            inputs = tf.concat([features[:, date, :], previous], axis=1)
            previous = network(inputs, training=training)
            holdings.append(previous)
        return tf.stack(holdings, axis=1)
