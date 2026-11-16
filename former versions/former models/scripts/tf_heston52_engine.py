"""Section 5.2 TensorFlow policy: 30 independent 4->17->17->2 networks.

All dates, including date zero, retain the paper network. BatchNorm is before
each ReLU, with standard training/inference behavior; no statistics repair or
initial scalar is introduced. Exact numerical initializer/BN settings are
declared conventions because the paper does not supply them.
"""
import numpy as np
import tensorflow as tf


class Heston52Hedge(tf.keras.Model):
    def __init__(self,seed=5201,recurrent=True,bn_epsilon=.001,bn_momentum=.99):
        super().__init__(name='heston52_hedge',dtype='float32')
        self.recurrent=bool(recurrent)
        self.nets=[]
        for k in range(30):
            def dense(n,index):
                return tf.keras.layers.Dense(n,kernel_initializer=tf.keras.initializers.GlorotUniform(seed=seed+100*k+index),
                                             bias_initializer='zeros',dtype='float32',name=f'dense_{index}')
            def bn(index):
                return tf.keras.layers.BatchNormalization(epsilon=bn_epsilon,momentum=bn_momentum,
                                                          dtype='float32',name=f'bn_{index}')
            self.nets.append(tf.keras.Sequential([dense(17,0),bn(0),tf.keras.layers.ReLU(),
                                                  dense(17,1),bn(1),tf.keras.layers.ReLU(),dense(2,2)],name=f'date_{k}'))
        self.eta=self.add_weight(name='cvar_intercept',shape=(),initializer='zeros',trainable=True,dtype='float32')

    def call(self,features,training=False):
        features=tf.ensure_shape(tf.convert_to_tensor(features,dtype=tf.float32),[None,30,2])
        previous=tf.zeros_like(features[:,0,:])
        holdings=[]
        for k,net in enumerate(self.nets):
            inputs=tf.concat([features[:,k,:],previous],axis=1) if self.recurrent else features[:,k,:]
            previous=net(inputs,training=training)
            holdings.append(previous)
        return tf.stack(holdings,axis=1)

    def initialize(self):
        self(tf.zeros((2,30,2)),training=False)


def training_step(model,learning_rate=.005,alpha=.5):
    model.initialize()
    variables=tuple(model.trainable_variables)
    indices=[[i for i,v in enumerate(variables) if any(v is x for x in net.trainable_variables)] for net in model.nets]
    assert all(indices) and any(v is model.eta for v in variables)
    optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate,beta_1=.9,beta_2=.999,epsilon=1e-7)
    optimizer.build(list(variables))

    @tf.function(input_signature=[tf.TensorSpec([None,30,2],tf.float32),tf.TensorSpec([None,30,2],tf.float32),tf.TensorSpec([None],tf.float32)])
    def step(features,changes,payoff):
        with tf.GradientTape() as tape:
            holding=model(features,training=True)
            loss=payoff-tf.reduce_sum(holding*changes,axis=[1,2])
            objective=model.eta+tf.reduce_mean(tf.nn.relu(loss-model.eta))/(1-alpha)
        gradients=tape.gradient(objective,variables)
        if any(g is None for g in gradients):raise RuntimeError('Disconnected policy parameter')
        tf.debugging.check_numerics(objective,'Nonfinite OCE')
        for g in gradients:tf.debugging.check_numerics(g,'Nonfinite gradient')
        norms=tf.stack([tf.linalg.global_norm([gradients[i] for i in ix]) for ix in indices])
        optimizer.apply_gradients(zip(gradients,variables))
        return {'objective':objective,'network_gradient_norms':norms,'eta':tf.identity(model.eta),
                'max_abs_holdings':tf.reduce_max(tf.abs(holding),axis=[0,1])}
    return step


def predict(model,features,chunk=2048):
    return np.concatenate([model(features[i:i+chunk],training=False).numpy() for i in range(0,len(features),chunk)])


def mode_diagnostic(model,features,batch=256):
    """Only a diagnostic on simulation data; restore all BN buffers afterwards."""
    buffers=list(model.non_trainable_variables);saved=[v.numpy().copy() for v in buffers]
    try:
        inference=predict(model,features,chunk=batch)
        train=np.concatenate([model(features[i:i+batch],training=True).numpy() for i in range(0,len(features),batch)])
    finally:
        for v,x in zip(buffers,saved):v.assign(x)
    assert all(np.array_equal(v.numpy(),x) for v,x in zip(buffers,saved))
    return inference,train,{'mean_abs_difference_stock':float(np.abs(inference[:,:,0]-train[:,:,0]).mean()),
                           'mean_abs_difference_swap':float(np.abs(inference[:,:,1]-train[:,:,1]).mean()),
                           'first_input_holding_dimension':4 if model.recurrent else 2,
                           'bn_buffers_restored':True}
