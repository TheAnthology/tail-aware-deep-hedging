"""Simpler 30-date CVaR50 / Probability expert, with independent BN calibration."""
import numpy as np
import tensorflow as tf

class CVaRExpert(tf.keras.Model):
    def __init__(self,seed=5201,bn_epsilon=.001,bn_momentum=.99):
        super().__init__(name='heston52_hedge',dtype='float32')
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
            inputs=features[:,k,:]
            previous=net(inputs,training=training)
            holdings.append(previous)
        return tf.stack(holdings,axis=1)

    def initialize(self):
        self(tf.zeros((2,30,2)),training=False)

def predict(model,features,chunk=2048):
    return np.concatenate([model(features[i:i+chunk],training=False).numpy() for i in range(0,len(features),chunk)])
def calibrate(model,features):
    previous=np.zeros((len(features),2),dtype='float32'); records=[]
    for k,net in enumerate(model.nets):
        x=features[:,k,:]
        record={'date':k}
        for di,bi,ri in [(0,1,2),(3,4,5)]:
            z=np.concatenate([net.layers[di](x[i:i+256]).numpy() for i in range(0,len(x),256)])
            mu=z.astype('float64').mean(0);var=z.astype('float64').var(0)
            net.layers[bi].moving_mean.assign(mu);net.layers[bi].moving_variance.assign(var)
            x=net.layers[ri](net.layers[bi](z,training=False))
            record[str(bi)]={'mean':mu.tolist(),'variance':var.tolist()}
        previous=net.layers[6](x).numpy();records.append(record)
    return records
def make_step(model,name,q):
 if name not in ['C','P']: raise ValueError('Expected C or P preference')
 # Eta is an OCE auxiliary variable; frequency-only has no eta to optimize.
 variables=tuple(v for v in model.trainable_variables if name!='P' or v is not model.eta)
 optimizer=tf.keras.optimizers.Adam(.005,beta_1=.9,beta_2=.999,epsilon=1e-7);optimizer.build(list(variables))
 indices=[[i for i,v in enumerate(variables) if any(v is x for x in net.trainable_variables)] for net in model.nets]
 @tf.function(input_signature=[tf.TensorSpec([256,30,2],tf.float32),tf.TensorSpec([256,30,2],tf.float32),tf.TensorSpec([256],tf.float32)])
 def step(f,changes,payoff):
  with tf.GradientTape() as tape:
   h=model(f,training=True)
   # Keep the original currency OCE intercept convention so A can reproduce
   # the old baseline exactly; eta_loss = eta_training-q.
   liability_loss=payoff-tf.reduce_sum(h*changes,axis=[1,2]);loss=liability_loss-q
   oce=model.eta+2*tf.reduce_mean(tf.nn.relu(liability_loss-model.eta))-q
   soft=tf.reduce_mean(tf.math.sigmoid(loss/.1))
   objective=oce if name=='C' else soft
  gradients=tape.gradient(objective,variables)
  if any(g is None for g in gradients):raise RuntimeError('Disconnected gradient')
  tf.debugging.check_numerics(objective,'objective')
  for g in gradients:tf.debugging.check_numerics(g,'gradient')
  norms=tf.stack([tf.linalg.global_norm([gradients[i] for i in ix]) for ix in indices])
  empirical=tf.reduce_mean(tf.sort(loss)[128:]);hard=tf.reduce_mean(tf.cast(loss>0,tf.float32))
  optimizer.apply_gradients(zip(gradients,variables))
  return objective,oce,soft,empirical,hard,tf.linalg.global_norm(gradients),norms
 return step,optimizer
