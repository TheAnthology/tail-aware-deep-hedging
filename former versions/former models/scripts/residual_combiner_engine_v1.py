"""Causal two-asset residual combiner and direct-MLP ablation.

The frozen preference ensembles are evaluated outside the trainable graph.
Only their within-preference mean positions enter the residual policy.  The
direct policy receives no expert values and has no expert anchor; its output
uses the same fixed training-derived currency/position scales for numerical
comparability.  Neither policy clips positions or adds a regularizer.
"""
import os
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')
from pathlib import Path
import tempfile
import time
import numpy as np
import tensorflow as tf
from heston52_market import empirical_cvar
from moe_gate_v1 import tail95_tf

KEYS = ['f', 'changes', 'hc', 'hp', 'liability', 'payoff']


def project(ensemble, market):
    """Current features, frozen mean positions, true two-asset increments.

    This calls the existing validated position-averaging implementation once;
    no averaging of parameters or reported risk measures is involved.
    """
    bundle, _, holdings = ensemble.project(market, individual=True)
    changes = np.stack([np.diff(market['spot'], axis=1),
                        np.diff(market['swap'], axis=1)], axis=-1)
    return {'f': np.asarray(bundle['f'], dtype='float32'),
            'changes': np.asarray(changes, dtype='float64'),
            'hc': np.asarray(holdings['C'], dtype='float64'),
            'hp': np.asarray(holdings['P'], dtype='float64'),
            'liability': np.asarray(bundle['liability'], dtype='float64'),
            'payoff': np.asarray(bundle['payoff'], dtype='float64')}


class ResidualCombiner(tf.keras.Model):
    """Shared date-independent 6/4 -> 32 -> 32 -> 2 tanh MLP.

    Residual: h_C + D F(z, (h_P-h_C)/D).
    Direct:   D F(z), with no h_C anchor or expert-output input.
    z uses the current actual-strategy surplus, before the current increment.
    """
    def __init__(self, mode, scales, budget, q, seed=6501):
        if mode not in ['residual', 'direct']:
            raise ValueError('mode must be residual or direct')
        super().__init__(name=f'{mode}_combiner', autocast=False)
        scales = np.asarray(scales, dtype='float64')
        if scales.shape != (2,) or not np.isfinite(scales).all() or np.any(scales <= 0):
            raise ValueError('Two finite strictly positive fixed position scales required')
        if not np.isfinite(budget) or budget <= 0 or not np.isfinite(q):
            raise ValueError('Finite premium and strictly positive risk budget required')
        self.mode = mode
        self.scales = scales.tolist()
        self.scale_tensor = tf.constant(scales, dtype=tf.float64)
        self.budget = float(budget)
        self.q = float(q)
        self.seed = int(seed)
        self.a = tf.keras.layers.Dense(32, activation='tanh', dtype='float32',
            kernel_initializer=tf.keras.initializers.GlorotUniform(seed=seed),
            bias_initializer='zeros')
        self.b = tf.keras.layers.Dense(32, activation='tanh', dtype='float32',
            kernel_initializer=tf.keras.initializers.GlorotUniform(seed=seed+1),
            bias_initializer='zeros')
        self.out = tf.keras.layers.Dense(2, dtype='float32',
            kernel_initializer='zeros', bias_initializer='zeros')
        self.a(tf.zeros([1, 6 if mode == 'residual' else 4], tf.float32))
        self.b(tf.zeros([1, 32], tf.float32))
        self.out(tf.zeros([1, 32], tf.float32))

    def call(self, inputs):
        f, changes, hc, hp, liability, payoff = inputs
        n = tf.shape(f)[0]
        gain = tf.zeros([n], tf.float64)
        positions, corrections, surpluses = [], [], []
        for k in range(30):
            surplus = self.q + gain - liability[:, k]
            feature = [(f[:, k, 0]-np.log(100.))/.2,
                       f[:, k, 1]/.04,
                       tf.cast(surplus/self.budget, tf.float32),
                       tf.fill([n], tf.cast(k/30., tf.float32))]
            z = tf.stack(feature, axis=1)
            if self.mode == 'residual':
                difference = tf.cast((hp[:, k, :]-hc[:, k, :])/self.scale_tensor, tf.float32)
                z = tf.concat([z, difference], axis=1)
            normalized_output = self.out(self.b(self.a(z)))
            output = self.scale_tensor*tf.cast(normalized_output, tf.float64)
            holding = hc[:, k, :]+output if self.mode == 'residual' else output
            correction = holding-hc[:, k, :]
            positions.append(holding)
            corrections.append(correction)
            surpluses.append(surplus)
            # Preserve graph through actual combined prior gains. Expert
            # positions are cached constants; no future increment enters z_k.
            gain = gain+tf.reduce_sum(holding*changes[:, k, :], axis=1)
        return (self.q+gain-payoff, tf.stack(positions, axis=1),
                tf.stack(corrections, axis=1), tf.stack(surpluses, axis=1))

    def initialize(self):
        self((tf.zeros([2,30,2], tf.float32), tf.zeros([2,30,2], tf.float64),
              tf.zeros([2,30,2], tf.float64), tf.zeros([2,30,2], tf.float64),
              tf.fill([2,30], tf.constant(self.q, tf.float64)), tf.zeros([2], tf.float64)))


def tensors(data, ix=None):
    return tuple(tf.convert_to_tensor(data[k] if ix is None else data[k][ix],
                 dtype=tf.float32 if k == 'f' else tf.float64) for k in KEYS)


def make_forward(model):
    @tf.function(reduce_retracing=True)
    def forward(f, changes, hc, hp, liability, payoff):
        return model((f, changes, hc, hp, liability, payoff))
    return forward


def evaluate(model, data, chunk=2048, forward=None):
    if forward is None:
        forward = make_forward(model)
    results = [[], [], [], []]
    for lo in range(0, len(data['payoff']), chunk):
        values = forward(*tensors(data, slice(lo, lo+chunk)))
        for target, value in zip(results, values):
            target.append(value.numpy())
    return tuple(np.concatenate(v) for v in results)


def make_step(model):
    """Exactly the earlier scalar Gate's objective/controller and optimizer.

    Returned scalar6 is diagnostic normalized correction RMS only; it is not
    added to the objective. No position clipping or extra regularization.
    """
    optimizer = tf.keras.optimizers.Adam(.001)
    dual = tf.Variable(1., trainable=False, dtype=tf.float64)
    @tf.function(reduce_retracing=True)
    def step(f, changes, hc, hp, liability, payoff):
        with tf.GradientTape() as tape:
            pl, holding, correction, surplus = model((f, changes, hc, hp, liability, payoff))
            soft = tf.reduce_mean(tf.sigmoid(-pl/.1))
            risk = tail95_tf(-pl)
            g = risk/model.budget-1
            objective = soft+tf.stop_gradient(dual)*g+5*tf.nn.relu(g)**2
        gradients = tape.gradient(objective, model.trainable_variables)
        if any(grad is None for grad in gradients):
            raise RuntimeError('Disconnected combiner parameter')
        tf.debugging.check_numerics(objective, 'combiner objective')
        for grad in gradients:
            tf.debugging.check_numerics(grad, 'combiner gradient')
        clipped, norm = tf.clip_by_global_norm(gradients, 10.)
        optimizer.apply_gradients(zip(clipped, model.trainable_variables))
        dual.assign(tf.maximum(tf.constant(0., tf.float64), dual+.05*tf.stop_gradient(g)))
        correction_rms = tf.sqrt(tf.reduce_mean((correction/model.scale_tensor)**2))
        return soft, risk, g, dual, tf.cast(norm, tf.float64), correction_rms
    return step, optimizer, dual


def _mock_data(n=64, seed=6591):
    rng = np.random.default_rng(seed)
    hc = rng.normal(size=(n,30,2))*np.array([.3,100.])
    return {'f': np.stack([np.log(100.)+rng.normal(0,.02,(n,30)),
                          rng.uniform(0,.1,(n,30))], axis=-1).astype('float32'),
            'changes': rng.normal(size=(n,30,2))*np.array([.1,.0002]),
            'hc': hc, 'hp': hc+rng.normal(size=(n,30,2))*np.array([.5,150.]),
            'liability': rng.uniform(0,2,(n,30)), 'payoff': rng.uniform(0,3,n)}


def checks(smoke_steps=100):
    """Small synthetic integration checks, never training/test evidence."""
    q = 1.691833925110384; B = 1.238727; scales = [.5, 300.]
    data = _mock_data(); rng = np.random.default_rng(6592)
    residual = ResidualCombiner('residual', scales, B, q)
    direct = ResidualCombiner('direct', scales, B, q)
    base = residual(tensors(data))
    expected = q+np.sum(data['hc']*data['changes'], axis=(1,2))-data['payoff']
    assert np.array_equal(base[1].numpy(), data['hc'])
    assert np.array_equal(base[2].numpy(), np.zeros_like(data['hc']))
    assert np.allclose(base[0].numpy(), expected, rtol=1e-13, atol=1e-13)
    assert np.allclose(base[3].numpy()[:,0], q-data['liability'][:,0], atol=1e-13)
    first_gain = np.sum(data['hc'][:,0]*data['changes'][:,0], axis=1)
    assert np.allclose(base[3].numpy()[:,1], q+first_gain-data['liability'][:,1], atol=1e-13)
    # Zero direct initialization equals the zero hedge, not the CVaR anchor.
    d0 = direct(tensors(data))
    assert np.array_equal(d0[1].numpy(), np.zeros_like(data['hc']))
    assert np.array_equal(d0[0].numpy(), q-data['payoff'])
    # D output scaling/inverse scaling are asset-specific and literal.
    residual.out.bias.assign([.2,-.3])
    scaled = residual(tensors(data))
    correction = scales*np.array([.2,-.3], dtype='float32').astype('float64')
    assert np.allclose(scaled[2].numpy(), np.broadcast_to(correction,data['hc'].shape), atol=1e-13)
    assert np.allclose(scaled[2].numpy()/scales, [.2,-.3], atol=2e-8)
    # Nonconstant output makes causal and expert-input checks substantive.
    residual.out.kernel.assign(rng.normal(0,.03,residual.out.kernel.shape))
    direct.out.kernel.assign(rng.normal(0,.03,direct.out.kernel.shape))
    before = residual(tensors(data))
    altered = {k:v.copy() for k,v in data.items()}
    altered['changes'][:,10:] += np.array([2.,.1])
    altered['payoff'] += 999.
    altered['f'][:,11:,0] += .4
    altered['hc'][:,11:] += np.array([2.,500.])
    altered['hp'][:,11:] -= np.array([3.,900.])
    altered['liability'][:,11:] += 10.
    changed = residual(tensors(altered))
    assert np.array_equal(before[1].numpy()[:,:11],changed[1].numpy()[:,:11])
    assert np.array_equal(before[3].numpy()[:,:11],changed[3].numpy()[:,:11])
    # A direct policy's decisions, gains, and surplus do not consume experts.
    changed_experts = {k:v.copy() for k,v in data.items()}
    changed_experts['hc'] += np.array([100.,10000.])
    changed_experts['hp'] -= np.array([200.,20000.])
    db = direct(tensors(data)); dc = direct(tensors(changed_experts))
    for i in [0,1,3]:assert np.array_equal(db[i].numpy(),dc[i].numpy())
    # Explicit two-instrument accounting against the rollout's holdings.
    recalc = q+np.sum(before[1].numpy()*data['changes'],axis=(1,2))-data['payoff']
    assert np.allclose(before[0].numpy(),recalc,rtol=1e-12,atol=1e-12)
    # Finite differences include the entire path through the surplus state.
    with tf.GradientTape() as tape:
        pl = residual(tensors(data))[0]
        value = tf.reduce_mean(pl*pl)
    grads = tape.gradient(value,residual.trainable_variables)
    idx = next(i for i,v in enumerate(residual.trainable_variables) if v is residual.out.bias)
    original = residual.out.bias.numpy().copy(); eps=.001
    finite=[]
    for asset in [0,1]:
        offset=np.zeros(2,dtype='float32');offset[asset]=eps
        residual.out.bias.assign(original+offset)
        plus=float(tf.reduce_mean(residual(tensors(data))[0]**2))
        residual.out.bias.assign(original-offset)
        minus=float(tf.reduce_mean(residual(tensors(data))[0]**2))
        finite.append((plus-minus)/(2*eps))
    residual.out.bias.assign(original)
    assert np.allclose(grads[idx].numpy(),finite,rtol=.015,atol=.002)
    x=rng.normal(size=1024)
    assert np.isclose(float(tail95_tf(tf.constant(x))),empirical_cvar(x,.95),rtol=1e-12)
    # Quantile loss gradient agrees with its weighted active tail away from ties.
    xx=tf.constant(x)
    with tf.GradientTape() as tape:
        tape.watch(xx); cv=tail95_tf(xx)
    grad=tape.gradient(cv,xx).numpy();order=np.argsort(x)[::-1]
    reference=np.zeros_like(x);reference[order[:51]]=1/51.2;reference[order[51]]=.2/51.2
    assert np.allclose(grad,reference,atol=1e-13)
    # Small trained save/load smoke, keeping only temporary files.
    smoke={}
    for mode in ['residual','direct']:
        model=ResidualCombiner(mode,scales,B,q,seed=6501)
        step,optimizer,dual=make_step(model);start=time.monotonic()
        for _ in range(smoke_steps):values=step(*tensors(data))
        assert int(optimizer.iterations.numpy())==smoke_steps
        assert all(np.isfinite(float(v.numpy())) for v in values)
        outputs=model(tensors(data))
        with tempfile.TemporaryDirectory(prefix='residual_combiner_check_') as directory:
            path=Path(directory)/'roundtrip.weights.h5'
            model.save_weights(path)
            restored=ResidualCombiner(mode,scales,B,q,seed=6501);restored.initialize()
            restored.load_weights(path)
            check=restored(tensors(data))
            assert all(np.array_equal(a.numpy(),b.numpy()) for a,b in zip(outputs,check))
        smoke[mode]={'steps':smoke_steps,'seconds':time.monotonic()-start,
                     'finite_last_scalars':[float(v.numpy()) for v in values],
                     'parameter_scalars':sum(int(np.prod(v.shape)) for v in model.trainable_variables)}
    return {'zero_residual_exact_C_positions':True,'zero_direct_is_zero_hedge':True,
            'direct_policy_independent_of_expert_outputs':True,
            'asset_specific_D_inverse_and_output_scales':True,
            'causal_current_surplus_and_future_mutation_checks':True,
            'two_asset_terminal_PnL_identity':True,
            'full_surplus_recurrence_gradient_finite_difference':True,
            'fractional_CVaR95_value_and_gradient':True,
            'save_load_bitwise_output_roundtrip':True,
            'no_added_regularization_or_position_clipping':True,
            'mock_smoke_only':smoke}


if __name__=='__main__':
    import json
    tf.config.threading.set_intra_op_parallelism_threads(1)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    print(json.dumps(checks(),indent=2),flush=True)
