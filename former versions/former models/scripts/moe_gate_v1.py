"""Frozen preference ensembles and causal differentiable surplus gating."""
import os
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL','3')
from pathlib import Path
import json
import numpy as np
import tensorflow as tf
from scipy.special import expit
from scipy.optimize import brentq, minimize_scalar
from tf_heston52_engine import Heston52Hedge,predict
from heston52_market import sample_tensors,empirical_cvar,pnl
from moe_liability_v1 import HestonLiability
from run_risk_ablation_v1 import summarize

ROOT=Path(__file__).resolve().parents[1]

def stats(pl):
    r=summarize(pl)
    r['probability_profit_gt_1']=float(np.mean(pl>1))
    r['pnl_standard_deviation']=float(np.std(pl))
    return r

class FrozenEnsembles:
    def __init__(self,manifest):
        self.models={}
        for pref in ['C','P']:
            for row in manifest['preferences'][pref]:
                m=Heston52Hedge(recurrent=False);m.initialize()
                m.load_weights(row['calibrated_weights']);m.trainable=False
                self.models[f'{pref}_{row["member"]}']=m
        self.liability=HestonLiability();self.q=self.liability.q

    def project(self,market,individual=False):
        f,_,_=sample_tensors(market)
        changes=np.stack([np.diff(market['spot'],axis=1),np.diff(market['swap'],axis=1)],axis=-1)
        hold={};singles={};ens_hold={}
        for pref in ['C','P']:
            members=[]
            for i in [1,2,3]:
                name=f'{pref}_{i}';h=predict(self.models[name],f)
                members.append(h.astype('float64'))
                if individual:singles[name]=pnl(market,h,self.q)
            # Average positions strictly within preference, before Gate.
            h=np.mean(members,axis=0);ens_hold[pref]=h
            hold[pref]=np.sum(h*changes,axis=-1)
            if individual:
                direct=pnl(market,h,self.q)
                identity=self.q+hold[pref].sum(axis=1)-np.maximum(market['spot'][:,-1]-100,0)
                assert np.allclose(direct,identity,atol=1e-10,rtol=1e-10)
                assert np.allclose(direct,np.mean([singles[f'{pref}_{i}'] for i in [1,2,3]],axis=0),atol=1e-9)
        bundle={'f':f,'gc':hold['C'],'gp':hold['P'],'liability':self.liability.price(market),
                'payoff':np.maximum(market['spot'][:,-1]-100,0)}
        if individual:return bundle,singles,ens_hold
        return bundle

def constant_pnl(data,q,w):
    return q+((1-w)*data['gc']+w*data['gp']).sum(axis=1)-data['payoff']

def choose_constant(data,q,budget):
    c=constant_pnl(data,q,0);p=constant_pnl(data,q,1)
    lc=-c;lp=-p
    mass=len(lc)*.05;whole=int(np.floor(mass));fraction=mass-whole
    order=np.argsort(lc)[::-1];direction=lp-lc
    derivative=float((direction[order[:whole]].sum()+fraction*direction[order[whole]])/mass)
    def risk(w):return empirical_cvar((1-w)*lc+w*lp,.95)
    def objective(w):return float(expit(((1-w)*lc+w*lp)/.1).mean())
    # Risk convexity gives interval feasibility. Under positive initial slope,
    # and a budget exactly equal to risk(0), only zero is feasible.
    if derivative>0 and np.isclose(budget,risk(0),atol=1e-12,rtol=1e-12):
        return {'weight':0.,'feasible_upper_weight':0.,'tail_right_derivative_at_zero':derivative,
                'soft_objective':objective(0),'validation':stats(c),
                'method':'Convex empirical CVaR, positive right derivative at0; only w0 feasible',
                'feasibility_tolerance':'Floating arithmetic only, no economic relaxation'}
    if risk(1)<=budget:upper=1.
    else:
        lo,hi=0.,1.
        for _ in range(60):
            mid=(lo+hi)/2
            if risk(mid)<=budget:lo=mid
            else:hi=mid
        upper=lo
    grid=np.linspace(0,upper,1001);values=np.array([objective(w) for w in grid])
    choices=[(float(values[i]),float(grid[i])) for i in np.argsort(values)[:5]]
    for i in np.argsort(values)[:5]:
        lo=grid[max(i-1,0)];hi=grid[min(i+1,1000)]
        if hi>lo:
            opt=minimize_scalar(objective,bounds=(lo,hi),method='bounded',options={'xatol':1e-12})
            if risk(opt.x)<=budget:choices.append((float(opt.fun),float(opt.x)))
    value,w=min(choices)
    return {'weight':w,'feasible_upper_weight':upper,'tail_right_derivative_at_zero':derivative,
            'soft_objective':value,'validation':stats(constant_pnl(data,q,w)),
            'method':'1001-point feasible interval grid plus5localboundedrefinements; no global soft-objective certificate'}

class PreferenceGate(tf.keras.Model):
    def __init__(self,surplus,budget,q,seed=6501,initial_weight=1e-6):
        super().__init__(name='surplus_gate' if surplus else 'market_gate',autocast=False)
        self.surplus=bool(surplus);self.budget=float(budget);self.q=float(q)
        init=float(np.clip(initial_weight,1e-6,1-1e-6))
        self.a=tf.keras.layers.Dense(16,activation='tanh',kernel_initializer=tf.keras.initializers.GlorotUniform(seed=seed))
        self.b=tf.keras.layers.Dense(16,activation='tanh',kernel_initializer=tf.keras.initializers.GlorotUniform(seed=seed+1))
        self.out=tf.keras.layers.Dense(1,kernel_initializer='zeros',bias_initializer=tf.keras.initializers.Constant(np.log(init/(1-init))))
        self.a(tf.zeros([1,4 if surplus else 3]));self.b(tf.zeros([1,16]));self.out(tf.zeros([1,16]))

    def call(self,inputs):
        f,gc,gp,liability,payoff=inputs
        n=tf.shape(f)[0];gain=tf.zeros([n],tf.float64)
        weights=[];surpluses=[]
        for k in range(30):
            # All features precede this date's gain; preserve gradient through
            # actual combined prior gains for the surplus-aware policy.
            surplus=self.q+gain-liability[:,k]
            feature=[(f[:,k,0]-np.log(100.))/.2,f[:,k,1]/.04,
                     tf.fill([n],tf.cast(k/30.,tf.float32))]
            if self.surplus:feature.append(tf.cast(surplus/self.budget,tf.float32))
            x=tf.stack(feature,axis=1)
            w=tf.cast(tf.sigmoid(tf.squeeze(self.out(self.b(self.a(x))),axis=-1)),tf.float64)
            weights.append(w);surpluses.append(surplus)
            gain=gain+(1-w)*gc[:,k]+w*gp[:,k]
        return self.q+gain-payoff,tf.stack(weights,axis=1),tf.stack(surpluses,axis=1)

    def initialize(self):
        self((tf.zeros([2,30,2],tf.float32),tf.zeros([2,30],tf.float64),
              tf.zeros([2,30],tf.float64),tf.fill([2,30],tf.constant(self.q,tf.float64)),
              tf.zeros([2],tf.float64)))

def tensors(data,ix=None):
    return tuple(tf.convert_to_tensor(data[k] if ix is None else data[k][ix],
                 dtype=tf.float32 if k=='f' else tf.float64) for k in ['f','gc','gp','liability','payoff'])

def make_predict(gate):
    @tf.function(reduce_retracing=True)
    def forward(f,gc,gp,liability,payoff):return gate((f,gc,gp,liability,payoff))
    return forward

def evaluate_gate(gate,data,chunk=2048,forward=None):
    if forward is None:forward=make_predict(gate)
    pl=[];weights=[];surplus=[]
    for lo in range(0,len(data['payoff']),chunk):
        values=forward(*tensors(data,slice(lo,lo+chunk)))
        for dest,v in zip([pl,weights,surplus],values):dest.append(v.numpy())
    return np.concatenate(pl),np.concatenate(weights),np.concatenate(surplus)

def tail95_tf(loss):
    n=tf.size(loss);mass=tf.cast(n,tf.float64)*.05
    whole=tf.cast(tf.floor(mass),tf.int32);frac=mass-tf.cast(whole,tf.float64)
    values=tf.sort(loss,direction='DESCENDING')
    return (tf.reduce_sum(values[:whole])+frac*values[whole])/mass

def make_train_step(gate):
    optimizer=tf.keras.optimizers.Adam(.001)
    dual=tf.Variable(1.,trainable=False,dtype=tf.float64)
    @tf.function(reduce_retracing=True)
    def step(f,gc,gp,liability,payoff):
        with tf.GradientTape() as tape:
            pl,w,r=gate((f,gc,gp,liability,payoff))
            soft=tf.reduce_mean(tf.sigmoid(-pl/.1));risk=tail95_tf(-pl)
            g=risk/gate.budget-1
            objective=soft+tf.stop_gradient(dual)*g+5*tf.nn.relu(g)**2
        grads=tape.gradient(objective,gate.trainable_variables)
        if any(v is None for v in grads):raise RuntimeError('Disconnected gate parameter')
        tf.debugging.check_numerics(objective,'gate objective')
        for grad in grads:tf.debugging.check_numerics(grad,'gate gradient')
        clipped,norm=tf.clip_by_global_norm(grads,10.)
        optimizer.apply_gradients(zip(clipped,gate.trainable_variables))
        dual.assign(tf.maximum(tf.constant(0.,tf.float64),dual+.05*tf.stop_gradient(g)))
        return soft,risk,g,dual,tf.cast(norm,tf.float64),tf.reduce_mean(w)
    return step,optimizer,dual

def integrity_checks():
    rng=np.random.default_rng(6491);n=32;q=1.691833925110384
    data={'f':np.stack([np.log(100.)+rng.normal(0,.02,(n,30)),rng.uniform(0,.1,(n,30))],axis=-1).astype('float32'),
          'gc':rng.normal(0,.1,(n,30)),'gp':rng.normal(0,2,(n,30)),
          'liability':rng.uniform(0,3,(n,30)),'payoff':rng.uniform(0,3,n)}
    # Uniform gate identity and exact trading recursion.
    gate=PreferenceGate(True,1.2,q,initial_weight=.3)
    pl,w,r=gate(tensors(data));expected=constant_pnl(data,q,.3)
    assert np.allclose(pl.numpy(),expected,atol=2e-6)
    assert np.allclose(r.numpy()[:,0],q-data['liability'][:,0],atol=1e-12)
    assert np.allclose(r.numpy()[:,1],q+.7*data['gc'][:,0]+.3*data['gp'][:,0]-data['liability'][:,1],atol=2e-6)
    # Nonconstant gate makes causality check substantive.
    gate.out.kernel.assign(rng.normal(0,.1,gate.out.kernel.shape))
    base=gate(tensors(data));alter={k:v.copy() for k,v in data.items()}
    alter['gc'][:,10:]+=100;alter['gp'][:,10:]-=100;alter['payoff']+=999
    alter['f'][:,11:,0]+=.2;alter['liability'][:,11:]+=30
    changed=gate(tensors(alter))
    assert np.array_equal(base[1].numpy()[:,:11],changed[1].numpy()[:,:11])
    # Finite-difference one gate coefficient checks full recurrent surplus graph.
    with tf.GradientTape() as tape:
        out=gate(tensors(data))[0];objective=tf.reduce_mean(out*out)
    gradients=tape.gradient(objective,gate.trainable_variables)
    variable=gate.out.bias;original=variable.numpy().copy();eps=.001
    variable.assign(original+eps);plus=float(tf.reduce_mean(gate(tensors(data))[0]**2))
    variable.assign(original-eps);minus=float(tf.reduce_mean(gate(tensors(data))[0]**2))
    variable.assign(original)
    idx=next(i for i,v in enumerate(gate.trainable_variables) if v is variable)
    assert np.isclose(float(gradients[idx][0]),(plus-minus)/(2*eps),rtol=.02,atol=.002)
    x=rng.normal(size=1024)
    assert np.isclose(float(tail95_tf(tf.constant(x))),empirical_cvar(x,.95),rtol=1e-12)
    return {'constant_gate_matches_fixed_positions':True,'gain_precedes_next_surplus':True,
            'routing_causal_under_future_gain_payoff_state_mutation':True,
            'surplus_recurrence_gradient_matches_finite_difference':True,
            'fractional_empirical_CVaR95_matches_numpy':True,'experts_precomputed_frozen':True}
