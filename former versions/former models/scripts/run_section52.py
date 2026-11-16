"""Published Section 5.2 baseline reconstruction; no extension objective."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL','2')
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('OMP_NUM_THREADS','1')
os.environ.setdefault('PYTHONDONTWRITEBYTECODE','1')
import numpy as np
import tensorflow as tf

from heston52_market import (PARAMETERS,sample_paths,path_chunks,sample_tensors,pnl,metrics,
                             empirical_cvar,simulation_diagnostics,self_checks)
from tf_heston52_engine import Heston52Hedge,training_step,predict,mode_diagnostic
from heston52_benchmark import HestonGridHedge,QuantLibReference

ROOT=Path(__file__).resolve().parents[1]


def dump(path,value):path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def engine_checks():
    result=self_checks()
    data=sample_paths(256,np.random.default_rng(5233))
    features,changes,payoff=sample_tensors(data)
    model=Heston52Hedge();model.initialize()
    # Capture actual t0 network input (including previous two holdings).
    captured=[];original_call=model.nets[0].call
    def capture(inputs,*args,**kwargs):
        captured.append(inputs.numpy().copy())
        return original_call(inputs,*args,**kwargs)
    model.nets[0].call=capture
    positions=model(features,training=False).numpy()
    model.nets[0].call=original_call
    expected=np.array([np.log(100.),.04,0.,0.],dtype='float32')
    assert np.array_equal(captured[-1],np.tile(expected,(256,1)))
    assert positions.shape==(256,30,2)
    # Probe both traded-asset branches at final date, away from the hinge.
    with tf.GradientTape() as tape:
        h=model(features,training=False)
        loss=tf.constant(np.full(256,100.,dtype='float32'))-tf.reduce_sum(h*changes,axis=[1,2])
        tf.debugging.assert_greater(tf.reduce_min(loss),0.)
        obj=model.eta+tf.reduce_mean(tf.nn.relu(loss-model.eta))/.5
    gradient=tape.gradient(obj,model.nets[-1].layers[-1].bias).numpy()
    expected_gradient=-2*np.mean(changes[:,-1,:],axis=0)
    assert np.allclose(gradient,expected_gradient,rtol=2e-5,atol=1e-7)
    previous=[v.numpy().copy() for v in model.trainable_variables]
    step=training_step(model);first=step(features,changes,payoff)
    changed=sum(not np.array_equal(v.numpy(),old) for v,old in zip(model.trainable_variables,previous))
    assert changed>0 and np.all(first['network_gradient_norms'].numpy()>0)
    before=[v.numpy().copy() for v in model.weights]
    _,_,mode=mode_diagnostic(model,features)
    assert mode['bn_buffers_restored'] and all(np.array_equal(v.numpy(),old) for v,old in zip(model.weights,before))
    return dict(market=result,actual_initial_input=captured[-1][0].tolist(),output_shape=list(positions.shape),
                both_asset_terminal_gradients=gradient.tolist(),expected_both_asset_gradients=expected_gradient.tolist(),
                all30_network_gradient_norms=first['network_gradient_norms'].numpy().tolist(),
                trainable_variables_changed=changed,bn_diagnostic_state_restored=True)


def simulation_gate(reference,cfg):
    """Aggregate a bounded independent market check, before NN training."""
    n=262144;sums=np.zeros(8);zeros=0
    for data in path_chunks(n,8192,cfg['market_check_seed']):
        s=data['spot'][:,-1];swap=data['swap'][:,-1];v=data['variance'][:,-1]
        call=np.maximum(s-100,0)
        sums+=np.array([s.sum(),(s*s).sum(),swap.sum(),(swap*swap).sum(),call.sum(),(call*call).sum(),v.sum(),len(s)])
        zeros+=int(np.count_nonzero(data['variance']==0))
    def mean_se(i):
        mean=sums[i]/n;se=np.sqrt(max(0,sums[i+1]/n-mean*mean)/(n-1))
        return float(mean),float(se)
    stock,se_s=mean_se(0);swap,se_v=mean_se(2);call,se_c=mean_se(4)
    q=reference.price(100,.04,30);initial_swap=.04*30/365
    # Daily integral quadrature has weak bias; don't demand an exact continuous
    # price match from the explicitly discretized simulation.
    assert abs(stock-100)<6*se_s+.01
    assert abs(swap-initial_swap)<6*se_v+1e-6
    assert abs(call-q)<6*se_c+.025
    return dict(paths=n,stock_mean=stock,stock_mean_se=se_s,swap_mean=swap,swap_mean_se=se_v,
                call_mean=call,call_mean_se=se_c,quantlib_q=q,initial_swap=initial_swap,
                terminal_variance_mean=float(sums[6]/n),zero_variance_states=zeros,
                note='Exact CIR skeleton, approximate integrated variance; .025 price-bias allowance is a declared diagnostic tolerance, not an authors setting.')


def check_grid(grid):
    reports=[grid.validation_report(remaining_days=tuple(range(30,0,-1)),points_per_day=8,seed=seed) for seed in (7101,7111,7112,7113)]
    report=reports[0]
    report['checks']=[dict(r,seed=seed) for seed,part in zip((7101,7111,7112,7113),reports) for r in part['checks']]
    report['independent_seeds']=[7101,7111,7112,7113]
    # Check interpolation separately from model fit. Values and both Greek
    # holdings use direct QuantLib off-grid comparisons, including V near zero.
    report['acceptance_tolerances']={'price_abs_max':.001,'stock_holding_abs_max':.02,
                                     'swap_holding_abs_max':10.}
    report['accepted']=all(r['price_abs_max']<.001 and r['stock_holding_abs_max']<.02
                           and r['swap_holding_abs_max']<10 for r in report['checks'])
    return report


def summarize_pnls(values):
    values=np.asarray(values,dtype=float)
    return dict(samples=len(values),mean_pnl=float(values.mean()),mse=float(np.mean(values**2)),
                probability_loss=float(np.mean(values<0)),cvar50=empirical_cvar(-values,.5),
                cvar95=empirical_cvar(-values,.95),cvar99=empirical_cvar(-values,.99),
                minimum_pnl=float(values.min()),maximum_pnl=float(values.max()))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--check-only',action='store_true')
    parser.add_argument('--tag',default='')
    args=parser.parse_args()
    tf.config.threading.set_intra_op_parallelism_threads(1)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    protocol_path=ROOT/'protocol_section52_v1.json';protocol=json.loads(protocol_path.read_text())
    cfg=protocol['declared_conventions'];spec=protocol['paper_explicit']
    out=ROOT/'results'/('section52_v1'+('_'+args.tag if args.tag else ''))
    out.mkdir(parents=True,exist_ok=False)
    dump(out/'protocol.json',protocol)
    files=[Path(__file__),ROOT/'scripts/heston52_market.py',ROOT/'scripts/heston52_benchmark.py',
           ROOT/'scripts/tf_heston52_engine.py',protocol_path]
    dump(out/'implementation.json',{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files})
    start=time.monotonic()
    reference=QuantLibReference()
    gate=simulation_gate(reference,cfg);dump(out/'simulation_gate.json',gate)
    print('MARKET '+json.dumps(gate),flush=True)
    grid=HestonGridHedge(reference=reference,cache_dir=ROOT/'results/section52_quantlib_cache',
                         sensitive_variance=cfg['benchmark_sensitive_variance'])
    grid_checks=check_grid(grid);dump(out/'benchmark_validation.json',grid_checks)
    print('BENCHMARK '+json.dumps(grid_checks),flush=True)
    if not grid_checks['accepted']:raise RuntimeError('QuantLib grid not accurate enough; do not train against an unverified benchmark')
    checks=engine_checks();dump(out/'engine_checks.json',checks)
    print('ENGINE '+json.dumps(checks),flush=True)
    if args.check_only:return
    # Prebuild all benchmark dates before spending the formal training budget.
    for days in range(30,0,-1):
        grid.build(days)
        print(f'GRID remaining_days={days}',flush=True)
    validation=sample_paths(cfg['validation_paths'],np.random.default_rng(cfg['validation_seed']))
    val_features,_,_=sample_tensors(validation)
    q=grid.q0
    tf.keras.utils.set_random_seed(cfg['network_seed'])
    model=Heston52Hedge(seed=cfg['network_seed'],bn_epsilon=cfg['bn_epsilon'],bn_momentum=cfg['bn_momentum'])
    model.initialize();step=training_step(model,learning_rate=spec['learning_rate'],alpha=spec['cvar_alpha'])
    val_model=grid.hedge_paths(validation)
    benchmark_metrics=metrics(validation,val_model,q)
    dump(out/'model_hedge_validation.json',benchmark_metrics)
    stock_min=float(val_model[:,:,0].min());stock_max=float(val_model[:,:,0].max())
    benchmark_gate=dict(stock_min=stock_min,stock_max=stock_max,
                        max_absolute_swap_holding=float(np.abs(val_model[:,:,1]).max()),
                        model_pnl_mean=benchmark_metrics['mean_pnl'],model_pnl_mse=benchmark_metrics['mse'],
                        note='Loose catastrophic-numerics gates on actual simulation states, not calibration to paper headline results.')
    benchmark_gate['accepted']=bool(stock_min>=-.01 and stock_max<=1.01
                                     and benchmark_gate['max_absolute_swap_holding']<1e8
                                     and abs(benchmark_metrics['mean_pnl'])<.1 and benchmark_metrics['mse']<10)
    dump(out/'benchmark_actual_state_gate.json',benchmark_gate)
    if not benchmark_gate['accepted']:
        raise RuntimeError('Model hedge failed actual-state numerical sanity checks; no training allowed')
    initial=metrics(validation,predict(model,val_features),q)
    rng=np.random.default_rng(cfg['training_seed']);updates=0;first=None;train_sum=0.
    milestones={256,1024,4096,10000,cfg['training_updates']}
    for data in path_chunks(cfg['training_paths'],cfg['chunk_paths'],cfg['training_seed']):
        features,changes,payoff=sample_tensors(data)
        for lo in range(0,len(payoff),spec['batch_size']):
            result=step(features[lo:lo+256],changes[lo:lo+256],payoff[lo:lo+256]);updates+=1
            train_sum+=float(result['objective'].numpy())
            if first is None:first={k:v.numpy().tolist() for k,v in result.items()}
            if updates in milestones:
                holding=predict(model,val_features)
                vm=metrics(validation,holding,q)
                a,b,mode=mode_diagnostic(model,val_features[:cfg['validation_mode_subset_paths']])
                sub={k:v[:cfg['validation_mode_subset_paths']] for k,v in validation.items()}
                record=dict(updates=updates,paths_seen=updates*256,validation=vm,mode=mode,
                            training_mode_subset=metrics(sub,b,q),inference_mode_subset=metrics(sub,a,q),
                            eta=float(model.eta.numpy()),elapsed_seconds=time.monotonic()-start)
                with (out/'training_diagnostics.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
                print(f'TRAIN {updates}/{cfg["training_updates"]}: val_CVaR50={vm["cvar50"]:.5f}, stock_mode_gap={mode["mean_abs_difference_stock"]:.5f}, swap_mode_gap={mode["mean_abs_difference_swap"]:.5f}, seconds={time.monotonic()-start:.1f}',flush=True)
    assert updates==cfg['training_updates']
    model.save_weights(out/'final.weights.h5')
    names=['deep_cvar50','heston_model_hedge','unhedged'];collected={name:[] for name in names}
    tested=0
    for data in path_chunks(spec['test_paths'],cfg['chunk_paths'],cfg['test_seed']):
        f,_,_=sample_tensors(data)
        holding=predict(model,f);benchmark=grid.hedge_paths(data)
        for name,h in [('deep_cvar50',holding),('heston_model_hedge',benchmark),('unhedged',np.zeros_like(holding))]:
            collected[name].append(pnl(data,h,q))
        tested+=len(f)
        if tested%131072==0:print(f'TEST {tested}/{spec["test_paths"]}',flush=True)
    distributions={name:np.concatenate(chunks) for name,chunks in collected.items()}
    summaries={name:summarize_pnls(x) for name,x in distributions.items()}
    edges=np.linspace(-8,8,161)
    histogram={'edges':edges.tolist(),'note':'Histogram window -8..8; outside counts retained; risk metrics use every sample.'}
    for name,x in distributions.items():
        histogram[name]={'counts':np.histogram(x,bins=edges)[0].tolist(),'below':int(np.sum(x<edges[0])),'above':int(np.sum(x>edges[-1]))}
    dump(out/'histogram.json',histogram)
    result=dict(target='Section5.2 Figure2 primary baseline',q=q,parameters=PARAMETERS,
                training_updates=updates,training_paths=updates*256,test_paths=tested,first_training_step=first,
                initial_validation=initial,metrics=summaries,elapsed_seconds=time.monotonic()-start,
                benchmark_configuration=grid.configuration(),benchmark_fallback_points=grid.fallback_points,
                benchmark_total_evaluated_points=grid.total_points,
                benchmark_grid_build_seconds=grid.build_seconds,
                claim_limit='Paper explicit setup retained; training budget/BN/initialization/integral endpoint/benchmark interpolation are declared numerical conventions. Not an exact paper-result replication.')
    dump(out/'metrics.json',result);print('RESULT '+json.dumps(result),flush=True)


if __name__=='__main__':main()
