"""Frozen BN calibration and matched million-path comparison; no training."""
import os
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL','3')
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('OMP_NUM_THREADS','1')
from pathlib import Path
import hashlib,json,time
import numpy as np
import tensorflow as tf
from heston52_market import sample_paths,path_chunks,sample_tensors,pnl,metrics
from tf_heston52_engine import Heston52Hedge,predict,mode_diagnostic
from run_section52 import summarize_pnls

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/section52_calibrated_million_v1'

def dump(p,x):p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def calibrate(model,features):
    previous=np.zeros((len(features),2),dtype='float32'); records=[]
    for k,net in enumerate(model.nets):
        x=np.concatenate([features[:,k,:],previous],axis=1) if model.recurrent else features[:,k,:]
        record={'date':k}
        for di,bi,ri in [(0,1,2),(3,4,5)]:
            z=np.concatenate([net.layers[di](x[i:i+256]).numpy() for i in range(0,len(x),256)])
            mu=z.astype('float64').mean(0);var=z.astype('float64').var(0)
            net.layers[bi].moving_mean.assign(mu);net.layers[bi].moving_variance.assign(var)
            x=net.layers[ri](net.layers[bi](z,training=False))
            record[str(bi)]={'mean':mu.tolist(),'variance':var.tolist()}
        previous=net.layers[6](x).numpy();records.append(record)
    return records

def distribution(x):
    r=summarize_pnls(x);levels=[0,.001,.01,.05,.25,.5,.75,.95,.99,.999,1]
    r['pnl_quantiles']={str(p):float(v) for p,v in zip(levels,np.quantile(x,levels))};return r


def main():
    tf.config.threading.set_intra_op_parallelism_threads(1);tf.config.threading.set_inter_op_parallelism_threads(1)
    OUT.mkdir(exist_ok=False);start=time.monotonic()
    base=ROOT/'results/section52_v1_final';simple=ROOT/'results/section52_fig4_control'
    old=json.loads((base/'protocol.json').read_text());cfg=old['declared_conventions']
    paths={'recurrent':base/'final.weights.h5','simpler':simple/'final.weights.h5'}
    sources=[Path(__file__),ROOT/'scripts/tf_heston52_engine.py',ROOT/'scripts/heston52_market.py',ROOT/'scripts/heston52_benchmark.py',*paths.values()]
    hashes={str(p.relative_to(ROOT)):sha(p) for p in sources}
    for file,expected in json.loads((base/'implementation.json').read_text()).items():assert sha(ROOT/file)==expected
    protocol={'id':'section52_calibrated_million_v1','parent_protocol':old,
      'calibration':{'paths':8192,'seed':5291,'same_pool_for_both_models':True,'method':'date/layer sequential population BN moments; upstream inference and float64 moment accumulation; no learned parameter changes'},
      'test':{'paths':1000000,'seed':5213,'chunk':8192,'shared_with_original_million_test':True},
      'diagnostics':{'seen_training_paths':32768,'training_replay_seed':5211,'training_replay_chunk':8192,'validation_paths':32768,'validation_seed':5212,'fresh_paths':32768,'fresh_seed':5292,'holdings_sample_paths':8192,'figure3_date_indices':[14,15]},
      'training_updates':0,'selection':'Fixed final checkpoints, no tuning or model selection on test results',
      'interpretation':'Previously examined shared test is reused for diagnostic comparison; not a newly sealed confirmation. Calibration differs from unavailable original author BN protocol.',
      'outputs':'full P&L arrays; million-path k15 holdings; sample of complete holdings; aggregated per-asset/date gain differences. No full market paths.',
      'convergence_limit':'Original logs have only five validation milestones; no complete minibatch loss curve or intermediate checkpoints retained.'}
    dump(OUT/'protocol.json',protocol);dump(OUT/'implementation.json',hashes)
    from heston52_benchmark import HestonGridHedge
    grid=HestonGridHedge(cache_dir=ROOT/'results/section52_quantlib_cache',sensitive_variance=cfg['benchmark_sensitive_variance']);q=grid.q0
    calibration=sample_paths(8192,np.random.default_rng(5291));cf,_,_=sample_tensors(calibration);del calibration
    models={};diagnostics={}
    for name,path in paths.items():
        m=Heston52Hedge(seed=cfg['network_seed'],recurrent=name=='recurrent',bn_epsilon=cfg['bn_epsilon'],bn_momentum=cfg['bn_momentum']);m.initialize();m.load_weights(path)
        learned=[v.numpy().copy() for v in m.trainable_variables]
        records=calibrate(m,cf)
        assert all(np.array_equal(v.numpy(),x) for v,x in zip(m.trainable_variables,learned))
        dump(OUT/f'{name}_bn_statistics.json',records);m.save_weights(OUT/f'{name}_calibrated.weights.h5');models[name]=m
        diagnostics[name]={}
        pools={'training_seen_replay':list(path_chunks(32768,8192,5211)),
               'validation':[sample_paths(32768,np.random.default_rng(5212))],
               'fresh':[sample_paths(32768,np.random.default_rng(5292))]}
        for label,chunks in pools.items():
            ps=[];train_ps=[]
            for market in chunks:
                f,_,_=sample_tensors(market);hinf,htrain,mode=mode_diagnostic(m,f,batch=256)
                ps.append(pnl(market,hinf,q));train_ps.append(pnl(market,htrain,q))
            diagnostics[name][label]={'inference':distribution(np.concatenate(ps)),'training_mode_diagnostic':distribution(np.concatenate(train_ps))}
        del pools
        logs=base/'training_diagnostics.jsonl' if name=='recurrent' else simple/'training_diagnostics.jsonl'
        diagnostics[name]['archived_training_milestones']=[json.loads(x) for x in logs.read_text().splitlines()]
        print('DIAGNOSTIC '+name+' '+json.dumps({k:v['inference']['cvar50'] for k,v in diagnostics[name].items() if k!='archived_training_milestones'}),flush=True)
    dump(OUT/'diagnostics.json',diagnostics)
    arrays={name:[] for name in [*models,'heston']};k15={name:[] for name in arrays};k14={name:[] for name in arrays}
    sums={name:{key:np.zeros((30,2)) for key in ['holding_diff_sum','holding_diff_sq','holding_diff_abs','holding_diff_min','holding_diff_max','gain_diff_sum','gain_diff_sq','gain_diff_times_total_error']} for name in models}
    for s in sums.values():s['holding_diff_min'].fill(np.inf);s['holding_diff_max'].fill(-np.inf)
    samples={};tested=0
    for market in path_chunks(1000000,8192,5213):
        f,_,_=sample_tensors(market);hs={name:predict(m,f) for name,m in models.items()};hs['heston']=grid.hedge_paths(market)
        changes=np.stack([np.diff(market['spot'],axis=1),np.diff(market['swap'],axis=1)],axis=-1)
        for name,h in hs.items():
            arrays[name].append(pnl(market,h,q));k15[name].append(h[:,15,:].astype('float32'));k14[name].append(h[:,14,:].astype('float32'))
            if tested==0:samples[name]=h.astype('float32')
        for name in models:
            d=hs[name].astype('float64')-hs['heston'];g=d*changes;e=g.sum(axis=(1,2));s=sums[name]
            s['holding_diff_sum']+=d.sum(0);s['holding_diff_sq']+=(d*d).sum(0);s['holding_diff_abs']+=np.abs(d).sum(0)
            s['holding_diff_min']=np.minimum(s['holding_diff_min'],d.min(0));s['holding_diff_max']=np.maximum(s['holding_diff_max'],d.max(0))
            s['gain_diff_sum']+=g.sum(0);s['gain_diff_sq']+=(g*g).sum(0);s['gain_diff_times_total_error']+=(g*e[:,None,None]).sum(0)
            assert np.allclose(e,arrays[name][-1]-arrays['heston'][-1],rtol=1e-8,atol=1e-10)
        tested+=len(f)
        if tested%131072==0:print(f'TEST {tested}/1000000; seconds={time.monotonic()-start:.1f}',flush=True)
    pl={name:np.concatenate(x) for name,x in arrays.items()};summary={name:distribution(x) for name,x in pl.items()}
    assert tested==1000000
    # Reproduced reference must match the previously evaluated shared paths.
    previous=json.loads((base/'metrics.json').read_text())['metrics']['heston_model_hedge']
    for key in ['mean_pnl','mse','cvar50','cvar95']:assert np.isclose(summary['heston'][key],previous[key],rtol=1e-10,atol=1e-10)
    comparison={};positions={};attribution={}
    for name in models:
        error=pl[name]-pl['heston'];mean=error.mean();var=error.var();s=sums[name]
        cov=s['gain_diff_times_total_error']/tested-s['gain_diff_sum']/tested*mean
        assert np.isclose(cov.sum(),var,rtol=1e-10,atol=1e-10)
        attribution[name]={'mean_holding_difference':(s['holding_diff_sum']/tested).tolist(),'rms_holding_difference':np.sqrt(s['holding_diff_sq']/tested).tolist(),'mean_absolute_holding_difference':(s['holding_diff_abs']/tested).tolist(),'min_holding_difference':s['holding_diff_min'].tolist(),'max_holding_difference':s['holding_diff_max'].tolist(),'mean_gain_difference':(s['gain_diff_sum']/tested).tolist(),'rms_gain_difference':np.sqrt(s['gain_diff_sq']/tested).tolist(),'covariance_with_total_error':cov.tolist(),'sum_covariance':float(cov.sum()),'total_error_variance':float(var),'interpretation':'Covariance contributions sum exactly to variance of NN-minus-Heston terminal gain; signed and correlated, not additive CVaR contributions.'}
        for k,store in [(14,k14),(15,k15)]:
            nn=np.concatenate(store[name]);hh=np.concatenate(store['heston']);diff=nn-hh;levels=[0,.01,.05,.5,.95,.99,1]
            positions[f'{name}_k{k}']={'quantile_levels':levels,'nn':np.quantile(nn,levels,axis=0).tolist(),'heston':np.quantile(hh,levels,axis=0).tolist(),'difference':np.quantile(diff,levels,axis=0).tolist(),'samples':tested}
    paired=pl['recurrent']-pl['simpler'];comparison={'paired_pnl_difference_mean':float(paired.mean()),'paired_pnl_difference_rms':float(np.sqrt(np.mean(paired**2))),'cvar50_ratio_recurrent_over_simpler':summary['recurrent']['cvar50']/summary['simpler']['cvar50'],'note':'No arbitrary closeness threshold; compare distributions and tail gaps directly.'}
    edges=np.linspace(-8,8,161);hist={'edges':edges.tolist(),'note':'Full-sample normalized histogram; outside counts explicit; all metrics use full sample.'}
    for name,x in pl.items():hist[name]={'counts':np.histogram(x,edges)[0].tolist(),'below':int((x<edges[0]).sum()),'above':int((x>edges[-1]).sum())}
    np.savez_compressed(OUT/'pnl.npz',**pl);np.savez_compressed(OUT/'holdings_first8192.npz',**samples)
    np.savez_compressed(OUT/'holdings_k15.npz',**{name:np.concatenate(v) for name,v in k15.items()})
    dump(OUT/'histogram.json',hist);dump(OUT/'holdings_attribution.json',attribution);dump(OUT/'holdings_k14_k15_quantiles.json',positions)
    assert all(sha(ROOT/p)==expected for p,expected in hashes.items())
    result={'protocol':'section52_calibrated_million_v1','q':q,'test_paths':tested,'training_updates':0,'calibration_paths_each':8192,'metrics':summary,'paired_comparison':comparison,'elapsed_seconds':time.monotonic()-start,'original_sources_and_checkpoints_unchanged':True,'replayed_reference_matches_original':True,'claim_limit':protocol['interpretation']}
    dump(OUT/'metrics.json',result);print('RESULT '+json.dumps(result),flush=True)

if __name__=='__main__':main()
