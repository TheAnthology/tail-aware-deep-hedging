"""Paper-structure Section5.5 retry, isolated from frozen adapted baselines.
Only added numerical convention: independent sequential population BN stats.
"""
import os
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL','3');os.environ.setdefault('OPENBLAS_NUM_THREADS','1');os.environ.setdefault('OMP_NUM_THREADS','1')
from pathlib import Path
import json,hashlib,time,subprocess,sys
import numpy as np
import tensorflow as tf
from paper_market import window_data,fit_garch,paths,sample_tensors,observed_path,bs_premium,bs_hedge,pnl,cvar
from tf_paper_engine import PaperHedge,make_adam,create_train_step,predict_positions,compare_training_and_inference
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'results/paper55_recurrent_bnfix_v3_window000'
def dump(p,x):p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def verify_freeze():
    x=json.loads((ROOT/'results/sp500_adapted_frozen_v1/freeze_manifest.json').read_text())
    for p,h in x['files'].items():assert sha(ROOT/p)==h,p

def calibrate(model,features):
    learned=[a.numpy().copy() for a in model.trainable_variables];previous=np.zeros((len(features),1),dtype='float32');records=[]
    for k,net in enumerate(model.nets):
        x=np.concatenate([features[:,k:k+1],previous],axis=1);record={'date':k,'input_columns':['log_absolute_S','own_previous_total_holding']}
        for di,bi,ri in [(0,1,2),(3,4,5)]:
            z=np.concatenate([net.layers[di](x[i:i+256]).numpy() for i in range(0,len(x),256)])
            mu=z.astype('float64').mean(0);var=z.astype('float64').var(0)
            net.layers[bi].moving_mean.assign(mu);net.layers[bi].moving_variance.assign(var)
            x=net.layers[ri](net.layers[bi](z,training=False));record[str(bi)]={'mean':mu.tolist(),'variance':var.tolist()}
        previous=net.layers[6](x).numpy();records.append(record)
    assert all(np.array_equal(a.numpy(),b) for a,b in zip(model.trainable_variables,learned))
    return records

def metrics(pl):
    pl=np.asarray(pl,dtype='float64');assert np.isfinite(pl).all()
    return {'paths':len(pl),'mean_pnl':float(pl.mean()),'mse':float(np.mean(pl*pl)),'probability_loss':float(np.mean(pl<0)),
            **{f'cvar{int(a*100)}':cvar(-pl,a) for a in [.5,.75,.95]},'pnl_quantiles':{str(a):float(np.quantile(pl,a)) for a in [0,.01,.05,.5,.95,.99,1]}}

def evaluate(model,prices,strike,q,days):
    f=sample_tensors(prices,strike,days)[0];h=predict_positions(model,f);return metrics(pnl(prices,h,q,strike,days)),h

def main():
    tf.config.threading.set_intra_op_parallelism_threads(1);tf.config.threading.set_inter_op_parallelism_threads(1)
    verify_freeze();OUT.mkdir(exist_ok=False);start=time.monotonic();old=json.loads((ROOT/'protocol_paper_tf_v2.json').read_text());cfg=old['declared_conventions_not_specified_in_paper'];spec=old['paper_requirements'];days=spec['hedge_days']
    sources=[Path(__file__),ROOT/'scripts/tf_paper_engine.py',ROOT/'scripts/paper_market.py',ROOT/'protocol_paper_tf_v2.json',ROOT/'results/sp500_adapted_frozen_v1/freeze_manifest.json',ROOT/'data/processed/paper_rolling_windows_300.csv',ROOT/'data/processed/sp500_prices_20070612_20180102.csv']
    hashes={str(p.relative_to(ROOT)):sha(p) for p in sources}
    protocol={'id':'S55_PAPER_STRUCTURE_RECURRENT_C75_BNFIX_V3','parent':old,'version_separation':'Distinct from frozen S55_ADAPTED_SIMPLE_C75_V1 / RESIDUAL_V1 / BS_V1; no ensembles or extensions',
        'window_id':0,'training_paths':20000000,'updates':78125,'batch':256,'inputs':['log_absolute_S_k','previous_total_holding'],'initial_previous_holding':0,
        'architecture':'6 independent2->20BNReLU->20BNReLU->1;raw holdings outputs;recurrence remains in gradient graph',
        'premium_and_loss':'Original index-price currency;BS q;train CVaR75 OCE excluding constant q;no premium/path scaling',
        'training':'Same fixed init seed2701,training RNG1701,original v2 chunks/milestones;one fresh20M-path pass,not continued optimizer from old checkpoint',
        'bn_repair':{'paths':8192,'seed':5704,'procedure':'Only moving moments replaced; layers/dates sequential,upstream eval rollout,ownpreviousholdings,float64 moments;independent of validation/test'},
        'validation':{'paths':65536,'seed':5702,'used_for_selection':False},'test':{'paths':100000,'seed':5713,'fresh_stream_relative_to_old5703':True},
        'selection':'Fixed final trained weights;raw and BN-calibrated outputs both retained,no historical/checkpoint selection or untrained fallback',
        'historical':'Already inspected origin0,diagnostic only;not rolling300 or sealed historical test',
        'claim_limit':'Paper-specified architecture/features/setup restored, plus explicit BN numerical repair. Original missing details remain local choices;exact authors code/result recovery not claimed.',
        'frozen_adapted_baselines_unchanged':True,'source_hashes':hashes}
    dump(OUT/'protocol.json',protocol)
    w,history,r=window_data(0);p,state,fit=fit_garch(r[:1187]);s0=strike=float(history[1187]);q=bs_premium(s0,fit['initial_sigma0_daily']);dump(OUT/'garch_fit.json',fit)
    cal,_=paths(p,state,s0,8192,np.random.default_rng(5704));cf=sample_tensors(cal,strike,days)[0];del cal
    val,_=paths(p,state,s0,65536,np.random.default_rng(5702));vf=sample_tensors(val,strike,days)[0]
    model=PaperHedge(seed=2701);model.initialize_without_bn_updates();optimizer=make_adam(.0001);step=create_train_step(model,optimizer,.75)
    evaluator=PaperHedge(seed=2701);evaluator.initialize_without_bn_updates()
    rng=np.random.default_rng(1701);updates=seen=0;all_finite=True;first=None;milestones=[65536,1048576,5000192,10000128,20000000];last=None
    for milestone in milestones:
        while seen<milestone:
            take=min(65536,milestone-seen);sp,_=paths(p,state,s0,take,rng);f,ch,z=sample_tensors(sp,strike,days)
            for lo in range(0,take,256):
                result=step(f[lo:lo+256],ch[lo:lo+256],z[lo:lo+256]);updates+=1
                values={k:v.numpy() for k,v in result.items()};assert all(np.isfinite(v).all() for v in values.values())
                if first is None:
                    first={k:v.tolist() for k,v in values.items()};assert (values['network_gradient_norms']>0).all();dump(OUT/'first_update.json',first)
            seen+=take
        evaluator.set_weights(model.get_weights());calibrate(evaluator,cf)
        before=[a.copy() for a in model.get_weights()];raw,_=evaluate(model,val,strike,q,days);corrected,_=evaluate(evaluator,val,strike,q,days)
        assert all(np.array_equal(a,b) for a,b in zip(before,model.get_weights()))
        last={'updates':updates,'paths':seen,'raw_validation':raw,'calibrated_validation':corrected,'eta':float(model.eta.numpy()),'elapsed_seconds':time.monotonic()-start}
        with (OUT/'training_milestones.jsonl').open('a') as f:f.write(json.dumps(last)+'\n')
        print(f'PROGRESS {seen}/20M: raw CV75={raw["cvar75"]:.5f}; calibrated CV75={corrected["cvar75"]:.5f}',flush=True)
    assert seen==20000000 and updates==78125
    model.save_weights(OUT/'raw_final.weights.h5');tf.train.Checkpoint(model=model,optimizer=optimizer).write(str(OUT/'raw_final_adam'))
    original=PaperHedge(seed=2701);original.initialize_without_bn_updates();original.load_weights(ROOT/'results/paper_tf_v2_paper_window000/final.weights.h5')
    oldmatch={'bitwise_same_raw_weights_as_v2':all(np.array_equal(a,b) for a,b in zip(model.get_weights(),original.get_weights())),
              'maximum_absolute_raw_weight_difference':max(float(np.max(abs(a-b))) for a,b in zip(model.get_weights(),original.get_weights()))}
    repaired=PaperHedge(seed=2701);repaired.initialize_without_bn_updates();repaired.set_weights(model.get_weights());bn=calibrate(repaired,cf);dump(OUT/'bn_population_statistics.json',bn);repaired.save_weights(OUT/'calibrated_final.weights.h5')
    assert all(np.array_equal(a.numpy(),b.numpy()) for a,b in zip(model.trainable_variables,repaired.trainable_variables))
    checks={'same_learned_parameters_raw_vs_calibrated':True,'all_training_scalars_finite':True,'all_six_first_network_gradients_nonzero':True,'optimizer_iterations':int(optimizer.iterations.numpy()),**oldmatch}
    h=predict_positions(repaired,vf[:4096],batch_size=256);hh=predict_positions(repaired,vf[:4096],batch_size=4096);maxgap=float(np.max(abs(h-hh)));assert np.allclose(h,hh,rtol=1e-5,atol=1e-4);checks['eval_batch_size_max_position_gap']=maxgap
    shuffled=np.random.default_rng(5714).permutation(4096);unshuffled=predict_positions(repaired,vf[:4096][shuffled])[np.argsort(shuffled)];assert np.allclose(h,unshuffled,rtol=1e-5,atol=1e-4)
    singleton=predict_positions(repaired,vf[:16],batch_size=1);assert np.allclose(singleton,h[:16],rtol=1e-5,atol=1e-4)
    changed=vf[:256].copy();changed[:,3:]+=1.;assert np.array_equal(predict_positions(repaired,changed)[:,:3],h[:256,:3])
    checks.update(singleton_shuffled_batch_invariance=True,causal_features=True,initial_holding_same_all_paths=bool(np.max(np.ptp(h[:,0]))<1e-6))
    logs={}
    for name,m in [('raw',model),('calibrated',repaired)]:
        mode=compare_training_and_inference(m,vf[:4096]);pl_train=pnl(val[:4096],mode['training_mode_positions'],q,strike,days);pl_eval=pnl(val[:4096],mode['inference_positions'],q,strike,days)
        logs[name]={'training_mode_diagnostic':metrics(pl_train),'inference':metrics(pl_eval),'mean_abs_holdings_gap_per_date':mode['mean_abs_difference_per_date'].tolist(),'warning':'Batch-stat mode is diagnostic only,not deployable'}
    dump(OUT/'verification.json',checks);dump(OUT/'mode_diagnostics.json',logs);dump(OUT/'frozen_before_test.json',{'selection':'Fixed final;raw/control and calibrated/repaired both committed','training_complete':True,'historical_outcome_used_for_selection':False})
    test,tv=paths(p,state,s0,100000,np.random.default_rng(5713));tftr=sample_tensors(test,strike,days)[0];arrays={};positions={}
    for name,m in [('paper_raw',model),('paper_bn_fixed',repaired)]:positions[name]=predict_positions(m,tftr)
    positions['BS']=bs_hedge(test,tv,strike,days)
    for name,h in positions.items():arrays[name]=pnl(test,h,q,strike,days)
    simulated={name:metrics(a) for name,a in arrays.items()};np.savez_compressed(OUT/'test_pnl.npz',**arrays)
    hist,hv=observed_path(p,state,s0,r[1187:]);np.testing.assert_allclose(hist[0],history[1187:],atol=1e-10)
    historical={}
    for name,m in [('paper_raw',model),('paper_bn_fixed',repaired),('BS',None)]:
        h=predict_positions(m,sample_tensors(hist,strike,days)[0]) if m is not None else bs_hedge(hist,hv,strike,days)
        pl=float(pnl(hist,h,q,strike,days)[0]);historical[name]={'index_pnl':pl,'normalized_pnl':pl/s0*100,'holdings':h[0].tolist()}
    for row in simulated.values():
        row['normalized_cvar75']=row['cvar75']/s0*100;row['normalized_mse']=row['mse']/(s0/100)**2
    assert all(sha(ROOT/p)==h for p,h in hashes.items());verify_freeze()
    dump(OUT/'metrics.json',{'window':w,'q':q,'original_spot':s0,'training_paths':seen,'updates':updates,'simulated_test':simulated,'historical':historical,'checks':checks,'elapsed_seconds':time.monotonic()-start,'frozen_adapted_baselines_unchanged':True,'scope':protocol['claim_limit']})
    dump(OUT/'STATUS.json',{'execution':'COMPLETE','adapted_baselines_frozen':True,'paper_structure_restored':True,'bn_protocol_explicit_local_repair':True,'full_300_window_reproduction':False})
    print('COMPLETE '+json.dumps({'simulated':simulated,'historical':historical,'checks':checks}),flush=True)
if __name__=='__main__':main()
