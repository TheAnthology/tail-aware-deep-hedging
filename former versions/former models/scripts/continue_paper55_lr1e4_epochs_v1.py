"""Fixed five-pass continuation at paper LR; regenerate identical finite pool."""
import os
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL','3')
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('OMP_NUM_THREADS','1')
from pathlib import Path
import json, hashlib, time
import numpy as np
import tensorflow as tf
from paper_market import window_data,fit_garch,paths,sample_tensors,observed_path,bs_premium,bs_hedge,pnl,cvar
from tf_paper_engine import PaperHedge,make_adam,create_train_step,predict_positions,compare_training_and_inference
from run_paper55_recurrent_bnfix_v3 import calibrate,metrics,evaluate,verify_freeze
ROOT=Path(__file__).resolve().parents[1]
PARENT=ROOT/'results/paper55_recurrent_bnfix_v3_window000'
OUT=ROOT/'results/paper55_lr1e4_five_epochs_v1_window000'
DAYS=[0,5,10,15,20,25]
MILESTONES=[65536,1048576,5000192,10000128,20000000]
def read(p):return json.loads(p.read_text())
def dump(p,x):p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def verify_parent():
 for name,h in read(PARENT/'artifact_manifest.json').items():assert sha(PARENT/name)==h,name
 verify_freeze()
def threshold(pl,q,eta):
 loss=q-pl;optimal=float(np.quantile(loss,.75));quantile=float(np.mean(loss<=eta))
 return {'learned_eta':eta,'validation_loss_quantile75':optimal,'learned_eta_validation_quantile':quantile,'intercept_only_OCE_gap':float(eta+4*np.maximum(loss-eta,0).mean()-optimal-4*np.maximum(loss-optimal,0).mean())}
def main():
 tf.config.threading.set_intra_op_parallelism_threads(1);tf.config.threading.set_inter_op_parallelism_threads(1)
 verify_parent();assert not OUT.exists();start=time.monotonic()
 model=PaperHedge(seed=2701);model.initialize_without_bn_updates();model.load_weights(PARENT/'raw_final.weights.h5');before=[v.copy() for v in model.get_weights()]
 optimizer=make_adam(.0001);step=create_train_step(model,optimizer,.75)
 ckpt=tf.train.Checkpoint(model=model,optimizer=optimizer)
 ckpt.read(str(PARENT/'raw_final_adam')).assert_consumed()
 assert all(np.array_equal(a,b) for a,b in zip(before,model.get_weights()))
 assert int(optimizer.iterations.numpy())==78125
 assert np.isclose(float(optimizer.learning_rate.numpy()),.0001)
 reader=tf.train.load_checkpoint(str(PARENT/'raw_final_adam'));slot_checks=0
 for i,v in enumerate(optimizer.variables):
  key=f'optimizer/_variables/{i}/.ATTRIBUTES/VARIABLE_VALUE'
  if reader.has_tensor(key):np.testing.assert_array_equal(v.numpy(),reader.get_tensor(key));slot_checks+=1
 assert slot_checks>100
 OUT.mkdir()
 protocol={'id':'S55_PAPER_LR1E4_FIVE_EPOCHS_V1','parent':str(PARENT.relative_to(ROOT)),'learning_rate':.0001,'neural_and_eta_same_lr':True,'eta_initialization':'Original zero initialization; restored epoch1 state, no reset','unique_training_paths':20000000,'total_epochs':5,'additional_epochs':4,'total_processed_paths':100000000,'total_updates':390625,'data_reuse':'Reset RNG1701 and reproduce original milestone-dependent chunk boundaries each epoch; identical sample order, no shuffle. Finite-pool repeated training, not100M distinct paths.','stopping':'Fixed epoch5, no test/historical checkpoint selection','bn':'Independent8192-path seed5704 sequential calibration on evaluation clone only; raw training EMA never overwritten','validation':{'n':65536,'seed':5702},'development_test':{'n':100000,'seed':5713,'reused':True,'evaluated':'Only after prescribed final epoch'},'historical':'Single already-inspected origin0; not300 rolling windows','scope':'Tests stopping-budget assumption at paper LR. Epoch count/order and BN repair are declared local conventions, not recovered author training settings.','retention':'Raw/calibrated weights and Adam state at each full epoch; scalar summaries at original milestones; no path files','source_hashes':{str(p.relative_to(ROOT)):sha(p) for p in [Path(__file__),ROOT/'scripts/tf_paper_engine.py',ROOT/'scripts/paper_market.py',ROOT/'scripts/run_paper55_recurrent_bnfix_v3.py',PARENT/'artifact_manifest.json']}}
 dump(OUT/'protocol.json',protocol)
 dump(OUT/'resume_verification.json',{'checkpoint_fully_consumed':True,'raw_weights_and_BN_exact_H5_match':True,'optimizer_iterations':78125,'optimizer_slot_variables_exact_match':slot_checks,'actual_learning_rate':float(optimizer.learning_rate.numpy())})
 w,hist,r=window_data(0);p,state,fit=fit_garch(r[:1187]);s0=float(hist[1187]);q=bs_premium(s0,fit['initial_sigma0_daily']);dump(OUT/'garch_fit.json',fit)
 cal,_=paths(p,state,s0,8192,np.random.default_rng(5704));cf=sample_tensors(cal,s0,DAYS)[0];del cal
 val,vv=paths(p,state,s0,65536,np.random.default_rng(5702));vf=sample_tensors(val,s0,DAYS)[0]
 evaluator=PaperHedge(seed=2701);evaluator.initialize_without_bn_updates()
 baseline=read(PARENT/'training_milestones.jsonl') if False else json.loads((PARENT/'training_milestones.jsonl').read_text().splitlines()[-1])
 baseline.update(epoch=1,threshold=read(PARENT/'validation_intercept_diagnostic.json'))
 dump(OUT/'epoch1_reference.json',baseline)
 epoch_hashes=[]
 for epoch in range(2,6):
  rng=np.random.default_rng(1701);seen=0;digest=hashlib.sha256();maxpos=0.;min_grad=np.full(6,np.inf)
  for milestone in MILESTONES:
   while seen<milestone:
    take=min(65536,milestone-seen);sp,_=paths(p,state,s0,take,rng);f,ch,z=sample_tensors(sp,s0,DAYS)
    for a in (f,ch,z):digest.update(a.tobytes())
    for lo in range(0,take,256):
     v={k:a.numpy() for k,a in step(f[lo:lo+256],ch[lo:lo+256],z[lo:lo+256]).items()}
     assert all(np.isfinite(a).all() for a in v.values())
     maxpos=max(maxpos,float(v['max_abs_position']));min_grad=np.minimum(min_grad,v['network_gradient_norms'])
    seen+=take
   dump(OUT/'STATUS.json',{'execution':'RUNNING','epoch':epoch,'paths_in_epoch':seen,'completed_epochs':epoch-1,'total_updates':int(optimizer.iterations.numpy())})
   print(f'PROGRESS epoch{epoch}/5 paths{seen}/20M eta={float(model.eta.numpy()):.5f}',flush=True)
  epoch_hashes.append(digest.hexdigest());assert len(set(epoch_hashes))==1
  assert int(optimizer.iterations.numpy())==epoch*78125
  ep=OUT/f'epoch_{epoch:02d}';ep.mkdir();model.save_weights(ep/'raw.weights.h5');ckpt.write(str(ep/'adam'))
  before=[a.copy() for a in model.get_weights()];evaluator.set_weights(before);bn=calibrate(evaluator,cf);evaluator.save_weights(ep/'calibrated.weights.h5');dump(ep/'bn_statistics.json',bn)
  raw,_=evaluate(model,val,s0,q,DAYS);fixed,hold=evaluate(evaluator,val,s0,q,DAYS);pl=pnl(val,hold,q,s0,DAYS)
  assert all(np.array_equal(a,b) for a,b in zip(before,model.get_weights()))
  modes=compare_training_and_inference(model,vf[:4096]);modefix=predict_positions(evaluator,vf[:4096]);modeinfo={'training_mode':metrics(pnl(val[:4096],modes['training_mode_positions'],q,s0,DAYS)),'raw_inference':metrics(pnl(val[:4096],modes['inference_positions'],q,s0,DAYS)),'calibrated_inference':metrics(pnl(val[:4096],modefix,q,s0,DAYS)),'raw_mean_abs_holdings_gap':modes['mean_abs_difference_per_date'].tolist(),'calibrated_mean_abs_holdings_gap':np.mean(abs(modefix-modes['training_mode_positions']),axis=0).tolist()}
  info={'epoch':epoch,'updates':int(optimizer.iterations.numpy()),'processed_paths':epoch*20000000,'raw_validation':raw,'calibrated_validation':fixed,'threshold':threshold(pl,q,float(model.eta.numpy())),'maximum_training_batch_abs_position':maxpos,'minimum_per_network_gradient_norm':min_grad.tolist(),'training_tensor_sha256':digest.hexdigest(),'elapsed_seconds':time.monotonic()-start}
  dump(ep/'metrics.json',info);dump(ep/'mode_diagnostics.json',modeinfo)
  with (OUT/'epochs.jsonl').open('a') as fp:fp.write(json.dumps(info)+'\n')
  print(f'EPOCH COMPLETE {epoch}: rawCV75={raw["cvar75"]:.5f} calibratedCV75={fixed["cvar75"]:.5f} eta={float(model.eta.numpy()):.5f}',flush=True)
 dump(OUT/'frozen_before_test.json',{'final_epoch':5,'updates':390625,'selection':'Fixed final epoch only'})
 test,tv=paths(p,state,s0,100000,np.random.default_rng(5713));ft=sample_tensors(test,s0,DAYS)[0]
 positions={'paper_raw':predict_positions(model,ft),'paper_bn_fixed':predict_positions(evaluator,ft),'BS':bs_hedge(test,tv,s0,DAYS)}
 arrays={name:pnl(test,hold,q,s0,DAYS) for name,hold in positions.items()}
 with np.load(PARENT/'test_pnl.npz') as old:np.testing.assert_array_equal(arrays['BS'],old['BS'])
 for pl in arrays.values():
  loss=-pl;eta=np.quantile(loss,.75);assert abs(cvar(loss,.75)-(eta+4*np.maximum(loss-eta,0).mean()))<1e-8
 np.savez_compressed(OUT/'test_pnl.npz',**arrays)
 h,hv=observed_path(p,state,s0,r[1187:]);np.testing.assert_allclose(h[0],hist[1187:],atol=1e-10)
 historical={}
 for name,m in [('paper_raw',model),('paper_bn_fixed',evaluator),('BS',None)]:
  pos=predict_positions(m,sample_tensors(h,s0,DAYS)[0]) if m is not None else bs_hedge(h,hv,s0,DAYS)
  historical[name]={'index_pnl':float(pnl(h,pos,q,s0,DAYS)[0]),'holdings':pos[0].tolist()}
 hh=predict_positions(evaluator,vf[:4096],batch_size=256);np.testing.assert_array_equal(hh,predict_positions(evaluator,vf[:4096],batch_size=4096))
 np.testing.assert_allclose(hh[:16],predict_positions(evaluator,vf[:16],batch_size=1),atol=1e-4,rtol=1e-5)
 verify_parent()
 for name,d in protocol['source_hashes'].items():assert sha(ROOT/name)==d,name
 dump(OUT/'metrics.json',{'simulated_test':{k:metrics(v) for k,v in arrays.items()},'historical':historical,'q':q,'original_spot':s0,'elapsed_seconds':time.monotonic()-start,'epochs':5,'updates':390625,'unique_paths':20000000,'processed_paths':100000000,'checks':{'identical_tensor_hashes_all_repeated_epochs':epoch_hashes,'all_training_scalars_finite':True,'paired_BS_bitwise_equal':True,'CVaR_OCE_sort_identity':True,'batch_size_invariance':True,'parent_and_frozen_baselines_unchanged':True}})
 dump(OUT/'STATUS.json',{'execution':'COMPLETE','completed_epochs':5,'total_updates':390625,'full300historical_replication':False})
 print('COMPLETE '+json.dumps(read(OUT/'metrics.json')),flush=True)
if __name__=='__main__':main()
