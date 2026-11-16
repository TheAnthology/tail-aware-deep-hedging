"""Predeclared matched A/B/C risk-objective pilot, frozen Simpler architecture."""
import os
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL','3');os.environ.setdefault('OPENBLAS_NUM_THREADS','1');os.environ.setdefault('OMP_NUM_THREADS','1')
from pathlib import Path
import json,hashlib,time,csv
import numpy as np
import tensorflow as tf
from heston52_market import sample_paths,path_chunks,sample_tensors,pnl,metrics
from tf_heston52_engine import Heston52Hedge,predict
from run_section52_calibrated import calibrate,distribution
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'results/risk_ablation_simpler_v1'
def dump(p,x):p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def terms(loss,eta,alpha=.5,a=0.,tau=.1,scale=1.,weight=1.):
 cvar=eta+tf.reduce_mean(tf.nn.relu(loss-eta))/(1-alpha)
 probability=tf.reduce_mean(tf.math.sigmoid((loss-a)/tau))
 return cvar/scale,probability,cvar/scale+weight*probability

def loss_checks(q):
 loss=tf.constant([-3.,-1.,.5,2.],tf.float64);eta=tf.constant(-.25,tf.float64)
 cv,prob,combined=terms(loss,eta)
 assert np.isclose(cv.numpy(),1.25)
 x=loss.numpy();reference=np.mean(1/(1+np.exp(-x/.1)))
 assert np.isclose(prob.numpy(),reference)
 assert np.isclose(combined.numpy(),cv.numpy()+prob.numpy())
 a0,b0,c0=terms(loss,eta,weight=0.);assert a0.numpy()==c0.numpy()
 shifted=terms(loss-q,eta-q)
 assert np.isclose(shifted[0].numpy(),cv.numpy()-q)
 assert not np.isclose(shifted[1].numpy(),prob.numpy())
 with tf.GradientTape() as tape:
  tape.watch(loss);cv,prob,combined=terms(loss,eta)
 grad=tape.gradient(combined,loss).numpy();assert np.all(grad>=0)
 eps=1e-5;finite=[]
 for i in range(4):
  plus=x.copy();minus=x.copy();plus[i]+=eps;minus[i]-=eps
  finite.append((terms(tf.constant(plus),eta)[2].numpy()-terms(tf.constant(minus),eta)[2].numpy())/(2*eps))
 assert np.allclose(grad,finite,atol=1e-6)
 return {'oce_matches_worst_half':True,'sigmoid_matches_numpy':True,'combined_lambda0_matches_cvar':True,'premium_enters_probability':True,'combined_loss_monotone_in_each_path_loss':True,'loss_gradient_matches_finite_difference':True}

def make_step(model,name,q):
 # Eta is an OCE auxiliary variable; frequency-only has no eta to optimize.
 variables=tuple(v for v in model.trainable_variables if name!='B' or v is not model.eta)
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
   objective=oce if name=='A' else soft if name=='B' else oce+soft
  gradients=tape.gradient(objective,variables)
  if any(g is None for g in gradients):raise RuntimeError('Disconnected gradient')
  tf.debugging.check_numerics(objective,'objective')
  for g in gradients:tf.debugging.check_numerics(g,'gradient')
  norms=tf.stack([tf.linalg.global_norm([gradients[i] for i in ix]) for ix in indices])
  empirical=tf.reduce_mean(tf.sort(loss)[128:]);hard=tf.reduce_mean(tf.cast(loss>0,tf.float32))
  optimizer.apply_gradients(zip(gradients,variables))
  return objective,oce,soft,empirical,hard,tf.linalg.global_norm(gradients),norms
 return step,optimizer

def summarize(x):
 r=distribution(x)
 r['tail_probabilities']={str(a):float((x<-a).mean()) for a in [0.,.5,1.,2.]}
 r['soft_tail_probability_a0_tau01']=float(np.mean(tf.math.sigmoid(tf.constant(-x/.1,tf.float64)).numpy()))
 return r

def main():
 tf.config.threading.set_intra_op_parallelism_threads(1);tf.config.threading.set_inter_op_parallelism_threads(1)
 OUT.mkdir(exist_ok=False);start=time.monotonic();parent=json.loads((ROOT/'protocol_section52_v1.json').read_text());q=json.loads((ROOT/'results/section52_v1_final/metrics.json').read_text())['q']
 protocol={'id':'risk_ablation_simpler_v1','scope':'Single-configuration, single-initialization matched pilot; Recurrent work deferred; no transaction costs',
  'parent_market':parent['paper_explicit'],'architecture_override':'30 independent 2->17->17->2; logS,V inputs only, stock/swap total-position outputs; same Dense/BN/ReLU implementation',
  'losses':{'A':'CVaR50(-PL)','B':'mean(sigmoid((-PL-a)/tau))','C':'CVaR50(-PL)/s + lambda*mean(sigmoid((-PL-a)/tau))'},
  'risk_constants':{'alpha':.5,'a':0.,'tau':.1,'s':1.,'lambda':1.,'q':q,'units':'Currency with spot100; a0 means any loss, not severe loss; s1 is fixed one-currency scale'},
  'optimization':{'updates_each':20000,'fresh_paths_each':5120000,'batch_size':256,'lr':.005,'adam_betas':[.9,.999],'adam_epsilon':1e-7,'same_initial_network_seed':5201,'same_training_seed':5211,'chunk_paths':8192,'A_C_initial_liability_OCE_intercept':0.,'B_OCE_intercept':'unused, excluded from optimizer'},
  'bn':{'epsilon':.001,'momentum':.99,'calibration_paths':8192,'calibration_seed':5291,'method':'Same date/layer sequential population statistics; infer-mode upstream, float64 moments; evaluation clone only'},
  'validation':{'paths':32768,'seed':5212,'milestones':[0,5000,10000,15000,20000],'selection':'None; final prescribed weights only'},
  'test':{'paths':1000000,'seed':5403,'chunk_paths':8192,'same_paths_for_A_B_C':True,'new_stream_not_used_in_prior_experiments':True,'evaluation_order':'Only after all three models finish; no parameter changes based on results'},
  'heston_reference':{'scope':'First32768 of this new million-path stream only, clearly separate subset comparison'},
  'metrics':['hard P(PL<-a) at a0,.5,1,2','CVaR50','CVaR95','CVaR99','MSE','mean P&L','P&L quantiles','turnover and holdings magnitudes','full-sample histogram'],
  'limits':['One seed and one lambda/threshold/temperature configuration, not a hyperparameter search.','No transaction costs or position constraints; probability-only may accept severe rare losses.','BN calibration is a declared numerical protocol, not recovered author settings.','No original convex-risk guarantees claimed for B/C.','Monte Carlo uncertainty conditional on trained weights does not measure training-seed uncertainty.']}
 dump(OUT/'protocol.json',protocol)
 files=[Path(__file__),ROOT/'scripts/heston52_market.py',ROOT/'scripts/tf_heston52_engine.py',ROOT/'scripts/run_section52_calibrated.py',ROOT/'scripts/heston52_benchmark.py']
 hashes={str(p.relative_to(ROOT)):sha(p) for p in files};dump(OUT/'implementation.json',hashes);dump(OUT/'loss_checks.json',loss_checks(q))
 models={name:Heston52Hedge(seed=5201,recurrent=False,bn_epsilon=.001,bn_momentum=.99) for name in ['A','B','C']}
 for m in models.values():m.initialize()
 initial=[v.numpy().copy() for v in models['A'].weights]
 assert all(all(np.array_equal(v.numpy(),x) for v,x in zip(m.weights,initial)) for m in models.values())
 dump(OUT/'initialization.json',{'identical_all_weights_and_buffers':True,'trainable_tensors_network_only':300,'trainable_scalars_network_only':13830,'A_C_auxiliary_eta_scalars':1,'B_unused_eta':True})
 cal=sample_paths(8192,np.random.default_rng(5291));cf,_,_=sample_tensors(cal);del cal
 val=sample_paths(32768,np.random.default_rng(5212));vf,_,_=sample_tensors(val)
 evaluator=Heston52Hedge(recurrent=False);evaluator.initialize()
 def evaluate(name,updates):
  m=models[name];saved=[v.numpy().copy() for v in m.weights];evaluator.set_weights(m.get_weights());calibrate(evaluator,cf);h=predict(evaluator,vf)
  stats=summarize(pnl(val,h,q));stats['mean_turnover_stock']=metrics(val,h,q)['mean_turnover_stock'];stats['mean_turnover_swap']=metrics(val,h,q)['mean_turnover_swap']
  row={'model':name,'updates':updates,'validation':stats,'max_abs_holdings':np.abs(h).max(axis=(0,1)).tolist(),'swap_median_k15':float(np.median(h[:,15,1])),'elapsed_seconds':time.monotonic()-start}
  if updates==20000:
   m.save_weights(OUT/f'{name}_training.weights.h5');evaluator.save_weights(OUT/f'{name}_calibrated.weights.h5')
  assert all(np.array_equal(v.numpy(),x) for v,x in zip(m.weights,saved))
  with (OUT/'validation.jsonl').open('a') as s:s.write(json.dumps(row,allow_nan=False)+'\n')
  print(f'VALIDATE {name} {updates}: CVaR50={stats["cvar50"]:.5f} CVaR95={stats["cvar95"]:.5f} P(loss)={stats["probability_loss"]:.5f}',flush=True)
 for name in models:evaluate(name,0)
 steps={};optimizers={}
 for name,m in models.items():steps[name],optimizers[name]=make_step(m,name,q)
 updates=0
 with (OUT/'training_batches.csv').open('w',newline='') as stream:
  writer=csv.writer(stream);writer.writerow(['update','model','objective','oce_minus_q','soft_probability','batch_cvar50','batch_probability_loss','gradient_norm'])
  for market in path_chunks(5120000,8192,5211):
   f,ch,z=sample_tensors(market)
   for lo in range(0,len(z),256):
    updates+=1
    for name in models:
     values=steps[name](f[lo:lo+256],ch[lo:lo+256],z[lo:lo+256]);writer.writerow([updates,name,*[float(v.numpy()) for v in values[:6]]])
     if updates==1:
      norms=values[-1].numpy();assert (norms>0).all();dump(OUT/f'{name}_first_gradients.json',{'all30_network_gradient_norms':norms.tolist(),'optimizer_variables':len(optimizers[name].variables)})
    if updates%100==0:stream.flush()
    if updates in [5000,10000,15000,20000]:
     for name in models:evaluate(name,updates)
 assert updates==20000
 # Independent baseline reproduction check, before accessing new test data.
 old=Heston52Hedge(recurrent=False);old.initialize();old.load_weights(ROOT/'results/section52_fig4_control/final.weights.h5')
 deviations=[float(np.max(np.abs(v.numpy()-w.numpy()))) for v,w in zip(models['A'].weights,old.weights)]
 replay={'max_abs_deviation':max(deviations),'bitwise_identical':all(x==0 for x in deviations),'source':'original matched Simpler raw final checkpoint'}
 dump(OUT/'baseline_replay.json',replay)
 if not np.isclose(max(deviations),0,atol=1e-5):raise RuntimeError('Matched A baseline replay failed; do not test before explaining discrepancy')
 for name,m in models.items():m.load_weights(OUT/f'{name}_calibrated.weights.h5')
 from heston52_benchmark import HestonGridHedge
 grid=HestonGridHedge(cache_dir=ROOT/'results/section52_quantlib_cache',sensitive_variance=.001)
 collected={name:[] for name in models};reference=[];subsets={name:[] for name in models}
 aggregates={name:{'turnover_sum':np.zeros(2),'holding_abs_max':np.zeros(2),'holding_sum':np.zeros((30,2))} for name in models};tested=0
 for market in path_chunks(1000000,8192,5403):
  f,_,_=sample_tensors(market)
  for name,m in models.items():
   h=predict(m,f);pl=pnl(market,h,q);collected[name].append(pl)
   a=aggregates[name];bounded=np.concatenate([np.zeros_like(h[:,:1,:]),h,np.zeros_like(h[:,:1,:])],axis=1)
   a['turnover_sum']+=np.abs(np.diff(bounded,axis=1)).sum(axis=(0,1));a['holding_abs_max']=np.maximum(a['holding_abs_max'],np.abs(h).max(axis=(0,1)));a['holding_sum']+=h.sum(axis=0)
   if tested<32768:subsets[name].append(pl)
  if tested<32768:reference.append(pnl(market,grid.hedge_paths(market),q))
  tested+=len(f)
  if tested%131072==0:print(f'TEST {tested}/1000000 seconds={time.monotonic()-start:.1f}',flush=True)
 assert tested==1000000
 arrays={n:np.concatenate(v) for n,v in collected.items()};result={n:summarize(x) for n,x in arrays.items()}
 for n,r in result.items():
  a=aggregates[n];r['mean_turnover_stock']=float(a['turnover_sum'][0]/tested);r['mean_turnover_swap']=float(a['turnover_sum'][1]/tested);r['max_abs_holdings']=a['holding_abs_max'].tolist();r['mean_holdings_by_date']=(a['holding_sum']/tested).tolist()
 edges=np.linspace(-8,8,161);hist={'edges':edges.tolist(),'note':'Full-sample normalized; all observations included in metrics, outside counts retained.'}
 for n,x in arrays.items():hist[n]={'counts':np.histogram(x,edges)[0].tolist(),'below':int((x<edges[0]).sum()),'above':int((x>edges[-1]).sum())}
 reference_array=np.concatenate(reference);subset_result={n:summarize(np.concatenate(v)) for n,v in subsets.items()};subset_result['heston']=summarize(reference_array)
 dump(OUT/'reference_subset.json',{'paths':len(reference_array),'metrics':subset_result});dump(OUT/'histogram.json',hist);np.savez_compressed(OUT/'pnl.npz',**arrays)
 assert all(sha(ROOT/p)==expected for p,expected in hashes.items())
 dump(OUT/'metrics.json',{'protocol_id':protocol['id'],'test_paths':tested,'q':q,'training_updates_each':updates,'training_paths_each':updates*256,'metrics':result,'baseline_replay':replay,'elapsed_seconds':time.monotonic()-start,'original_sources_unchanged':True,'claim_limit':'Matched single-seed pilot only; no efficacy claim across other a,tau,lambda,seeds/costs.'})
 print('RESULT '+json.dumps({n:{k:r[k] for k in ['cvar50','cvar95','mse','probability_loss']} for n,r in result.items()}),flush=True)
if __name__=='__main__':main()
