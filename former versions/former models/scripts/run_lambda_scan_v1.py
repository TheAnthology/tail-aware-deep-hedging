"""Validation-selected lambda scan; sequential isolated training workers."""
import os
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL','3');os.environ.setdefault('OPENBLAS_NUM_THREADS','1');os.environ.setdefault('OMP_NUM_THREADS','1')
from pathlib import Path
import json,hashlib,time,sys,subprocess,argparse
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'results/risk_lambda_scan_v1';OLD=ROOT/'results/risk_ablation_simpler_v1'
GRID=[0.,.01,.03,.1,.3,1.,3.]
def tag(weight):return 'lambda_'+str(weight).replace('.','p')
def dump(p,x):p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def worker(weight):
 import numpy as np
 import tensorflow as tf
 from heston52_market import sample_paths,path_chunks,sample_tensors,pnl
 from tf_heston52_engine import Heston52Hedge,predict
 from run_section52_calibrated import calibrate
 from run_risk_ablation_v1 import summarize
 tf.config.threading.set_intra_op_parallelism_threads(1);tf.config.threading.set_inter_op_parallelism_threads(1)
 out=OUT/tag(weight);out.mkdir(exist_ok=False);start=time.monotonic()
 q=json.loads((OLD/'metrics.json').read_text())['q']
 model=Heston52Hedge(seed=5201,recurrent=False);model.initialize()
 evaluator=Heston52Hedge(recurrent=False);evaluator.initialize()
 cal=sample_paths(8192,np.random.default_rng(5291));cf,_,_=sample_tensors(cal);del cal
 val=sample_paths(32768,np.random.default_rng(5212));vf,_,_=sample_tensors(val)
 variables=tuple(model.trainable_variables);optimizer=tf.keras.optimizers.Adam(.005,beta_1=.9,beta_2=.999,epsilon=1e-7);optimizer.build(list(variables))
 indices=[[i for i,v in enumerate(variables) if any(v is x for x in net.trainable_variables)] for net in model.nets]
 @tf.function(input_signature=[tf.TensorSpec([256,30,2],tf.float32),tf.TensorSpec([256,30,2],tf.float32),tf.TensorSpec([256],tf.float32)])
 def step(f,ch,z):
  with tf.GradientTape() as tape:
   h=model(f,training=True);liability=z-tf.reduce_sum(h*ch,axis=[1,2]);loss=liability-q
   oce=model.eta+2*tf.reduce_mean(tf.nn.relu(liability-model.eta))-q;soft=tf.reduce_mean(tf.math.sigmoid(loss/.1));objective=oce+weight*soft
  gradients=tape.gradient(objective,variables)
  if any(g is None for g in gradients):raise RuntimeError('Disconnected gradient')
  tf.debugging.check_numerics(objective,'objective')
  for g in gradients:tf.debugging.check_numerics(g,'gradient')
  norms=tf.stack([tf.linalg.global_norm([gradients[i] for i in ix]) for ix in indices])
  optimizer.apply_gradients(zip(gradients,variables))
  return objective,oce,soft,tf.linalg.global_norm(gradients),norms
 def evaluate(updates):
  saved=[v.numpy().copy() for v in model.weights];evaluator.set_weights(model.get_weights());calibrate(evaluator,cf);pl=pnl(val,predict(evaluator,vf),q)
  row={'lambda':weight,'updates':updates,'validation':summarize(pl),'elapsed_seconds':time.monotonic()-start}
  with (out/'validation.jsonl').open('a') as s:s.write(json.dumps(row)+'\n')
  assert all(np.array_equal(v.numpy(),old) for v,old in zip(model.weights,saved))
  if updates==20000:
   model.save_weights(out/'training.weights.h5');evaluator.save_weights(out/'calibrated.weights.h5')
   np.savez_compressed(out/'validation_pnl.npz',pnl=pl)
  print(f'LAMBDA {weight} UPDATE {updates}: val P(loss)={row["validation"]["probability_loss"]:.5f} CVaR50={row["validation"]["cvar50"]:.5f} CVaR95={row["validation"]["cvar95"]:.5f} MSE={row["validation"]["mse"]:.5f}',flush=True)
  return row
 import csv
 updates=0
 with (out/'training.csv').open('w',newline='') as s:
  w=csv.writer(s);w.writerow(['update','objective','oce_minus_q','soft_probability','gradient_norm'])
  for data in path_chunks(5120000,8192,5211):
   f,ch,z=sample_tensors(data)
   for lo in range(0,len(z),256):
    values=step(f[lo:lo+256],ch[lo:lo+256],z[lo:lo+256]);updates+=1;w.writerow([updates,*[float(v.numpy()) for v in values[:4]]])
    if updates==1:
     norms=values[-1].numpy();assert (norms>0).all();dump(out/'first_gradients.json',{'all30_nonzero':True,'norms':norms.tolist()})
    if updates%100==0:s.flush()
    if updates in [5000,10000,15000,20000]:final=evaluate(updates)
 assert updates==20000
 tf.train.Checkpoint(model=model,optimizer=optimizer).write(str(out/'final_adam_checkpoint'))
 # Complete state = training HDF5 (all BN buffers) + optimizer checkpoint.
 dump(out/'result.json',{'lambda':weight,'updates':updates,'training_paths':updates*256,'final_validation':final['validation'],'elapsed_seconds':time.monotonic()-start,'optimizer_iterations':int(optimizer.iterations.numpy()),'restoration_bundle':'training.weights.h5 plus final_adam_checkpoint; preserve both for future exact optimizer-state continuation'})


def select(validation):
 base=validation[tag(0.)];keys=['cvar50','cvar95','mse'];choices={}
 for name,tolerance in [('strict_no_worse',0.),('allow_5percent',.05)]:
  feasible=[w for w in GRID if all(validation[tag(w)][k]<=base[k]*(1+tolerance) for k in keys)]
  best=min(feasible,key=lambda w:(validation[tag(w)]['probability_loss'],w))
  choices[name]={'lambda':best,'feasible_lambdas':feasible,'relative_tolerance':tolerance,'risk_constraints':keys,'objective':'Minimum hard P(loss>0), tie smaller lambda'}
 # Four-metric Pareto set, empirical validation values only.
 keys=['probability_loss','cvar50','cvar95','mse'];frontier=[]
 for w in GRID:
  a=validation[tag(w)];dominated=False
  for other in GRID:
   if other==w:continue
   b=validation[tag(other)]
   if all(b[k]<=a[k] for k in keys) and any(b[k]<a[k] for k in keys):dominated=True;break
  if not dominated:frontier.append(w)
 return {'rules':choices,'validation_pareto_lambdas':frontier,'note':'Chosen using validation only, before reading new test data; tolerance is a predeclared preference, not a claim of no performance sacrifice.'}


def evaluate_all():
 import numpy as np
 import tensorflow as tf
 from heston52_market import path_chunks,sample_tensors,pnl,sample_paths
 from tf_heston52_engine import Heston52Hedge,predict
 from run_risk_ablation_v1 import summarize
 tf.config.threading.set_intra_op_parallelism_threads(1);tf.config.threading.set_inter_op_parallelism_threads(1)
 q=json.loads((OLD/'metrics.json').read_text())['q'];models={};validation={}
 # Existing endpoints are re-evaluated on the very same validation pool.
 val=sample_paths(32768,np.random.default_rng(5212));vf,_,_=sample_tensors(val)
 for weight in GRID:
  if weight in [0.,1.]:file=OLD/('A_calibrated.weights.h5' if weight==0 else 'C_calibrated.weights.h5')
  else:file=OUT/tag(weight)/'calibrated.weights.h5'
  m=Heston52Hedge(recurrent=False);m.initialize();m.load_weights(file);models[tag(weight)]=m
  pl=pnl(val,predict(m,vf),q);validation[tag(weight)]=summarize(pl)
  if weight in [0.,1.]:
   records=[json.loads(s) for s in (OLD/'validation.jsonl').read_text().splitlines()];old=next(x['validation'] for x in records if x['model']==('A' if weight==0 else 'C') and x['updates']==20000)
   for k in ['cvar50','cvar95','mse','probability_loss']:assert np.isclose(validation[tag(weight)][k],old[k],rtol=1e-12,atol=1e-12)
 decision=select(validation);dump(OUT/'validation_metrics.json',validation);dump(OUT/'selection.json',decision)
 print('VALIDATION SELECTION '+json.dumps(decision),flush=True)
 # Only after the selected lambda is committed do we access the new test.
 arrays={k:[] for k in models};tested=0
 for data in path_chunks(1000000,8192,5503):
  f,_,_=sample_tensors(data)
  for key,m in models.items():arrays[key].append(pnl(data,predict(m,f),q))
  tested+=len(f)
  if tested%131072==0:print(f'SCAN TEST {tested}/1000000',flush=True)
 assert tested==1000000
 pl={k:np.concatenate(v) for k,v in arrays.items()};metrics={k:summarize(v) for k,v in pl.items()}
 np.savez_compressed(OUT/'test_pnl.npz',**pl)
 dump(OUT/'test_metrics.json',{'test_paths':tested,'seed':5503,'q':q,'metrics':metrics,'selection':decision,'note':'All grid results reported; do not reselect a winner using this test table.'})
 print('TEST RESULT '+json.dumps({str(w):{k:metrics[tag(w)][k] for k in ['probability_loss','cvar50','cvar95','mse']} for w in GRID}),flush=True)


def main():
 OUT.mkdir(exist_ok=False);start=time.monotonic()
 protocol={'id':'risk_lambda_scan_v1','lambda_grid':GRID,'new_train_lambdas':[.01,.03,.1,.3,3.],'reused_endpoints':{'0':'risk_ablation_simpler_v1/A','1':'risk_ablation_simpler_v1/C'},'fixed_risk':{'alpha':.5,'a':0.,'tau':.1,'s':1.},'architecture_data_bn':'Frozen Simpler baseline unchanged; parent docs/simpler_risk_baseline_v1.md','training':{'updates_each':20000,'paths_each':5120000,'seed':5211,'network_seed':5201,'batch':256,'chunk':8192,'Adam_lr':.005,'beta1':.9,'beta2':.999,'epsilon':1e-7,'same_initial_weights_and_draws_as_endpoints':True},'calibration':{'paths':8192,'seed':5291,'procedure':'Same sequential date/layer population BN; evaluator clones only'},'validation':{'paths':32768,'seed':5212,'checkpoint_selection':'None; final prescribed weights','primary_weight_selection':'Minimum P(loss>0), subject to CVaR50,CVaR95,MSE <= lambda0 values, tie lower lambda','secondary_weight_selection':'Same rule but permit at most5% increase in each of those risk/performance metrics','fallback':'lambda0 always feasible','pareto':'Report frequency,CVaR50,CVaR95,MSE empirical validation frontier'},'test':{'paths':1000000,'seed':5503,'chunk':8192,'all_grid_same_paths':True,'new_stream':True,'access_after_selection_committed':True,'all_metrics_reported_no_reselection':True},'limits':['Best only within this finite grid and one seed, not a globally optimal lambda.','a0 remains any-loss frequency, not severe-loss frequency.','5% tolerance is a stated preference and is not cost-free improvement.','Training budgets equal; convergence may vary by lambda.','No architecture, feature scaling, transaction costs or premium changes.'],'execution':'Sequential isolated worker processes to bound TF RAM and file churn'}
 dump(OUT/'protocol.json',protocol)
 files=[Path(__file__),ROOT/'scripts/tf_heston52_engine.py',ROOT/'scripts/heston52_market.py',ROOT/'scripts/run_section52_calibrated.py',ROOT/'scripts/run_risk_ablation_v1.py',OLD/'A_calibrated.weights.h5',OLD/'C_calibrated.weights.h5']
 hashes={str(p.relative_to(ROOT)):sha(p) for p in files};dump(OUT/'implementation.json',hashes)
 for weight in [.01,.03,.1,.3,3.]:subprocess.run([sys.executable,'-u',str(Path(__file__).resolve()),'--worker',str(weight)],cwd=ROOT,check=True)
 assert all(sha(ROOT/p)==expected for p,expected in hashes.items())
 # Evaluation in a clean process, after every isolated training worker exits.
 subprocess.run([sys.executable,'-u',str(Path(__file__).resolve()),'--evaluate'],cwd=ROOT,check=True)
 assert all(sha(ROOT/p)==expected for p,expected in hashes.items())
 dump(OUT/'completion.json',{'execution':'COMPLETE','elapsed_seconds':time.monotonic()-start,'frozen_sources_match':True,'new_models':5,'grid_models':7,'test_paths_each':1000000})
if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('--worker',type=float);parser.add_argument('--evaluate',action='store_true');args=parser.parse_args()
 if args.worker is not None:worker(args.worker)
 elif args.evaluate:evaluate_all()
 else:main()
