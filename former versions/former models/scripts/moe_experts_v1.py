"""Preference-specific expert members for the frozen Phase1 MoE experiment.

Reuses original A/B as seed1, recalibrates copies, and runs four new experts
in isolated processes. No ensemble parameters are averaged. No test is read.
"""
import os
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')
from pathlib import Path
import argparse
import csv
import hashlib
import json
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/preference_moe_v1/experts'
OLD = ROOT / 'results/risk_ablation_simpler_v1'
SEEDS = {1: (5201, 5211), 2: (6202, 6212), 3: (6203, 6213)}


def dump(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def worker(preference, member):
    import numpy as np
    import tensorflow as tf
    from heston52_market import sample_paths, path_chunks, sample_tensors, pnl
    from tf_heston52_engine import Heston52Hedge, predict
    from run_section52_calibrated import calibrate
    from run_risk_ablation_v1 import make_step, summarize
    tf.config.threading.set_intra_op_parallelism_threads(1)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    start = time.monotonic()
    out = OUT / f'{preference}_{member}'
    out.mkdir(exist_ok=False)
    net_seed, train_seed = SEEDS[member]
    q = json.loads((OLD/'metrics.json').read_text())['q']
    source_name = 'A' if preference == 'C' else 'B'
    model = Heston52Hedge(seed=net_seed, recurrent=False)
    model.initialize()
    evaluator = Heston52Hedge(recurrent=False)
    evaluator.initialize()
    cal = sample_paths(8192, np.random.default_rng(6291))
    cf, _, _ = sample_tensors(cal)
    del cal
    val = sample_paths(32768, np.random.default_rng(6214))
    vf, _, _ = sample_tensors(val)
    def evaluate(updates, save=False):
        before = [v.numpy().copy() for v in model.weights]
        evaluator.set_weights(model.get_weights())
        calibrate(evaluator, cf)
        h = predict(evaluator, vf)
        pl = pnl(val, h, q)
        stats = summarize(pl)
        stats['pnl_standard_deviation'] = float(np.std(pl))
        stats['probability_profit_gt_1'] = float(np.mean(pl > 1))
        row = {'preference':preference, 'member':member, 'updates':updates,
               'validation':stats, 'elapsed_seconds':time.monotonic()-start}
        assert all(np.array_equal(v.numpy(), x) for v,x in zip(model.weights,before))
        with (out/'validation.jsonl').open('a') as stream:
            stream.write(json.dumps(row,allow_nan=False)+'\n')
        if save:
            model.save_weights(out/'training.weights.h5')
            evaluator.save_weights(out/'calibrated.weights.h5')
            np.savez_compressed(out/'validation_pnl.npz', pnl=pl)
        print(f'EXPERT {preference}_{member} UPDATE {updates}: '
              f'P(loss)={stats["probability_loss"]:.6f} '
              f'CVaR50={stats["cvar50"]:.6f} CVaR95={stats["cvar95"]:.6f}',flush=True)
        return row
    if member == 1:
        raw_source = OLD/f'{source_name}_training.weights.h5'
        before_sha = sha(raw_source)
        model.load_weights(raw_source)
        learned = [v.numpy().copy() for v in model.trainable_variables]
        final = evaluate(20000,save=True)
        assert all(np.array_equal(v.numpy(),x) for v,x in zip(model.trainable_variables,learned))
        assert sha(raw_source) == before_sha
        result = {'preference':preference,'member':member,'network_seed':net_seed,
                  'training_seed':train_seed,'training_updates':20000,'new_updates':0,
                  'source_original':str(raw_source.relative_to(ROOT)),
                  'source_sha256':before_sha,'calibration_seed':6291,'validation_seed':6214,
                  'final_validation':final['validation'],'optimizer_state':'Original pilot did not save optimizer; no claim of exact continuation',
                  'elapsed_seconds':time.monotonic()-start}
    else:
        step, optimizer = make_step(model, source_name, q)
        updates = 0
        with (out/'training.csv').open('w',newline='') as stream:
            writer = csv.writer(stream)
            writer.writerow(['update','objective','oce_minus_q','soft_probability',
                             'batch_cvar50','batch_probability_loss','gradient_norm'])
            for market in path_chunks(5120000,8192,train_seed):
                f,ch,z = sample_tensors(market)
                for lo in range(0,len(z),256):
                    values = step(f[lo:lo+256],ch[lo:lo+256],z[lo:lo+256])
                    updates += 1
                    writer.writerow([updates,*[float(v.numpy()) for v in values[:6]]])
                    if updates == 1:
                        norms = values[-1].numpy()
                        assert (norms > 0).all()
                        dump(out/'first_gradients.json',{'all30_nonzero':True,'norms':norms.tolist()})
                    if updates % 100 == 0:
                        stream.flush()
                    if updates in [5000,10000,15000,20000]:
                        final = evaluate(updates,save=updates==20000)
                        dump(out/'STATUS.json',{'execution':'TRAINING','updates':updates})
        assert updates == 20000
        tf.train.Checkpoint(model=model,optimizer=optimizer).write(str(out/'final_adam_checkpoint'))
        result = {'preference':preference,'member':member,'network_seed':net_seed,
                  'training_seed':train_seed,'training_updates':updates,'new_updates':updates,
                  'training_paths':updates*256,'calibration_seed':6291,'validation_seed':6214,
                  'final_validation':final['validation'],
                  'optimizer_iterations':int(optimizer.iterations.numpy()),
                  'restoration_bundle':'training.weights.h5 plus final_adam_checkpoint (all BN buffers require HDF5)',
                  'elapsed_seconds':time.monotonic()-start}
    dump(out/'result.json', result)
    dump(out/'STATUS.json',{'execution':'COMPLETE','updates':20000})


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    if (OUT/'protocol.json').exists():
        raise RuntimeError('Expert run already initialized; refusing to overwrite')
    start = time.monotonic()
    dump(OUT/'protocol.json',{
        'id':'preference_moe_experts_v1','preference_members':{'C':3,'P':3},
        'objectives':{'C':'CVaR50(-PL)','P':'mean(sigmoid(-PL/.1))'},
        'q':json.loads((OLD/'metrics.json').read_text())['q'],
        'architecture':'Frozen Simpler 30 independent2->17->17->2',
        'member_seeds':{str(k):{'network_seed':v[0],'training_seed':v[1]} for k,v in SEEDS.items()},
        'training':{'updates_each':20000,'paths_each':5120000,'batch':256,'chunk':8192,
                    'Adam_lr':.005,'beta1':.9,'beta2':.999,'epsilon':1e-7,
                    'member1_reused':'Existing original A and B raw weights; same training budget',
                    'pairing':'C and P member m use the same original initialization and draws'},
        'calibration':{'paths':8192,'seed':6291,'all_six_members':'Independent member BN population statistics on same calibration pool; copied member1 raw recalibrated without modifying originals'},
        'validation':{'paths':32768,'seed':6214,'selection':'None, final prescribed20k weights'},
        'test':'No final test read in this expert-only script',
        'aggregation':'Average hedging positions strictly within preference only; later pipeline computes actual ensemble P&L',
        'prior_experiments':'Preserved unchanged; old lambda scan remains paused'})
    files=[Path(__file__),ROOT/'scripts/heston52_market.py',ROOT/'scripts/tf_heston52_engine.py',
           ROOT/'scripts/run_risk_ablation_v1.py',ROOT/'scripts/run_section52_calibrated.py',
           OLD/'A_training.weights.h5',OLD/'B_training.weights.h5']
    hashes={str(p.relative_to(ROOT)):sha(p) for p in files}
    dump(OUT/'implementation.json',hashes)
    tasks=[('C',1),('P',1),('C',2),('P',2),('C',3),('P',3)]
    for preference,member in tasks:
        dump(OUT/'STATUS.json',{'execution':'RUNNING','current':f'{preference}_{member}'})
        subprocess.run([sys.executable,'-u',str(Path(__file__).resolve()),'--worker',preference,str(member)],
                       cwd=ROOT,check=True)
    manifest={'preferences':{'C':[],'P':[]},'calibration_seed':6291,'validation_seed':6214,
              'calibration_paths':8192,'validation_paths':32768,
              'elapsed_seconds':time.monotonic()-start,'test_accessed':False}
    for preference,member in tasks:
        out=OUT/f'{preference}_{member}'
        result=json.loads((out/'result.json').read_text())
        result['raw_weights']=str(out/'training.weights.h5')
        result['calibrated_weights']=str(out/'calibrated.weights.h5')
        manifest['preferences'][preference].append(result)
    assert all(sha(ROOT/p)==s for p,s in hashes.items())
    dump(OUT/'manifest.json',manifest)
    dump(OUT/'STATUS.json',{'execution':'COMPLETE','members':6,'new_trained':4,'test_accessed':False,
                          'original_hashes_unchanged':True,'elapsed_seconds':time.monotonic()-start})
    print('EXPERTS COMPLETE '+json.dumps({'manifest':str(OUT/'manifest.json'),
          'seconds':time.monotonic()-start}),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--worker',nargs=2,metavar=('PREFERENCE','MEMBER'))
    args=parser.parse_args()
    if args.worker:
        worker(args.worker[0],int(args.worker[1]))
    else:
        main()
