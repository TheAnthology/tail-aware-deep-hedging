"""One fixed-weight hinge-penalty approximation of the CVaR-constrained problem.

Finite penalty and stochastic Adam do not guarantee feasibility. Preserve the
cap, report the final candidate even if infeasible, and never select on test.
"""
import os
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')
from pathlib import Path
import csv
import hashlib
import json
import time
import numpy as np
import tensorflow as tf
from heston52_market import sample_paths, path_chunks, sample_tensors, pnl
from tf_heston52_engine import Heston52Hedge, predict
from run_section52_calibrated import calibrate
from run_risk_ablation_v1 import summarize

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/risk_constrained_beta10_v1'
OLD = ROOT / 'results/risk_ablation_simpler_v1'
BETA = 10.

def dump(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def objective_terms(loss, eta_loss, cap):
    oce = eta_loss + 2 * tf.reduce_mean(tf.nn.relu(loss - eta_loss))
    soft = tf.reduce_mean(tf.math.sigmoid(loss / .1))
    excess = tf.nn.relu(oce - cap)
    return soft + BETA * excess, oce, soft, excess

def check_objective():
    x = tf.constant([-3., -1., .5, 2.], tf.float64)
    eta = tf.constant(-.25, tf.float64)
    for cap in [.3, 2.]:
        with tf.GradientTape() as tape:
            tape.watch(x)
            obj, oce, soft, excess = objective_terms(x, eta, cap)
        grad = tape.gradient(obj, x).numpy()
        assert np.isclose(oce.numpy(), 1.25)
        assert np.isclose(obj.numpy(), soft.numpy() + 10 * max(1.25-cap, 0))
        finite = []
        for i in range(4):
            plus = x.numpy().copy(); minus = plus.copy()
            plus[i] += 1e-5; minus[i] -= 1e-5
            finite.append((objective_terms(tf.constant(plus), eta, cap)[0].numpy()
                           - objective_terms(tf.constant(minus), eta, cap)[0].numpy()) / 2e-5)
        assert np.allclose(grad, finite, atol=1e-6)
        assert np.all(grad >= 0)
        assert (excess.numpy() == 0) == (cap > 1.25)
    # Aggregated excess, not per-path exceedance or a CVaR95 constraint.
    return {'active_and_inactive_hinge_correct': True,
            'loss_gradient_matches_finite_difference': True,
            'cvar_sign_and_worst_half_check': True,
            'finite_penalty_is_not_a_hard_feasibility_guarantee': True}

def make_step(model, q, cap):
    variables = tuple(model.trainable_variables)
    optimizer = tf.keras.optimizers.Adam(.005, beta_1=.9, beta_2=.999, epsilon=1e-7)
    optimizer.build(list(variables))
    indices = [[i for i,v in enumerate(variables) if any(v is w for w in net.trainable_variables)]
               for net in model.nets]
    @tf.function(input_signature=[tf.TensorSpec([256,30,2],tf.float32),
                                  tf.TensorSpec([256,30,2],tf.float32),
                                  tf.TensorSpec([256],tf.float32)])
    def step(f, ch, z):
        with tf.GradientTape() as tape:
            h = model(f, training=True)
            liability = z - tf.reduce_sum(h * ch, axis=[1,2])
            # Preserve A's liability-unit intercept convention and arithmetic.
            oce = model.eta + 2*tf.reduce_mean(tf.nn.relu(liability-model.eta)) - q
            soft = tf.reduce_mean(tf.math.sigmoid((liability-q)/.1))
            excess = tf.nn.relu(oce-cap)
            objective = soft + BETA*excess
        gradients = tape.gradient(objective, variables)
        if any(g is None for g in gradients):
            raise RuntimeError('Disconnected gradient')
        tf.debugging.check_numerics(objective, 'objective')
        for g in gradients:
            tf.debugging.check_numerics(g, 'gradient')
        norms = tf.stack([tf.linalg.global_norm([gradients[i] for i in ix]) for ix in indices])
        optimizer.apply_gradients(zip(gradients, variables))
        return objective, oce, soft, excess, tf.linalg.global_norm(gradients), norms
    return step, optimizer

def scalar_ci(x):
    d = float(np.mean(x)); se = float(np.std(x, ddof=1)/np.sqrt(len(x)))
    return {'difference': d, 'standard_error': se, 'approximate_95_interval': [d-1.96*se, d+1.96*se]}

def main():
    tf.config.threading.set_intra_op_parallelism_threads(1)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    OUT.mkdir(exist_ok=False)
    start = time.monotonic()
    records = [json.loads(s) for s in (OLD/'validation.jsonl').read_text().splitlines()]
    base_val = next(r['validation'] for r in records if r['model']=='A' and r['updates']==20000)
    cap = base_val['cvar50']
    q = json.loads((OLD/'metrics.json').read_text())['q']
    protocol = {
        'id': OUT.name, 'target': 'min soft P(PL<0), subject to population CVaR50(-PL)<=c_A',
        'solver': 'Single fixed-weight stochastic hinge-penalty approximation; not a certified constrained solver',
        'objective': 'mean(sigmoid(-PL/.1)) + beta*relu(OCE50(-PL,eta)-c_A)/s',
        'beta': BETA, 's': 1., 'a': 0., 'tau': .1, 'q': q, 'cap_c_A': cap,
        'cap_source': 'Frozen A final calibrated validation CVaR50; same 32768 paths seed5212',
        'cap_rule': 'Fixed before training, no 5% tolerance, no test-dependent cap or relaxation',
        'architecture_market_bn': 'docs/simpler_risk_baseline_v1.md, unchanged',
        'training': {'updates':20000, 'fresh_paths':5120000, 'network_seed':5201,
                     'training_seed':5211, 'chunk':8192, 'batch':256, 'Adam_lr':.005,
                     'Adam_betas':[.9,.999], 'epsilon':1e-7,
                     'start':'Same original initialization as A, not a warm start'},
        'bn': {'calibration_paths':8192, 'seed':5291, 'procedure':'Frozen sequential population calibration of evaluation clones'},
        'validation': {'paths':32768, 'seed':5212, 'milestones':[0,5000,10000,15000,20000],
                       'selection':'None; final20000 candidate only, including failure'},
        'test': {'paths':1000000, 'seed':5603, 'chunk':8192, 'models':['A','candidate'],
                 'access':'Only after final candidate frozen; new paths shared by both models'},
        'limitations': [
            'A finite beta does not guarantee feasibility or global optimality.',
            'Hinge applied to minibatch OCE is a noisy conservative surrogate, not an unbiased estimator of hinge(population CVaR-cap).',
            'Training BN and calibrated inference differ; only full calibrated evaluation determines empirical feasibility.',
            'Only CVaR50 constrained; CVaR95/MSE need not be preserved.',
            'Cap is an empirical baseline validation estimate, not a known population bound.',
            'One seed/weight; paired test intervals conditional on fixed policies do not include training uncertainty.'],
        'penalty_reference':'https://see.stanford.edu/materials/lsocoee364b/10-seq_slides.pdf pp14-15',
        'old_lambda_scan':'Remains paused; no files or processes resumed'}
    dump(OUT/'protocol.json', protocol)
    files = [Path(__file__), ROOT/'scripts/tf_heston52_engine.py', ROOT/'scripts/heston52_market.py',
             ROOT/'scripts/run_section52_calibrated.py', ROOT/'scripts/run_risk_ablation_v1.py',
             OLD/'A_training.weights.h5', OLD/'A_calibrated.weights.h5', OLD/'validation.jsonl',
             ROOT/'results/risk_lambda_scan_v1/STATUS.json']
    hashes = {str(p.relative_to(ROOT)):sha(p) for p in files}
    dump(OUT/'implementation.json', hashes)
    dump(OUT/'objective_checks.json', check_objective())
    dump(OUT/'STATUS.json', {'execution':'TRAINING', 'updates':0})
    model = Heston52Hedge(seed=5201, recurrent=False); model.initialize()
    evaluator = Heston52Hedge(recurrent=False); evaluator.initialize()
    baseline = Heston52Hedge(recurrent=False); baseline.initialize()
    baseline.load_weights(OLD/'A_calibrated.weights.h5')
    cal = sample_paths(8192, np.random.default_rng(5291)); cf,_,_ = sample_tensors(cal); del cal
    val = sample_paths(32768, np.random.default_rng(5212)); vf,_,_ = sample_tensors(val)
    replay = summarize(pnl(val, predict(baseline,vf),q))
    for k in ['cvar50','cvar95','mse','probability_loss']:
        assert np.isclose(replay[k],base_val[k],atol=1e-12,rtol=1e-12)
    dump(OUT/'baseline_validation_replay.json',replay)
    def evaluate(updates):
        saved = [v.numpy().copy() for v in model.weights]
        evaluator.set_weights(model.get_weights()); calibrate(evaluator,cf)
        pl = pnl(val,predict(evaluator,vf),q); stats = summarize(pl)
        row = {'updates':updates, 'validation':stats, 'cap':cap,
               'constraint_violation':stats['cvar50']-cap,
               'empirically_feasible':bool(stats['cvar50']<=cap),
               'elapsed_seconds':time.monotonic()-start}
        with (OUT/'validation.jsonl').open('a') as s:
            s.write(json.dumps(row)+'\n')
        assert all(np.array_equal(v.numpy(),x) for v,x in zip(model.weights,saved))
        if updates:
            model.save_weights(OUT/f'update_{updates}_training.weights.h5')
            evaluator.save_weights(OUT/f'update_{updates}_calibrated.weights.h5')
        if updates==20000:
            np.savez_compressed(OUT/'validation_pnl.npz',candidate=pl)
        dump(OUT/'STATUS.json', {'execution':'TRAINING' if updates<20000 else 'TRAINING_COMPLETE',
                                'updates':updates, 'latest_validation':row})
        print('VALIDATION '+json.dumps(row),flush=True)
        return row
    evaluate(0)
    step,optimizer = make_step(model,q,cap)
    updates = 0
    with (OUT/'training.csv').open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['update','objective','oce_minus_q','soft_probability','cvar_excess','gradient_norm'])
        for data in path_chunks(5120000,8192,5211):
            f,ch,z = sample_tensors(data)
            for lo in range(0,len(z),256):
                values = step(f[lo:lo+256],ch[lo:lo+256],z[lo:lo+256]); updates += 1
                writer.writerow([updates,*[float(v.numpy()) for v in values[:5]]])
                if updates==1:
                    norms = values[-1].numpy(); assert (norms>0).all()
                    dump(OUT/'first_gradients.json',{'all30_nonzero':True,'norms':norms.tolist()})
                if updates%100==0: stream.flush()
                if updates in [5000,10000,15000,20000]: final = evaluate(updates)
    assert updates==20000
    tf.train.Checkpoint(model=model,optimizer=optimizer).write(str(OUT/'final_adam_checkpoint'))
    dump(OUT/'frozen_final.json',{'updates':updates,'validation':final,
                               'selection':'Final prescribed weights only; no selection or retry based on test',
                               'optimizer_iterations':int(optimizer.iterations.numpy())})
    # Test begins only after the candidate and cap have been committed.
    model.load_weights(OUT/'update_20000_calibrated.weights.h5')
    del val, vf
    arrays = {'A':[], 'candidate':[]}; tested = 0
    max_hold = {k:np.zeros(2) for k in arrays}
    turnover = {k:np.zeros(2) for k in arrays}
    for data in path_chunks(1000000,8192,5603):
        f,_,_ = sample_tensors(data)
        for name,m in [('A',baseline),('candidate',model)]:
            h = predict(m,f); arrays[name].append(pnl(data,h,q))
            max_hold[name] = np.maximum(max_hold[name],np.abs(h).max(axis=(0,1)))
            padded = np.concatenate([np.zeros_like(h[:,:1]),h,np.zeros_like(h[:,:1])],axis=1)
            turnover[name] += np.abs(np.diff(padded,axis=1)).sum(axis=(0,1))
        tested += len(f)
        if tested%131072==0:
            print(f'TEST {tested}/1000000 elapsed={time.monotonic()-start:.1f}',flush=True)
            dump(OUT/'STATUS.json',{'execution':'TESTING','test_paths':tested})
    assert tested==1000000
    pl = {k:np.concatenate(v) for k,v in arrays.items()}
    stats = {k:summarize(v) for k,v in pl.items()}
    for k in stats:
        stats[k]['max_abs_holdings'] = max_hold[k].tolist()
        stats[k]['mean_turnover_stock_swap'] = (turnover[k]/tested).tolist()
    np.savez_compressed(OUT/'test_pnl.npz',**pl)
    paired = {}
    for threshold in [0.,1.]:
        paired[f'probability_loss_gt_{threshold}'] = scalar_ci(
            (pl['candidate'] < -threshold).astype(float)-(pl['A'] < -threshold).astype(float))
    for alpha in [.5,.95]:
        influences = {}
        for k,x in pl.items():
            loss = -x; eta = np.quantile(loss,alpha)
            y = eta+np.maximum(loss-eta,0)/(1-alpha)
            influences[k] = y-y.mean()
        metric = 'cvar'+str(int(alpha*100))
        ci = scalar_ci(influences['candidate']-influences['A'])
        d = stats['candidate'][metric]-stats['A'][metric]; se=ci['standard_error']
        ci.update(difference=d,approximate_95_interval=[d-1.96*se,d+1.96*se])
        paired[metric]=ci
    paired['mse'] = scalar_ci(pl['candidate']**2-pl['A']**2)
    result = {'q':q,'beta':BETA,'cap':cap,'test_paths':tested,'test_seed':5603,
              'validation':{'A':base_val,'candidate':final['validation']},'test':stats,
              'feasibility':{'validation_candidate_le_fixed_cap':final['empirically_feasible'],
                             'test_candidate_le_fixed_cap':bool(stats['candidate']['cvar50']<=cap),
                             'test_candidate_le_paired_A':bool(stats['candidate']['cvar50']<=stats['A']['cvar50']),
                             'test_excess_over_fixed_cap':stats['candidate']['cvar50']-cap},
              'paired_test_intervals':paired,'elapsed_seconds':time.monotonic()-start,
              'original_sources_unchanged':all(sha(ROOT/p)==s for p,s in hashes.items())}
    assert result['original_sources_unchanged']
    dump(OUT/'metrics.json',result)
    dump(OUT/'STATUS.json',{'execution':'COMPLETE','updates':updates,'test_paths':tested,
                           'feasibility':result['feasibility']})
    print('FINAL '+json.dumps(result),flush=True)

if __name__=='__main__':
    main()
