"""Matched diagnostic controls; original paper networks and neural LR unchanged.

These optimization choices are not claimed to be recovered author settings.
No external code is executed and no existing model/results are overwritten.
"""
import os
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')
from pathlib import Path
import hashlib
import json
import time
import numpy as np
import tensorflow as tf
from paper_market import window_data, fit_garch, paths, sample_tensors, observed_path, bs_premium, bs_hedge, pnl
from tf_paper_engine import PaperHedge, make_adam, create_train_step, cvar_objective, predict_positions, compare_training_and_inference
from run_paper55_recurrent_bnfix_v3 import calibrate, metrics, evaluate, verify_freeze

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/paper55_risk_optimizer_controls_v1_window000'
PARENT = ROOT / 'results/paper55_recurrent_bnfix_v3_window000'
DAYS = [0, 5, 10, 15, 20, 25]
MILESTONES = [65536, 1048576, 5000192, 10000128, 20000000]

def dump(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')

def read(path):
    return json.loads(path.read_text())

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def build_step(model, mode, currency_scale):
    """Separate eta optimizer is an exact positive coordinate rescaling.

    Adam on eta / scale with epsilon e is equivalent to Adam on raw eta
    with LR multiplied by scale and epsilon divided by scale. The neural
    gradient remains that of the original, unscaled OCE objective.
    """
    model.initialize_without_bn_updates()
    variables = tuple(v for v in model.trainable_variables if v is not model.eta)
    neural = make_adam(.0001)
    neural.build(list(variables))
    threshold = tf.keras.optimizers.Adam(
        learning_rate=.0001 * currency_scale, beta_1=.9, beta_2=.999,
        epsilon=1e-7 / currency_scale, amsgrad=False)
    threshold.build([model.eta])
    optimized = variables + ((model.eta,) if mode == 'scaled_eta_oce' else ())
    groups = [[i for i, v in enumerate(variables)
               if any(v is w for w in net.trainable_variables)] for net in model.nets]

    @tf.function(input_signature=[tf.TensorSpec([None, 6], tf.float32),
                                  tf.TensorSpec([None, 6], tf.float32),
                                  tf.TensorSpec([None], tf.float32)])
    def step(features, changes, payoff):
        with tf.GradientTape() as tape:
            holdings = model(features, training=True)
            loss = payoff - tf.reduce_sum(holdings * changes, axis=1)
            if mode == 'scaled_eta_oce':
                objective = cvar_objective(loss, model.eta, .75)
            else:
                n = tf.size(loss)
                tf.debugging.assert_equal(n % 4, 0)
                objective = tf.reduce_mean(tf.math.top_k(loss, k=n // 4).values)
            tf.debugging.check_numerics(objective, 'Nonfinite objective')
        gradients = tape.gradient(objective, optimized)
        if any(g is None for g in gradients):
            raise RuntimeError('Disconnected optimized parameter')
        for g in gradients:
            tf.debugging.check_numerics(g, 'Nonfinite gradient')
        neural.apply_gradients(zip(gradients[:len(variables)], variables))
        if mode == 'scaled_eta_oce':
            threshold.apply_gradients([(gradients[-1], model.eta)])
        return {'objective': objective, 'eta': tf.identity(model.eta),
                'gradient_norms': tf.stack([tf.linalg.global_norm([gradients[i] for i in ix]) for ix in groups]),
                'max_abs_position': tf.reduce_max(tf.abs(holdings))}
    return step, neural, threshold

def verify_step(features, changes, payoff):
    """At unit scale, split optimizer must reproduce the original updates."""
    a, b = PaperHedge(seed=2701), PaperHedge(seed=2701)
    a.initialize_without_bn_updates(); b.initialize_without_bn_updates()
    original = create_train_step(a, make_adam(.0001), .75)
    split, _, _ = build_step(b, 'scaled_eta_oce', 1.)
    for _ in range(3):
        original(features[:256], changes[:256], payoff[:256])
        split(features[:256], changes[:256], payoff[:256])
    gap = max(float(np.max(np.abs(x - y))) for x, y in zip(a.get_weights(), b.get_weights()))
    assert gap < 1e-6, gap
    losses = tf.Variable(np.random.default_rng(7001).normal(size=256), dtype=tf.float32)
    eta = float(np.quantile(losses.numpy(), .75))
    with tf.GradientTape() as t:
        sorted_risk = tf.reduce_mean(tf.math.top_k(losses, k=64).values)
    g_sorted = t.gradient(sorted_risk, losses).numpy()
    with tf.GradientTape() as t:
        oce_risk = cvar_objective(losses, tf.constant(eta, tf.float32), .75)
    g_oce = t.gradient(oce_risk, losses).numpy()
    np.testing.assert_array_equal(g_sorted, g_oce)
    np.testing.assert_allclose(sorted_risk.numpy(), oce_risk.numpy(), atol=1e-6)
    return {'split_optimizer_unit_scale_max_weight_gap': gap,
            'sorted_tail_equals_batch_optimal_OCE_value_and_gradient_without_ties': True}

def eta_diagnostic(pl, premium, eta):
    losses = premium - pl
    optimum = float(np.quantile(losses, .75))
    return {'eta': eta, 'own_calibrated_validation_quantile75': optimum,
            'eta_quantile': float(np.mean(losses <= eta)),
            'OCE_excess_above_optimal_eta': float(eta + 4 * np.maximum(losses - eta, 0).mean()
                                                 - optimum - 4 * np.maximum(losses - optimum, 0).mean())}

def main():
    tf.config.threading.set_intra_op_parallelism_threads(1)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    verify_freeze()
    assert not OUT.exists(), 'Never overwrite a previous experiment'
    parent_manifest = read(PARENT / 'artifact_manifest.json')
    for name, digest in parent_manifest.items():
        assert sha(PARENT / name) == digest
    _, history, returns = window_data(0)
    params, state, fit = fit_garch(returns[:1187])
    spot = float(history[1187]); premium = bs_premium(spot, fit['initial_sigma0_daily'])
    scale = spot / 100.
    # Independent training-only smoke paths; never reused as validation/test.
    smoke, _ = paths(params, state, spot, 256, np.random.default_rng(7011))
    checks = verify_step(*sample_tensors(smoke, spot, DAYS))
    OUT.mkdir()
    sources = [Path(__file__), ROOT / 'scripts/tf_paper_engine.py', ROOT / 'scripts/paper_market.py',
               ROOT / 'scripts/run_paper55_recurrent_bnfix_v3.py', PARENT / 'artifact_manifest.json']
    protocol = {'id': 'S55_RISK_OPTIMIZER_CONTROLS_V1', 'window': 0, 'seed_network': 2701,
                'training_paths': 20000000, 'epochs': 1, 'batch_size': 256, 'updates': 78125,
                'neural_learning_rate': .0001, 'architecture': 'Unmodified PaperHedge, including initial-date network and BN',
                'modes': {'scaled_eta_oce': {'eta_initial': 0., 'currency_scale': scale,
                          'eta_lr_original_units': .0001 * scale, 'eta_lr_normalized_units': .0001,
                          'interpretation': 'Original currency OCE neural objective; eta optimized in units with initial spot 100'},
                          'sorted_batch_cvar': {'alpha': .75, 'tail_fraction': .25, 'eta_optimized': False,
                          'interpretation': 'Mean of largest 64 of 256 losses; PFHedge-style empirical ES estimator; different stochastic estimator from joint OCE'}},
                'train_data': 'RNG1701, original milestone/chunk boundaries; identical 20M stream shared by both controls and prior one-pass run',
                'BN': 'Same independent8192-path seed5704 sequential calibration on clones; raw training BN unchanged',
                'validation': {'n': 65536, 'seed': 5702},
                'development_test': {'n': 100000, 'seed': 5713, 'previously_inspected': True},
                'selection': 'Fixed one-pass final weights for both; no historical/test-based selection',
                'limits': 'Single origin/seed optimization diagnostic; local optimizer conventions, not exact author recovery. Prior five-pass result has a larger budget.',
                'retention': 'Final weights/checkpoints, scalar logs, paired terminal P&L only; no full training path storage',
                'source_hashes': {str(p.relative_to(ROOT)): sha(p) for p in sources}}
    dump(OUT / 'protocol.json', protocol); dump(OUT / 'preflight_checks.json', checks)
    dump(OUT / 'garch_fit.json', fit)
    cal, _ = paths(params, state, spot, 8192, np.random.default_rng(5704))
    cal_features = sample_tensors(cal, spot, DAYS)[0]
    val, _ = paths(params, state, spot, 65536, np.random.default_rng(5702))
    val_features = sample_tensors(val, spot, DAYS)[0]
    models = {}; steppers = {}; optimizers = {}; evaluators = {}
    for mode in protocol['modes']:
        models[mode] = PaperHedge(seed=2701)
        steppers[mode], neural, eta_opt = build_step(models[mode], mode, scale)
        optimizers[mode] = (neural, eta_opt)
        evaluators[mode] = PaperHedge(seed=2701)
        evaluators[mode].initialize_without_bn_updates()
        (OUT / mode).mkdir()
    assert all(np.array_equal(a,b) for a,b in zip(*(m.get_weights() for m in models.values())))
    digest = hashlib.sha256(); seen = 0; start = time.monotonic()
    rng = np.random.default_rng(1701)
    max_positions = {mode: 0. for mode in models}
    for milestone in MILESTONES:
        while seen < milestone:
            take = min(65536, milestone - seen)
            prices, _ = paths(params, state, spot, take, rng)
            features, changes, payoff = sample_tensors(prices, spot, DAYS)
            for array in (features, changes, payoff): digest.update(array.tobytes())
            for lo in range(0, take, 256):
                for mode, step in steppers.items():
                    values = {k: v.numpy() for k,v in step(features[lo:lo+256], changes[lo:lo+256], payoff[lo:lo+256]).items()}
                    assert all(np.isfinite(v).all() for v in values.values())
                    max_positions[mode] = max(max_positions[mode], float(values['max_abs_position']))
                    if seen == 0 and lo == 0:
                        assert (values['gradient_norms'] > 0).all()
                        dump(OUT / mode / 'first_update.json', {k: v.tolist() for k,v in values.items()})
            seen += take
        row = {'paths': seen, 'elapsed_seconds': time.monotonic() - start, 'models': {}}
        for mode, model in models.items():
            evaluator = evaluators[mode]
            evaluator.set_weights(model.get_weights()); calibrate(evaluator, cal_features)
            raw, _ = evaluate(model, val, spot, premium, DAYS)
            corrected, holdings = evaluate(evaluator, val, spot, premium, DAYS)
            row['models'][mode] = {'raw_validation': raw, 'calibrated_validation': corrected,
                'eta_diagnostic': eta_diagnostic(pnl(val, holdings, premium, spot, DAYS), premium, float(model.eta.numpy())) if mode == 'scaled_eta_oce' else None}
        with (OUT / 'milestones.jsonl').open('a') as fp: fp.write(json.dumps(row) + '\n')
        dump(OUT / 'STATUS.json', {'execution': 'RUNNING', 'paths_per_model': seen, 'target_per_model': 20000000})
        print(json.dumps(row), flush=True)
    prior_hash = read(ROOT / 'results/paper55_lr1e4_five_epochs_v1_window000/epoch_02/metrics.json')['training_tensor_sha256']
    assert digest.hexdigest() == prior_hash
    for mode, model in models.items():
        neural, eta_opt = optimizers[mode]
        assert int(neural.iterations.numpy()) == 78125
        assert int(eta_opt.iterations.numpy()) == (78125 if mode == 'scaled_eta_oce' else 0)
        model.save_weights(OUT / mode / 'raw.weights.h5')
        tf.train.Checkpoint(model=model, neural_optimizer=neural, eta_optimizer=eta_opt).write(str(OUT / mode / 'checkpoint'))
        evaluators[mode].save_weights(OUT / mode / 'calibrated.weights.h5')
    dump(OUT / 'frozen_before_evaluation.json', {'selection': 'Fixed one-pass final for both controls', 'training_tensor_sha256': digest.hexdigest()})
    test, test_var = paths(params, state, spot, 100000, np.random.default_rng(5713))
    test_features = sample_tensors(test, spot, DAYS)[0]
    arrays = {'BS': pnl(test, bs_hedge(test, test_var, spot, DAYS), premium, spot, DAYS)}
    with np.load(PARENT / 'test_pnl.npz') as old:
        np.testing.assert_array_equal(arrays['BS'], old['BS'])
        arrays['original_oce_one_pass'] = old['paper_bn_fixed'].copy()
    hist, hist_var = observed_path(params, state, spot, returns[1187:])
    np.testing.assert_allclose(hist[0], history[1187:], atol=1e-10)
    historical = {'BS': float(pnl(hist, bs_hedge(hist, hist_var, spot, DAYS), premium, spot, DAYS)[0]),
                  'original_oce_one_pass': read(PARENT / 'metrics.json')['historical']['paper_bn_fixed']['index_pnl']}
    modes = {}
    for mode, model in models.items():
        fixed = evaluators[mode]
        for label, current in [(mode, fixed), (mode + '_raw', model)]:
            arrays[label] = pnl(test, predict_positions(current, test_features), premium, spot, DAYS)
        historical[mode] = float(pnl(hist, predict_positions(fixed, sample_tensors(hist, spot, DAYS)[0]), premium, spot, DAYS)[0])
        replay = PaperHedge(seed=2701); replay.initialize_without_bn_updates()
        replay.load_weights(OUT / mode / 'calibrated.weights.h5')
        reference = predict_positions(fixed, val_features[:4096])
        np.testing.assert_array_equal(reference, predict_positions(replay, val_features[:4096]))
        np.testing.assert_allclose(reference, predict_positions(fixed, val_features[:4096], 256), atol=1e-4, rtol=1e-5)
        np.testing.assert_allclose(reference[:16], predict_positions(fixed, val_features[:16], 1), atol=1e-4, rtol=1e-5)
        mode_comparison = compare_training_and_inference(model, val_features[:4096])
        modes[mode] = {'batch_stat_diagnostic': metrics(pnl(val[:4096], mode_comparison['training_mode_positions'], premium, spot, DAYS)),
                       'raw_inference': metrics(pnl(val[:4096], mode_comparison['inference_positions'], premium, spot, DAYS)),
                       'calibrated_inference': metrics(pnl(val[:4096], reference, premium, spot, DAYS))}
    results = {name: metrics(array) for name,array in arrays.items()}
    paired = {}
    reference = arrays['original_oce_one_pass']
    for mode in models:
        # Mean paired P&L SE is descriptive; tail risk comparison uses full common sample.
        differences = arrays[mode] - reference
        paired[mode] = {'mean_pnl_difference': float(differences.mean()),
                        'mean_pnl_difference_standard_error': float(differences.std(ddof=1) / np.sqrt(len(differences))),
                        'loss_to_nonloss': int(np.sum((reference < 0) & (arrays[mode] >= 0))),
                        'nonloss_to_loss': int(np.sum((reference >= 0) & (arrays[mode] < 0)))}
    np.savez_compressed(OUT / 'development_pnl.npz', **arrays)
    verify_freeze()
    for name, value in protocol['source_hashes'].items(): assert sha(ROOT / name) == value
    for name, value in parent_manifest.items(): assert sha(PARENT / name) == value
    checks.update(identical_training_tensor_hash_to_previous_run=True, paired_BS_bitwise_equal=True,
                  all_gradients_finite=True, all_six_networks_receive_gradient=True, reload_and_batch_invariance=True,
                  parent_and_frozen_baselines_unchanged=True)
    dump(OUT / 'metrics.json', {'development': results, 'historical_single_origin': historical,
         'paired_vs_original_one_pass': paired, 'mode_diagnostics': modes,
         'max_training_abs_positions': max_positions, 'checks': checks, 'elapsed_seconds': time.monotonic() - start})
    lines = ['# Risk optimizer controls: single-origin development diagnostics', '',
             'Same six paper networks, neural LR 1e-4, 20M paths, 78,125 updates, init seed and BN calibration. All currency metrics are original index points; MSE is squared points.', '',
             '| Strategy | CVaR75 | CVaR95 | MSE | Loss probability | Mean P&L | Historical P&L (one origin) |',
             '|---|---:|---:|---:|---:|---:|---:|']
    for name in ['BS', 'original_oce_one_pass', *models]:
        m = results[name]
        lines.append(f'| {name} | {m["cvar75"]:.6f} | {m["cvar95"]:.6f} | {m["mse"]:.6f} | {m["probability_loss"]:.3%} | {m["mean_pnl"]:.6f} | {historical[name]:.6f} |')
    lines += ['', '## Interpretation boundaries', '',
              '- scaled_eta_oce preserves the population OCE formula and network gradients but changes optimizer coordinates for eta. This is a local numerical convention, not a recovered author detail.',
              '- sorted_batch_cvar is the empirical top-quarter loss. It equals the batch-optimal OCE value, but its finite-batch stochastic gradient estimator differs from jointly learned global eta.',
              '- All branches retain the same explicit BN repair. Raw-inference metrics and mode diagnostics are retained in metrics.json; improvement of calibrated performance does not establish resolution of raw BN.',
              '- These are reused development paths and one inspected historical origin, not independent confirmation or a 300-window reproduction. Historical P&L did not select checkpoints.',
              '- The earlier five-pass OCE result has five times the optimization budget and is not a matched one-pass comparator.',
              '- Frozen adapted BS/CVaR/Residual files and original paper-style one-pass artifacts passed hash verification. The paused rolling job was not resumed.', '']
    (OUT / 'REPORT.md').write_text('\n'.join(lines))
    dump(OUT / 'STATUS.json', {'execution': 'COMPLETE', 'models': 2, 'paths_per_model': seen, 'updates_per_model': 78125})
    dump(OUT / 'artifact_manifest.json', {str(p.relative_to(OUT)): sha(p) for p in OUT.rglob('*') if p.is_file()})
    print('COMPLETE ' + str(OUT), flush=True)

if __name__ == '__main__':
    main()
