"""Bounded fresh-run CLI for the two retained experiments.

Archived results are separate; this curated orchestration has smoke and parity
coverage, not a new full-budget numerical reproduction. All output is new.
"""
import argparse
import importlib
import json
import time
from pathlib import Path
import numpy as np
import tensorflow as tf
from .io import dump
from .risk import metrics


def position_pnl(data, holding, q):
    gains = np.asarray(holding, dtype='float64') * data['changes']
    return q + gains.sum(axis=tuple(range(1, gains.ndim))) - data['payoff']


def project(exp, models, data):
    result = {k: v for k,v in data.items() if k != '_raw'}
    for preference in ['C', 'P']:
        result['h'+preference.lower()] = np.mean([
            exp.predict(m, data['f']).astype('float64') for m in models[preference]], axis=0)
    return result


def main(section):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--window', type=int, default=0)
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--cache-dir', default='runs/cache/heston')
    args = parser.parse_args()
    cfg = json.loads((Path(__file__).resolve().parents[1]/section/'config.json').read_text())
    if args.smoke:
        cfg.update(expert_paths=512, chunk=512, calibration_paths=256, validation_paths=512,
                   residual_pool_paths=512, residual_updates=2, validation_interval=1,
                   test_paths=512, members=cfg['members'][:1], residual_seeds=cfg['residual_seeds'][:1])
    if args.out.exists():
        raise FileExistsError('Choose a new output directory; previous runs are never overwritten')
    if not 0 <= args.window < 300:
        raise ValueError('window must be in 0..299')
    exp = importlib.import_module(section+'.experiment').Experiment(args.window, args.cache_dir)
    args.out.mkdir(parents=True)
    dump(args.out/'protocol.json', {**cfg, 'window': args.window, 'smoke_only': args.smoke,
                                   'selection': 'Validation only; fixed expert final weights',
                                   'runner': 'Curated fresh-run orchestration; archived results were generated before publication'})
    dump(args.out/'market.json', exp.metadata)
    dump(args.out/'status.json', {'status': 'RUNNING'})
    start = time.monotonic()
    cal = exp.sample(cfg['calibration_paths'], np.random.default_rng(cfg['calibration_seed']))['f']
    val_raw = exp.sample(cfg['validation_paths'], np.random.default_rng(cfg['validation_seed']), True)
    models = {'C': [], 'P': []}
    # Paired C/P members use identical initialization and market streams.
    for index, (network_seed, train_seed) in enumerate(cfg['members'], 1):
        pair = {p: exp.expert(seed=network_seed) for p in ['C','P']}
        for m in pair.values(): m.initialize()
        steps = {p: exp.expert_step(m,p,exp.q) for p,m in pair.items()}
        rng = np.random.default_rng(train_seed)
        seen = 0
        while seen < cfg['expert_paths']:
            n = min(cfg['chunk'], cfg['expert_paths']-seen)
            data = exp.sample(n, rng)
            for lo in range(0,n,256):
                for p in ['C','P']:
                    values = steps[p][0](data['f'][lo:lo+256], data['changes'][lo:lo+256].astype('float32'), data['payoff'][lo:lo+256].astype('float32'))
                    if not all(np.isfinite(v.numpy()).all() for v in values):
                        raise FloatingPointError('Nonfinite expert update')
            seen += n
            if seen == n or seen == cfg['expert_paths'] or seen % (cfg['chunk']*16) == 0:
                print(f'expert pair {index}: {seen}/{cfg["expert_paths"]} paths', flush=True)
        for p, m in pair.items():
            prefix = args.out/f'{p}_{index}'
            m.save_weights(str(prefix)+'.training.weights.h5')
            tf.train.Checkpoint(model=m, optimizer=steps[p][1]).write(str(prefix)+'.adam')
            before = [v.numpy().copy() for v in m.trainable_variables]
            exp.calibrate(m,cal)
            assert all(np.array_equal(v.numpy(),x) for v,x in zip(m.trainable_variables,before))
            m.save_weights(str(prefix)+'.calibrated.weights.h5')
            m.trainable = False
            models[p].append(m)
    val = project(exp,models,val_raw)
    c_val = metrics(position_pnl(val,val['hc'],exp.q))
    budget = c_val['cvar95'] if cfg['risk_budget'] == 'validation_C' else float(cfg['risk_budget'])
    if not np.isfinite(budget) or budget <= 0:
        raise ValueError('Invalid CVaR95 budget; never silently floor it')
    train = project(exp,models,exp.sample(cfg['residual_pool_paths'],np.random.default_rng(cfg['pool_seed']),True))
    scale = np.sqrt(np.mean(train['hc']**2, axis=tuple(range(train['hc'].ndim-1)))) if train['hc'].ndim == 3 else float(np.sqrt(np.mean(train['hc']**2)))
    dump(args.out/'risk_budget.json', {'budget': budget, 'scale': np.asarray(scale).tolist(), 'premium': exp.q})
    selections = []
    for index, (network_seed,batch_seed) in enumerate(cfg['residual_seeds'],1):
        model = exp.residual(scale,budget,exp.q,network_seed)
        model.initialize()
        step,optimizer,dual = exp.residual_step(model)
        forward = exp.forward(model)
        selection = {'source': 'C_fallback' if c_val['cvar95'] <= budget else 'no_feasible_candidate',
                     'soft_probability': c_val['soft_probability'] if c_val['cvar95'] <= budget else None,
                     'update': 0, 'validation': c_val}
        rng = np.random.default_rng(batch_seed)
        for update in range(1,cfg['residual_updates']+1):
            ix = rng.integers(0,len(train['payoff']),cfg['residual_batch'])
            values = step(*exp.tensors(train,ix))
            if not all(np.isfinite(v.numpy()).all() for v in values): raise FloatingPointError('Nonfinite residual')
            if update % cfg['validation_interval'] == 0 or update == cfg['residual_updates']:
                pl,hold,corr,surplus = exp.evaluate(model,val,forward=forward)
                m = metrics(pl)
                row = {'seed': index, 'update': update, 'metrics': m,
                       'max_abs_holding': float(np.max(np.abs(hold))), 'dual': float(dual.numpy())}
                with (args.out/'validation.jsonl').open('a') as f: f.write(json.dumps(row)+'\n')
                if m['cvar95'] <= budget and (selection['soft_probability'] is None or m['soft_probability'] < selection['soft_probability']-1e-12):
                    selection = {'source': 'learned_residual','update':update,'soft_probability':m['soft_probability'],'validation':m}
                    model.save_weights(args.out/f'R_{index}.selected.weights.h5')
        model.save_weights(args.out/f'R_{index}.final.weights.h5')
        tf.train.Checkpoint(model=model,optimizer=optimizer,dual=dual).write(str(args.out/f'R_{index}.adam'))
        selections.append(selection)
    dump(args.out/'selection_before_test.json',selections)
    del train
    test_raw = exp.sample(cfg['test_paths'],np.random.default_rng(cfg['test_seed']),True)
    benchmark = exp.benchmark(test_raw)
    test = project(exp,models,test_raw)
    arrays = {'benchmark':benchmark, 'CVaR':position_pnl(test,test['hc'],exp.q)}
    for index,selection in enumerate(selections,1):
        if selection['source'] == 'learned_residual':
            model = exp.residual(scale,budget,exp.q,0)
            model.initialize()
            model.load_weights(args.out/f'R_{index}.selected.weights.h5')
            arrays[f'Residual_{index}'] = exp.evaluate(model,test)[0]
        elif selection['source'] == 'C_fallback':
            arrays[f'Residual_{index}'] = arrays['CVaR'].copy()
    result = {name:{**metrics(pl),'constraint_pass':bool(metrics(pl)['cvar95']<=budget)} for name,pl in arrays.items()}
    hist_raw = exp.historical()
    historical = None
    if hist_raw is not None:
        hist = project(exp,models,hist_raw)
        historical = {'benchmark':float(exp.benchmark(hist_raw)[0]),'CVaR':float(position_pnl(hist,hist['hc'],exp.q)[0])}
        for index,selection in enumerate(selections,1):
            if selection['source'] == 'learned_residual':
                model = exp.residual(scale,budget,exp.q,0); model.initialize(); model.load_weights(args.out/f'R_{index}.selected.weights.h5')
                historical[f'Residual_{index}'] = float(exp.evaluate(model,hist)[0][0])
            elif selection['source'] == 'C_fallback': historical[f'Residual_{index}'] = historical['CVaR']
    np.savez_compressed(args.out/'pnl.npz',**arrays)
    dump(args.out/'metrics.json',{'simulated':result,'historical_single_contract':historical,'seconds':time.monotonic()-start,'smoke_only':args.smoke})
    dump(args.out/'status.json',{'status':'COMPLETE','smoke_only':args.smoke,'selections':selections})
    print(json.dumps(result,indent=2))
