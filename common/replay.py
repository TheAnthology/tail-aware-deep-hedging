"""Evaluate the released frozen policies, without training or recalibrating BN."""
import argparse
import importlib
import json
from pathlib import Path
import numpy as np
from .io import load_weights, dump
from .risk import metrics
from .run import project, position_pnl


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('section', choices=['sec53_theoretical','sec55_sp500'])
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--full', action='store_true', help='Use the complete recorded 1M/100k evaluation stream')
    parser.add_argument('--cache-dir', default='runs/cache/heston')
    args = parser.parse_args()
    if args.out.exists(): raise FileExistsError(args.out)
    root = Path(__file__).resolve().parents[1]
    spec = json.loads((root/args.section/'checkpoints/manifest.json').read_text())
    exp = importlib.import_module(args.section+'.experiment').Experiment(0,args.cache_dir)
    np.testing.assert_allclose(exp.q,spec['q'],atol=1e-10,rtol=0)
    experts = {'C':[],'P':[]}
    for preference in experts:
        for name in spec['experts'][preference]:
            model=exp.expert(); model.initialize()
            load_weights(model,root/args.section/'checkpoints'/name)
            model.trainable=False; experts[preference].append(model)
    residuals=[]
    for row in spec['residuals']:
        model=exp.residual(spec['scale'],spec['budget'],spec['q'],0)
        load_weights(model,root/args.section/'checkpoints'/row['weights'])
        residuals.append((row['name'],model,exp.forward(model)))
    total=spec['test_paths'] if args.full else 512
    # Both archived evaluations generated market paths in chunks of8192.
    # Chunk boundaries matter for assignment of random draws to paths/dates.
    chunk=8192
    rng=np.random.default_rng(spec['test_seed']); arrays={'CVaR':[],'benchmark':[]}
    arrays.update({name:[] for name,_,_ in residuals})
    seen=0
    while seen<total:
        generated=min(chunk,spec['test_paths']-seen)
        raw=exp.sample(generated,rng,True)
        take=min(generated,total-seen)
        benchmark=exp.benchmark(raw)[:take]
        data=project(exp,experts,raw)
        arrays['benchmark'].append(benchmark)
        arrays['CVaR'].append(position_pnl(data,data['hc'],exp.q)[:take])
        for name,model,forward in residuals:
            arrays[name].append(exp.evaluate(model,data,forward=forward)[0][:take])
        seen+=take
    arrays={name:np.concatenate(parts) for name,parts in arrays.items()}
    args.out.mkdir(parents=True)
    result={name:{**metrics(pl),'constraint_pass':bool(metrics(pl)['cvar95']<=spec['budget'])} for name,pl in arrays.items()}
    dump(args.out/'metrics.json',{'frozen_replay':True,'full_recorded_stream':args.full,'metrics':result})
    np.savez_compressed(args.out/'pnl.npz',**arrays)
    print(json.dumps(result,indent=2))

if __name__=='__main__': main()
