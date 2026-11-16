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
    parser.add_argument("section", choices=["sec53_theoretical", "sec55_sp500"])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--full",
        action="store_true",
        help="Use the complete recorded 1M/100k evaluation stream",
    )
    parser.add_argument("--cache-dir", default="runs/cache/heston")
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    root = Path(__file__).resolve().parents[1]
    spec = json.loads((root / args.section / "checkpoints/manifest.json").read_text())
    exp = importlib.import_module(args.section + ".experiment").Experiment(
        0, args.cache_dir
    )
    np.testing.assert_allclose(exp.q, spec["q"], atol=1e-10, rtol=0)
    paper_model = exp.paper_model()
    load_weights(
        paper_model,
        root / args.section / "checkpoints" / spec["paper_deep_hedge"]["weights"],
    )
    experts = {"C": [], "P": []}
    for preference in experts:
        for name in spec["experts"][preference]:
            model = exp.expert()
            model.initialize()
            load_weights(model, root / args.section / "checkpoints" / name)
            model.trainable = False
            experts[preference].append(model)
    selected_policies = []
    for row in spec["novel_model_1"]:
        model = exp.novel_model_1(spec["scale"], spec["budget"], spec["q"], 0)
        load_weights(model, root / args.section / "checkpoints" / row["weights"])
        selected_policies.append((row["name"], model, exp.forward(model)))
    total = spec["test_paths"] if args.full else 512
    # Both archived evaluations generated market paths in chunks of 8192.
    # Chunk boundaries matter for assignment of random draws to paths/dates.
    chunk = 8192
    rng = np.random.default_rng(spec["test_seed"])
    arrays = {"CVaR": [], "benchmark": [], "paper_deep_hedge": []}
    arrays.update({name: [] for name, _, _ in selected_policies})
    seen = 0
    while seen < total:
        generated = min(chunk, spec["test_paths"] - seen)
        raw = exp.sample(generated, rng, True)
        take = min(generated, total - seen)
        benchmark = exp.benchmark(raw)[:take]
        data = project(exp, experts, raw)
        paper_holdings = exp.paper_holdings(paper_model, raw)
        arrays["paper_deep_hedge"].append(
            position_pnl(data, paper_holdings, exp.q)[:take]
        )
        arrays["benchmark"].append(benchmark)
        arrays["CVaR"].append(position_pnl(data, data["hc"], exp.q)[:take])
        for name, model, forward in selected_policies:
            arrays[name].append(exp.evaluate(model, data, forward=forward)[0][:take])
        seen += take
    arrays = {name: np.concatenate(parts) for name, parts in arrays.items()}
    args.out.mkdir(parents=True)
    result = {
        name: {
            **metrics(pl),
            "constraint_pass": bool(metrics(pl)["cvar95"] <= spec["budget"]),
        }
        for name, pl in arrays.items()
    }
    dump(
        args.out / "metrics.json",
        {"frozen_replay": True, "full_recorded_stream": args.full, "metrics": result},
    )
    np.savez_compressed(args.out / "pnl.npz", **arrays)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
