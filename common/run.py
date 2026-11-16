"""Train experts, select residuals on validation data, then evaluate once.

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
    """Premium + trading gains - terminal payoff, summed over dates/assets."""
    gains = np.asarray(holding, dtype="float64") * data["changes"]
    return q + gains.sum(axis=tuple(range(1, gains.ndim))) - data["payoff"]


def project(exp, models, data):
    """Cache mean expert holdings: hc for CVaR, hp for loss probability.

    Average holdings within each preference, keeping each member's own frozen
    BatchNorm state. Model weights and the two preferences are never averaged.
    """
    projected = {key: value for key, value in data.items() if key != "_raw"}
    for preference in ["C", "P"]:
        member_holdings = [
            exp.predict(model, data["f"]).astype("float64")
            for model in models[preference]
        ]
        projected["h" + preference.lower()] = np.mean(member_holdings, axis=0)
    return projected


def _load_config(section, smoke):
    config_path = Path(__file__).resolve().parents[1] / section / "config.json"
    config = json.loads(config_path.read_text())
    if smoke:
        config.update(
            expert_paths=512,
            chunk=512,
            calibration_paths=256,
            validation_paths=512,
            residual_pool_paths=512,
            residual_updates=2,
            validation_interval=1,
            test_paths=512,
            members=config["members"][:1],
            residual_seeds=config["residual_seeds"][:1],
        )
    return config


def _train_experts(exp, config, calibration_features, output_dir):
    """Train paired C/P experts, calibrate BatchNorm, then freeze inference."""
    models = {"C": [], "P": []}
    for index, (network_seed, train_seed) in enumerate(config["members"], 1):
        # Paired preferences share initialization and market draws.
        pair = {preference: exp.expert(seed=network_seed) for preference in ["C", "P"]}
        for model in pair.values():
            model.initialize()
        training_steps = {
            preference: exp.expert_step(model, preference, exp.q)
            for preference, model in pair.items()
        }
        rng = np.random.default_rng(train_seed)
        paths_seen = 0
        while paths_seen < config["expert_paths"]:
            count = min(config["chunk"], config["expert_paths"] - paths_seen)
            data = exp.sample(count, rng)
            for start in range(0, count, 256):
                batch = slice(start, start + 256)
                for preference in ["C", "P"]:
                    step, _ = training_steps[preference]
                    diagnostics = step(
                        data["f"][batch],
                        data["changes"][batch].astype("float32"),
                        data["payoff"][batch].astype("float32"),
                    )
                    if not all(
                        np.isfinite(value.numpy()).all() for value in diagnostics
                    ):
                        raise FloatingPointError("Nonfinite expert update")
            paths_seen += count
            if (
                paths_seen == count
                or paths_seen == config["expert_paths"]
                or paths_seen % (config["chunk"] * 16) == 0
            ):
                print(
                    f'expert pair {index}: {paths_seen}/{config["expert_paths"]} paths',
                    flush=True,
                )

        for preference, model in pair.items():
            prefix = output_dir / f"{preference}_{index}"
            model.save_weights(str(prefix) + ".training.weights.h5")
            tf.train.Checkpoint(
                model=model, optimizer=training_steps[preference][1]
            ).write(str(prefix) + ".adam")
            learned_weights = [
                value.numpy().copy() for value in model.trainable_variables
            ]
            exp.calibrate(model, calibration_features)
            assert all(
                np.array_equal(value.numpy(), before)
                for value, before in zip(model.trainable_variables, learned_weights)
            )
            model.save_weights(str(prefix) + ".calibrated.weights.h5")
            model.trainable = False
            models[preference].append(model)
    return models


def _position_scale(cvar_holdings):
    """Training-only RMS holdings: one scale per asset, or a scalar for stock."""
    if cvar_holdings.ndim == 3:
        path_and_date_axes = tuple(range(cvar_holdings.ndim - 1))
        return np.sqrt(np.mean(cvar_holdings**2, axis=path_and_date_axes))
    return float(np.sqrt(np.mean(cvar_holdings**2)))


def _train_residuals(
    exp, config, train, validation, cvar_metrics, scale, budget, output_dir
):
    """Select only validation-feasible improvements over the CVaR fallback."""
    selections = []
    for index, (network_seed, batch_seed) in enumerate(config["residual_seeds"], 1):
        model = exp.residual(scale, budget, exp.q, network_seed)
        model.initialize()
        step, optimizer, dual = exp.residual_step(model)
        forward = exp.forward(model)
        fallback_feasible = cvar_metrics["cvar95"] <= budget
        selection = {
            "source": "C_fallback" if fallback_feasible else "no_feasible_candidate",
            "soft_probability": (
                cvar_metrics["soft_probability"] if fallback_feasible else None
            ),
            "update": 0,
            "validation": cvar_metrics,
        }
        rng = np.random.default_rng(batch_seed)
        for update in range(1, config["residual_updates"] + 1):
            indices = rng.integers(0, len(train["payoff"]), config["residual_batch"])
            diagnostics = step(*exp.tensors(train, indices))
            if not all(np.isfinite(value.numpy()).all() for value in diagnostics):
                raise FloatingPointError("Nonfinite residual")
            if (
                update % config["validation_interval"] == 0
                or update == config["residual_updates"]
            ):
                pnl, holdings, _, _ = exp.evaluate(model, validation, forward=forward)
                candidate = metrics(pnl)
                row = {
                    "seed": index,
                    "update": update,
                    "metrics": candidate,
                    "max_abs_holding": float(np.max(np.abs(holdings))),
                    "dual": float(dual.numpy()),
                }
                with (output_dir / "validation.jsonl").open("a") as log:
                    log.write(json.dumps(row) + "\n")

                feasible = candidate["cvar95"] <= budget
                improves_probability = (
                    selection["soft_probability"] is None
                    or candidate["soft_probability"]
                    < selection["soft_probability"] - 1e-12
                )
                if feasible and improves_probability:
                    selection = {
                        "source": "learned_residual",
                        "update": update,
                        "soft_probability": candidate["soft_probability"],
                        "validation": candidate,
                    }
                    model.save_weights(output_dir / f"R_{index}.selected.weights.h5")
        model.save_weights(output_dir / f"R_{index}.final.weights.h5")
        tf.train.Checkpoint(model=model, optimizer=optimizer, dual=dual).write(
            str(output_dir / f"R_{index}.adam")
        )
        selections.append(selection)
    return selections


def _load_selected_residual(exp, scale, budget, output_dir, index):
    model = exp.residual(scale, budget, exp.q, 0)
    model.initialize()
    model.load_weights(output_dir / f"R_{index}.selected.weights.h5")
    return model


def _evaluate_simulated(exp, models, raw, selections, scale, budget, output_dir):
    """Evaluate the benchmark, CVaR policy and already selected residuals."""
    benchmark = exp.benchmark(raw)
    data = project(exp, models, raw)
    pnl_by_policy = {
        "benchmark": benchmark,
        "CVaR": position_pnl(data, data["hc"], exp.q),
    }
    for index, selection in enumerate(selections, 1):
        if selection["source"] == "learned_residual":
            model = _load_selected_residual(exp, scale, budget, output_dir, index)
            pnl_by_policy[f"Residual_{index}"] = exp.evaluate(model, data)[0]
        elif selection["source"] == "C_fallback":
            pnl_by_policy[f"Residual_{index}"] = pnl_by_policy["CVaR"].copy()
    return pnl_by_policy


def _evaluate_historical(exp, models, selections, scale, budget, output_dir):
    """Return one realized contract's P&L when a historical path is available."""
    raw = exp.historical()
    if raw is None:
        return None
    data = project(exp, models, raw)
    pnl_by_policy = {
        "benchmark": float(exp.benchmark(raw)[0]),
        "CVaR": float(position_pnl(data, data["hc"], exp.q)[0]),
    }
    for index, selection in enumerate(selections, 1):
        if selection["source"] == "learned_residual":
            model = _load_selected_residual(exp, scale, budget, output_dir, index)
            pnl_by_policy[f"Residual_{index}"] = float(exp.evaluate(model, data)[0][0])
        elif selection["source"] == "C_fallback":
            pnl_by_policy[f"Residual_{index}"] = pnl_by_policy["CVaR"]
    return pnl_by_policy


def main(section):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--window", type=int, default=0)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--cache-dir", default="runs/cache/heston")
    args = parser.parse_args()
    config = _load_config(section, args.smoke)
    if args.out.exists():
        raise FileExistsError(
            "Choose a new output directory; previous runs are never overwritten"
        )
    if not 0 <= args.window < 300:
        raise ValueError("window must be in 0..299")
    exp = importlib.import_module(section + ".experiment").Experiment(
        args.window, args.cache_dir
    )
    args.out.mkdir(parents=True)
    dump(
        args.out / "protocol.json",
        {
            **config,
            "window": args.window,
            "smoke_only": args.smoke,
            "selection": "Validation only; fixed expert final weights",
            "runner": "Curated fresh-run orchestration; archived results were generated before publication",
        },
    )
    dump(args.out / "market.json", exp.metadata)
    dump(args.out / "status.json", {"status": "RUNNING"})
    start = time.monotonic()

    # Each stage has its own seeded market stream; keep their roles separate.
    calibration_features = exp.sample(
        config["calibration_paths"], np.random.default_rng(config["calibration_seed"])
    )["f"]
    validation_raw = exp.sample(
        config["validation_paths"],
        np.random.default_rng(config["validation_seed"]),
        True,
    )
    models = _train_experts(exp, config, calibration_features, args.out)
    validation = project(exp, models, validation_raw)
    cvar_metrics = metrics(position_pnl(validation, validation["hc"], exp.q))
    budget = (
        cvar_metrics["cvar95"]
        if config["risk_budget"] == "validation_C"
        else float(config["risk_budget"])
    )
    if not np.isfinite(budget) or budget <= 0:
        raise ValueError("Invalid CVaR95 budget; never silently floor it")
    train = project(
        exp,
        models,
        exp.sample(
            config["residual_pool_paths"],
            np.random.default_rng(config["pool_seed"]),
            True,
        ),
    )
    scale = _position_scale(train["hc"])
    dump(
        args.out / "risk_budget.json",
        {
            "budget": budget,
            "scale": np.asarray(scale).tolist(),
            "premium": exp.q,
        },
    )
    selections = _train_residuals(
        exp, config, train, validation, cvar_metrics, scale, budget, args.out
    )
    # Freeze checkpoint choices before drawing or inspecting final-test paths.
    dump(args.out / "selection_before_test.json", selections)
    del train
    test_raw = exp.sample(
        config["test_paths"], np.random.default_rng(config["test_seed"]), True
    )
    pnl_by_policy = _evaluate_simulated(
        exp, models, test_raw, selections, scale, budget, args.out
    )
    result = {
        name: {
            **metrics(pnl),
            "constraint_pass": bool(metrics(pnl)["cvar95"] <= budget),
        }
        for name, pnl in pnl_by_policy.items()
    }
    historical = _evaluate_historical(exp, models, selections, scale, budget, args.out)
    np.savez_compressed(args.out / "pnl.npz", **pnl_by_policy)
    dump(
        args.out / "metrics.json",
        {
            "simulated": result,
            "historical_single_contract": historical,
            "seconds": time.monotonic() - start,
            "smoke_only": args.smoke,
        },
    )
    dump(
        args.out / "status.json",
        {
            "status": "COMPLETE",
            "smoke_only": args.smoke,
            "selections": selections,
        },
    )
    print(json.dumps(result, indent=2))
