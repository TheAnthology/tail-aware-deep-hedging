# Sec 5.3 — Theoretical / simulated market

Project section number; the frictionless Heston market originates from the paper's Section 5.2.

The market has S0=100, V0=0.04, kappa=1, theta=0.04, vol-of-vol=2 and correlation=-0.7. Thirty daily decisions use dt=1/365, stock plus a variance-swap instrument, an ATM European call and zero explicit costs. The CIR variance transition is sampled exactly; integrated variance uses a declared left-endpoint approximation. This is not exact simulation of the full joint Heston path.

## Retained models

1. Heston model hedge from QuantLib pricing, explicit finite differences and checked grid interpolation/direct fallbacks.
2. CVaR ensemble: three Simpler experts with independent per-date 2→17→17→2 networks, BN before ReLU, inputs log S and V. Each expert trains CVaR50. Holdings are averaged within preference.
3. Residual Combiner: two-asset nonlinear correction to the frozen CVaR ensemble, with three residual seeds retained.

The three Probability experts supply internal contrast features. They are not a standalone headline strategy; unconstrained frequency-only training can generate very large losses.

## Recorded results

All rows use the same fresh 1,000,000 paths, fixed premium 1.6918339251 and risk budget **CVaR95 ≤ 1.238727**. Results use the released selected checkpoints, not post-test reselection.

| Policy | Loss probability | CVaR50 | CVaR95 | MSE | Budget pass |
|---|---:|---:|---:|---:|---|
| CVaR ensemble |43.9424%|0.289786|1.244369|0.196832|No|
| Residual seed 1 |35.8455%|0.392033|1.241409|0.297610|No|
| Residual seed 2 |35.9128%|0.385161|1.225682|0.296747|Yes|
| Residual seed 3 |34.9112%|0.382277|1.222973|0.297174|Yes|

Residual reduces loss frequency, but increases CVaR50 and MSE. Only two of three residual seeds pass the fixed budget. Even the CVaR ensemble slightly exceeds the development-derived cap on this fresh stream. No cap was increased after testing.

These residual replicates share the same frozen expert collection; they are not three independent whole-pipeline replications. Historical development paths remain separate. Exact metrics: [recorded_metrics.json](results/recorded_metrics.json).

## Run

From repository root:

```bash
python -m common.replay sec53_theoretical --out runs/heston_replay --full
python -m sec53_theoretical.run --smoke --out runs/heston_smoke
```

The first call may build the pricing cache. All six expert checkpoints and three selected residual checkpoints are included in compact NPZ format, including every BN buffer. [Config](config.json), [checkpoint manifest](checkpoints/manifest.json), [protocol](../PROTOCOL.md).
