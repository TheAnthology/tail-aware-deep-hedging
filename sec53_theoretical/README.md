# Sec 5.3 — Theoretical / simulated market

Project section number; the frictionless Heston market originates from the paper's Section 5.2.

The market has S0=100, V0=0.04, kappa=1, theta=0.04, vol-of-vol=2 and correlation=-0.7. Thirty daily decisions use dt=1/365, stock plus a variance-swap instrument, an ATM European call and zero explicit costs. The CIR variance transition is sampled exactly; integrated variance uses a declared left-endpoint approximation. This is not exact simulation of the full joint Heston path.

## Active comparison

1. **Novel_model_1** learns a stock and variance-swap correction to a frozen CVaR ensemble. The three selected seeds share three CVaR50 experts and three probability experts.
2. **Paper Deep Hedging** uses a semi-recurrent network with log spot, variance and both previous holdings. Frozen replay loads the CVaR50 checkpoint from the independent Heston reconstruction.
3. **Heston model hedge** computes positions from model-price sensitivities.

The recorded table compares the new policy with its CVaR fallback component. Frozen replay also evaluates the paper policy and the Heston benchmark on shared simulated paths.

## Recorded results

All rows use the same fresh 1,000,000 paths, fixed premium 1.6918339251 and risk budget **CVaR95 ≤ 1.238727**. Results use the released selected checkpoints, not post-test reselection.

| Policy | Loss probability | CVaR50 | CVaR95 | MSE | Budget pass |
|---|---:|---:|---:|---:|---|
| CVaR ensemble |43.9424%|0.289786|1.244369|0.196832|No|
| Novel_model_1 seed 1 |35.8455%|0.392033|1.241409|0.297610|No|
| Novel_model_1 seed 2 |35.9128%|0.385161|1.225682|0.296747|Yes|
| Novel_model_1 seed 3 |34.9112%|0.382277|1.222973|0.297174|Yes|

Novel_model_1 reduces loss frequency, but increases CVaR50 and MSE. Only two of three Novel_model_1 seeds pass the fixed budget. Even the CVaR ensemble slightly exceeds the development-derived cap on this fresh stream. No cap was increased after testing.

These Novel_model_1 replicates share the same frozen expert collection; they are not three independent whole-pipeline replications. Historical development paths remain separate. Exact metrics: [recorded_metrics.json](results/recorded_metrics.json).

## Run

From repository root:

```bash
python -m common.replay sec53_theoretical --out runs/heston_replay --full
python -m sec53_theoretical.run --smoke --out runs/heston_smoke
```

The first call may build the pricing cache. All six expert checkpoints and three selected Novel_model_1 checkpoints are included in compact NPZ format, including every BN buffer. [Config](config.json), [checkpoint manifest](checkpoints/manifest.json), [protocol](../PROTOCOL.md).
