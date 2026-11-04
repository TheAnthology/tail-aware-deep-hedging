# Sec 5.5 — S&P 500 historical data

Each origin uses 1,187 past daily log returns to estimate GJR-GARCH with variance-standardized Student-t innovations. Training uses simulated paths from that fitted model. The next 30 actual daily returns provide one realized historical ATM call P&L per strategy. Stock positions are updated every five trading days; there are no explicit costs.

## Retained models

1. Black–Scholes delta using causally filtered volatility and the same fixed premium.
2. One CVaR75 Simpler expert per origin: six independent 2→20→20→1 networks, inputs log S and annualized conditional variance, independently calibrated BN.
3. One Residual Combiner per origin, anchored to the frozen CVaR expert. A single Probability expert is retained as an internal input dependency.

This is an adapted Section 5.5 experiment. The added variance feature and Simpler architecture differ from the original paper's semi-recurrent network. No preference ensemble is used in this section.

## Historical results: 10 of 50 origins completed

| Policy | Mean realized P&L, original index points |
|---|---:|
| Black–Scholes |2.080992|
| CVaR expert |-0.925050|
| Selected Residual |6.032871|

Using each contract's S0=100 normalization:

| Policy | Loss probability | CVaR75 | MSE | Mean P&L |
|---|---:|---:|---:|---:|
| CVaR expert |60%|0.221246|0.023681|-0.035443|
| Selected Residual |20%|0.014937|0.094370|0.236267|

Residual P&L is higher in 7 contracts and equal in 3. The equal cases deploy the validation-selected CVaR fallback; they are not learned residual improvements. The ten 30-day contracts overlap because origins move weekly. Tail estimates and apparent historical gains are provisional. Positive mean shifts can raise MSE around zero, so MSE is reported separately from downside risk.

Completed origins are 0–9. The latest-origin frozen C/P/Residual checkpoints are included; the other origins' outcomes are retained in [historical_10_windows.json](results/historical_10_windows.json). Those other checkpoints are not bundled, and this repo does not claim it can replay all ten archived policies from the three latest-origin files.

The historical index path is real; the option payoff is synthetic. Historical traded option prices, execution slippage and bid–ask costs are not part of this study. Per-origin simulated CVaR checks are separate from statistics across ten historical outcomes.

## Run

```bash
python -m sec55_sp500.download_data
python -m common.replay sec55_sp500 --full --out runs/sp500_frozen_origin0
python -m sec55_sp500.run --smoke --window 0 --out runs/sp500_smoke
```

For full training, omit `--smoke`. Each expert consumes 20M simulated paths; Residual trains for 5,000 updates on a fixed 131,072-path pool. [Config](config.json), [protocol](../PROTOCOL.md), [data provenance](../docs/DATA.md).
