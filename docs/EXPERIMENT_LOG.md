# Deep hedging experiment log

Snapshot as of 2026-11-19. Historical S&P 500 results are reported in original index points. Simulated S&P 500 risk budgets use S₀ = K = 100. Loss is negative terminal P&L.

## S&P 500 historical evaluation

All main rolling comparisons use 1,187 prior returns for GJR-GARCH calibration, a 30-trading-day realized path and five-day rebalancing. Strategies in a window receive the same Black–Scholes premium. Neighboring windows overlap.

| Run | Status | Evaluation |
|---|---|---|
| Figure 13 semi-recurrent reconstruction and Black–Scholes | 300/300 scored | Mean P&L 5.3877 and 3.4199. The published means are 4.3 and 3.2. Input data and local training conventions remain sources of replication uncertainty. |
| Original Novel 1 / Novel 2 | 300/300 scored each | Novel 1 uses Simpler experts. Novel 2 uses semi-recurrent experts. Both use the same correction objective and validation-time fallback design. |
| Aligned Novel 1 / Novel 3 development pilot | 3/3 planned windows scored | 65,536 expert paths, five passes and 1,000 correction updates per strategy. Used for a pipeline and projection check. |
| Aligned Novel 1 / Novel 3 larger pilot at origin 1 | 1/1 planned window scored | 524,288 expert paths, with the other pilot settings held fixed. |
| Aligned Novel 1 / Novel 3 full-budget rolling run | Paused after 2/300 windows | Origins 0 and 1 complete. 20 million expert paths, five passes, and 5,000 correction updates per strategy. 298 origins remain. |

### Original Novel 1 / Novel 2 and matched benchmarks

| Strategy | Mean P&L ↑ | Loss rate ↓ | CVaR95 ↓ | CVaR99 ↓ |
|---|---:|---:|---:|---:|
| Original Novel 1 | 3.153 | 33.33% | 57.190 | 90.479 |
| Original Novel 2 | 3.211 | 33.67% | 55.510 | 101.445 |
| Figure 13 semi-recurrent reconstruction | 5.388 | 33.00% | 31.142 | 56.587 |
| Black–Scholes | 3.420 | 33.33% | 46.108 | 71.376 |
| Original Novel 1 standalone C expert | 4.265 | 36.67% | 37.194 | 58.597 |
| Original Novel 2 standalone C expert | 3.902 | 34.00% | 35.575 | 76.224 |

Novel 1 fell back to C in 66 windows and Novel 2 in 75. Compared with their own C experts, their loss counts fell by 10 and 1 windows, respectively, while historical CVaR95 rose by 20.00 and 19.94 points. Novel 2 led Novel 1 in 142 of 300 paired windows, with a mean P&L difference of +0.058 points. CVaR99 reflects three outcomes.

### Aligned Novel 1 / Novel 3

Both policies share one Simpler C/P expert pair per window, the same simulated paths, budget and validation rule. Expert training uses float64, centered BatchNorm, random batches drawn from a fixed pool, and independent BatchNorm calibration. Novel 3 adds a CVaR95 projection after Adam, with a step-norm cap during recovery. Current rolling scores are from two completed full-budget windows.

| Origin | Original Novel 1 P&L ↑ | Aligned Novel 1 P&L ↑ | Novel 3 P&L ↑ | Figure 13 P&L ↑ | Black–Scholes P&L ↑ |
|---|---:|---:|---:|---:|---:|
| 0 | +0.902 | +1.010 | +1.010 | +1.568 | +6.095 |
| 1 | +4.424 | +5.799 | −2.486 | −4.428 | +1.064 |

At origin 0 both aligned policies fell back to the same expert. At origin 1 both used learned corrections. On independent simulations for origin 1, the loss rates were 42.94% and 58.97%, and CVaR95 was 3.092 and 3.079 for aligned Novel 1 and Novel 3, respectively, against a budget of 3.164.

The three-origin pilot had mean historical P&L of +43.319 and +35.553 for aligned Novel 1 and Novel 3. Those overlapping origins were all rising markets. At origin 1, increasing the expert sample from 65,536 to 524,288 paths changed the two scores from +54.671 / +35.277 to +13.752 / +13.867. The pilot and full-budget runs remain distinct experiments.

## Heston simulation

| Experiment | Status | Notes |
|---|---|---|
| Theoretical Novel 1, formerly Selected Residual Combiner | Recorded three-seed experiment | Frozen C/P experts and validation fallback, with a model-based Heston hedge as reference. See the [theoretical study](../sec53_theoretical/README.md) and [recorded metrics](../sec53_theoretical/results/recorded_metrics.json). |
| Figures 2 and 4, zero-cost distribution comparisons | Local million-path evaluations | Compared Heston theoretical hedge, calibrated semi-recurrent variants and Simpler NN. Precision, BatchNorm and simulator controls were separate development runs. |
| Figure 5, 1% proportional transaction costs and CVaR99 | Complete | Both networks trained to 160,000 updates. The semi-recurrent run resumed from 20,000. Pricing used 262,144 paired paths and independent testing used 1,000,000 paired paths. |

| Figure 5 metric | Semi-recurrent | Simpler |
|---|---:|---:|
| Risk-adjusted price ↓ | 4.4525 | 4.5041 |
| Paper price | 5.5137 | 6.7446 |
| Mean loss before premium ↓ | 2.9091 | 3.1072 |
| Mean loss after premium ↓ | −1.5434 | −1.3969 |
| Test CVaR99 after premium | −0.0138 | −0.0080 |
| Mean transaction cost ↓ | 1.2060 | 1.4045 |

The price ordering agrees with Figure 5. The gap is 0.0516 in this run and 1.2309 in the paper. Both prices are below the paper values. The mean loss after premium includes the quote charged by each strategy.

## Earlier development and handoffs

- [Former model archive](../former%20versions/former%20models/README.md) indexes the C/P and weighted-loss ablations, fixed mixtures, market and surplus gates, direct MLP, early S&P reconstructions, CVaR optimizer diagnostics and spline-flow data generation pilot. The Heston lambda scan is partial, with no final lambda selected.
- The [project protocol](../PROTOCOL.md) and the [original S&P development records](../sec55_sp500/README.md) describe earlier splits and baselines. The older ten-window results remain a separate development comparison.
- Data generation and the longer-horizon rolling forecast are separate workstreams. This snapshot records the spline-flow pilot. It includes no completed 30/240-day forecast-update comparison.

### Run artifacts

Local run exports are identified by their filenames for team handoff.

| Run | Export and files |
|---|---|
| Figure 13 | `SP500_Figure13_results/summary.json` and `rolling_protocol.json` |
| Original Novel 1 / Novel 2 | `SP500_Novel1_2_results/results/comparison.json`, both per-model summaries, and the rolling protocol |
| Aligned Novel 1 / Novel 3 | `results_full/summary.json`, `rolling_protocol.json`, and the two completed origin directories with weights and hedge ledgers, for use with the matching cloud bundle |
| Heston Figure 5 | `Heston_Figure5_final_results/results_gpu/summary.json`, `figure5_table.csv`, `loss_histogram.csv`, and the 160,000-update checkpoints |
