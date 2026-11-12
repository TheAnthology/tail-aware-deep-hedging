# Reading the code

The project has two market adapters and one shared experiment workflow.
Read the workflow first, then follow the adapter for the market you need.

## Suggested reading order

1. [`common/run.py`](../common/run.py): `main` connects data generation,
   expert training, residual selection, and final evaluation. The private
   helpers keep these stages separate.
2. [`sec53_theoretical/experiment.py`](../sec53_theoretical/experiment.py) or
   [`sec55_sp500/experiment.py`](../sec55_sp500/experiment.py): each `Experiment`
   exposes the same operations to the shared runner.
3. The section's `market.py`: market simulation and benchmark accounting.
   The S&P adapter also calibrates GJR-GARCH using past returns only.
4. [`sec53_theoretical/experts.py`](../sec53_theoretical/experts.py) or
   [`sec55_sp500/models.py`](../sec55_sp500/models.py): the expert predicts
   total holdings at each decision date.
5. [`sec53_theoretical/residual.py`](../sec53_theoretical/residual.py) or
   `StockResidual` in the S&P models module: the residual adjusts CVaR holdings
   using current market state and the combined strategy's running surplus.

For inference only, start with [`common/replay.py`](../common/replay.py).
It loads released weights and evaluates them without training or recalibrating
BatchNorm.

## Shared training workflow

`main` in `common/run.py` follows this order:

1. Read the section configuration and create a new output directory.
2. Generate separate BatchNorm-calibration and validation paths.
3. `_train_experts`: train paired CVaR/probability experts, calibrate each
   member's BatchNorm statistics, and freeze its inference state.
4. Cache expert holdings on validation paths; establish the CVaR95 budget.
5. Generate the residual's training pool; fit position scales on that pool.
6. `_train_residuals`: train each seed and select checkpoints on validation.
7. Save `selection_before_test.json` before drawing final-test paths.
8. `_evaluate_simulated` and `_evaluate_historical`: evaluate the frozen
   choices and save P&L and metrics.

Each stage uses its own configured random seed. C/P expert pairs share
initialization and training paths. `--smoke` reduces the budgets for an
integration check; it does not reproduce the archived research results.

## Data names and shapes

`N` is the number of paths and `T` is the number of decision dates:
30 for Heston, 6 for S&P. The Heston asset axis is ordered stock, variance swap.
The compact dictionary keys are shared across adapters and tensor helpers.

| Key / name | Meaning | Heston shape | S&P shape |
|---|---|---|---|
| `f` / `features` | Current log spot and variance | `(N, 30, 2)` | `(N, 6, 2)` |
| `changes` | Price changes over the next holding interval | `(N, 30, 2)` | `(N, 6)` |
| `hc` / `cvar_holdings` | Frozen CVaR expert holdings | `(N, 30, 2)` | `(N, 6)` |
| `hp` / `probability_holdings` | Frozen probability expert holdings | `(N, 30, 2)` | `(N, 6)` |
| `liability` | Model call mark at the current decision date | `(N, 30)` | `(N, 6)` |
| `payoff` | Call payoff at maturity | `(N,)` | `(N,)` |
| `_raw` | Original simulated or observed market arrays for the benchmark | Section-specific | Section-specific |

`C` and `P` identify the two expert preferences. `q` is the fixed option premium,
shared by the compared policies. Heston features use instantaneous variance;
S&P features use annualized daily variance, `252 * sigma_daily**2`.
Features are `float32`; residual accounting and evaluation use `float64`.

`project` averages **holdings within a preference**. Every member retains its own
weights and BatchNorm buffers. The probability expert supplies an internal
direction for the residual; it is not a separately reported main policy.

## Holdings and P&L

At date `k`, a network outputs total holdings for the next interval. For Heston,
sum over both assets as well as dates:

```text
trading_gain[k] = sum_assets(holding[k] * changes[k])
terminal_pnl = q + sum_dates(trading_gain) - payoff
loss = -terminal_pnl
```

The residual builds its state before using the current interval's change:

```text
surplus[k] = q + cumulative_gain_before_k - liability[k]
expert_difference[k] = (probability_holdings[k] - cvar_holdings[k]) / scale
holding[k] = cvar_holdings[k] + scale * network(current_state, expert_difference[k])
cumulative_gain_after_k = cumulative_gain_before_k + trading_gain[k]
```

This order matters: current holdings depend on gains from actual combined
holdings in previous intervals. The current price increment and final payoff
are not network inputs. The output head starts at zero, so the initial combined
policy is exactly the CVaR policy.

The layer attributes `a`, `b`, and `out` are the first hidden layer, second hidden
layer, and output head. Their existing names and order are retained for saved
model compatibility.

## Objectives and checkpoint selection

The CVaR expert uses a trainable threshold `eta` in liability units. Its objective
is `eta + mean(relu(liability_loss - eta)) / (1 - alpha) - q`, where
`liability_loss = payoff - trading_gains`. Here `alpha` is 0.5 for Heston and
0.75 for S&P. The probability expert minimizes `mean(sigmoid(loss / 0.1))` and
does not optimize `eta`.

The residual uses a fixed CVaR95 budget:

```text
risk_excess = CVaR95(loss) / budget - 1
objective = soft_loss_probability + dual * risk_excess + 5 * relu(risk_excess)**2
dual = max(0, dual + 0.05 * risk_excess)
```

Positive `risk_excess` means the budget is exceeded. The dual update raises the
penalty in that case. The normalized correction RMS returned by the training
step is a diagnostic; it does not enter the objective.

A residual checkpoint replaces the current selection only if its validation
CVaR95 meets the budget and its soft loss probability improves by more than
`1e-12`. If no learned checkpoint improves on a feasible CVaR fallback, the
fallback remains selected. If neither is feasible, the runner records
`no_feasible_candidate`. Final-test results never change that choice.

See [`PROTOCOL.md`](../PROTOCOL.md) for experimental boundaries and differences
between fresh runs and archived experiments. Validation feasibility is not a
guarantee of test or population feasibility.
