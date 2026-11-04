# Tail-Aware Deep Hedging

Preference-specific CVaR experts and residual nonlinear hedging policies in a simulated Heston market and on historically calibrated S&P 500 paths.

The research question is whether a residual policy can reduce the frequency of hedging losses while respecting a predetermined CVaR95 budget. Results are preliminary: lower loss frequency does not imply uniformly better hedging, and individual risk-budget failures are reported.

## Two study sections

| Section | Market and data | Retained models | Evidence |
|---|---|---|---|
| [Sec 5.3 — Theoretical / simulated market](sec53_theoretical/README.md) | Heston simulation; stock and variance swap; daily hedging | Heston model hedge, CVaR ensemble, Residual Combiner | Three residual seeds on a shared million-path evaluation |
| [Sec 5.5 — S&P 500 historical data](sec55_sp500/README.md) | Historical daily returns → GJR-GARCH/Student-t calibration → simulated training; real underlying paths for historical evaluation | Black–Scholes, one CVaR expert, one Residual Combiner per origin | 10 of 50 planned overlapping historical contracts completed |

**Section numbering is this project's organization.** The frictionless Heston setup comes from Section 5.2 of Buehler et al. (2019). The project heading “Sec 5.3” does not claim reproduction of the original paper's transaction-cost section. “Theoretical” denotes a controlled simulated market, not a new theorem.

中文导航：Sec 5.3 为 Heston 理论模拟；Sec 5.5 为 S&P 500 历史数据实验。Probability Expert 只作为 Residual 的内部依赖保留，不增加第三个学习模型的主结果。

## Model design

Within each preference, average expert **holdings**, then apply the residual:

\[
\delta_k=\bar\delta_k^C+D F_\phi\!\left(z_k,D^{-1}(\bar\delta_k^P-\bar\delta_k^C)\right).
\]

The residual has two 32-unit tanh hidden layers and a zero-initialized output layer. It starts exactly at the CVaR policy. Its surplus uses gains from its own actual combined holdings. Expert weights and independently calibrated BatchNorm buffers are frozen. Position scales are fitted on training paths only.

Training minimizes a soft loss probability using a normalized CVaR95 primal-dual penalty. This numerical method does **not** guarantee constraint feasibility. Checkpoint selection uses validation only; selected fallbacks and test violations remain visible. [Full protocol](PROTOCOL.md).

## What is included

```text
sec53_theoretical/   Heston market, model hedge, experts, residual, config,
                     frozen checkpoints and recorded results
sec55_sp500/         data preparation, GJR-GARCH, BS, experts, residual,
                     config, latest-origin checkpoints and 10-origin results
common/             risk metrics, checkpoint loading, training and replay CLIs
tests/              numerical/causality checks and checkpoint-parity record
docs/               provenance, references and cloud execution notes
```

Only the main expert/residual families and necessary benchmarks are retained. The earlier scalar Gate, direct MLP, Flow experiments, learning-rate searches and unstable legacy pilots are outside this release.

## Setup

Use Python 3.11. The pinned dependencies reflect the tested local research environment; Linux/GPU parity and speed are not yet certified.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
```

Fetch and validate the attributed historical snapshot before S&P experiments:

```bash
python -m sec55_sp500.download_data
```

Raw market data is acquired separately and excluded from Git. Its exact SHA256 is checked; a changed upstream snapshot is rejected. See [data provenance](docs/DATA.md).

## Replay the released models

These commands use frozen calibrated weights, without training. The default preview uses 512 evaluated paths; it is a wiring check, not the recorded full experiment.

```bash
python -m common.replay sec53_theoretical --out runs/heston_preview
python -m common.replay sec55_sp500 --out runs/sp500_preview
```

Add `--full` for the recorded million-path Heston or 100,000-path S&P stream. Heston price grids are built in `runs/cache/heston` on first use, with explicit direct-pricing fallbacks. The S&P checkpoints cover origin 0 only; the historical results file covers all ten completed origins.

## Train a new experiment

Start with a short integration run:

```bash
python -m sec53_theoretical.run --smoke --out runs/heston_smoke
python -m sec55_sp500.run --smoke --window 0 --out runs/sp500_smoke
```

Omit `--smoke` to use the section's full configuration. Each output directory must be new. S&P `--window` selects an independently calibrated origin in 0–299. Full-budget runs are substantial; no cloud job is launched by installation or import.

The publication runners are curated refactors. They have short-run and frozen-forward checks; a complete fresh run of every archived experiment has not been repeated after refactoring. Archived results are immutable evidence from the original runs, not outputs claimed to have been generated by these new runners. See [cloud notes](docs/CLOUD.md) for differences and restoration requirements.

## Main limitations

- No explicit transaction costs in these experiments.
- Heston expert ensembles and S&P single experts are different experimental settings.
- Independent BN population calibration is a declared numerical addition, not recovered author inference code.
- Ten overlapping historical contracts cannot establish stable extreme-tail performance or broad market superiority.
- Actual S&P underlying prices are observed; the ATM option liability is synthetic, with model-based premiums/marks rather than historical option quotes.
- The original paper's 300-origin neural historical result has not been reproduced.

The repository makes no blanket claim that Residual outperforms CVaR or Black–Scholes. See the two section result tables, including every residual seed and failure.

[References and attribution](docs/REFERENCES.md). No software license has been selected for this initial public release; visibility alone is not an additional license grant. Third-party dependencies retain their own licenses.
