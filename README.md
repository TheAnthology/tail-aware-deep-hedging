# Tail-Aware Deep Hedging

We compare `Novel_model_1` with the semi-recurrent Deep Hedging strategy from Buehler et al. and a classical hedge in each market. The research question is whether the new strategy reduces the probability of a terminal hedging loss within a fixed CVaR95 budget.

| Study | Our strategy | Paper comparison | Classical benchmark |
|---|---|---|---|
| [Heston simulation](sec53_theoretical/README.md) | `Novel_model_1`, three trained seeds | Semi-recurrent Deep Hedging, CVaR50 | Heston model hedge using stock and a variance swap |
| [S&P 500](sec55_sp500/README.md) | `Novel_model_1`, selected per origin | Semi-recurrent Deep Hedging, CVaR75 | Black–Scholes delta using filtered volatility |

The paper's main architecture passes the previous holding into each date's network. Its Heston figures also compare a simpler architecture that omits this input. The retained paper models are independent implementations with calibrated BatchNorm. Paper-level numerical reproduction remains incomplete. The Heston experiment follows the paper's Section 5.2 and uses Section 5.3 in this project's numbering. [Deep Hedging, Sections 5.1–5.2](https://arxiv.org/pdf/1802.03042) · [Published paper](https://doi.org/10.1080/14697688.2019.1571683)

Previously reported as **Selected Residual Combiner**, `Novel_model_1` includes the validation-time fallback. The CVaR and probability experts supply its frozen inputs. The CVaR component's P&L is retained as a diagnostic for that fallback. Standalone Simpler NN, gate, mixture and direct-network comparisons are in [former versions/former models](former%20versions/former%20models/README.md).

## Model design

Within each preference, average expert **holdings**, then apply the policy:

\[
\delta_k=\bar\delta_k^C+D F_\phi\!\left(z_k,D^{-1}(\bar\delta_k^P-\bar\delta_k^C)\right).
\]

The policy has two 32-unit tanh hidden layers and a zero-initialized output layer. It starts exactly at the CVaR policy. Its surplus uses gains from its own actual combined holdings. Expert weights and independently calibrated BatchNorm buffers are frozen. Position scales are fitted on training paths only.

Training minimizes a soft loss probability using a normalized CVaR95 primal-dual penalty. This numerical method does **not** guarantee constraint feasibility. Checkpoint selection uses validation only; selected fallbacks and test violations remain visible. [Full protocol](PROTOCOL.md).

## What is included

```text
sec53_theoretical/   Heston market, model hedge, experts, Novel_model_1, config,
                     paper comparator, frozen checkpoints and recorded results
sec55_sp500/         data preparation, GJR-GARCH, BS, experts, Novel_model_1,
                     paper comparator, latest-origin checkpoints and 10-origin results
common/             risk metrics, checkpoint loading, training and replay CLIs
tests/              numerical/causality checks and checkpoint-parity record
docs/               provenance, references and cloud execution notes
```

## Setup

For a guided reading order, data shapes, and the key accounting equations, see
[Reading the code](docs/CODE_GUIDE.md). Start at `common/run.py` for training or
`common/replay.py` for frozen-model evaluation.

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

Frozen replay evaluates the classical benchmark (`benchmark`), the paper comparator (`paper_deep_hedge`) and each selected `Novel_model_1_seed_*` policy on shared paths. The additional `CVaR` row measures the fallback component. The default preview uses 512 evaluated paths; it is a wiring check, not the recorded full experiment.

```bash
python -m common.replay sec53_theoretical --out runs/heston_preview
python -m common.replay sec55_sp500 --out runs/sp500_preview
```

Add `--full` for the recorded million-path Heston or 100,000-path S&P stream. Heston price grids are built in `runs/cache/heston` on first use, with explicit direct-pricing fallbacks. The S&P replay covers origin 0 only; the historical results file covers all ten completed origins.

## Train a new experiment

These commands train the experts and `Novel_model_1`. The frozen paper comparison is available through the replay commands above.

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

The repository makes no blanket claim that Novel_model_1 outperforms CVaR or Black–Scholes. See the two section result tables, including every Novel_model_1 seed and failure.

[References and attribution](docs/REFERENCES.md). No software license has been selected for this initial public release; visibility alone is not an additional license grant. Third-party dependencies retain their own licenses.

