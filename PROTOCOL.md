# Experimental protocol and boundaries

| Item | Sec 5.3 theoretical | Sec 5.5 S&P 500 |
|---|---|---|
| Expert preference | CVaR50; soft probability internally | CVaR75; soft probability internally |
| Expert members per preference | 3 | 1 per origin |
| Expert learning rate / batch | Adam0.005 /256 | Adam0.0001 /256 |
| Training paths per expert |5,120,000|20,000,000|
| Time decisions |30 daily|6 five-trading-day intervals|
| Instruments |Stock and variance swap|Stock only|
| BN calibration |8,192 independent paths, per date/layer|Same principle, separate origin-specific paths|
| Residual |6→32tanh→32tanh→2|5→32tanh→32tanh→1|
| Residual risk cap |Fixed1.238727|Frozen CVaR expert's validation CVaR95|
| Residual training |5,000 updates, batch1,024, fixed131,072-path pool|Same budget|
| Final simulated evaluation |1,000,000 shared paths|100,000 shared paths per origin|

Residual objective is soft_probability + lambda*g +5*relu(g)^2, where g=CVaR95/B-1. Dual starts at1 and is updated by projected ascent0.05*g. Adam LR0.001 and global gradient-norm clipping10 are retained. Holdings and path losses are not clipped. The soft probability temperature is0.1 currency units. No extra regularizer is added.

Premium is fixed and identical across compared policies. Total holdings, not trade increments, are network outputs. Surplus is q plus actual accumulated combined trading gains minus a causal model liability mark, before the next price increment. It is not based on expert-only gains or future payoff information.

Expert ensembles average holdings strictly within each preference. Each member keeps its own independently calibrated BN state. Neither learned parameters nor risk statistics are averaged to define a policy.

## Checkpoint selection and feasibility

Experts use their prescribed final trained weights. Every500 residual updates, validation determines feasibility and the soft-probability ranking. A fallback is explicitly labeled. Final-test and realized historical outcomes do not select a checkpoint or adjust the budget. Pass/fail is reported for each seed. A finite empirical penalty and validation-feasible selection do not guarantee population or new-test feasibility.

## Publication refactor versus archived experiments

The release preserves only CVaR/Probability experts, residual policies and benchmarks. Core numerical kernels were extracted from the original workspace; frozen model outputs and all weight arrays are checked for exact parity on fixed synthetic probes. The archived result files remain evidence from the original runs, not a claim that all full training budgets were repeated after refactoring.

The fresh Heston runner deliberately uses a CVaR fallback, while the original comparison permitted a previously optimized constant-weight fallback. All three archived selected Heston residuals are learned models, so their published checkpoint outputs are unaffected. New full runs therefore define the documented curated protocol, not an exact replay of the original fallback search. No scalar Gate or constant-weight model is retained.

Heston original experts reused the first trained member; the fresh runner trains every member from its declared initialization. Checkpoints for all original members remain available for evaluation. Published fresh-run test generation is a separate run from the archived stream; use `common.replay --full` for the archived stream's chunk convention.

## Interpretation

- The original authors' undisclosed initialization, stopping and BN inference procedures have not been recovered.
- BN population calibration is an explicit local addition.
- The S&P dates and market calibration are broadly consistent with the paper; exact vendor-data identity is unverified.
- There are no transaction-cost experiments in this release.
- Loss frequency, tail severity, mean P&L and MSE must all be reported; none substitutes for the others.
- Repeated research on the historical period makes existing windows development evidence, not a new sealed final confirmation.
