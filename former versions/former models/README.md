# Former models

These are frozen development comparisons from the Heston and S&P 500 studies. The original run names are preserved to identify the corresponding model definitions and weights. The active model lineup is in the [project README](../../README.md).

| Market | Earlier comparison | Archived artifacts |
|---|---|---|
| Heston | Simpler NN and training-length controls | [Calibrated Simpler NN](results/section52_calibrated_million_v1), [Figure 4 control](results/section52_fig4_control), [semi-recurrent continuation](results/section52_recurrent_continue20k_v1) |
| Heston | Standalone CVaR, probability, weighted combined loss and constraint experiments | [A/B/C experts](results/risk_ablation_simpler_v1), [partial lambda scan](results/risk_lambda_scan_v1), [CVaR50 penalty](results/risk_constrained_beta10_v1) |
| Heston | Fixed mixtures E/F, market gate G, surplus gate H | [Mixture weights, six calibrated experts and six gate runs](results/preference_moe_v1), [gate implementation](scripts/moe_gate_v1.py) |
| Heston | Direct MLP control | [Three trained final weights and input scales](results/residual_combiner_v1), [network implementation](scripts/residual_combiner_engine_v1.py). Use direct mode. Validation selected the fixed F fallback for all three trials. |
| S&P 500 | Early PyTorch and TensorFlow pilots | [Single-contract PyTorch](results/paper55_local_baseline_pilot_v1), [retained rolling checkpoints](results/paper55_local_rolling100_v1), [TensorFlow diagnostic](results/paper_tf_v2_diagnostic_window000), [one-pass TensorFlow pilot](results/paper_tf_v2_paper_window000) |
| S&P 500 | Paper-architecture training variants at origin 0 | [Learning rate 0.0003](results/paper55_recurrent_bnfix_lr3e4_v1_window000), [0.00065](results/paper55_recurrent_bnfix_lr6p5e4_v1_window000), [0.001](results/paper55_recurrent_bnfix_lr1e3_v1_window000), [five passes](results/paper55_lr1e4_five_epochs_v1_window000), [CVaR optimizer controls](results/paper55_risk_optimizer_controls_v1_window000) |
| S&P 500 | Spline-flow path-generator pilot | [Three trained flows](results/nf_dgp_pilot_v1), [generator implementation](scripts/nf_dgp_pilot_engine_v1.py) |

The archive contains model definitions, frozen weights and selected settings. Historical training scripts use the original run-directory layout. Large path arrays, optimizer histories, raw market data and pricing caches are kept separately. The active training and replay commands read only the section directories.

The main policy previously appeared as **Selected Residual Combiner**. Historical code keeps that name wherever it shared an implementation with an archived control. In current experiments its name is **Novel_model_1**.
