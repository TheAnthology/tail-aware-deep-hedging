# Cloud execution and retention

No cloud resources are provisioned by this repository. Rebuild the pinned Python environment on the target host; do not copy a macOS virtual environment to Linux.

1. Run unit checks and one smoke run in each section.
2. Replay the bundled calibrated policies on fixed paths. Compare P&L/CVaR and model/BN buffers before full training. GPU arithmetic need not be bitwise identical to the original CPU host.
3. Measure throughput and RAM for the real market simulation, pricing and tiny per-date networks before choosing a large job count. GPU speedup is not established by this release.
4. Use a new output directory for each origin/seed/configuration. The fresh runner refuses overwrites and has no automatic partial-stage resume. Completed origins can run as independent jobs; preserve all outcomes including failures.
5. Save both complete HDF5 model weights (including BN moving statistics) and the Adam checkpoint for continuation. The original Keras network-list layout may not restore all BN buffers from a TensorFlow checkpoint alone. The shipped compact NPZs are frozen inference models, not Adam-resume bundles.
6. Keep calibration, validation and final streams separate; freeze selection rules before expanding historical evaluation. Account for overlapping30-day windows in uncertainty estimates.

Full Heston runs can require several GB of RAM for simulation, expert projections and evaluation; pricing-cache construction is an additional first-run cost. Full paths remain in memory rather than being repeatedly written to disk. Retained run files contain weights, checkpoints, sparse validation logs, configuration, metrics and terminal P&L. Source data, generated runs and caches are ignored by Git.

The archived S&P study has ten completed origins; that is not a promise that a fresh run's output is interchangeable with the archive. The original paper's neural300-origin result remains outstanding.
