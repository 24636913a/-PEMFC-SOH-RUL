# Earliest divergence — evidence boundary

Historical aggregates first diverge at epoch4. Train standardized MSE:
E2 `1.0782116692174564`, F0 `1.0782116759907117`, difference
`6.773255289971303e-09`. Validation standardized RMSE:
E2 `0.9963804483413696`, F0 `0.9963801503181458`, difference
`2.980232238769531e-07`. Epoch1–3 agree exactly.

Historical logs contain epoch aggregates, not batch-level forward, gradient,
AdamW or RNG snapshots. The historical first divergent batch/operation and RNG
timing cannot be inferred from these aggregate values.

The present first Backward raises a strict deterministic implementation error
at epoch1/batch0; it is an execution obstruction, NOT an observed inter-path
divergence. No repeated trajectory pair exists. Do not report epoch1/batch0 as
the first historical divergence, or interpret zero completed batch records as
zero differences.

Full historical epoch comparison and parameter/module differences are preserved
under `outputs/historical_audit`. See final report for the scientific handoff.
