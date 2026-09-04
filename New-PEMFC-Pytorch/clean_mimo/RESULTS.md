# Phase-1 results

This file is updated after the single-seed FC1 and FC2 runs.  The authoritative
numerical artifacts are the CSV/JSON files under `outputs/`; this document does
not substitute aggregate R² for the required h=1…4 metrics.

## Required conclusion rule

For each stack and each horizon, compare MIMO-LSTM against both Persistence and
Linear trend using the saved MAE/RMSE/R² and MSE skill versus Persistence.
Only state that Phase 1 supports MIMO if the comparison is stable across the
two stacks and later across pre-declared seeds.  If MIMO fails to improve, first
diagnose hourly aggregation, causal feature definition, scaling, and split
coverage; do not add a more complex model to compensate.

## Single-seed confirmation (seed 42, fixed configuration)

Both data pipelines passed the automated audit.  Exact logs and metadata:

- FC1: `outputs/FC1/seed_42/leakage_audit.md`
- FC2: `outputs/FC2/seed_42/leakage_audit.md`

The following are arithmetic means over the four separately evaluated horizons;
they are a compact diagnosis only.  Consult `metrics_per_horizon.csv` for the
required h=1…4 values.

| Stack | Model | mean MAE (V) | mean RMSE (V) | mean R² | mean RMSE improvement vs Persistence |
|---|---:|---:|---:|---:|
| FC1 | Persistence | 0.002019 | 0.004771 | 0.826927 | 0.000000 |
| FC1 | Linear trend | 0.004263 | 0.007981 | 0.542178 | -0.759343 |
| FC1 | vanilla MIMO-LSTM | 0.042120 | 0.044704 | -12.812314 | -9.509882 |
| FC2 | Persistence | 0.006598 | 0.041147 | -0.080985 | 0.000000 |
| FC2 | Linear trend | 0.010353 | 0.035546 | 0.191964 | +0.136321 |
| FC2 | vanilla MIMO-LSTM | 0.030962 | 0.045586 | -0.328975 | -0.108192 |

## Phase-1 conclusion

**The clean vanilla MIMO-LSTM does not currently outperform the analytical
baselines on either FC1 or FC2.**  The first acceptance criterion (causal,
reproducible pipeline and executable three-model comparison) is satisfied; the
success criterion of stable MIMO superiority is not.  Therefore do not run a
ten-seed or Loop50 study yet, do not test the residual version, and do not add
OCM-MIMO-LSTM.

The immediate diagnosis is data-definition/model fit, not model complexity:

- FC1 targets shift downward over time: training target mean is 3.2995 V,
  validation is 3.2496 V, and test is 3.2276 V.  Direct absolute-voltage
  prediction under a train-only scaler is forced to extrapolate beyond the
  training target distribution, while Persistence correctly starts from the
  observed current voltage.
- FC2 retains the intended hourly current variability, but the later test
  region includes substantially different operating behavior.  Persistence is
  weak (mean R² −0.081), Linear trend gives positive skill (+13.6% mean RMSE
  improvement), and the plain LSTM still does not exploit the joint features.

The next action is a protocol-preserving diagnostic of feature representation
(including whether known elapsed operating time and/or legal residual targets
should be predeclared), training-target formulation, and validation/test drift.
It must be decided using validation—not test—and then rerun from a new fixed
configuration.  No complex architecture is justified before that diagnosis.

## Status

- FC1 leakage audit: pass
- FC2 leakage audit: pass
- FC1 single seed (Persistence / Linear / vanilla MIMO): complete
- FC2 single seed (Persistence / Linear / vanilla MIMO): complete
- Residual MIMO: deferred pending a data-definition decision
- OCM-MIMO-LSTM: not implemented by Phase-1 design
