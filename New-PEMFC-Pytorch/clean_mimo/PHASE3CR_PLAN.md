# Phase 3C-R plan — matched-origin history-length diagnostic

## Purpose

Phase 3C varied `N=15/24/48/72` at `H=24`, but each N generated its own legal split-contained windows. That was causal and leakage-free, yet longer N necessarily had fewer train/validation origins. Phase 3C-R removes this sample-composition confound before any architectural decision is made.

## Frozen protocol

- Stacks: FC1 and FC2.
- Development data only: chronological folds `0–30%/30–40%`, `0–40%/40–50%`, and `0–50%/50–60%`.
- Final `60–100%` partition: not constructed, read, predicted, scored, or used for selection.
- Forecasting: 1 h resolution, `H=24`, direct residual MIMO with legal reconstruction `Uhat(t+h)=U(t)+DeltaUhat(t+h)`.
- Input: hourly `Utot` history only; no relative representation and no elapsed-time feature.
- Encoder and training: one-layer LSTM, hidden size 64, MSE, Adam, existing learning rate/batch size/weight decay, validation early stopping, seed 42.
- Histories: only `N=15`, `N=24`, `N=48`. `N=72` is excluded because it is outside this matched-origin design.

## Matched-origin rule

For every fold and both partitions, generate the complete legal origin list with `Nmax=48` first. For a given origin `t`:

- `N=48` receives `U(t-47:t)`;
- `N=24` receives the last 24 rows of that exact Nmax window, `U(t-23:t)`;
- `N=15` receives the last 15 rows of that exact Nmax window, `U(t-14:t)`.

All three variants therefore share identical train and validation origins, target timestamps, target voltage, origin anchors, and residual labels. Each N still fits its own input scaler on its own train-only representation. Since residual targets are identical, the train-only residual scaler population is identical across N.

## Evaluation and predeclared summaries

- Per horizon: MAE, RMSE, R², Skill versus Persistence, Skill versus Linear Trend, and Skill versus the lower-RMSE analytical baseline.
- `MeanSkill_all`: mean `Skill_Best` over h=1–24.
- `MeanSkill_late_h13_h24`: mean `Skill_Best` over h=13–24.
- Stable-positive horizon count: number of horizons with `Skill_Best>0` in every fold.
- Crossover horizon: first of three consecutive stable-positive horizons.
- Pairwise model-only contrast: `DeltaRMSE=RMSE(first_N)-RMSE(second_N)` at matched targets for `N24−N15`, `N48−N15`, and `N48−N24`.
- Paired uncertainty: 2,000 deterministic moving-block bootstrap resamples of fixed validation origins, with 24-hour blocks drawn separately within each fold. A negative delta favors the first N.

## Audit and stop boundary

The fail-closed audit checks causal hourly aggregation, Nmax split containment, train-only input/residual scalers, legal anchors, no final-partition access, output isolation, suffix slicing, and cross-N equality of origins/targets/labels.

This phase will stop after the matched-origin development report. It will not run a final test, relative-history input, a matrix-memory encoder, multi-scale input, uncertainty model, multi-seed robustness, or RUL.
