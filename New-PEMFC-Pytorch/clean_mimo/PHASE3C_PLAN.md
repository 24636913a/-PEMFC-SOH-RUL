# Phase 3C — history-length diagnostic at H=24

## Purpose

Phase 3B found development-only positive residual-MIMO skill at longer forecast
ranges. Phase 3C tests whether that advantage is strengthened by a longer
observed voltage history, without changing architecture, representation, or
the prediction task.

## Frozen variables

- FC1 and FC2; causal 1-hour aggregation.
- Direct H=24 residual MIMO with a legal observed `Utot(t)` persistence anchor.
- Utot-only history, one-layer LSTM hidden 64, MSE, Adam, seed 42, original
  batch size, learning rate, weight decay, and validation early stopping.
- Development-only folds: 0–30/30–40%, 0–40/40–50%, 0–50/50–60%.
- No final 40% test split or result is constructed or read by the experiment.

The sole changed variable is `N in {15, 24, 48, 72}`.

## Required summaries

For every fold and h=1..24, report MAE, RMSE, R2, Persistence and Linear
skills, and `Skill_Best`. Each N records:

- `MeanSkill_all`: mean Skill_Best over h=1..24 and all folds;
- `MeanSkill_late`: mean Skill_Best over h=13..24 and all folds;
- stable-positive count: horizons positive against the best baseline in all
  three folds;
- crossover `h_c`: earliest start of three consecutive horizons positive in
  all three folds.

## Stop boundary

Phase 3C determines whether long history has a precondition for later
multi-scale or matrix-memory work. It does not implement M-LSTM, multi-scale
history, relative history, uncertainty, multiple seeds, a final test, or RUL.
