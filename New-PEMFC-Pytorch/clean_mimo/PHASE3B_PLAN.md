# Phase 3B — horizon-scale diagnostic

## Goal

Phase 3B tests whether a learned residual MIMO model becomes useful only after
the short H=4 range, where FC1 Persistence and FC2 Linear Trend are already
strong. It changes **only** the direct prediction horizon.

## Frozen protocol

- Stacks: FC1 and FC2.
- Input: observed `Utot_mean` history only.
- Target: `Utot(t+h)-Utot(t)`; restoration uses observed origin `Utot(t)`.
- N=15 h, one-layer LSTM hidden 64, MSE, Adam, seed 42, original early stop
  and train-only scalers.
- H in `{4, 8, 12, 24}`; no other feature, representation, or architecture
  changes.
- Development prefix only: 0–60% of each hourly sequence, with folds
  0–30/30–40%, 0–40/40–50%, and 0–50/50–60% train/validation boundaries.
- The final 40% test partition is not constructed, trained on, predicted, or
  evaluated.

## Comparators and primary metric

Every fold and forecast horizon compares Persistence, 15-hour Linear Trend,
and Residual MIMO-LSTM. The primary horizon-specific score is

```text
Skill_Best,h = 1 - RMSE_model,h / min(RMSE_persistence,h, RMSE_linear,h)
```

MAE, RMSE, R2, Skill_Persistence, Skill_Linear, the best baseline name, and
Skill_Best are all saved. Phase 3B inspects `h -> Skill_Best,h` for a
reproducible horizon crossover; it does not select a model using final test.

## Stop boundary

After this development-only diagnostic, do not change N, use relative inputs,
add OCM/attention/Transformer/Full-26, conduct multi-seed studies, or evaluate
the final test until the horizon evidence is reviewed.
