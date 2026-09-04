# Phase 3D plan — prediction-origin relative-history diagnostic

## Question

At fixed causal origins, does removing absolute lifecycle voltage position from the recurrent input while preserving the local voltage trajectory improve H=24 residual forecasting?

## Frozen protocol

- FC1 and FC2; seed 42.
- Three chronological development folds: `0–30%/30–40%`, `0–40%/40–50%`, `0–50%/50–60%`.
- The final `60–100%` partition is neither read nor evaluated.
- 1 h data; `N=15`; `H=24`; Utot-only; direct residual MIMO target `U(t+h)-U(t)`; reconstruction through the legal observed origin voltage `U(t)`.
- One-layer LSTM, hidden size 64, MSE, Adam, unchanged batch size, learning rate, weight decay, and validation early stopping.
- To keep Phase 3D directly comparable with Phase 3C-R N=15, the legal origin schedule is first generated with `Nmax=48`; the network itself receives only the last 15 inputs. Nmax is never an input feature.

## Only experimental variable

Absolute input:

`[U(t-14), ..., U(t)]`

Prediction-origin-relative input:

`[U(t-14)-U(t), ..., U(t-1)-U(t), 0]`

Target, origin, future target times, anchors, baselines, network, optimizer, seed, and folds are identical between the pair.

## Evaluation

Report per-horizon MAE, RMSE, R², Skill_Persistence, Skill_Linear, and Skill_Best, plus mean RMSE, mean/late Skill_Best, stable-positive count, crossover horizon, and per-horizon three-fold mean/std.

For each input lag, calculate raw-voltage train/validation Wasserstein distance for both representations. Mean the 15 lag-specific distances per fold and compare `W_relative-W_absolute`.

For Relative minus Absolute RMSE, use 2,000 paired moving-block bootstrap resamples of fixed validation origins, with 24-hour blocks sampled within each fold. The key result is late `h=13–24` pooled delta and its 95% interval; negative favors relative input.

## Predeclared interpretation rule

Relative history is eligible for a later multi-seed robustness study only when at least two of the following improve without material reversal of the third:

1. mean Skill_Best;
2. late h=13–24 Skill_Best;
3. stable-positive horizon count increases or crossover occurs earlier.

Its paired late RMSE interval should preferably be below zero. If the interval spans zero and the directional improvements are not stable, record no robust representation advantage and stop representation search.

## Stop boundary

This phase creates no final-test result and does not implement another representation, a new encoder, multi-seed robustness, probability output, or RUL.
