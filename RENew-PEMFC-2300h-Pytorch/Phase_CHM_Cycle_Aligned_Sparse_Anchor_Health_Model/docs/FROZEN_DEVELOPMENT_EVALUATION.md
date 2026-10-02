# Frozen Development Evaluation

## Retrospective outer LOAO

The outer folds are 300, 600, 850, 1150, 1500, and 1750 h. In each fold, the
held-out non-BoL anchor is unavailable to fitting, lambda selection, early
stopping, uncertainty estimation, and calibration. The remaining five
non-BoL anchors plus BoL normalization are the only sparse training targets.

For each of the 27 lambda combinations, perform inner LOAO on the five outer
training anchors. Its score is mean absolute anchor error averaged over seeds
42, 2024, and 3407. Select the lowest inner MAE; scores within `1e-8` use
protocol order: `λ_anchor`, then `λ_obs`, then `λ_smooth`. Refit each selected
outer candidate on the five permitted anchors. The three seed predictions are
reduced by their median.

## Frozen reporting unit

For each held-out checkpoint, report only: prediction, target, signed error,
absolute error, squared error, 2.5/50/97.5% uncertainty values, and whether the
95% interval contains the target. Per-anchor Spearman is prohibited.

After six outer predictions, report aggregate MAE, RMSE, bias, Spearman rho,
calibration slope, and calibration intercept. Spearman and calibration use
`n=6` only and are small-sample descriptive diagnostics, not significance tests.

## Qualification gates

`DEVELOPMENT_HEALTH_MODEL_QUALIFIED` requires all of the following:

1. 6/6 legal LOAO predictions;
2. `MAE_CHM < MAE_B0`;
3. `MAE_CHM ≤ min(MAE_B1, MAE_B2) + 1e-6`;
4. zero complete-cycle monotonic violations; and
5. at least 5/6 held-out anchors within their LOAO 95% interval.

The fifth gate is named a **small-sample coverage gate**, never validated 95%
calibration. A near-miss does not authorize protocol revision.

## S1 robustness

The full all-six Development fit and the S1 fit use identical architecture,
lambda selection, and optimization. S1 uses only 300, 850, and 1750 h anchor
supervision. It has `S1_NO_CONCLUSION_REVERSAL=true` only if all are true across
the 146 complete-cycle trajectory:

\[
\rho(H_{primary},H_{S1})\ge0.95,
\quad \operatorname{median}|H_{primary}-H_{S1}|\le0.01,
\quad |H_{primary,end}-H_{S1,end}|\le0.01.
\]

S1 is a robustness sensitivity and cannot select a model.

## Causal-online secondary diagnostic

300 h is `COLD_START_NO_PRIOR_NONBOL_ANCHOR`; 600 h and 850 h are
`INSUFFICIENT_PRIOR_ANCHORS_FOR_PREFIX_MODEL_SELECTION`. These are reported but
not aggregated. Prefix-only fits at 1150, 1500, and 1750 h may respectively use
only their earlier non-BoL anchors. Their aggregate MAE/RMSE has `n=3`, is
secondary only, and cannot select retrospective hyperparameters or use future
anchors.
