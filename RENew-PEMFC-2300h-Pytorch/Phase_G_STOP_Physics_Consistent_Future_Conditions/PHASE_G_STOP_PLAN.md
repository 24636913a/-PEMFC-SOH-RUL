# Phase G-STOP plan

## Research question

Can strictly causal forecasts of physically consistent future operating conditions improve the frozen 12 h PEMFC voltage forecast without using future voltage, true future conditions, or Test information?

## Frozen starting point and task

- Research start: `06f157d28da767f7085e7f0bff9c907fea4b7214`.
- Dataset: RealHyFC durability load-cycle data.
- Resolution: 0.5 h.
- History: 48 steps (24 h).
- Horizon: 24 steps (12 h).
- Target: `DeltaV(t+h) = V(t+h) - V(t)`.
- Historical input: frozen F2 14-feature set.
- Future-condition input: frozen 9-feature set.
- Validation: the original three chronological expanding Development folds.
- Test boundary: `[1819.997738, 2256.800000] h`; Test targets and features remain unopened.

All alternatives use identical validation origins, target timestamps, masks, units, horizons, preprocessing scope, and frozen SA-TimeXer weights. Test cannot be used for diagnosis, threshold selection, or performance evaluation.

## G1 preregistration

G1 precedes all new forecasting models. The audit uses the frozen Development prefix only. Raw-data validation stops before the Test boundary and reads only current, current density, ageing coordinate, and operation-point fields needed for the registered identities.

The zero-current definition is frozen to the Phase N1 implementation: a raw sample is assigned to the zero-current set when `abs(I.S.act) <= 0.10 A`. Bin standard deviations are population standard deviations (`ddof=0`). Missing current values are not counted as zero but remain in the Phase N1 denominator; their frequency is audited.

Evidence is sufficient to proceed beyond G1 only when all conditions hold:

1. At least 30 independent mixed-current segments exist across Development.
2. Each Development validation fold contains at least 10 independent mixed-current segments.
3. Both nonzero-to-mixed and mixed-to-nonzero transitions have at least 30 independent occurrences.
4. Current and current-density statistics come from identical raw bins, and a fixed active area estimated from legal Fold-1 Train data validates `J=I/A`. For each of mean, population standard deviation, and peak-to-peak, the zero-intercept slope must be within 0.01% of the Train-only area, Pearson correlation must be at least 0.99999, and median relative identity error must not exceed `2e-4` after a 1 A denominator floor.
5. Recomputed raw-bin means, population standard deviations, peak-to-peak values, and zero-current fractions agree with the frozen causal table to numerical tolerance (`1e-6` absolute, except recorded source rounding).
6. The mixed-current mean/variance reconstruction is reported quantitatively. A relation is used downstream only if its median absolute error is no greater than `1e-3 A` for the mean and `1e-2 A` for the standard deviation.

The `q=1` class is evaluated separately. Its absence does not authorize invented examples. If absent or sparse, q=1 Precision/Recall/F1 and continuous-zero diagnostics are `NOT_ESTIMABLE`, and P2 is frozen as a two-part `q>0` occurrence plus conditional-q regression model. If the mixed-current sufficiency conditions fail, phase status is `G1_INCONCLUSIVE` and G2-G5 are `NOT_RUN`.

### G1 criterion amendment recorded before downstream evaluation

The initial draft used a `1e-4 A` absolute current/current-density identity tolerance. The raw audit showed that this threshold is below the export quantization: the zero-intercept slopes were approximately 280.000 cm² and correlations exceeded 0.9999996, while the median absolute discrepancy was about 0.007 A. Before any P0/P1/P2 downstream inference was run, the criterion was replaced by the slope/correlation/relative-error rule above. This amendment is based only on raw-source resolution and cannot be changed after downstream results are available.

## Independent event and regime definitions

Let `q` be the 30 min zero-current fraction.

- `continuous_nonzero`: adjacent valid bins in one temporal segment with `q=0` in both bins.
- `nonzero_to_mixed`: `q=0` followed by `0<q<1`.
- `mixed_to_nonzero`: `0<q<1` followed by `q=0`.
- `continuous_zero`: adjacent valid bins with `q=1` in both bins.
- `mixed_current_window`: one valid bin with `0<q<1`.

Consecutive state runs define independent segments. Event counts use unique bins/transitions, not overlapping forecasting windows. Voltage-error summaries may use all matched target points but report the number of unique target bins and independent events separately.

## G1 oracle diagnostics

The frozen N2.8F M0 checkpoint is evaluated with:

- Ridge future conditions (formal baseline).
- True future `q` only (diagnostic and potentially inconsistent).
- True future physical current group: q, current mean/std/peak-to-peak, and corresponding current-density statistics (diagnostic only).
- All true future conditions (diagnostic upper bound only).

No diagnostic true-future input can become a deployable result or select a model.

## G2 future-condition schemes

- P0: exact repaired N2.8F multi-output Ridge (`alpha=1.0`), unchanged.
- P1: P0 followed by only verified projections: nonnegative current/statistical ranges, `q` clipped to `[0,1]`, nonnegative transition counts, and current-density statistics derived from current statistics with a Fold-Train-only fixed area. Exact `q=1` handling is applied only if such predictions occur and the raw audit validates compatibility.
- P2: if q=1 is not estimable, a fixed two-part model predicts `q>0` with Logistic Regression (`C=1`, balanced class weights, threshold 0.5) and conditional q with Ridge (`alpha=1`). Conditional nonzero-current mean and population standard deviation are forecast by Ridge and combined only through raw-validated mixture identities. Peak-to-peak remains independently forecast by Ridge. The result receives the same verified projection as P1.

All P2 choices above are frozen before frozen-SA-TimeXer downstream results are inspected. No model family, threshold, or regularization parameter will be changed after downstream evaluation.

## Causal construction

Training pseudo-futures use five chronological blocks. For every prediction block:

`max(training target completion time) <= min(prediction origin time)`.

Scalers and imputers are fit only on completed training windows. P0, P1, and P2 use identical legal origins. Any overlap or shape mismatch stops the run.

## G3 frozen-SA-TimeXer comparison

The N2.8F M0 seed-42 checkpoints are loaded without retraining. P0 must reproduce the recorded mean RMSE `24.538245 mV` within `0.001 mV`; otherwise G3 stops. Only future-condition tensors vary.

Gate B passes only if one candidate satisfies every registered condition:

- mean RMSE improves at least 5% versus P0;
- at least two folds improve;
- pooled P99 worsens no more than 5%;
- 9.5 h RMSE worsens no more than 10%;
- physical violation rate decreases.

Paired uncertainty uses a 19-origin moving-block bootstrap, 2000 repetitions, seed 42. Gate B is determined by the point estimates above; the bootstrap interval describes uncertainty and does not replace a failed point-estimate gate.

If both candidates fail, status is `G_STOP_NO_DOWNSTREAM_GAIN`; G4 and G5 are `NOT_RUN`.

## Conditional stages

- G4 runs only after Gate B. It retrains the unchanged SA-TimeXer architecture and training protocol for P0 and the unique winning scheme, seed 42 first. Three seeds `[42, 2024, 3407]` run only if seed 42 lowers mean RMSE by at least 5%, improves at least two folds, and does not worsen Fold3, P95, or P99 by more than 5%.
- G5 runs only after G3 passes and G4 completes its required control. It uses the frozen strong LSTM protocol and a small causal residual head trained from chronological Train-only OOF residuals. Validation residuals are prohibited.

Before any G5 head is fitted or scored, the qualitative LSTM promotion rule is
fixed numerically. L2 must improve mean RMSE by at least 2% versus both L0 and
L1, improve at least two folds versus each reference, and worsen neither Fold3
by more than 3% nor pooled P95/P99 by more than 5%. Seed 42 is the screen; only
a passing result triggers seeds 2024 and 3407 under the same matched-origin
protocol. The head is one standardized Ridge (`alpha=1`) per horizon using only
the nine predicted future-condition features. Its target is a strictly
chronological OOF LSTM residual; no Validation residual is used for fitting.
For the earliest 64-completed-window OOF block, the internal early-stopping
training subset requires at least 16 windows after its own 12 h
target-completion purge; the internal validation subset also requires 16.
These feasibility minima were recorded after a fail-closed pre-training check
and before any G5 prediction or score existed.

## Paper and claim policy

The paper package follows an empirical IMRaD structure. It describes dynamic multi-horizon voltage forecasting, not SOH or RUL prediction. Claims are synchronized to completed evidence. Literature citations, cross-stack generalization, SOH, RUL, and Test performance remain pending or unsupported unless separately verified later.

## Stop conditions

The phase stops immediately at the first applicable condition: unverifiable physical identities, insufficient mixed-current events, irreproducible P0, no Gate B winner, no retraining gain, or no LSTM-transfer gain. No new architecture family, PINN, Attention, Mamba, uncertainty gate, Test evaluation, or follow-on phase is authorized.
