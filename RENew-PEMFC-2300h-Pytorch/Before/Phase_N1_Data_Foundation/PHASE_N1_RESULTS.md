# Phase N1 — Data Foundation, Sampling Resolution and SOH Target Construction

中文：阶段 N1——数据基础、采样分辨率与 SOH 目标构建

## Phase conclusion

**Phase N1: PASS**

The dataset foundation is sufficiently complete to enter **Phase N2 — Baseline Forecasting and Modern Temporal Model Screening**, subject to preserving the frozen Test firewall and the task definitions in this report. No neural network, temporal model, hyperparameter search, continuous SOH interpolation, EOL label, or RUL model was trained in Phase N1.

| Decision item | Frozen Phase N1 result |
| --- | --- |
| Primary sampling resolution | **0.5 h causal aggregation** |
| Robustness resolution | **1 h causal aggregation** |
| Active cells | **U1–U5 and U7–U11**; U6 excluded as non-physical |
| Primary SOH anchor | **j\* = 1.0 A/cm²** |
| Secondary SOH sensitivity anchor | **j\* = 1.6 A/cm²** |
| BoL comparability | **CONDITIONAL overall**; nominal 10-cell per-cell polarization is directly comparable, while the separate BoL sensitivity/EIS 8-cell stack requires configuration-aware treatment |
| Primary first-round lookback | **24 h** |
| Lookback sensitivities | **12 h and 48 h** |
| Primary first-round horizon | **12 h** |
| Horizon sensitivities | **4 h, 8 h, and 24 h** |
| Primary future target | **Multi-horizon voltage increment, ΔV(t+h)=V(t+h)−V(t)** |
| Strict RUL ground truth | **NO** |
| Phase N2 gate | **GO** |

## 1. Scope, safeguards, and provenance

All new scripts, outputs, logs, figures, tables, configurations, and reports were written under the new project directory only. The old PHM2014 project was treated as read-only. The original 2300 h data, associated TXT/CSV files, and the publication PDF were also treated as read-only.

The primary ageing coordinate is `OpHrs2 [h]`. `DateTime` is retained only for gap detection, startup/shutdown detection, characterization interruptions, and wall-clock continuity auditing. The anomalous `OpHrs [h]` field is not used as the lifetime axis.

All selection decisions in this phase use Train + Validation only. Test is used only for range, count, structure, and future-anchor inventory checks. Test values do not determine sampling resolution, j\*, SOH normalization, features, lookback, horizon, preprocessing, model, loss, or hyperparameters.

## 2. Raw durability data and active-cell audit

The durability stream comprises 17 Loadcycle CSV files and 8,604,654 raw rows, sampled at approximately 1 s. The observed `OpHrs2` span is **72.788690–2256.800000 h**.

The cell-channel audit was calculated on 6,708,040 Train + Validation rows before downstream cell statistics were constructed. U6 is decisively inconsistent with a physical cell-voltage channel:

| U6 statistic | Value |
| --- | ---: |
| Median | −0.008728 V |
| Mean | −0.009072 V |
| Standard deviation | 0.005585 V |
| p1 / p99 | −0.023865 / −0.000427 V |
| Negative ratio | 99.9941% |
| Flatline ratio | 56.7240% |

The mean of the ten channels excluding U6 agrees with `U.S.AveCell [V]` with a mean absolute error of **0.000061 V**. Including U6 increases that error to **0.053511 V**. Therefore, the frozen active-cell mask is:

`U1, U2, U3, U4, U5, U7, U8, U9, U10, U11`

All derived cell-voltage mean, median, standard deviation, range, minimum, and maximum features use only these ten active channels. `U.S.AveCell [V]` is retained separately as the exported average-cell voltage and is not used to override the recomputed ten-cell statistics.

## 3. Causal feature tables

Two independently aggregated causal tables were created:

| Resolution | Rows | Columns | Train bins | Validation bins | Test bins | Boundary-excluded bins |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.5 h | 3,843 | 214 | 2,248 | 755 | 836 | 4 |
| 1 h | 1,927 | 214 | 1,127 | 378 | 418 | 4 |

Each physical bin uses only observations whose `OpHrs2` values lie inside that bin. No centered smoother, centered rolling statistic, backward fill, future fill, cross-gap interpolation, or later-bin imputation is used. Incomplete bins remain incomplete and are described by sample count, expected count, coverage ratio, zero-current fraction, startup/shutdown flags, gap-before/gap-after flags, operation-point count, and operation-point transition count.

The feature set includes active-cell voltage state, actual load and current-density summaries, separate actual/set operating-condition summaries, flow and stoichiometry summaries, event indicators, timing metadata, and diagnostic-only fields. The feature dictionary records raw source, unit, aggregation, causal availability, online-input eligibility, and diagnostic-only status. The availability table separately labels past-known, origin-known, future-unknown, future-known-by-protocol, diagnostic-only, and offline-only variables.

## 4. Frozen chronological split

The blocked split is calculated from the observed `OpHrs2` span, using `min + 60% × span` and `min + 80% × span`:

| Split | Frozen `OpHrs2` interval | DateTime range | 30 min bins (valid) | 1 h bins (valid) | Characterization checkpoints |
| --- | --- | --- | ---: | ---: | --- |
| Train | **[72.788690, 1383.195476) h** | 2025-01-27 12:24:06.005 to 2025-05-22 17:21:20.005 | 2,248 (2,236) | 1,127 (1,116) | BoL, 300, 600, 850, 1150 h |
| Validation | **[1383.195476, 1819.997738) h** | 2025-05-22 17:51:21.005 to 2025-06-29 13:04:55.005 | 755 (750) | 378 (374) | 1500, 1750 h |
| Test | **[1819.997738, 2256.800000] h** | 2025-06-29 13:58:41.005 to 2025-07-26 11:48:14.005 | 836 (832) | 418 (415) | 2000, 2300 h |

A bin is assigned only when its complete physical interval is contained within one split. Bins that straddle a split cut are marked `boundary_excluded`, not reassigned.

## 5. Polarization configuration compatibility

### 5.1 Evidence

The durability test used a generic HyFaB **10-cell**, **280 cm² active-area** stack with the project MEA. The nominal polarization CSV metadata report 10 cells, and Train + Validation files report 280 cm². The publication describes the nominal RealHyFC polarization protocol as 16 load points over 0–1.8 A/cm², measured in descending and ascending directions with defined dwell times and conditioning.

The publication also states that the separate BoL EIS/sensitivity campaign used an **8-cell** stack because of test-bench limitations, albeit with MEAs from the same batch. The raw sensitivity exports report 10 cells, which conflicts with the publication’s physical-stack description. This conflict is preserved as an audit finding rather than silently reconciled.

The 2000 h and nominal 2300 h Test polarization exports omit the active-area metadata field. This does not affect Train + Validation selection and is retained as a Test-side metadata limitation.

### 5.2 Decision

**BoL comparability is CONDITIONAL overall.**

- The nominal BoL polarization data and durability polarization checkpoints are directly comparable on a per-cell, current-density basis because they belong to the nominal 10-cell polarization series and use a common protocol/MEA context.
- The separate BoL sensitivity/EIS 8-cell stack must not be merged with the 10-cell durability stack as unqualified absolute stack-voltage data. It is used only for configuration-aware sensitivity interpretation.
- Consequently, the nominal BoL polarization checkpoint is a valid `SOH_reference = 1` for the selected per-cell polarization anchor. The broad statement “all BoL data are directly comparable to all durability data” is false.

## 6. Sparse polarization SOH anchors

The j\* candidate set was evaluated at 0.8, 1.0, 1.2, 1.4, and 1.6 A/cm². Selection used only the seven non-after-Air-Air Train + Validation checkpoints. The 2000 h and 2300 h Test checkpoints were excluded from coverage, correlation, monotonicity, ranking, formula selection, and weighting.

Selection combined coverage, Train/Validation checkpoint support, repeated-measurement variability, ageing Spearman correlation, monotonicity, non-increasing fraction, degradation-to-repeat-noise ratio, Air-Air-start sensitivity, and distance above the 500 mV test lower limit. It did not select the candidate with maximum Spearman correlation alone.

| Candidate | Rank | Coverage | Spearman ρ | Monotonicity | Non-increasing fraction | Repeated std | Margin above 500 mV | Air-Air shift at 1500 h |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **1.0 A/cm²** | **1** | 1.000 | −0.964 | 0.960 | 0.833 | 0.000554 V | 0.1792 V | 0.003916 V |
| **1.6 A/cm²** | **2** | 1.000 | −1.000 | 1.000 | 1.000 | 0.001366 V | 0.1076 V | 0.006455 V |

The frozen recommendation is **primary j\* = 1.0 A/cm²**, with **secondary j\* = 1.6 A/cm²** for sensitivity analysis. The primary candidate sacrifices a small amount of ideal monotonicity in exchange for substantially lower repeat variability, a larger lower-limit margin, and lower Air-Air sensitivity.

The SOH definition is:

`SOH_V(t_i) = V(j*, t_i) / V(j*, t_ref)`

with `j* = 1.0 A/cm²` and `t_ref = BoL` from the nominal 10-cell per-cell polarization series. The reference voltage is **0.698953 V**.

The requested Candidate B,

`1 − [V_ref − V(t_i)] / V_ref`,

is algebraically identical to Candidate A. It is retained as a numerical cross-check, not presented as an independent health label.

Only sparse diagnostic anchors are created. No 0.5 h or 1 h interpolation is treated as ground truth. No unconfirmed EOL voltage, `SOH_EOS = 96%`, or `RUL = 2300 − t` target is used. The Test anchors remain labeled as future validation anchors and do not retroactively affect the design.

## 7. Sampling-resolution comparison

The 0.5 h and 1 h datasets were compared on the same physical task. For example, a past-24 h to future-24 h task uses L=48/H=48 at 0.5 h and L=24/H=24 at 1 h.

All decision metrics below use Train + Validation only:

| Metric | 0.5 h | 1 h | Interpretation |
| --- | ---: | ---: | --- |
| Total bins | 3,003 | 1,505 | 0.5 h retains twice the temporal resolution |
| Valid-bin fraction | 99.434% | 99.003% | No coverage penalty for 0.5 h |
| Low-coverage-bin fraction | 0.400% | 0.665% | 0.5 h is slightly better |
| Gap-affected-bin fraction | 0.633% | 1.196% | 0.5 h is better |
| Legal 24 h → 24 h windows | 1,680 | 844 | 0.5 h provides materially more legal sequences |
| Within-bin voltage temporal std | 0.040839 V | 0.041474 V | Comparable |
| Matched-condition local noise | 0.001876 V | 0.001814 V | Comparable |
| Matched voltage-age Spearman ρ | −0.777 | −0.805 | Both show a clear ageing trend |
| Matched degradation amplitude | 0.007778 V | 0.007533 V | Comparable |
| Noise/degradation ratio | 0.2411 | 0.2408 | Essentially identical |
| Current-profile preservation score | 0.4518 | 0.3417 | 0.5 h preserves dynamics better |
| Points per empirical duty cycle | 19.0 | 9.5 | 0.5 h resolves the load cycle better |

Every pre-frozen decision rule passes for 0.5 h: more legal windows, no material coverage deterioration, no material trendability deterioration, no material noise/degradation deterioration, and improved dynamic resolution. Therefore, **0.5 h is the primary sampling resolution** and **1 h remains a required robustness/sensitivity comparison**.

## 8. Load cycle, ACF, and temporal dependence

Train-only current/current-density autocorrelation and spectral analysis identify a fundamental period of approximately **9.5 h on the `OpHrs2` operating-age axis**. The paper reports a roughly 12 h wall-clock duty cycle that includes three short stops and one long stop. Because `OpHrs2` counts time under load, the shorter empirical operating-age period is physically consistent and is not replaced by the wall-clock estimate.

At the 9.5 h lag, Train-only ACF values are:

| Signal | ACF at 9.5 h |
| --- | ---: |
| Current density | 0.979 |
| Current | 0.979 |
| Operation-point mode | 0.928 |
| Operation-point exact match | 0.948 |
| Active-cell voltage mean | 0.943 |
| Matched-condition voltage | 0.843 |
| Cell-voltage standard deviation | 0.981 |

These values provide direct evidence of temporal dependence and repeatable duty-cycle structure. L>1 is therefore supported empirically, not merely theoretically. ACF is treated as time-scale evidence; it is not used alone to claim an optimal lookback.

## 9. Lookback/horizon feasibility

The fail-closed window audit evaluates both resolutions, lookbacks of 12/24/48 h, horizons of 4/8/12/24 h, and an additional 48 h diagnostic horizon. Every accepted sequence must lie entirely in one split, remain strictly time ordered, avoid characterization gaps and unacceptable missing regions, satisfy coverage requirements, and avoid split boundaries.

For the recommended 0.5 h resolution and 24 h lookback, legal sequence counts are:

| Horizon | Train | Validation | Test, count-only |
| --- | ---: | ---: | ---: |
| 4 h | 1,686 | 530 | 612 |
| 8 h | 1,606 | 498 | 580 |
| **12 h** | **1,526** | **466** | **548** |
| 24 h | 1,310 | 370 | 474 |

For 1 h and the same physical 24 h lookback:

| Horizon | Train | Validation | Test, count-only |
| --- | ---: | ---: | ---: |
| 4 h | 846 | 266 | 307 |
| 8 h | 806 | 250 | 291 |
| **12 h** | **766** | **234** | **275** |
| 24 h | 658 | 186 | 238 |

The first-round recommendation is **24 h lookback**, with **12 h and 48 h sensitivity lookbacks**. Twenty-four hours contains about 2.53 empirical operating-age duty cycles, captures repeated operating-point context, and retains ample legal Train and Validation windows. This is a physically justified candidate, not a claim of optimality.

The first-round primary forecast horizon is **12 h**, approximately 1.26 empirical operating-age duty cycles. It is long enough to test cycle-scale forecasting while preserving 1,526 Train and 466 Validation windows at the primary resolution. Horizons of **4, 8, and 24 h** remain required sensitivity tasks; 48 h remains diagnostic rather than primary.

## 10. Future forecasting task definitions

### Primary task: degradation-increment forecasting

Input: a legal continuous history over 12/24/48 h, with 24 h primary.

Target:

`ΔV(t+h) = V(t+h) − V(t),  h = 1, …, H`

Reconstruction:

`V_hat(t+h) = V(t) + ΔV_hat(t+h)`

This residual/increment target is primary because the per-cell voltage level is dominated by the repeatable duty cycle and operating condition, while degradation is a much smaller long-term component. Absolute-voltage forecasting remains a secondary benchmark and reconstruction metric, not the preferred learning target.

### Secondary task: health-trajectory forecasting

Direct SOH-trajectory forecasting is deferred until a defensible continuous or semi-supervised target exists. The sparse polarization anchors may be used for checkpoint-level validation or weak supervision, but they must not be linearly interpolated and called ground truth.

### Future covariate scenarios

- **Scenario A — Future-unknown forecasting:** future actual current, temperature, pressure, RH, flow, and stoichiometry are unavailable. This is the strict default.
- **Scenario B — Known-duty-cycle forecasting:** only set points that are demonstrably predefined by the experimental protocol may be supplied. This scenario must be evaluated separately and clearly labeled.

Actual future operating variables must never leak into the strict task. Train-only fitting is required for every scaler, transformation, condition reference, or learned preprocessing step.

## 11. Phase N2 candidate model register

No model was trained in Phase N1. The Phase N2 screening register is:

- Analytical/statistical: persistence, linear trend, operating-point seasonal naive, Ridge lag model, and state-space/Kalman-style baseline.
- Classical temporal: LSTM, GRU, and TCN.
- Modern temporal: iTransformer, TimeMixer, and PatchTST.

The list is a screening plan, not a ranking. Analytical baselines must be established before modern models are interpreted.

## 12. Explicit answers to Q1–Q10

### Q1 — Which sampling resolution should be primary?

**0.5 h.** It yields materially more legal same-physics windows, slightly better coverage/gap statistics, nearly identical noise/degradation ratio, comparable matched-condition ageing visibility, and better duty-cycle/current-profile retention. Keep **1 h** as a robustness and sensitivity dataset.

### Q2 — Does the new dataset support L>1?

**YES.** Train-only ACF shows strong dependence at the empirical 9.5 h duty-cycle period, the load cycle is repeatable, degradation remains visible under matched conditions, coverage is high, and legal sequence counts remain substantial for all proposed physical histories.

### Q3 — What physical lookback is most reasonable initially?

Use **24 h as the primary first-round candidate**, with **12 h and 48 h** as sensitivity candidates. Twenty-four hours spans about 2.53 operating-age duty cycles while retaining ample legal windows. No claim of optimality is made.

### Q4 — Which polarization j\* is the primary health anchor?

Use **j\* = 1.0 A/cm²** as primary and **1.6 A/cm²** as secondary sensitivity. The decision uses Train + Validation checkpoints only.

### Q5 — Are BoL and the durability stack directly comparable?

**CONDITIONAL.** The nominal BoL and durability polarization series are directly comparable on a per-cell/current-density basis. The separate BoL 8-cell sensitivity/EIS stack is not unconditionally interchangeable with the 10-cell durability stack and must remain configuration-aware.

### Q6 — How should current SOH be defined?

At sparse characterization checkpoints only:

`SOH_V(t_i) = V(1.0 A/cm², t_i) / V(1.0 A/cm², BoL)`

with nominal 10-cell per-cell polarization data and `V_ref = 0.698953 V`. No dense interpolation and no unconfirmed EOL normalization are allowed.

### Q7 — Is strict RUL ground truth available?

**NO.** The experiment ended after the planned durability campaign/characterization sequence; it does not establish a verified failure time or EOL threshold. `RUL = 2300 − t` is therefore prohibited.

### Q8 — What should the next forecasting target be?

Primary: **multi-horizon residual/degradation increment ΔV**. Secondary: reconstructed/absolute voltage for benchmarking. Sparse SOH is for anchor-level validation; continuous SOH forecasting remains deferred.

### Q9 — What are the final split boundaries?

- Train: **[72.788690, 1383.195476) h**
- Validation: **[1383.195476, 1819.997738) h**
- Test: **[1819.997738, 2256.800000] h**

All boundaries are on `OpHrs2 [h]`; the final 20% Test remains frozen.

### Q10 — Can the project enter Phase N2?

**GO.** The active-cell mask, causal feature tables, split, configuration-aware SOH anchor, sparse anchor table, sampling decision, temporal-scale evidence, and fail-closed window feasibility audit are complete. Phase N2 must preserve the Test firewall, use 0.5 h primary/1 h sensitivity, start with 24 h → 12 h as the primary physical task, benchmark simple baselines first, and avoid constructing synthetic dense SOH or RUL labels.

## 13. Key limitations carried forward

1. SOH remains sparse and characterization-based; it is not a continuous ground-truth series.
2. Strict RUL ground truth is absent because no verified EOL/failure threshold is observed.
3. The BoL sensitivity/EIS physical stack description conflicts with raw exported cell-count metadata; these data remain configuration-aware only.
4. Two Test polarization exports omit active-area metadata. They remain future validation anchors and were not needed for selection.
5. The empirical 9.5 h period is on operating age (`OpHrs2`), whereas the approximately 12 h published period is wall-clock including stops.

## 14. Core output index

### Planning and configuration

- `PHASE_N1_PLAN.md`
- `configs/phase_n1_config.yaml`

### Active cells and provenance

- `outputs/audits/active_cell_channel_audit.csv`
- `outputs/audits/ACTIVE_CELL_MASK.md`
- `outputs/audits/loadcycle_source_timeline.csv`
- `outputs/audits/source_file_manifest.csv`
- `outputs/audits/phase_n1_integrity_audit.csv`
- `outputs/audits/phase_n1_run_manifest.json`

### Causal datasets and feature metadata

- `outputs/causal_feature_tables/causal_features_30min.csv`
- `outputs/causal_feature_tables/causal_features_1h.csv`
- `outputs/causal_feature_tables/causal_feature_dictionary.csv`
- `outputs/causal_feature_tables/causal_feature_availability.csv`

### Split

- `outputs/split_manifest/split_manifest.csv`
- `outputs/split_manifest/FROZEN_DATA_SPLIT.md`

### Sampling and temporal scale

- `outputs/sampling_comparison/sampling_resolution_comparison.csv`
- `outputs/sampling_comparison/SAMPLING_RESOLUTION_DECISION.md`
- `outputs/temporal_scale_analysis/load_cycle_periodicity_analysis.csv`
- `outputs/temporal_scale_analysis/load_cycle_periodogram.csv`
- `outputs/temporal_scale_analysis/temporal_dependency_analysis.csv`
- `outputs/temporal_scale_analysis/window_feasibility_summary.csv`

### SOH anchors and compatibility

- `outputs/soh_anchor_analysis/polarization_configuration_compatibility.md`
- `outputs/soh_anchor_analysis/polarization_configuration_summary.csv`
- `outputs/soh_anchor_analysis/sensitivity_stack_configuration_summary.csv`
- `outputs/soh_anchor_analysis/polarization_soh_anchor_candidates.csv`
- `outputs/soh_anchor_analysis/SOH_ANCHOR_SELECTION.md`
- `outputs/soh_anchor_analysis/soh_reference_anchors.csv`

### Scripts

- `scripts/phase_n1_build_foundation.py`
- `scripts/phase_n1_analyze_polarization.py`
- `scripts/phase_n1_analyze_temporal_sampling.py`
- `scripts/phase_n1_window_audit.py`

### Required figures

All required sampling, durability, polarization/SOH, periodicity, ACF, and window-count figures are in `outputs/figures/`.

---

Phase N1 ends here. No Phase N2 model experiment was executed.
