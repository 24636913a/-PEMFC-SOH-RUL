# Phase F Final Summary

Protocol SHA256: `c8ee4be440b9828cad04b615ddd19c15781033fdb22bbadeb1df18a59cf74cbf`. Base: `d7190bc86a1507015207c99702de6153b1006ee6`. Seed42 only. Development exploratory, not independent Test.

PHASE_F_ASSESSMENT_INHERITED_SELECTION_BIAS; C1_D0_E0_REPRODUCIBILITY_UNRESOLVED; PHASE_F_TFT_ARCHITECTURE_EXPLORATION_ONLY; TIMESNET_FREEZE; BACKBONE_SEARCH_CLOSED.

|Model|Validation RMSE mV|Assessment RMSE mV|Val MAE mV|Assess MAE mV|Val R²|Assess R²|Params|Best epoch|Training s|
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|F0|9.741564|21.413261|5.945160|7.756553|0.997506|0.988014|323483|165|326.07|
|F1|9.445198|26.572484|4.679124|7.061643|0.997656|0.981542|332379|204|397.55|
|F2|5.746273|22.351705|4.362213|6.624243|0.999132|0.986940|332436|239|441.58|

- F1 minus F0: Validation -0.296365 mV; Assessment +5.159223 mV; Assessment relative reduction -24.094%; parameters +8896; total training-time ratio 1.219.
- F2 minus F1: Validation -3.698926 mV; Assessment -4.220779 mV; Assessment relative reduction +15.884%; parameters +57; total training-time ratio 1.111.
- F2 minus F0: Validation -3.995291 mV; Assessment +0.938444 mV; Assessment relative reduction -4.383%; parameters +8953; total training-time ratio 1.354.

|Model|12h RMSE mV|P95 mV|P99 mV|Max mV|Target-equal RMSE mV|Max-SSE row|Multiplicity|Max-row SSE share|Top5 SSE share|
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|F0|18.309446|17.763852|31.043233|317.225657|21.178687|2952|19|29.299%|84.255%|
|F1|30.598323|14.982781|25.090261|521.948291|25.723957|2504|24|41.300%|87.445%|
|F2|12.667647|14.223970|24.165475|310.136396|22.213520|2952|19|34.239%|90.744%|

`PHASE_F_F0_F1_F2_DEVELOPMENT_EXPLORATION_COMPLETED`

`SINGLE_SEED_DEVELOPMENT_STRUCTURAL_GAIN_OBSERVED`

No Test, SOH/RUL, new feature search, loss search, extra seeds, Optuna or architecture revision. Model science is limited to same-stage exploratory structure effects. Further research requires explicit human review; this execution stops here.

## Post-run interpretation and delivery clarification

This annotation does not alter metrics, frozen training code, protocol, checkpoints, or scientific decisions. `SINGLE_SEED_DEVELOPMENT_STRUCTURAL_GAIN_OBSERVED` applies narrowly to **F2 versus F1**, not to overall superiority over F0.

1. **F1 versus F0:** Validation RMSE improves by 0.296365 mV (3.042%), but Assessment worsens by 5.159223 mV (24.094%). A general dual-path benefit is **not established**.
2. **F2 versus F1:** Validation improves by 3.698926 mV (39.162%) and Assessment by 4.220779 mV (15.884%). This is preliminary, single-seed evidence of useful adaptive fusion relative to fixed fusion.
3. **F2 versus F0:** Validation improves by 3.995291 mV (41.013%), but Assessment worsens by 0.938444 mV (4.383%). Overall architecture superiority over the native baseline is **not established**. Lower F2 MAE does not override the frozen primary RMSE.
4. Directional agreement therefore holds only for F2 versus F1; the other two comparisons disagree between Validation and Assessment.
5. Errors are strongly concentrated: five target rows contribute 84.255% / 87.445% / 90.744% of Assessment SSE for F0/F1/F2. All targets remain in primary results. The diagnostic CSV field `independent_target_rows` counts **distinct target keys**, not statistically independent observations (447 on Assessment).
6. F2 Assessment gate is nonconstant: mean 0.411090, std 0.050180, range 0.329402–0.514853; P5/P50/P95 = 0.336757/0.405383/0.500224. No gate is within the registered near-0/1 extremes; 8.201% lie within ±0.01 of 0.5. No electrochemical mechanism is inferred.
7. F1 adds 8896 parameters (+2.750%) and costs 21.923% more total training time than F0. F2 adds 57 parameters to F1 (+0.017%) and costs 11.075% more than F1. F2 versus F0 adds 8953 parameters (+2.768%) and costs 35.425% more. Total-time ratios include different stopping epochs; consult per-epoch time and peak-memory ledgers before inferring compute efficiency.
8. Current evidence supports **human review of the conditional gate effect**, not automatic promotion, another search, or a claim of improved independent generalization. Assessment has inherited selection bias; all historical boundaries remain.

All three best checkpoints are small (~1.45–1.49 MB each), fit ordinary Git file-size rules, and are included in the Phase F delivery rather than silently omitted. File SHA256 and exact reload checks are recorded in `outputs/audits/F_ALL_CHECKPOINTS_FROZEN.json`. No retraining or Assessment rerun was used for this annotation.

Eight figures were exported as PNG (300 dpi) and PDF using the scientific-figure-making publication style. Font fallback warnings were nonfatal; rendered comparisons, convergence, scatter and gate figures were visually inspected. Metric scripts and frozen execution-code hashes remained unchanged after the formal run.
