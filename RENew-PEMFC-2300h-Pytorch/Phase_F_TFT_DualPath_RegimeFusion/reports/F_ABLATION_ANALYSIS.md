# Phase F Architecture Ablation

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

`SINGLE_SEED_DEVELOPMENT_STRUCTURAL_GAIN_OBSERVED`. Directional consistency must involve both Validation and Assessment; lowest Assessment alone does not prove an architecture. Added 8896 convolution parameters and 57 gate parameters, training-time/mean-epoch/memory ledgers are explicit confounders. A single seed does not establish stability or causal mechanism.

## Gate behavior

split,mean,std,min,max,P5,P50,P95,fraction_near_0_or_1,fraction_within_0p01_of_half
validation,0.4171304518129767,0.0490900640629343,0.3304469883441925,0.5090110898017883,0.3380867004394531,0.414761334657669,0.4982996940612793,0.0,0.0829268292682926
assessment,0.4110898699552294,0.0501801652654863,0.3294019401073455,0.51485276222229,0.3367574363946914,0.4053833037614822,0.5002242982387543,0.0,0.082010582010582

The context-conditioned gate distributions are recorded by predefined quartiles of the five masked/standardized historical regime means. Thresholds near 0/1 (0.05/0.95) and half (±0.01) are descriptive diagnostics only, never model-selection rules. Gate correlation is not electrochemical mechanism evidence.
