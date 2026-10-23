# Phase F Formal Training Report

Protocol SHA256: `c8ee4be440b9828cad04b615ddd19c15781033fdb22bbadeb1df18a59cf74cbf`. Base: `d7190bc86a1507015207c99702de6153b1006ee6`. Seed42 only. Development exploratory, not independent Test.

PHASE_F_ASSESSMENT_INHERITED_SELECTION_BIAS; C1_D0_E0_REPRODUCIBILITY_UNRESOLVED; PHASE_F_TFT_ARCHITECTURE_EXPLORATION_ONLY; TIMESNET_FREEZE; BACKBONE_SEARCH_CLOSED.

## Frozen protocol and input

Exactly 3007 prefix rows were read, then 3005 Development rows retained. Train [0,2103), Validation [2103,2404), Assessment [2404,3005). Legal windows: 1380/205/378. Metadata eligibility alone was counted for Assessment before training; its model arrays/targets were constructed only after all three best checkpoints froze.

Phase E/C window, Top10, CommonScaler, per-window EWMA and zero-mask functions were imported without modifying historical files. Shared unfiltered-Train scaler, target scale, fixed inputs and first shuffled batch were hashed. Native parameters/buffers and F1/F2 branch initialization matched exactly. Seed42 was reset independently for each group. Training dropout random-number paths need not match.

AdamW/MSE, batch32, max450, patience40, clip1, frozen LR/weight-decay. Validation standardized RMSE alone selected epoch using strict improvement >1e-12; Assessment never selected epochs. Train MSE is the mean of batch losses, matching Phase E. Each best checkpoint includes identities, scale, initialization and fixed Validation predictions and passed exact reload parity. All 3 froze before Assessment. See per-epoch CSV, run JSON, environment and source-code manifest.

|Model|Validation RMSE mV|Assessment RMSE mV|Val MAE mV|Assess MAE mV|Val R²|Assess R²|Params|Best epoch|Training s|
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|F0|9.741564|21.413261|5.945160|7.756553|0.997506|0.988014|323483|165|326.07|
|F1|9.445198|26.572484|4.679124|7.061643|0.997656|0.981542|332379|204|397.55|
|F2|5.746273|22.351705|4.362213|6.624243|0.999132|0.986940|332436|239|441.58|

## Technical verification

Independent scalar math.fsum recomputation from round-trip CSVs agrees within 1e-10. All keys, true ΔV, horizon sets, target rows and Development boundary passed. 9072 queries/group on Assessment, 4920/group on Validation. Checkpoints are retained locally and subject to repository file-size rules at delivery.

## Historical E2 comparison

F0 Validation difference from historical E2 6.064596 mV: +3.676968 mV; Assessment difference from 7.785821 mV: +13.627440 mV. Current protocol/input/checkpoint identities are self-consistent; no unverified explanation of cross-stage differences is asserted.
