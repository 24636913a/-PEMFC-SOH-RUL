# RENew-PEMFC-2300h-Pytorch

Research workspace for the 2300 h PEMFC durability dataset.

## Current completed phase

**Phase N1 — Data Foundation, Sampling Resolution and SOH Target Construction**

The phase establishes a leakage-controlled data foundation without training forecasting or health models. Its frozen decisions are:

- `OpHrs2 [h]` is the primary ageing coordinate.
- Active cells are U1–U5 and U7–U11; U6 is excluded.
- 0.5 h causal aggregation is primary; 1 h is retained for robustness.
- Train/Validation/Test use a chronological 60%/20%/20% blocked split.
- Polarization SOH uses sparse checkpoints at primary `j* = 1.0 A/cm²`, with `1.6 A/cm²` as sensitivity.
- The first Phase N2 physical task is a 24 h lookback and 12 h horizon, predicting multi-horizon voltage increments.
- The final Test interval remains selection-blind.
- No continuous SOH interpolation, strict RUL label, or neural-network training was created in Phase N1.

## Navigation

- Phase plan: `Phase_N1_Data_Foundation/PHASE_N1_PLAN.md`
- Full results: `Phase_N1_Data_Foundation/PHASE_N1_RESULTS.md`
- Reproducible scripts: `Phase_N1_Data_Foundation/scripts/`
- Frozen configuration: `Phase_N1_Data_Foundation/configs/phase_n1_config.yaml`
- Generated tables, audits, reports, and figures: `Phase_N1_Data_Foundation/outputs/`

Original data and the prior PHM2014 project are read-only dependencies and are not copied or modified here.
