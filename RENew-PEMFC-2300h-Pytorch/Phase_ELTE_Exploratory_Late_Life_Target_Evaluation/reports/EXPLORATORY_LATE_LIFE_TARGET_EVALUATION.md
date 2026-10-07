# Exploratory Late-Life Target Evaluation

## 1. Why this is exploratory rather than confirmatory

The historical AHC-TV one-shot remains aborted after Test execution started, with target unblinding false. This new phase is an explicitly authorized, one-shot exploratory opening; the 2000/2300 targets were protocol-exposed and are not a pristine independent Test. No post-target tuning was performed.

## 2. Frozen pre-target predictions

Predictions and provenance were SHA256-frozen before target I/O. Raw B2 coefficients were a=1.198811493832, b=-0.209036246312; the nominal-age coefficients were alpha=0.985326743626, beta=-0.000008395266. CN-P1S is secondary only.

## 3. Target extraction

The frozen Stage-T semantics were used: active-cell nanmean over U1–U5 and U7–U11, U6 excluded, j*=1.0 A/cm² within ±0.01 A/cm², then nanmedian across valid rows. Both nominal sources were opened consecutively after prediction freeze.

## 4. 2000 h result

Support is 0/3 (`OUT_OF_FROZEN_DEVELOPMENT_JOINT_SUPPORT`). Target Y=0.969496816; target voltage=0.677633006 V. Raw P1S B2: 0.969300949 (AE 0.137 mV); age: 0.968536211 (AE 0.671 mV); raw B0: 0.982921169 (AE 9.383 mV); CN Ridge: 0.969949279 (AE 0.316 mV). This is an out-of-frozen-support exploratory extrapolation.

## 5. 2300 h result

Support is 2/3 (`FROZEN_ALIGNMENT_COMPATIBLE`). Target Y=0.969998838; target voltage=0.677983895 V. Raw P1S B2: 0.970203049 (AE 0.143 mV); age: 0.966017631 (AE 2.783 mV); raw B0: 0.983673665 (AE 9.558 mV); CN Ridge: 0.970463229 (AE 0.325 mV). This is alignment-compatible exploratory evidence, not a pristine Test.

## 6. Raw P1S vs Age

E1/E2/E3 are True/True/True. Raw P1S B2 MAE=0.000200040 (0.140 mV); age MAE=0.002470906 (1.727 mV).

## 7. CN-P1S secondary comparison

CN Ridge MAE=0.000458427 (0.320 mV): `CN_P1S_EXPLORATORY_DEGRADATION`. Its prior Development result remains `CONDITION_NORMALIZED_P1S_NO_CLEAR_DEVELOPMENT_GAIN`; it was neither promoted nor retuned.

## 8. Error in normalized Y and mV

All absolute errors are saved in `outputs/ELTE_TARGET_RESULTS.csv`; mV errors are normalized-Y error × frozen BoL voltage × 1000.

## 9. Development-to-late-life trajectory

See `figures/Development_to_LateLife_Health_Trajectory.png`. Opened late-life targets are explicitly marked exploratory.

## 10. Scientific interpretation

A good 2000 h estimate cannot invalidate the support gate because it is only one out-of-support point. A good 2300 h estimate is stronger same-stack exploratory evidence but remains non-confirmatory.

## 11. Is the direction worth deeper study?

`STRONG_P1S_EXPLORATORY_LATE_LIFE_SIGNAL`; decision: `YES`. Any next study requires a separately authorized protocol and a new clean validation resource.

## 12. What evidence is still missing?

A clean independent Test, prospectively frozen success/failure criteria, and no target-exposed method refinement are still missing. This phase must not trigger automatic SOH, RUL, or model training.
