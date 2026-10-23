# CTA Reference Availability Erratum

CTA correctly selected T2, but its temporal scope was overbroad. The fixed denominator is `V_ref = median(P1S voltage of COMPLETE_001 through COMPLETE_010)`. The ten frozen reference voltages reproduce `0.71383023635 V`; the final required measurement ends at `171.82091 h`. Before this time, fixed-reference retention can only be reconstructed retrospectively.

PAA remains stricter than numerical denominator availability: all ten cycles are `REFERENCE_WARMUP_ONLY` and never formal prediction targets. The first automatically identified causal evaluation cycle is `COMPLETE_011` at `181.34366 h`.

## Corrected target

`POST_REFERENCE_RAW_P1S_RETENTION_AS_CAUSAL_HEALTH_TARGET`: `R_P1S(t) = V_P1S(t) / V_ref` only for non-reference evaluation cycles after `171.82091 h`.

Post-reference 24 h, 48 h and 96 h past-only retention slope prefix invariance is `True`, `True`, and `True`. First available times are 200.39166, 200.39166, and 200.39166 h. This establishes calculation-level causality only, not forecasting utility.

## Preserved CTA conclusions

`CTA_T2_DIRECTION_PRESERVED`; `CTA_T2_TEMPORAL_SCOPE_CORRECTED`.

Global B2: `RETROSPECTIVE_ONLY`. Full-sequence trends: `RETROSPECTIVE_ONLY`. Support: `STRATIFIED_MODELING_REQUIRED`.

## CTA ERRATUM DECISION

T2 direction: **PRESERVED**

Corrected target: **POST_REFERENCE_RAW_P1S_RETENTION_AS_CAUSAL_HEALTH_TARGET**

Reference warm-up: **FIRST_10_COMPLETE_P1S_CYCLES**

Fixed reference available after: **171.82091 h**

Reference cycles used as prediction targets: **NO**

Global B2: **RETROSPECTIVE_ONLY**

Full-sequence trends: **RETROSPECTIVE_ONLY**

Post-reference raw-retention slope: **CAUSAL_FEATURE_CANDIDATE_IF_PREFIX_REPLAY_PASS**

Support: **STRATIFIED_MODELING_REQUIRED**

Next phase: **CAUSAL_P1S_RETENTION_FORECASTING**
