# Phase CHM — Cycle-Aligned Sparse-Anchor Health Model

**Status:** `CHM_D_NEGATIVE_DEVELOPMENT_RESULT_FROZEN`

CHM-D has executed once under the frozen Development-only protocol. The
primary result is `DEVELOPMENT_HEALTH_MODEL_NOT_QUALIFIED`: Gate 3 failed
because the prespecified nonnegative affine P1S calibration (B2) outperformed
the neural latent-state CHM. Gate 5 was not evaluated because bootstrap was
not triggered after Gate 3 failed. This phase does not open Test, construct
validated SOH, or train RUL.

The proposed estimand remains a **Cycle-Aligned Latent Health State** (`H_c`),
not SOH and not RUL. The original
`PREREGISTERED_CHM_PROTOCOL_DRAFT.yaml`, Candidate V1, and Final V1 are
preserved as historical, non-executable records.

The protocol inherits, without modification:

- `TIMESNET_FREEZE` and `BACKBONE_SEARCH_CLOSED`;
- Phase RCR `RAW_CYCLE_STRUCTURE_QUALIFIED`: 146 frozen complete cycles;
- original dual-point RHA `CONDITION_COMMON_SUPPORT_INSUFFICIENT`;
- Phase P1 `P1_ANCHOR_ALIGNMENT_INCONCLUSIVE`;
- Phase PAA `P1S_PREANCHOR_COMPATIBILITY_EXPLORATORY_PASS`;
- Phase PAV `PAA_EXPLORATORY_PASS_TEMPORALLY_VERIFIED`.

The sole proposed dense observation is the frozen P1S operating-voltage
retention at `operation_point=221`, `I.S.set=221.200 A`, state token
`221||221.200`. It is a dense degradation-related observation, explicitly not
a dense SOH label.

The current executable authority is
`configs/PREREGISTERED_CHM_PROTOCOL_FINAL_V2.yaml` plus the narrowly scoped
`configs/CHM_FINAL_V2_EXECUTION_SUPPLEMENT_A.yaml`, governed by
`configs/CHM_PROTOCOL_AUTHORITY.yaml`. Final V2A freezes the 152-row HOU table,
the six sparse anchor targets, and the 18-row frozen anchor interval mapping.
CHM-D may consume only those tables, Final V2, and Supplement A after SHA256 validation; it
may not reconstruct raw HOU or parse a polarization CSV. The polarization term
remains a within-file local scatter proxy, never independent-repeat variance,
and anchor Gaussian perturbation remains disabled.

The execution record is retained without alteration in `final/FINAL_DECISION.yaml`,
the LOAO outputs, profiles, final latent trajectory, S1 trajectory, report, and
figures. The posthoc semantic qualification is in
`docs/POSTHOC_CHM_D_RESULT_CLARIFICATION.md` and
`final/POSTHOC_RESULT_QUALIFICATION.yaml`. It does not revise the original
execution-time result.

The qualified label for the historical `CHM_D_CAUSAL_ONLINE_DIAGNOSTIC.csv` is
`ANCHOR_PREFIX_FULL_HOU_RETROSPECTIVE_DIAGNOSTIC`: its sparse polarization
anchors are prefix-restricted, but each fit uses the full frozen Development
HOU sequence. It is not a strict causal-online or deployment evaluation.

No Test raw data, Test polarization, Test EIS, 2000 h/2300 h polarization,
after-Air-Air data, SOH interpolation, or RUL output is created by CHM-D.
