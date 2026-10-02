# Phase CHM — Cycle-Aligned Sparse-Anchor Health Model Protocol

**Status:** `FINAL_V2_FROZEN_AWAITING_CHM_D_EXECUTION`

This phase designs the next Development-only health-state experiment. It does
not execute it. The proposed estimand is a **Cycle-Aligned Latent Health
State** (`H_c`), not SOH and not RUL. The original
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
`configs/PREREGISTERED_CHM_PROTOCOL_FINAL_V2.yaml`, governed by
`configs/CHM_PROTOCOL_AUTHORITY.yaml`. Final V2 freezes the 152-row HOU table,
the six sparse anchor targets, and the 18-row frozen anchor interval mapping.
CHM-D may consume only those tables and Final V2 after SHA256 validation; it
may not reconstruct raw HOU or parse a polarization CSV. The polarization term
remains a within-file local scatter proxy, never independent-repeat variance,
and anchor Gaussian perturbation remains disabled.

This status is deliberately not an execution command. No CHM model, checkpoint,
LOAO prediction, bootstrap model fit, MAE/RMSE, SOH label, or RUL output has
been created. Only a subsequent explicit `Phase CHM-D — Development Execution`
command can initiate the frozen Development experiment.

No Test raw data, Test polarization, Test EIS, 2000 h/2300 h polarization,
after-Air-Air data, checkpoint, model metric, SOH interpolation, or RUL output
is created in this phase.
