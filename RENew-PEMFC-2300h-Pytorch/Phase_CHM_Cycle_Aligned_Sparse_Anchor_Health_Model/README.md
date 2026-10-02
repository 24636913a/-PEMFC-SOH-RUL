# Phase CHM — Cycle-Aligned Sparse-Anchor Health Model Protocol

**Status:** `FINAL_PROTOCOL_FROZEN_AWAITING_EXECUTION_COMMAND`

This phase designs the next Development-only health-state experiment. It does
not execute it. The proposed estimand is a **Cycle-Aligned Latent Health
State** (`H_c`), not SOH and not RUL. The original
`PREREGISTERED_CHM_PROTOCOL_DRAFT.yaml` is preserved as a historical design
record; Candidate V1 is the technically resolved successor and remains
non-executable pending final protocol review.

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

Candidate V1 is preserved unchanged as the historical technical-resolution
record. Final V1 freezes the executable Development-only protocol and its
allowlisted source hashes. The HOU table now contains 146 complete-cycle HOU
(141 hash-frozen PAA rows plus new Development-only cycles 152–156) and six
frozen partial HOU. The final polarization term is a within-file local scatter
proxy, not independent-repeat variance; anchor Gaussian perturbation is
disabled. See `configs/PREREGISTERED_CHM_PROTOCOL_FINAL.yaml`,
`configs/CHM_SOURCE_ALLOWLIST.yaml`, and `outputs/FINAL_PREEXECUTION_AUDIT.json`.

This status is deliberately not an execution command. No CHM model, checkpoint,
LOAO prediction, bootstrap model fit, MAE/RMSE, SOH label, or RUL output has
been created. A subsequent explicit Phase CHM-D command is required before any
Development execution.

No Test raw data, Test polarization, Test EIS, 2000 h/2300 h polarization,
after-Air-Air data, checkpoint, model metric, SOH interpolation, or RUL output
is created in this phase.
