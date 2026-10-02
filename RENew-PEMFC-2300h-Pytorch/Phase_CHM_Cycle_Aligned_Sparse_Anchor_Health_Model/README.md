# Phase CHM — Cycle-Aligned Sparse-Anchor Health Model Protocol

**Status:** `TECHNICALLY_RESOLVED_AWAITING_FINAL_FREEZE`

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

Candidate V1 resolves the HOU anchor functional, BoL/initial-state distinction,
transition parameterization, feature scaler, one-hidden-layer model, loss grid,
baselines, LOAO, online diagnostic, uncertainty, and Test wording. See
[technical resolution](docs/TECHNICAL_REVIEW_RESOLUTION.md). It is still not
execution-authorized: no model may be trained from this directory until final
protocol review explicitly freezes the candidate and its source/code hashes.

No Test raw data, Test polarization, Test EIS, 2000 h/2300 h polarization,
after-Air-Air data, checkpoint, model metric, SOH interpolation, or RUL output
is created in this phase.
