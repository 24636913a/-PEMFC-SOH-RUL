# Phase CHM — Cycle-Aligned Sparse-Anchor Health Model Protocol

**Status:** `PROTOCOL_DESIGN_ONLY` / `DRAFT_AWAITING_TECHNICAL_REVIEW`

This phase designs the next Development-only health-state experiment. It does
not execute it. The proposed estimand is a **Cycle-Aligned Latent Health
State** (`H_c`), not SOH and not RUL.

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

This draft is intentionally non-runnable: exact bootstrap block length, final
loss grid, baseline implementation, latent-step scale, anchor-noise estimator,
early stopping, and an optional condition-residual head require technical
review. See [open decisions](docs/OPEN_TECHNICAL_DECISIONS.md). No model may
be trained from this directory until a subsequent execution protocol freezes
those points.

No Test raw data, Test polarization, Test EIS, 2000 h/2300 h polarization,
after-Air-Air data, checkpoint, model metric, SOH interpolation, or RUL output
is created in this phase.
