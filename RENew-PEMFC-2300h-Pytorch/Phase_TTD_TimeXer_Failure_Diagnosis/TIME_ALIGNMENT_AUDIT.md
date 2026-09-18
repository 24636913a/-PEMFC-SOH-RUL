# Time alignment audit

The high-loss audit includes all Fold 1 top-D unique targets and the 50 largest Fold 3 continuous-nonzero pointwise D records. Every audited row satisfies `target_row = origin_row + horizon_step`; the future-condition row and voltage-label row equal that same target row; every origin precedes its target.

All 70 audited records have `alignment_status=PASS`; 12 h records have exactly 12.0 h ageing distance. No one-bin forward/backward label offset, duplicate timestamp substitution, or cross-segment index misuse was found. Mechanism D is therefore not supported.
