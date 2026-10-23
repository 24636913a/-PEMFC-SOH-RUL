# Implementation Plan — Pending Technical Review

This document is a future execution checklist, not authorization to run it.
Technical Resolution V1 has closed the draft-level mathematical choices; the
immediate next action is final protocol review, not training.

1. Conduct final review of Candidate V1 and record acceptance or rejection
   without reading Test.
2. If accepted, create an executable protocol with a content hash, fixed source
   hashes, and static-test record; do not alter Candidate V1 definitions.
3. Build one cycle-level Development table from the frozen RCR cycle identities
   and PAA middle-50% P1S rule. Validate exactly 146 complete cycles, the
   frozen ten-cycle reference, the seven mandatory conditions, and exclusion of
   age features.
4. Join only the frozen six PAA/PAV pre-characterization anchors. Validate
   their 6/6 temporal qualification and the 3/6 median-source support label.
5. Execute B0, B1, B2, and the single primary model under both explicitly
   labeled Development evaluation tracks. No backbone or token comparison is
   permitted.
6. Execute S1 only as a prespecified robustness sensitivity. It cannot select
   a model or rewrite the all-six primary estimand.
7. Produce a Development-only decision with per-anchor outputs, uncertainty
   diagnostics, invalid-bootstrap counts, and code/source hashes.
8. Stop. If the model is qualified, prepare a separate test-opening protocol
   (`Phase_THV`); do not open Test automatically and do not begin RUL.

Forbidden implementation work includes model checkpoints before protocol
freeze, Test loading, hyperparameter searches outside the approved finite grid,
SOH interpolation, phase/DTW realignment, new operation-point searches, and
RUL training.
