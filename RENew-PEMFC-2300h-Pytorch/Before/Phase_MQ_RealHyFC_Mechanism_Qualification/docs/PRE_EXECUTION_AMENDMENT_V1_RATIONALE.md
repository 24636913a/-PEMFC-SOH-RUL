# Phase MQ — pre-execution amendment V1

Status: **FROZEN_AFTER_PREREGISTRATION_BEFORE_FIRST_MQ_OUTCOME**.

The historical `configs/PREREGISTERED_MQ_PROTOCOL.yaml` remains immutable. Its
SHA256 is `1e934216843a8a72fe7d2ebe9975db2b573c0548605cee0496fdd445505e6c63`.
The effective protocol is that file plus only the explicit overrides in
`configs/PRE_EXECUTION_AMENDMENT_V1.yaml`. This amendment is to be committed
*before* creating an execution script or computing any MQ outcome.

## Timing and scope of the freeze audit

At remote GitHub `main` HEAD
`df46775590fa1d472b5b0c9d1a33d145171aa237`, the Phase MQ tree contained
only `README.md`, `SOURCE_MANIFEST.yaml`, `configs/`, and `docs/`.
No Phase MQ scripts, cycle results, gate results, residual tables, anchor
association, or terminal decision were committed. The original YAML text was
independently SHA256-checked against the committed checksum. This is a
**committed-repository audit**: unpushed work on the desktop is not observable
here. Before executing, desktop Codex must reject the run if its local working
tree or HEAD contradicts the freeze audit. No N1 dynamic row, polarization
measurement, EIS value, or Test result was read for this amendment.

The previous planning-layer accidental contact with a mixed Test-containing
polarization summary remains recorded in
`docs/PLANNING_LAYER_TEST_CONTACT_RECORD.md`. A later execution-specific
`phase_mq_execution_test_accessed=false` cannot erase it.

## Why these overrides precede outcomes

0. **Parser compatibility.** The immutable original YAML contains one
   unquoted flow-list entry, `active_cells: [U1 [V], ...]`, which PyYAML
   rejects because the cell labels themselves contain brackets. Hash the
   **unchanged original bytes** first. Require that the exact frozen line
   occurs once, quote only its ten labels in memory, and check the parsed
   list equals U1–U5 and U7–U11. Never write this normalized copy over
   the original or repair unrelated syntax automatically.
1. **End-of-cycle causality.** N1 computes `gap_after_flag` by shifting the
   following bin's `gap_before_flag` backward. Cycle identity, validity, and
   publication at the final bin's *confirmed completion* must not depend on
   this future-derived flag. An internal `gap_before_flag` is already known
   when that bin completes. A cycle beginning directly after a gap is excluded
   from primary degradation and anchor evidence, with a separate sensitivity
   record.
2. **Detection versus measurement.** A repeated load schedule can contain
   mixed 0.5 h transition bins. They remain explicit `TRANSITION` tokens in
   cycle detection. Only voltage and condition observations at the selected
   operation points require the original pure-bin contract. This prevents
   binning resolution from being mistaken for absence of a physical cycle.
3. **Frozen temporal calibration.** The first 300 chronological eligible
   EARLY bins, starting at the first eligible segment, define the only period
   search population. The 300 bins may span segments, but no lag or candidate
   cycle may cross a segment. The first complete chosen-period token sequence
   defines a provisional template. The first ten complete exact matches
   starting there constitute the calibration/reference cycles. The template,
   two positive points, voltage reference, and condition scaling become
   available only after both the 300th bin and tenth reference cycle have
   completed. Evaluation cycles must *start later* and the ten reference
   cycles cannot appear in the evaluation/generalization numerator. The
   choice of 300 is a project analysis design (`10 × 30`), not an
   outcome-optimized length.
4. **A0/A1/A2 interpretation.** A0 examines coarse set-current recurrence,
   A1 the operation-point/transition sequence, and A2 the availability of
   pure selected measurements. If A0 passes but A1/A2 fails *because mixed
   bins erase the required identity or pure measurement*, report
   `AGGREGATION_RESOLUTION_LIMITATION`; do not claim the physical schedule
   lacks repetition. An unrelated A1 exact-token failure is not automatically
   a resolution limitation. Fixed 19-bin tiling remains sensitivity only.
5. **Joint support versus phase benefit.** Gate C1 checks seven frozen
   operating variables in their raw units across EARLY evaluation, MIDDLE,
   and LATE; at least five different cycles per stratum and selected point
   must lie *simultaneously* inside all seven overlapping ranges. A variable
   constant in reference cycles may be dropped from a standardized numerical
   distance, but remains in raw support. A systematic later shift with no
   reference support fails C1.
6. **C2 paired-query semantics.** When the same selected point occurs in
   multiple template phases, a given LATE observation may be the common
   query for one same-phase and one different-phase comparison; otherwise
   the requested within-query comparison would be undefined. The EARLY
   counterpart observations in these two pools are disjoint, no observation
   is paired with itself, and the control counterpart must come from a
   different phase. Matching is deterministic, without replacement within
   each pool, using conditions alone. If the point occurs at only one
   template phase, a different-phase control does not exist. C1 can still
   pass, but no phase-alignment benefit may be claimed.
7. **Cycle and anchor units.** Gate D uses only completed evaluation cycles,
   two fixed selected points, the fixed ten-cycle reference, and the
   preregistered two/four-cycle block bootstrap. Gate E aligns only three
   completed *evaluation* cycles before a verified characterization start.
   BoL is a nominal polarization normalization reference without a
   pre-characterization durability cycle. None of this constructs dense SOH,
   proves irreversible degradation, or starts RUL work.

## Required precedence and terminal boundary

First verify the original protocol hash and this amendment's hash. Where this
amendment names an override, use its value; every untouched original field
remains binding. Commit this amendment, effective manifest, and timing audit
as one identifiable commit. Only a later distinct commit may add scripts or
outputs from Phase MQ execution. Stop at the first applicable A/B/C1
barrier; Gate D/E must remain `NOT_RUN` when an upstream barrier fails.
No model phase is automatic even if every mechanism gate passes.
