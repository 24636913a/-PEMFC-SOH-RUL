# P1S-B2 Causal Target Audit

## Calibration time availability

The all-six-anchor mapping used by PHI is explicitly classified as
`RETROSPECTIVE_REFERENCE_ONLY`.  Every one of the 141 PHI events occurs before
the final 1750 h anchor, so the global B2 function was not causally available
at any event.  Availability is recorded only from nominal checkpoint timing
and is labelled `NOMINAL_CHECKPOINT_AVAILABILITY_ONLY`; no unsupported claim is
made about the exact real-time availability timestamp of a characterization
result.

For expanding-fold validation, the frozen CSTL functions are chronology-safe:
fold 1 uses 300/600/850, and folds 2–3 use 300/600/850/1150.  No frozen fold
uses a future calibration anchor.  Across the validation events, mean absolute
global-versus-causal B2 revision is 0.000342503, P95 is
0.000419223, and the maximum is 0.000427403.
This measures label rewriting from future calibration information, not accuracy.

## Prefix trend replay

All PHI full-sequence trend methods revise at least some endpoint or history
when later P1S cycles are supplied: SG endpoint max revision =
0.00142568, isotonic =
0.00180682, and monotonic PCHIP =
0.00328968.  Thus full-sequence SG,
isotonic and PCHIP values are `RETROSPECTIVE_TREND_ONLY`.

The past-only raw-HI slope replay matches PHI exactly within 1e-12: 24 h =
True, 48 h = True, 96 h = True.  This establishes
calculation causality only; it does not establish predictive value.

The 35 upward raw-HI transitions remain empirical observations.  Zero upward
changes in the isotonic trend are `CONSTRAINT-INDUCED_MONOTONICITY`, not proof
of irreversible PEMFC degradation.

## Support-domain stratification

Support reproduction yields 82 `IN_SUPPORT` and 59 `OUT_OF_SUPPORT` events.
The most frequent frozen support violations are RH_violation, temperature_violation, anode_stoich_violation.
The summary is descriptive only: without per-cycle polarization ground truth it
cannot show that either group is more or less accurate.  Since out-of-support
events constitute a substantial portion of the series and occur over formal
time regions, future work must retain support as a stratum / quality variable
rather than delete those events.

## Cross-phase provenance clarification

CSHSE was planned as current same-cycle health estimation, but its final data
availability audit found `same_cycle_history_event_count = 0` and
`FAIL__FROZEN_P1S_MEASUREMENT_PRECEDES_TARGET_CYCLE_DYNAMIC_HISTORY`.
Its only defensible interpretation is
`DESCRIPTIVE_GUARD_CONSTRAINED_PRE_CYCLE_REPLAY_ONLY`; it is not evidence for
same-cycle current health estimation.

## FINAL TARGET DECISION

**T2**

**TARGET:** `RAW_P1S_RETENTION_AS_CAUSAL_HEALTH_TARGET`

**CALIBRATION:** `R_P1S = V_P1S / V_ref` is observable at the P1S cycle.  The
global six-anchor affine B2 mapping remains retrospective interpretation;
fold-specific B2 may only be used in chronology-safe expanding-fold evaluation
after its frozen nominal anchor set is available.

**TREND:** `FULL_SEQUENCE_TRENDS_RETROSPECTIVE_ONLY`

**ONLINE DEGRADATION RATE:** allowed as a causal raw-HI slope candidate;
predictive utility remains untested.

**SUPPORT HANDLING:** `STRATIFIED_MODELING_REQUIRED`

**NEXT PHASE:** `CAUSAL_DEGRADATION_FORECASTING_PROTOCOL`
