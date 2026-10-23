# Final P1 single-point identifiability audit

## Designation and preserved RHA result

This is a **POSTHOC_EXPLORATORY** audit of the RHA-frozen primary operation
point P1=`221`. The original dual-point RHA remains
`CONDITION_COMMON_SUPPORT_INSUFFICIENT`; it is not changed by this result.
P2 was not replaced or reselected. The representation is **P1 Cycle
Operating-Voltage Retention**, not SOH ground truth.

## Access qualification

Raw post-boundary rows accessed in the boundary-crossing chunk:
**31316**; used: **0**. These rows were
filtered before stable-window assignment, measurements, support, matching,
figures, thresholds, and decisions. Test polarization and EIS were not read.
Development polarization was accessed only after P1-A/B/C and stable-window
sensitivity passed: **True**.

## Final result

**`P1_ANCHOR_ALIGNMENT_INCONCLUSIVE`**

P1-A support reproduction: **True**. P1-B raw long-term
visibility: **True**. P1-C condition-matched robustness:
**True**. Last-25% direction sensitivity:
**True**. P1-D sparse-anchor
compatibility: **False**.

| Quantity | Value |
| --- | ---: |
| Raw EARLY–LATE retention contrast | -0.00706489 |
| Local cycle variation | 0.00019563 |
| Condition-matched contrast | -0.00658831 |
| Last-25% contrast | -0.00757636 |

Sparse-anchor results, if polarization access was allowed, are written below.
They can support only `SPARSE_POLARIZATION_ANCHOR_COMPATIBILITY`, because both
representations are age-associated; they are not independent SOH validation.

Polarization was accessed, but the frozen pre-characterization cycle requirement made anchor alignment unresolved.

No neural model, continuous SOH label, RUL model, or automatic next phase was
created.
