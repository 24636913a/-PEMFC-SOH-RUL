# Final pre-open implementation errata

This document records implementation corrections made before any real AHC-TV
Test file was opened. It does not alter the frozen protocol, P1S affine,
nominal-age affine, support bounds, 48 h selection window, or Rules A/B/C.

## Stage P

The prior runtime prototype had three technical deviations from frozen RCR/PAA
semantics:

- It used a two-second timestamp break instead of RCR's five-second break,
  and did not treat decreasing age or non-increasing timestamps as a break.
- It did not use the RCR source-file boundary, zero-current state flag, and
  template-qualified cycle logic before accepting an interval.
- It used `DataFrame.mean(skipna=False)` for active-cell voltage. Frozen PAA
  uses the ten-cell `numpy.nanmean` convention: a partially observed row is
  retained, while an all-missing row is `NaN`.

Stage P now uses the RCR token `operation_point || round(I.S.set [A], 3)`, the
frozen `15||14.000 → 15||0.000` anchors, RCR template qualification, canonical
`COMPLETE_###` identity, a cutoff-derived `PARTIAL_PRE_<checkpoint>` identity,
and PAA's inclusive middle-50% **global DateTime** membership (not an added
source-file restriction). These are implementation corrections,
not new scientific choices. The Development raw-to-anchor replay is the
evidence that the corrected implementation reproduces the frozen PAA result.

## Stage T

Stage T now uses the same ten-cell `numpy.nanmean` / local `nanmedian`
extractor as the frozen Development polarization pipeline. The AHC-TV frozen
BoL denominator is exactly `0.6989533066`. A historical CHM sparse-target
artifact used a higher-precision denominator and therefore differs in
normalized values at approximately `1e-8`; the historical value remains an
audit field only. The frozen AHC-TV denominator is the executable authority.

## One-shot execution

The formal CLI no longer accepts a runtime directory. It derives the sole
formal path `runtime/ONE_SHOT_TEST_RUN` and creates a durable phase-global
ledger before Reader M can access Test metadata. The ledger is exclusive-create
only, fsync-backed, forward-only, and is never deleted after a failure. Formal
artifacts are also exclusive-create only. After the paired target reads,
target/error values and the decision are immediately frozen with SHA256
sidecars; any later reporting must read the frozen result table rather than a
polarization source.
