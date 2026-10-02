# Posthoc Test-firewall deviation qualification

## Scope

This is a posthoc implementation qualification of Phase RCR. It does not
replace the original RCR protocol, its SHA256 sidecar, its final decision, or
any cycle-structure result.

## What occurred

The RCR protocol listed `Test_access_or_decision_use` as prohibited. The
implementation opened and streamed all 17 Loadcycle CSV files, parsed only the
six operating-state fields, and then applied the strict
`OpHrs2 [h] < 1819.997738` filter within each chunk. Therefore post-boundary
raw operating rows were accessed during source parsing before being discarded.

```text
post_boundary_rows_accessed = 1896614
post_boundary_rows_used = 0

test_raw_operating_rows_accessed = true
test_raw_operating_rows_used_for_cycle_discovery = false
test_raw_operating_rows_used_for_template = false
test_raw_operating_rows_used_for_statistics = false
test_raw_operating_rows_used_for_figures = false
test_raw_operating_rows_used_for_decision = false

test_polarization_accessed = false
test_EIS_accessed = false
```

The access deviation means RCR must not be described as having achieved *zero
Test access* or `FULL_PROTOCOL_COMPLIANCE`. It is recorded as
`TEST_ACCESS_FIREWALL_DEVIATION_RECORDED`.

## Scientific-result qualification

The retained Development rows alone formed every RCR operating-state run,
template, cycle catalogue, duration statistic, figure, and qualification gate.
No post-boundary row was retained in those objects. The scientific result is
therefore unchanged:

```text
RAW_CYCLE_STRUCTURE_QUALIFIED
146 complete cycles
EARLY / MIDDLE / LATE = 50 / 45 / 51
loaded-duration median = 9.52301 h
loaded-duration CV = 0.006413
```

## Timestamp-unit correction

Commit `b32c0280e1cda53a3c9ef99a34dc78866c9beced` corrected the nominal
timestamp-interval audit from an incorrect approximately `0.001 s` display to
`1.0 s`. Run boundaries already used timestamp `.total_seconds()` and were not
affected. The 146-cycle count, stratum counts, duration summary, frozen
template, and `RAW_CYCLE_STRUCTURE_QUALIFIED` status did not change.

## Consequence for subsequent work

RCR cycle identities remain frozen and may be reused. Any subsequent raw-data
phase must explicitly report raw post-boundary access and use a
boundary-aware early-stop stream; it must not claim `Test accessed = false`
when a crossing chunk has been parsed.
