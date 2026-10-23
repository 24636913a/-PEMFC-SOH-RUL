# TECHNICAL BLOCKER

Phase HA stopped before endpoint statistics, figures, report, and final research
status.  This is the required fail-closed result of the pre-statistics cache
integrity defense.

The first required frozen cache was:

```text
method: SimpleTrend
origin_row: 48
path: Phase_DF_Leakage_Safe_Decomposition_Features/cache/SimpleTrend/origin_48.npz
CACHE_PROVENANCE_MANIFEST.csv cache_sha256:
5681a869977c53700b46a470b5f22e0bff12ff74ee7a0593244ca386e990668c
current file SHA256:
c379953dc59369aca8da536625deb94325376cd0bb54e886ee8f86eef6b98a2f
```

The existing `LOCAL_ARTIFACT_MANIFEST.csv` records the latter SHA, whereas the
Phase DF cache-provenance manifest records the former.  This disagreement is an
artifact-provenance inconsistency, not evidence about EMD health sensitivity.
Phase HA does not choose one manifest after the fact, regenerate a cache, or
replace a historical Phase DF file.  No endpoint table, bootstrap, trend,
roughness, revision, ASR, RDR, recovery, SOH, RUL, or final status was produced.

```text
ha_common.AuditFailure: cache SHA mismatch: origin_48.npz
```
