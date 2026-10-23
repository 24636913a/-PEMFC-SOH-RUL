# TECHNICAL BLOCKER V2.1

The V2.1 retry stopped before issuing a health state.

```text
AttributeError: 'DataFrame' object has no attribute 'stratum'
```

The production traceback located this at `rdr_audit`: the empty matched-contrast
table correctly represented no matched-support result but did not retain its
ordinary result columns.  This is an execution handling error only; it is not
a cache, data boundary, hash, matching, bootstrap, or endpoint-statistics
failure.  No normal endpoint output was persisted.
