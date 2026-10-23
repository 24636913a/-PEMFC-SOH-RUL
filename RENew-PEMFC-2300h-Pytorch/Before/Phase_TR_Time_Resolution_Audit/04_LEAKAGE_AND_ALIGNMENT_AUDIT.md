# Leakage and alignment audit

- Raw chunks are Development-filtered before any aggregation or statistic; the final source also has a fixed Development-prefix limit.
- OpHrs2 is the only bin coordinate; DateTime is continuity diagnostic only.
- Bins are left-closed/right-open, observed-only, and invalid/gap bins break segments.
- Each window including its 24 h history and 12 h target lies fully within its own Fold region.
- Every scaler and model uses only Fold Train windows.
- Unified evaluation requires exact common origin endpoints and verified identical direct versus weighted targets.
- Test features, targets, statistics, and results are absent from all tables.
