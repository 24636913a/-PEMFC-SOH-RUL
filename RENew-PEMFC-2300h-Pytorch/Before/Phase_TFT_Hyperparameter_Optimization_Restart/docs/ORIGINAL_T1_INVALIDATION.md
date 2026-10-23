# Original T1 invalidation

Status: `T1_ORIGINAL_RUN_INVALIDATED`.

The original T1 run started before its decision-state and multi-seed execution
logic had been corrected. The archived SQLite metadata confirms Trial 0 and
Trial 1 completed and Trial 2 remained running when the process was stopped.
No outer prediction, formal metric, or checkpoint artifact was present. This
is a protocol and execution-logic invalidation, not a TFT performance result.

The original database is audit-only and must not be resumed, queried for its
trial values/parameters, or used to seed T1-R.
