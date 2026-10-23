# Invalidated CSHSE run archive — do not use

Everything beneath this directory was produced by the pre-repair CSHSE execution with status `INVALID_PROTOCOL_BREACH`.

Cause: a whole-file SHA256 operation read `causal_features_30min.csv` beyond the frozen Development prefix (`FULL_FILE_HASH_READ_BEYOND_DEVELOPMENT_PREFIX`). There is no evidence that Test labels were accessed, but the data-access boundary breach invalidates all archived metrics, predictions, figures, summaries, reports, and gates.

These files are retained solely to preserve the incident audit trail. They must not be cited as formal scientific results and must not be used as inputs to the clean rerun. The formal regenerated outputs are located in the phase-level `outputs/`, `figures/`, and `reports/` directories outside this archive.
