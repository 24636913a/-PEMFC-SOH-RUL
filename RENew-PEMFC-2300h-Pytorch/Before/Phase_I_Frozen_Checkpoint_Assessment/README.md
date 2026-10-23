# Phase I — Frozen Checkpoint Assessment

Nine unique Phase H MULTISEED_PRIMARY checkpoints (F0/F1/F2 × 42/52/62) are evaluated once on **Reused Development Assessment**. Repeat-A/B are excluded. No training, scaler refitting, architecture change, seed selection or formal Test access is authorized.

Base main: `1b83d05b94a4952d0a41df00142db710278830b4`.

Current execution status: **PHASE_I_EXECUTION_BLOCKED**. The first fixed Validation reload (F0 / Seed42) did not meet frozen bitwise parity; the ledger stopped before any new Assessment inference. All 22 post-freeze synthetic regression tests passed. No Assessment scores or scientific ranking were produced. Do not run the execution command again; the ledger remains immutable evidence of the failed attempt. See `reports/I_FINAL_SCIENTIFIC_CONCLUSION.md` and `outputs/I_SUMMARY.json`.

## Frozen execution

Run from this directory, with the same Python/CUDA environment as H:

```powershell
python -B -m unittest discover -s tests -v
python -B scripts/prepare_i_assessment.py --freeze
python -B scripts/run_i_assessment.py --execute
```

The entry point repeats the synthetic tests, independently checks all nine weights and their saved fixed Validation predictions, and starts Assessment only after those checks pass. The protocol and whole source identity are frozen **before the first new Assessment prediction**. The ledger refuses a second Assessment execution, including after failure. It records every completed cache and any exception. Do not delete it to rerun.

The only N1 read is the inherited `pd.read_csv(nrows=3007)`, followed by `bin_end_ophrs2_h < 1819.997738`, giving 3005 rows. Integrity uses the logical Development-prefix fingerprint, never a full N1 checksum. Persisted E scaler, inherited E EWMA/preprocessing and C window/metric functions are reused without fit calls. Assessment has 378 origins × 24 steps = 9072 queries per checkpoint.

## Audit and outputs

```powershell
python -B scripts/audit_i_predictions.py
```

This audit only rereads local prediction caches; it does not repeat inference. Each cache must have identical keys, truth, target ages and frozen identities. A second scalar/math.fsum implementation independently recomputes all eight metrics. Target-time-equal RMSE averages squared errors within target before averaging targets; it is not an ensemble. Primary metric remains forecast-query-weighted RMSE. Seed means and sample SD (ddof=1) are separate from descriptive pooled metrics.

`outputs/I_SUMMARY.json`, `outputs/metrics/`, `outputs/diagnostic/`, `reports/` and `figures/` are generated from audited outputs. Nine PNG/PDF figure pairs follow a publication-style layout; seed SD is not a confidence interval. Reports preserve inherited Assessment selection bias and unresolved historical reproducibility. No outlier-deleted ranking or significance claim is permitted. No automatic Phase J, Test, SOH or RUL follows this stage.
