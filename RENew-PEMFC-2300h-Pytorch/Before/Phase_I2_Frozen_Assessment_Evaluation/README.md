# Phase I2 — Frozen Assessment Evaluation

Independent one-shot evaluation of the nine frozen H F0/F1/F2 checkpoints, seeds 42/52/62. Restores I-R verified H Path A; original I remains BLOCKED. No training, no model/scaler updates, no Test.

Run from this phase with Python `-B` (historical phases are read-only):

```powershell
python -B -m unittest discover -s tests -v
python -B scripts/prepare_i2.py
python -B -m unittest discover -s tests -v
python -B scripts/run_i2_assessment.py --execute
```

The execution command is authorized once only. Never delete the ledger or rerun a failed worker. All nine fixed 32-origin Validation predictions must be bitwise equal before Assessment. Assessment: 378 origins ×24 queries per checkpoint, 81648 rows total. H `e.predict(model,x,32,device)` uses eval/no_grad with original requires_grad=True flags intact. Frozen checkpoint bytes and every state entry are audited unchanged.

Data access: prescribed nrows=3007 read, Development `<1819.997738` filter, 3005 rows. No full-N1 checksum. Saved E scaler only. Top10, per-window causal EWMA and float32 scaling unchanged.

Primary: forecast-query-weighted RMSE (mV); n=3 equal-seed mean and sample SD. Target-time-equal diagnostics average squared errors, never predictions. Negative F2−F0 means F2 is lower. All extreme targets retained. Figures export 300-dpi PNG and PDF with explicit reused Development label.

This Assessment has historical selection bias; it is not independent Test. No automatic next phase or Test opening. Detailed protocol, ledger, raw NPZ, CSV, per-state audits and independent recalculation evidence are retained.

## Completed execution

Status: `F2_REUSED_ASSESSMENT_MULTISEED_GAIN_OBSERVED`. Fixed Validation exact reload 9/9; Assessment caches 9/9; independent metric checks 72/72. Original I remains BLOCKED. Regression tests after freeze: 23/23, no skipped tests. No repeated Assessment inference is permitted.

Protocol SHA256: `d9d5dd2ad31e7fe8a6a28aaa19d706bea0a7ad16092b6de449171172bb239307`.

Execution code identity SHA256: `e1945efba676202247bf0600de00be7f3b11f9487ac6ed8827f1ec175398f2c3`.

See `reports/I2_FINAL_SCIENTIFIC_CONCLUSION.md`, `outputs/I2_SUMMARY.json`, `outputs/metrics/I2_ARTIFACT_INTEGRITY_AUDIT.json` and `outputs/execution_ledger/I2_EXECUTION_LEDGER.json`. Seven PNG/PDF figure pairs were visually inspected; their error bars are seed sample SD, not confidence intervals.
