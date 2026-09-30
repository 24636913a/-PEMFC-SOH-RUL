# Phase HA — EMD Health Representation Audit

This phase audits the online, origin-endpoint form of the frozen Phase DF
first-two-IMF-removed EMD remainder.  It is a Development-only representation
audit, not a forecasting re-run, SOH construction, RUL study, or Test evaluation.

Run from this directory after freezing the protocol:

```powershell
python scripts/freeze_ha_protocol.py
python -m unittest discover -s tests -v
python scripts/run_health_audit.py
```

The runner verifies the base commit and frozen source hashes, reads exactly the
first 3,007 rows of the N1 causal table, filters its Development prefix, and
uses the already saved Phase DF endpoint feature caches.  It does not load
forecast targets, predictions, test rows, or a neural model.

`PRE-REGISTERED_BEFORE_PHASE_HA_ENDPOINT_ANALYSIS` means that the Phase HA
protocol was written before its endpoint statistics, but after Phase DF's
forecasting and historical endpoint-revision evidence were known.

