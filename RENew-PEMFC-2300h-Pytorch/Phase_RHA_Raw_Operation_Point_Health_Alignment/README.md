# Phase RHA — Raw Operation-Point Health Alignment Audit

Phase RHA is a zero-training, Development-only health-evidence qualification audit. It reuses the 146 frozen complete cycles from Phase RCR without rediscovery, alignment changes, DTW, voltage-based phase selection, or changes to the RCR template.

Only after point support, seven-variable joint condition support, and condition-robust long-term voltage visibility pass may the audit access the allowlisted Development polarization anchors. Cycle operating-voltage retention is a descriptive representation, not SOH ground truth or a continuous SOH label.

The raw source stream uses a 50,000-row boundary-aware early-stop protocol. If a crossing chunk contains post-boundary rows, access is disclosed while all such rows are excluded from every analysis object and decision.

Run after protocol freeze:

```powershell
$env:PYTHONUTF8 = '1'
python scripts/run_raw_operation_point_health_audit.py
python -m unittest tests.test_raw_operation_point_health_audit
```
