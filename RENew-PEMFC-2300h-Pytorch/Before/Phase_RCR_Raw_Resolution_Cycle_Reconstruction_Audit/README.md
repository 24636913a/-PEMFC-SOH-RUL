# Phase RCR — Raw-Resolution Cycle Reconstruction Audit

This is a preregistered, Development-only data-structure audit of the RealHyFC
Loadcycle stream at raw time resolution.  It asks whether a reproducible,
auditable load-cycle structure can be reconstructed using operating-state
information only.

The phase is intentionally not a model-training, SOH, RUL, polarization, or
voltage-degradation analysis.  Cycle discovery reads only the frozen operating
fields in `configs/PREREGISTERED_RCR_PROTOCOL.yaml`; voltage columns are never
requested.  The formal analysis universe is `OpHrs2 [h] < 1819.997738`.

Historical Phase MQ outputs are read only for the final raw-to-30-minute
mapping explanation and are never modified.  The three known MQ documentation
issues are recorded as non-blocking debt in this phase rather than repaired or
rerun.

Run after protocol freeze:

```powershell
$env:PYTHONUTF8 = '1'
python scripts/run_raw_cycle_reconstruction_audit.py
python -m unittest tests.test_raw_cycle_reconstruction_audit
```

The audit fails closed when the raw schema, boundary firewall, frozen
operating-only input contract, or deterministic cycle-template qualification
cannot be verified.
