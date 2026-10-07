# Late-Life P1S Operating-Condition Support Drift Audit

This target-blind audit explains the frozen AHC-TV Stage-P alignment outcome
without reading a polarization target, running Stage T, changing the one-shot
ledger, or modifying any AHC-TV protocol input.  It consumes frozen
Development support bounds and the already-created target-blind P1S interval
tables only.

Run from the repository root:

```powershell
python -B RENew-PEMFC-2300h-Pytorch\Phase_AHC_TV_Late_Life_Operating_Condition_Support_Drift_Audit\scripts\run_late_life_support_drift_audit.py
```

The `48 h` rule, P1S construction, latest-three selection, Development joint
support, one-shot ledger, and Test status are descriptive inputs to this
audit; none is changed by it.
