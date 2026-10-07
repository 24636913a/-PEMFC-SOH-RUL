# Phase CTA — P1S-B2 Causal Target Audit

This Development-only audit determines whether the P1S-B2 health quantities
created in PHI can be used as causal prediction targets.  It performs no model
training, no target retuning, no forecasting and no RUL analysis.

Three questions are addressed:

1. whether the all-six-anchor B2 calibration was available at each P1S event;
2. whether full-sequence health trends change when future events arrive; and
3. whether frozen operating-condition support should be carried forward as a
   quality flag or an explicit modeling stratum.

Only previously derived Development artifacts and the already-unblinded ELTE
result table are read.  No raw 2000 h / 2300 h polarization target file is
configured or opened.

Run from repository root:

```powershell
python RENew-PEMFC-2300h-Pytorch/Phase_CTA_P1S_B2_Causal_Target_Audit/scripts/run_cta_audit.py
```

The audit's formal target decision is `RAW_P1S_RETENTION_AS_CAUSAL_HEALTH_TARGET`.
The frozen global six-anchor B2 trajectory remains a retrospective reference,
while fold-specific B2 is retained only for chronology-safe expanding-fold
evaluation after its relevant nominal anchor set is available.
