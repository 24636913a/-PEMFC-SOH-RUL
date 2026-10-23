# Phase CSTL — Chronology-Safe Pretrained TimesNet Health Transfer

This Development-only phase evaluates whether the frozen `ΔV`-forecasting
TimesNet representations from the final backbone screen transfer to the
P1S-B2 calibrated health proxy through a small chronological Ridge probe.

The phase does not train a neural health head, fine-tune TimesNet, reopen
backbone search, use multitask loss, access Test data, construct post-1819 h
labels, or train RUL.  It corrects the earlier scratch TSHE label definition:
each outer fold recalibrates B2 using only the sparse health anchors that were
available within that fold's chronological training period.

Run from the repository root:

```powershell
python RENew-PEMFC-2300h-Pytorch/Phase_CSTL_Chronology_Safe_Pretrained_TimesNet_Health_Transfer/scripts/run_cstl.py
```

The script first reproduces saved `ΔV` predictions using the original
checkpoint-specific outer-train scaler.  It stops before embedding extraction
if that consistency contract fails.
