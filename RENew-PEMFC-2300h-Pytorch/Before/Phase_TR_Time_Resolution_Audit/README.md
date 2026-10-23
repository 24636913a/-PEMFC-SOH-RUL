# Phase TR — RealHyFC Time-Resolution Audit

This isolated phase tests whether a causally constructed 10-minute representation
adds measurable information for forecasting the future zero-current fraction `q`,
relative to the frozen 30-minute representation.  It uses only the frozen
Development prefix (`OpHrs2 < 1819.997738 h`) and does not train a neural network.

Run from the repository root:

```powershell
python .\RENew-PEMFC-2300h-Pytorch\Phase_TR_Time_Resolution_Audit\scripts\run_time_resolution_audit.py
```

The configuration fixes 24 h history, 12 h forecast, the established three
expanding validation folds, zero-current threshold `abs(I.S.act) <= 0.10 A`, and
all decision criteria before validation results are inspected.
