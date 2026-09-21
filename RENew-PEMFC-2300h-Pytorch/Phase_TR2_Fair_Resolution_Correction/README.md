# Phase TR2 — Fair Resolution Correction

`POST_HOC_METHOD_CORRECTION` of Phase TR.  It preserves the former phase and
re-evaluates only its implementation defects: future-observation weighting during
prediction aggregation, non-identical mixed-state labels, and non-common training
origins.  The experiment uses Development-only causal state tables already frozen
in Phase TR; it never opens Test data and does not train a deep model.

Run:

```powershell
python .\RENew-PEMFC-2300h-Pytorch\Phase_TR2_Fair_Resolution_Correction\scripts\run_fair_resolution_correction.py
python .\RENew-PEMFC-2300h-Pytorch\Phase_TR2_Fair_Resolution_Correction\tests\test_corrections.py
```
