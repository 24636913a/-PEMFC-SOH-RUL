# Condition-Normalized P1S Development Study

This is a Development-only, exploratory condition-normalization study for the
frozen P1S state.  It uses the PAA Development interval artifacts, the six
frozen Development polarization-health anchors, and the historical frozen
Raw-P1S B2 baseline.  It does not open 2000/2300 polarization files, invoke
Stage T, alter the consumed AHC-TV ledger, or execute a confirmatory Test.

Run from the repository root while `HEAD` is the registered base commit:

```powershell
python -B RENew-PEMFC-2300h-Pytorch\Phase_CNP1S_Condition_Normalized_P1S_Development_Study\scripts\run_cn_p1s_development_study.py
```

The primary candidate is C2, a partial-linear Ridge condition model.  Age is
used only as a nuisance spline in the interval-voltage model; it is never an
input to the final nonnegative affine `R -> Y` health mapping.
