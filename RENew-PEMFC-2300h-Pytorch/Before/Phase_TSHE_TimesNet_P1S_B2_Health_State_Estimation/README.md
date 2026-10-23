# Phase TSHE — TimesNet P1S-B2 Health-State Estimation

This phase keeps the frozen TimesNet backbone and replaces only its former
24-step DeltaV output contract with a scalar health-regression head.  It asks
whether a preceding 24 h (`48 × 14`) Raw F2 operating-history window can
estimate the next **P1S-B2 calibrated SOH proxy** without exposing the model to
the target P1S dwell, target P1S retention, operating age, polarization values,
or future operating conditions.

`P1S + frozen B2` is supervision for a reference-condition health-state proxy;
it is not the final predictive model and it is not an absolute ground-truth
SOH.  TimesNet remains the sole time-series backbone.  No backbone search,
RUL-head training, or hyperparameter tuning is performed here.

Run from the repository root:

```powershell
python RENew-PEMFC-2300h-Pytorch/Phase_TSHE_TimesNet_P1S_B2_Health_State_Estimation/scripts/run_tshe.py
```

The runner verifies the frozen backbone settings, constructs the target-cycle
isolated event table, runs the three expanding Development folds for seeds 42,
2024, and 3407, and then writes the frozen outputs, figures, and report.
