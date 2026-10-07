# Phase CSHSE — Causal Same-Cycle Health-State Estimation

This Development-only phase tests a narrower question than the earlier target-cycle-isolated studies: can a frozen DeltaV-pretrained TimesNet representation estimate the **current** P1S-B2 calibrated health proxy using the latest legal 24 h of operating history before the target P1S measurement begins?

The input is allowed to enter the target cycle, but it ends at least 0.5 h before the earliest raw observation used by the frozen PAA `middle_50` P1S measurement. The target P1S dwell, all guard-region bins, operating age, future observations, Test observations, and all neural-weight updates are prohibited.

Run from the repository root:

```powershell
python RENew-PEMFC-2300h-Pytorch/Phase_CSHSE_Causal_SameCycle_Health_State_Estimation/scripts/run_cshse.py
```

The runner fails closed if P1S raw-window reconstruction fails to reproduce the frozen P1S voltage, if a history window crosses a segment or the guard, or if any source reaches the Development/Test boundary.
