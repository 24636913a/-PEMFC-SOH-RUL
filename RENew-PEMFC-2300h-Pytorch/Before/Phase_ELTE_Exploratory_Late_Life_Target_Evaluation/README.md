# Phase ELTE — Exploratory Late-Life Target Evaluation

This phase is the single authorized, exploratory opening of the RealHyFC 2000 h and 2300 h nominal-polarization targets. It evaluates the already frozen Raw P1S + B2 affine reference against the already frozen nominal-age affine baseline. The condition-normalized P1S Ridge result is secondary only.

The phase is explicitly **not** a pristine or confirmatory Test. No post-target refit, interval replacement, feature change, support-gate change, or model training is permitted.

## One-shot contract

Run only once from the repository root:

```powershell
python RENew-PEMFC-2300h-Pytorch/Phase_ELTE_Exploratory_Late_Life_Target_Evaluation/scripts/run_elte_one_shot.py
```

The runner creates `inputs/FROZEN_ELTE_PREDICTIONS_BEFORE_TARGET_OPENING.csv`, freezes its SHA256 and provenance, then creates `runtime/ELTE_ONE_SHOT_LEDGER.json` immediately before the paired target read. An existing ledger is fail-closed and prevents any second run.

The historical AHC-TV ledger is read-only and is never modified by this phase.
