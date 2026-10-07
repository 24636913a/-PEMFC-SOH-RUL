# Phase PHI — P1S-B2 Health Index Consolidation

This phase consolidates the frozen Raw P1S + B2 affine mapping into a dense,
cycle-aligned **P1S-B2 calibrated health indicator** (`HI_P1S`).  It is a
health proxy / degradation-state representation, not independently validated
true SOH and not an RUL model.

The primary input is the frozen `middle_50` P1S complete-cycle measurement:
this is the stable-window definition that exactly reproduces the historical
P1S values used by all six Development anchor alignments.  No raw Loadcycle or
polarization file is read by this phase.  The already-unblinded 2000 h and
2300 h ELTE result table is consumed only as immutable, exploratory validation
evidence; neither target is reopened and no model is refit after ELTE.

Run from the repository root:

```powershell
python RENew-PEMFC-2300h-Pytorch/Phase_PHI_P1S_B2_Health_Index_Consolidation/scripts/run_phi_consolidation.py
```

The selected primary trend is decreasing isotonic regression.  The raw health
indicator remains unmodified in `HI_RAW`; all trend fields are separate.

Important coverage limit: the frozen complete-cycle P1S table provides 141
unique legal P1S cycles ending at 1712.79 h, below the 1819.997738 h
Development boundary.  Thus PHI provides dense Development-cycle coverage and
two existing late-life exploratory validation checkpoints, but no newly
constructed late-life cycle-level HI observations.
