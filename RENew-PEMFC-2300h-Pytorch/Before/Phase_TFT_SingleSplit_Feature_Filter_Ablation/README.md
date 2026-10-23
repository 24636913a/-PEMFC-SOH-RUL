# Phase TFT Single-Split Feature/Filter Ablation

Development-only, Seed42 exploratory ablation of a frozen Canonical TFT under a single chronological 7:1:2 split.  It compares raw F2 inputs (C0), a train-only Spearman standardised-space feature mask (C1), an independently causal voltage filter selected only on Validation (C2), and their combination (C3).

The runner reads exactly the first 3007 rows of the mixed causal feature table and retains the 3005 legal Development rows.  It never reads the post-Development region or uses future operating covariates.

Run `python scripts/run_c_ablation.py --smoke` before `python scripts/run_c_ablation.py --run`.
