# Phase POT — Optuna Search Report

The search was frozen before any outer-fold evaluation. Twenty deterministic TPE trials were completed for each method using only Fold 1 outer-train's chronological inner fit/stopper partition; no outer validation or Test selection occurred.

| Method | Best trial | Inner RMSE (mV) | Selected epoch | Learning rate | Weight decay | Dropout | Batch size |
|---|---:|---:|---:|---:|---:|---:|---:|
| Raw F2 | 8 | 16.578 | 107 | 0.00056915 | 0.00000909 | 0.2 | 32 |
| EMD | 18 | 16.887 | 30 | 0.00175768 | 0.00036730 | 0.1 | 32 |

The frozen resolved configuration file was hashed before M3/M4 formal refits. Fold 2 and Fold 3 were never used for the search.
