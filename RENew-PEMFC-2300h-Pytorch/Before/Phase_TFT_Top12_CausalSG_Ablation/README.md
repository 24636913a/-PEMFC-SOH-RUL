# Phase D — TFT Top12 + Causal SG Ablation

This is a Development-only, Seed42 controlled ablation on the same chronological 7:1:2 split as the preceding C-series study.  It uses the frozen CanonicalStrictHistoryTFT and T1-R Fold2 hyperparameters without a new Optuna search.

D0/D1 isolate Top10 versus Top12 standardised-space masks. D2/D3 isolate a fixed, one-sided causal SG(7,2) transformation of only the historical average-cell-voltage channel. All groups share one scaler fitted solely to unfiltered Train windows and one common DeltaV target scale.
