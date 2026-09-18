# Future-condition amplitude audit

The high/low threshold is fold-specific and train-only: the 75th percentile of the N2.8F target-completion-purged causal Ridge OOF absolute current error. No voltage error participates in the threshold.

The mutually exclusive groups are state error S1, state-correct/high-amplitude-error S0-H, and state-correct/low-amplitude-error S0-L. Fold 1's harmful D is concentrated in two `S1_pred_mixed_true_nonmixed` points (net +288614.5 mV²); S0-L is net favorable to P2. Fold 3 has 54 such S1 points with net +3.703 million mV², while S0-L (6441 points) is net -1.314 million mV² and S0-H (7 points) is net -0.018 million mV².

This provides evidence for sparse state/amplitude failures at exceptional points, but not for a cross-object magnitude-error mechanism independent of state errors.
