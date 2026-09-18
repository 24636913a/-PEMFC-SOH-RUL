# Physical counterfactual audit

Development-only raw statistics reconstruct the causal-table current mean, current standard deviation and q with maximum absolute residual 7.105e-14 A/fraction; all 3,005 bins are Oracle-estimable. Every four-group check has q in [0,1], nonnegative current mean and standard deviation, exact J=I/A, and zero unauthorized change in peak-to-peak features.

CF00 reuses the official P2 future-condition tensor and reproduces the frozen three-fold P2 mean RMSE (21.346566 mV). CF10/CF01/CF11 are **ORACLE_DIAGNOSTIC_ONLY**, not deployable performance. Evidence: `outputs/physical/true_conditional_current_audit.csv`, `outputs/counterfactual/counterfactual_input_audit.csv`, and `counterfactual_fold_metrics.csv`.
