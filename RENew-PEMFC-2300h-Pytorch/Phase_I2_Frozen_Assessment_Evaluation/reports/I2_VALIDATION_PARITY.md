# Fixed Validation exact reload

Reused Development Assessment; n=3 frozen seeds. Not independent Test. All 9072 queries per run retained. No new training, no new model selection.

| run_id | model | seed | H_run_id | array_equal | unequal_elements | unequal_fraction | finite | actual_nan | actual_inf | saved_nan | saved_inf | max_abs_difference_standardized | mean_abs_difference_standardized | difference_RMSE_standardized | max_abs_difference_mV | difference_RMSE_mV | max_difference_batch_index_zero_based | max_difference_sample_index_zero_based | max_difference_horizon_step | fraction_abs_difference_gt_1e-08_standardized | fraction_abs_difference_gt_1e-07_standardized | fraction_abs_difference_gt_1e-06_standardized | fraction_abs_difference_gt_1e-05_standardized | state_unchanged |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| I2_F0_Seed42 | F0 | 42 | H-F0-Seed42 | True | 0 | 0 | True | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | True |
| I2_F0_Seed52 | F0 | 52 | H-F0-Seed52 | True | 0 | 0 | True | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | True |
| I2_F0_Seed62 | F0 | 62 | H-F0-Seed62 | True | 0 | 0 | True | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | True |
| I2_F1_Seed42 | F1 | 42 | H-F1-Seed42 | True | 0 | 0 | True | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | True |
| I2_F1_Seed52 | F1 | 52 | H-F1-Seed52 | True | 0 | 0 | True | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | True |
| I2_F1_Seed62 | F1 | 62 | H-F1-Seed62 | True | 0 | 0 | True | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | True |
| I2_F2_Seed42 | F2 | 42 | H-F2-Seed42 | True | 0 | 0 | True | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | True |
| I2_F2_Seed52 | F2 | 52 | H-F2-Seed52 | True | 0 | 0 | True | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | True |
| I2_F2_Seed62 | F2 | 62 | H-F2-Seed62 | True | 0 | 0 | True | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | True |

9/9 np.array_equal against saved checkpoint 32-origin references before any Assessment inference. Each worker is a fresh process with frozen H environment; detailed JSON and raw NPZ preserve all numeric differences and parameter/buffer hashes. This is not a full 205-origin Validation reevaluation. Original I BLOCKED ledger remains unchanged.
