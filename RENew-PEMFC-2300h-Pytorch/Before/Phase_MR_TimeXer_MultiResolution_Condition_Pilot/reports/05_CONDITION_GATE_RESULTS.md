# Condition gate results

| model | fold | scope | q_mae | q_rmse | mixed_brier | mixed_f1 | current_mean_mae_A | current_std_mae_A | target_values |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| P2_common | fold_1 | all_common_origins | 0.001335494716463538 | 0.017645537557710478 | 0.002602689639090785 | 0.9894470240607851 | 1.8643453662407738 | 1.8386526573754756 | 5592 |
| MR_10min_direct | fold_1 | all_common_origins | 0.010175618277184576 | 0.021691721438212495 | 0.00325519120347633 | 0.9898648648648649 | 3.106190162415339 | 3.1398336556540967 | 5592 |
| P2_common | fold_1 | fold1_12h_endpoint | 0.0043532203183033495 | 0.050082747123396415 | 0.00525402063339039 | 0.98 | 3.1913821325794007 | 2.756003749502345 | 233 |
| MR_10min_direct | fold_1 | fold1_12h_endpoint | 0.01133270304237639 | 0.026147594180149236 | 0.003709153871456905 | 0.98989898989899 | 3.9768681463965336 | 3.8799564606737746 | 233 |
| P2_common | fold_2 | all_common_origins | 0.0004646958132562793 | 0.009940340430487464 | 0.001275016180037115 | 0.9987849331713244 | 1.8840001148582297 | 2.077740559364317 | 5880 |
| MR_10min_direct | fold_2 | all_common_origins | 0.011420344118604002 | 0.024837460197820363 | 0.0006085037776480618 | 0.9987849331713244 | 3.3113388383124267 | 3.5819430027759265 | 5880 |
| P2_common | fold_3 | all_common_origins | 0.003428970918305786 | 0.03875106639663339 | 0.004894866182637541 | 0.9832662571489091 | 2.376945286814782 | 2.149259182710649 | 11184 |
| MR_10min_direct | fold_3 | all_common_origins | 0.014080755952725283 | 0.03679670399229077 | 0.004329852596053521 | 0.9864003399915002 | 3.7331858303792114 | 3.5136577759169465 | 11184 |
| P2_common | fold_3 | fold3_nonzero_current_target_bins | 0.003062853650327932 | 0.038479063274129796 | 0.004989310726877456 | 0.0 | 2.5269279334750987 | 2.225813659308007 | 8844 |
| MR_10min_direct | fold_3 | fold3_nonzero_current_target_bins | 0.010322826809591939 | 0.029025210440219797 | 0.004317520903265718 | 0.0 | 3.3803843517480123 | 3.8541715446019946 | 8844 |

## Physical checks

| model | fold | nonfinite_trajectory_rate | physical_violation_trajectory_rate | q_bounds_violation_rate | current_density_identity_max_abs | mixture_variance_nonfinite_rate |
| --- | --- | --- | --- | --- | --- | --- |
| P2_common | fold_1 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| MR_10min_direct | fold_1 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| P2_common | fold_2 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| MR_10min_direct | fold_2 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| P2_common | fold_3 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| MR_10min_direct | fold_3 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |

**CONDITION_GATE_FAIL**

MR versus common P2 q-RMSE gain percent {"fold_1": -22.93, "fold_2": -149.865, "fold_3": 5.043}; non-worse Brier folds 2; MR physical violation rate 0.0.
