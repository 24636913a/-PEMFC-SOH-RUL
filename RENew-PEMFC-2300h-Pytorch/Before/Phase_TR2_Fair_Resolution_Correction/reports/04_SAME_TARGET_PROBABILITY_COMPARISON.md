# Same-target probability comparison

A and B directly predict the same future parent 30-minute labels from the same common origins; they use same Logistic Regression configuration and threshold. C is q-only diagnostic and has no Brier result.

| experiment | fold | metric_scope | horizon_scope | n_target_points | q_mae | q_rmse | mixed_brier | mixed_precision | mixed_recall | mixed_f1 | mixed_positive_count |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A_30min_baseline | fold_1 | common_overlapping_targets | all_horizons | 5592 | 0.01935896341105339 | 0.04511239251160958 | 0.006104054544642538 | 0.9536208299430431 | 1.0 | 0.9762598917117867 | 1172 |
| B_10min_direct_30target | fold_1 | common_overlapping_targets | all_horizons | 5592 | 0.010642906741809977 | 0.022495201269540303 | 0.0032992876622098513 | 0.979933110367893 | 1.0 | 0.9898648648648649 | 1172 |
| A_30min_baseline | fold_2 | common_overlapping_targets | all_horizons | 5880 | 0.021989361618742483 | 0.0656298632411863 | 0.007957888445103084 | 0.9470452801227935 | 1.0 | 0.9728025226645645 | 1234 |
| B_10min_direct_30target | fold_2 | common_overlapping_targets | all_horizons | 5880 | 0.014987929292536289 | 0.030588368850228306 | 0.0032461776180681616 | 0.9793486894360603 | 0.9991896272285251 | 0.9891696750902527 | 1234 |
| A_30min_baseline | fold_3 | common_overlapping_targets | all_horizons | 12912 | 0.031094539051151764 | 0.08414706840982038 | 0.020234026337355743 | 0.926914968376669 | 0.9712812960235641 | 0.9485796476087738 | 2716 |
| B_10min_direct_30target | fold_3 | common_overlapping_targets | all_horizons | 12912 | 0.027698258698393572 | 0.07708230796391706 | 0.018340238775985857 | 0.9281437125748503 | 0.9701767304860088 | 0.9486948694869487 | 2716 |