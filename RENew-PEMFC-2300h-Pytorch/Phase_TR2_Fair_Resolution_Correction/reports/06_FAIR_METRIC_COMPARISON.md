# Fair metric comparison

Per-fold rows use the identical overlapping common target collection within each Fold. Fold means and pooled target collections are separate statistics. Event counts deduplicate target timestamps.

## Per fold

| experiment | fold | metric_scope | horizon_scope | n_target_points | q_mae | q_rmse | mixed_brier | mixed_precision | mixed_recall | mixed_f1 | mixed_positive_count |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A_30min_baseline | fold_1 | common_overlapping_targets | all_horizons | 5592 | 0.01935896341105339 | 0.04511239251160958 | 0.006104054544642538 | 0.9536208299430431 | 1.0 | 0.9762598917117867 | 1172 |
| A_30min_baseline | fold_1 | common_overlapping_targets | terminal_12h | 233 | 0.030048308239920128 | 0.05822472558773408 | 0.010977639612489752 | 0.9074074074074074 | 1.0 | 0.9514563106796117 | 49 |
| B_10min_direct_30target | fold_1 | common_overlapping_targets | all_horizons | 5592 | 0.010642906741809977 | 0.022495201269540303 | 0.0032992876622098513 | 0.979933110367893 | 1.0 | 0.9898648648648649 | 1172 |
| B_10min_direct_30target | fold_1 | common_overlapping_targets | terminal_12h | 233 | 0.012150204456609531 | 0.02693255897477637 | 0.004086500106192845 | 0.98 | 1.0 | 0.98989898989899 | 49 |
| C_10min_then_equal_mean | fold_1 | common_overlapping_targets | all_horizons | 5592 | 0.056694230838721375 | 0.11969502140036585 | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN |
| C_10min_then_equal_mean | fold_1 | common_overlapping_targets | terminal_12h | 233 | 0.058595809349519405 | 0.1231619520443394 | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN |
| D_10min_aggregated_history_control | fold_1 | common_overlapping_targets | all_horizons | 5592 | 0.01935896341105339 | 0.04511239251160958 | 0.006104054544642538 | 0.9536208299430431 | 1.0 | 0.9762598917117867 | 1172 |
| D_10min_aggregated_history_control | fold_1 | common_overlapping_targets | terminal_12h | 233 | 0.030048308239920128 | 0.05822472558773408 | 0.010977639612489752 | 0.9074074074074074 | 1.0 | 0.9514563106796117 | 49 |
| A_30min_baseline | fold_2 | common_overlapping_targets | all_horizons | 5880 | 0.021989361618742483 | 0.0656298632411863 | 0.007957888445103084 | 0.9470452801227935 | 1.0 | 0.9728025226645645 | 1234 |
| A_30min_baseline | fold_2 | common_overlapping_targets | terminal_12h | 245 | 0.03306598192149604 | 0.08011625842434714 | 0.013129791551250384 | 0.9122807017543859 | 1.0 | 0.9541284403669725 | 52 |
| B_10min_direct_30target | fold_2 | common_overlapping_targets | all_horizons | 5880 | 0.014987929292536289 | 0.030588368850228306 | 0.0032461776180681616 | 0.9793486894360603 | 0.9991896272285251 | 0.9891696750902527 | 1234 |
| B_10min_direct_30target | fold_2 | common_overlapping_targets | terminal_12h | 245 | 0.01802567371709207 | 0.041606756768537895 | 0.00962217805080831 | 0.9622641509433962 | 0.9807692307692307 | 0.9714285714285714 | 52 |
| C_10min_then_equal_mean | fold_2 | common_overlapping_targets | all_horizons | 5880 | 0.058825638168776784 | 0.12267211202120527 | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN |
| C_10min_then_equal_mean | fold_2 | common_overlapping_targets | terminal_12h | 245 | 0.06103815484310538 | 0.12591022530676163 | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN |
| D_10min_aggregated_history_control | fold_2 | common_overlapping_targets | all_horizons | 5880 | 0.021989361618742483 | 0.0656298632411863 | 0.007957888445103084 | 0.9470452801227935 | 1.0 | 0.9728025226645645 | 1234 |
| D_10min_aggregated_history_control | fold_2 | common_overlapping_targets | terminal_12h | 245 | 0.03306598192149604 | 0.08011625842434714 | 0.013129791551250384 | 0.9122807017543859 | 1.0 | 0.9541284403669725 | 52 |
| A_30min_baseline | fold_3 | common_overlapping_targets | all_horizons | 12912 | 0.031094539051151764 | 0.08414706840982038 | 0.020234026337355743 | 0.926914968376669 | 0.9712812960235641 | 0.9485796476087738 | 2716 |
| A_30min_baseline | fold_3 | common_overlapping_targets | terminal_12h | 538 | 0.044197681811803405 | 0.10350968448993161 | 0.028563014650634308 | 0.907563025210084 | 0.9557522123893806 | 0.9310344827586207 | 113 |
| B_10min_direct_30target | fold_3 | common_overlapping_targets | all_horizons | 12912 | 0.027698258698393572 | 0.07708230796391706 | 0.018340238775985857 | 0.9281437125748503 | 0.9701767304860088 | 0.9486948694869487 | 2716 |
| B_10min_direct_30target | fold_3 | common_overlapping_targets | terminal_12h | 538 | 0.03340651894420561 | 0.09118442675668953 | 0.024582713806947497 | 0.9 | 0.9557522123893806 | 0.927038626609442 | 113 |
| C_10min_then_equal_mean | fold_3 | common_overlapping_targets | all_horizons | 12912 | 0.06573166537334697 | 0.12790327506643137 | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN |
| C_10min_then_equal_mean | fold_3 | common_overlapping_targets | terminal_12h | 538 | 0.06802508482695044 | 0.1337779108523089 | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN |
| D_10min_aggregated_history_control | fold_3 | common_overlapping_targets | all_horizons | 12912 | 0.031094539051151764 | 0.08414706840982038 | 0.020234026337355743 | 0.926914968376669 | 0.9712812960235641 | 0.9485796476087738 | 2716 |
| D_10min_aggregated_history_control | fold_3 | common_overlapping_targets | terminal_12h | 538 | 0.044197681811803405 | 0.10350968448993161 | 0.028563014650634308 | 0.907563025210084 | 0.9557522123893806 | 0.9310344827586207 | 113 |

## Summary

| experiment | aggregation | horizon_scope | q_mae | q_rmse | mixed_brier |
| --- | --- | --- | --- | --- | --- |
| A_30min_baseline | fold_mean | all_horizons | 0.02414762136031588 | 0.06496310805420542 | 0.011431989775700457 |
| A_30min_baseline | pooled_common_targets | all_horizons | 0.02620757291425885 | 0.07249015969317493 | 0.0140333007356785 |
| B_10min_direct_30target | fold_mean | all_horizons | 0.017776364910913278 | 0.04338862602789523 | 0.00829523468542129 |
| B_10min_direct_30target | pooled_common_targets | all_horizons | 0.02072195189689848 | 0.059058932269639805 | 0.011251078743309044 |
| C_10min_then_equal_mean | fold_mean | all_horizons | 0.06041717812694838 | 0.12342346949600085 | NOT_RUN |
| C_10min_then_equal_mean | pooled_common_targets | all_horizons | 0.06199377274373334 | 0.12480816816365055 | NOT_RUN |
| D_10min_aggregated_history_control | fold_mean | all_horizons | 0.02414762136031588 | 0.06496310805420542 | 0.011431989775700457 |
| D_10min_aggregated_history_control | pooled_common_targets | all_horizons | 0.02620757291425885 | 0.07249015969317493 | 0.0140333007356785 |

## Events

| experiment | fold | status | unique_target_times | independent_true_mixed_events | predicted_mixed_events | missed_true_events | false_positive_events |
| --- | --- | --- | --- | --- | --- | --- | --- |
| A_30min_baseline | fold_1 | RUN | 256.0 | 53.0 | 53.0 | 0.0 | 0.0 |
| B_10min_direct_30target | fold_1 | RUN | 256.0 | 53.0 | 53.0 | 0.0 | 0.0 |
| C_10min_then_equal_mean | fold_1 | NOT_RUN |  |  |  |  |  |
| D_10min_aggregated_history_control | fold_1 | RUN | 256.0 | 53.0 | 53.0 | 0.0 | 0.0 |
| A_30min_baseline | fold_2 | RUN | 268.0 | 57.0 | 57.0 | 0.0 | 0.0 |
| B_10min_direct_30target | fold_2 | RUN | 268.0 | 57.0 | 57.0 | 1.0 | 1.0 |
| C_10min_then_equal_mean | fold_2 | NOT_RUN |  |  |  |  |  |
| D_10min_aggregated_history_control | fold_2 | RUN | 268.0 | 57.0 | 57.0 | 0.0 | 0.0 |
| A_30min_baseline | fold_3 | RUN | 607.0 | 126.0 | 126.0 | 5.0 | 5.0 |
| B_10min_direct_30target | fold_3 | RUN | 607.0 | 126.0 | 124.0 | 5.0 | 5.0 |
| C_10min_then_equal_mean | fold_3 | NOT_RUN |  |  |  |  |  |
| D_10min_aggregated_history_control | fold_3 | RUN | 607.0 | 126.0 | 126.0 | 5.0 | 5.0 |
