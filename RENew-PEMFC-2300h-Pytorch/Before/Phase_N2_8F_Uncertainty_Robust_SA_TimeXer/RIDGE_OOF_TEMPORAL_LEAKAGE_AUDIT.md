# Ridge OOF temporal-leakage audit

Every fit target must end no later than the earliest prediction origin in its block. The CSV below reports the repaired target-completion purge. All rows passed; any `overlap_violation=true` would terminate this phase before M0 training.

| stage | fold | block | fit_origin_start_h | fit_origin_end_h | fit_count | max_fit_target_end_h | min_prediction_origin_h | max_prediction_origin_h | target_end_before_prediction_origin | overlap_violation | purged_index_rows_before_block | warmup_dropped_rows | scaler_fit_scope | fit_segment_crossings | prediction_segment_crossings |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ridge_oof | fold_1 | 1 | 97.000000 | 128.500000 | 64 | 140.500000 | 140.500000 | 272.500000 | True | False | 23 | 87 | completed_fit_history_only | 0 | 0 |
| ridge_oof | fold_1 | 2 | 97.000000 | 261.000000 | 257 | 273.000000 | 273.000000 | 420.000000 | True | False | 23 | 87 | completed_fit_history_only | 0 | 0 |
| ridge_oof | fold_1 | 3 | 97.000000 | 408.500000 | 449 | 420.500000 | 420.500000 | 552.500000 | True | False | 23 | 87 | completed_fit_history_only | 0 | 0 |
| ridge_oof | fold_1 | 4 | 97.000000 | 541.000000 | 641 | 553.000000 | 553.000000 | 734.000000 | True | False | 23 | 87 | completed_fit_history_only | 0 | 0 |
| ridge_oof | fold_1 | 5 | 97.000000 | 722.500000 | 833 | 734.500000 | 734.500000 | 930.000000 | True | False | 23 | 87 | completed_fit_history_only | 0 | 0 |
| confidence_oof | fold_1 | 1 | 140.500000 | 172.000000 | 64 | 184.000000 | 184.000000 | 358.500000 | True | False | 110 | 174 | completed_residual_history_only | 0 | 0 |
| confidence_oof | fold_1 | 2 | 140.500000 | 300.000000 | 248 | 312.000000 | 359.000000 | 482.500000 | True | False | 101 | 174 | completed_residual_history_only | 0 | 0 |
| confidence_oof | fold_1 | 3 | 140.500000 | 471.000000 | 414 | 483.000000 | 483.000000 | 655.500000 | True | False | 110 | 174 | completed_residual_history_only | 0 | 0 |
| confidence_oof | fold_1 | 4 | 140.500000 | 644.000000 | 589 | 656.000000 | 656.000000 | 743.000000 | True | False | 110 | 174 | completed_residual_history_only | 0 | 0 |
| confidence_oof | fold_1 | 5 | 140.500000 | 731.500000 | 764 | 743.500000 | 743.500000 | 930.000000 | True | False | 110 | 174 | completed_residual_history_only | 0 | 0 |
| ridge_oof | fold_2 | 1 | 97.000000 | 128.500000 | 64 | 140.500000 | 140.500000 | 295.500000 | True | False | 23 | 87 | completed_fit_history_only | 0 | 0 |
| ridge_oof | fold_2 | 2 | 97.000000 | 284.000000 | 303 | 296.000000 | 296.000000 | 503.000000 | True | False | 23 | 87 | completed_fit_history_only | 0 | 0 |
| ridge_oof | fold_2 | 3 | 97.000000 | 491.500000 | 542 | 503.500000 | 503.500000 | 708.000000 | True | False | 23 | 87 | completed_fit_history_only | 0 | 0 |
| ridge_oof | fold_2 | 4 | 97.000000 | 696.500000 | 781 | 708.500000 | 708.500000 | 927.500000 | True | False | 23 | 87 | completed_fit_history_only | 0 | 0 |
| ridge_oof | fold_2 | 5 | 97.000000 | 916.000000 | 1020 | 928.000000 | 928.000000 | 1118.500000 | True | False | 23 | 87 | completed_fit_history_only | 0 | 0 |
| confidence_oof | fold_2 | 1 | 140.500000 | 172.000000 | 64 | 184.000000 | 184.000000 | 382.000000 | True | False | 110 | 174 | completed_residual_history_only | 0 | 0 |
| confidence_oof | fold_2 | 2 | 140.500000 | 370.500000 | 286 | 382.500000 | 382.500000 | 529.000000 | True | False | 110 | 174 | completed_residual_history_only | 0 | 0 |
| confidence_oof | fold_2 | 3 | 140.500000 | 517.500000 | 507 | 529.500000 | 529.500000 | 725.500000 | True | False | 110 | 174 | completed_residual_history_only | 0 | 0 |
| confidence_oof | fold_2 | 4 | 140.500000 | 714.000000 | 729 | 726.000000 | 726.000000 | 1008.000000 | True | False | 110 | 174 | completed_residual_history_only | 0 | 0 |
| confidence_oof | fold_2 | 5 | 140.500000 | 930.000000 | 961 | 942.000000 | 1008.500000 | 1118.500000 | True | False | 99 | 174 | completed_residual_history_only | 0 | 0 |
| ridge_oof | fold_3 | 1 | 97.000000 | 128.500000 | 64 | 140.500000 | 140.500000 | 371.500000 | True | False | 23 | 87 | completed_fit_history_only | 0 | 0 |
| ridge_oof | fold_3 | 2 | 97.000000 | 360.000000 | 352 | 372.000000 | 372.000000 | 552.000000 | True | False | 23 | 87 | completed_fit_history_only | 0 | 0 |
| ridge_oof | fold_3 | 3 | 97.000000 | 540.500000 | 640 | 552.500000 | 552.500000 | 781.500000 | True | False | 23 | 87 | completed_fit_history_only | 0 | 0 |
| ridge_oof | fold_3 | 4 | 97.000000 | 770.000000 | 928 | 782.000000 | 782.000000 | 1097.500000 | True | False | 23 | 87 | completed_fit_history_only | 0 | 0 |
| ridge_oof | fold_3 | 5 | 97.000000 | 1086.000000 | 1216 | 1098.000000 | 1098.000000 | 1371.000000 | True | False | 23 | 87 | completed_fit_history_only | 0 | 0 |
| confidence_oof | fold_3 | 1 | 140.500000 | 172.000000 | 64 | 184.000000 | 184.000000 | 406.500000 | True | False | 110 | 174 | completed_residual_history_only | 0 | 0 |
| confidence_oof | fold_3 | 2 | 140.500000 | 395.000000 | 335 | 407.000000 | 407.000000 | 663.500000 | True | False | 110 | 174 | completed_residual_history_only | 0 | 0 |
| confidence_oof | fold_3 | 3 | 140.500000 | 652.000000 | 605 | 664.000000 | 664.000000 | 835.000000 | True | False | 110 | 174 | completed_residual_history_only | 0 | 0 |
| confidence_oof | fold_3 | 4 | 140.500000 | 794.000000 | 889 | 806.000000 | 835.500000 | 1106.000000 | True | False | 97 | 174 | completed_residual_history_only | 0 | 0 |
| confidence_oof | fold_3 | 5 | 140.500000 | 1094.500000 | 1146 | 1106.500000 | 1106.500000 | 1371.000000 | True | False | 110 | 174 | completed_residual_history_only | 0 | 0 |
