# Future-condition reconstruction

MR replaces only P2 q with the fixed TR2-B-style 10-minute-history Ridge prediction. It preserves P2 conditional on-current mean and standard-deviation predictions; total current mean and population standard deviation are recomputed with the inherited mixture identities. Peak-to-peak remains P2 independent Ridge. Current-density mean/std are deterministically derived with the frozen active area.

## Variable-level P2-to-MR reconstruction audit

| fold | feature | reconstruction_rule | mean_signed_change | mean_absolute_change | max_absolute_change |
| --- | --- | --- | --- | --- | --- |
| fold_1 | actual_current_A_mean | recomputed from MR q and preserved P2 conditional on-current mean | -1.0406410319769637 | 2.0233465056540685 | 90.1620412187633 |
| fold_1 | actual_current_A_std | recomputed from MR q and preserved P2 conditional on-current standard deviation | 1.5673910188880624 | 1.9167967054221888 | 30.087602750891918 |
| fold_1 | actual_current_A_peak_to_peak | preserved P2 independent Ridge prediction | 0.0 | 0.0 | 0.0 |
| fold_1 | actual_current_density_A_cm2_mean | derived from reconstructed current mean through J=I/A | -0.0037165583644275025 | 0.007226204953150696 | 0.32200583885190287 |
| fold_1 | actual_current_density_A_cm2_std | derived from reconstructed current standard deviation through J=I/A | 0.005597799839307054 | 0.006845671667309017 | 0.10745523983132221 |
| fold_1 | actual_current_density_A_cm2_peak_to_peak | preserved P2 independent Ridge prediction | 0.0 | 0.0 | 0.0 |
| fold_1 | load_transition_count | preserved P2 auxiliary prediction | 0.0 | 0.0 | 0.0 |
| fold_1 | operation_point_transitions | preserved P2 auxiliary prediction | 0.0 | 0.0 | 0.0 |
| fold_1 | zero_current_fraction | replaced by direct MR 10-minute-history Ridge prediction | 0.004699877933726854 | 0.010084909816868942 | 0.5894411801507388 |
| fold_2 | actual_current_A_mean | recomputed from MR q and preserved P2 conditional on-current mean | -1.1770582105619474 | 2.154436372543478 | 29.81438604849066 |
| fold_2 | actual_current_A_std | recomputed from MR q and preserved P2 conditional on-current standard deviation | 1.9446451062928196 | 2.1272975989926697 | 27.942044533891917 |
| fold_2 | actual_current_A_peak_to_peak | preserved P2 independent Ridge prediction | 0.0 | 0.0 | 0.0 |
| fold_2 | actual_current_density_A_cm2_mean | derived from reconstructed current mean through J=I/A | -0.004203754400363098 | 0.007694369998114306 | 0.10647931888231005 |
| fold_2 | actual_current_density_A_cm2_std | derived from reconstructed current standard deviation through J=I/A | 0.006945119917918262 | 0.007597446381498774 | 0.09979242454662604 |
| fold_2 | actual_current_density_A_cm2_peak_to_peak | preserved P2 independent Ridge prediction | 0.0 | 0.0 | 0.0 |
| fold_2 | load_transition_count | preserved P2 auxiliary prediction | 0.0 | 0.0 | 0.0 |
| fold_2 | operation_point_transitions | preserved P2 auxiliary prediction | 0.0 | 0.0 | 0.0 |
| fold_2 | zero_current_fraction | replaced by direct MR 10-minute-history Ridge prediction | 0.005141667559468494 | 0.011283978857477396 | 0.19683166182973366 |
| fold_3 | actual_current_A_mean | recomputed from MR q and preserved P2 conditional on-current mean | -0.8845443601801023 | 2.6570372732156913 | 67.74394353603965 |
| fold_3 | actual_current_A_std | recomputed from MR q and preserved P2 conditional on-current standard deviation | 1.956420658091685 | 2.288641060854186 | 36.51103478742309 |
| fold_3 | actual_current_A_peak_to_peak | preserved P2 independent Ridge prediction | 0.0 | 0.0 | 0.0 |
| fold_3 | actual_current_density_A_cm2_mean | derived from reconstructed current mean through J=I/A | -0.0031590675651355514 | 0.009489360451592098 | 0.24194116699303 |
| fold_3 | actual_current_density_A_cm2_std | derived from reconstructed current standard deviation through J=I/A | 0.006987173648905731 | 0.00817366778768425 | 0.13039575057942768 |
| fold_3 | actual_current_density_A_cm2_peak_to_peak | preserved P2 independent Ridge prediction | 0.0 | 0.0 | 0.0 |
| fold_3 | load_transition_count | preserved P2 auxiliary prediction | 0.0 | 0.0 | 0.0 |
| fold_3 | operation_point_transitions | preserved P2 auxiliary prediction | 0.0 | 0.0 | 0.0 |
| fold_3 | zero_current_fraction | replaced by direct MR 10-minute-history Ridge prediction | 0.0037781652278185388 | 0.01379452351397876 | 0.44745231763463383 |
