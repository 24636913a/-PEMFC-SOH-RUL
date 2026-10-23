# Active Cell Mask

Status: **CONFIRMED**

Decision scope: Train + Validation only (`OpHrs2 < 1819.997738 h`). Test distributions were not used.

U6 [V] remains non-physical in Train+Validation and excluding it gives the closest reconstruction of U.S.AveCell [V]. The active mask is frozen to 10 channels.

Active channels: `U1 [V], U2 [V], U3 [V], U4 [V], U5 [V], U7 [V], U8 [V], U9 [V], U10 [V], U11 [V]`

Inactive channel: `U6 [V]`

Best single-channel exclusion against exported average: `U6 [V]`

All-11 mean MAE against exported average: `0.0535108 V`

Quantiles use a deterministic every-10th-valid-observation sample; mean, standard deviation, zero/negative ratios and flatline ratios are streaming calculations over all eligible rows.

## Channel summary

| channel | median | mean | std | p1 | p99 | zero_ratio | negative_ratio | flatline_ratio | loaded_median_V | excluded_mean_vs_exported_mae_V | is_active |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| U1 [V] | 0.70748001 | 0.57895462 | 0.28118064 | 0.001526 | 0.84469098 | 0.0 | 1.327e-05 | 0.17846293 | 0.73134601 | 0.05880865 | True |
| U2 [V] | 0.70778501 | 0.5785286 | 0.28144212 | 0.00116 | 0.84212798 | 4.5e-07 | 1.371e-05 | 0.16657762 | 0.73158997 | 0.05876599 | True |
| U3 [V] | 0.70955497 | 0.57959313 | 0.28250835 | -0.000366 | 0.84297651 | 0.00428322 | 0.02745959 | 0.17945712 | 0.732566 | 0.05887249 | True |
| U4 [V] | 0.70937198 | 0.57961549 | 0.28221046 | 0.000305 | 0.84224999 | 0.00140697 | 0.00104367 | 0.17786186 | 0.73293298 | 0.05887456 | True |
| U5 [V] | 0.70656502 | 0.57801041 | 0.28094526 | 0.001709 | 0.844935 | 1.79e-06 | 2.013e-05 | 0.1756317 | 0.73158997 | 0.05871394 | True |
| U6 [V] | -0.008728 | -0.00907194 | 0.00558535 | -0.023865 | -0.000427 | 4.32e-06 | 0.99994141 | 0.56724005 | -0.009095 | 6.115e-05 | False |
| U7 [V] | 0.70863998 | 0.57985376 | 0.28154848 | 0.001282 | 0.84444702 | 3e-07 | 6.26e-06 | 0.16614649 | 0.73220003 | 0.05889816 | True |
| U8 [V] | 0.71083701 | 0.58131794 | 0.28193639 | 0.00177 | 0.84560698 | 1.5e-07 | 6e-07 | 0.16603946 | 0.73366499 | 0.05904448 | True |
| U9 [V] | 0.71059299 | 0.58127302 | 0.28185704 | 0.002136 | 0.84627801 | 0.0 | 3.13e-06 | 0.17622979 | 0.734092 | 0.05903998 | True |
| U10 [V] | 0.71095902 | 0.58080847 | 0.28263788 | 0.001099 | 0.84353203 | 4.5e-07 | 3.43e-06 | 0.18061126 | 0.733787 | 0.05899362 | True |
| U11 [V] | 0.70522201 | 0.57686367 | 0.28054573 | 0.00116 | 0.84572899 | 7.5e-07 | 4.17e-06 | 0.17720623 | 0.73036897 | 0.05859911 | True |
