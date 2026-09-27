# Overlap weighting audit

**SIGN_REVERSAL_CONFIRMED**

`forecast_query_weighted_RMSE` remains the formal historical metric. `target_time_equal_weight_RMSE` is a mandatory robustness sensitivity only; neither result selects a model. All P2−P0 attribution is in additive MSE space, not summed RMSE differences. No target times are treated as independent samples.

## Three-fold macro metrics

| fold | evaluation_unit | unit_count | p0_mse_mV2 | p2_mse_mV2 | p2_minus_p0_delta_mse_mV2 | p0_RMSE_mV | p2_RMSE_mV | p2_minus_p0_delta_RMSE_mV |
|---|---|---|---|---|---|---|---|---|
| three_fold_macro | forecast_query_weighted_RMSE | 22656 | 608.368620 | 496.963972 | -111.404649 | 24.538245 | 21.346566 | -3.191680 |
| three_fold_macro | target_time_equal_weight_RMSE | 1082 | 919.383311 | 1242.107905 | 322.724594 | 30.262011 | 34.997329 | 4.735318 |

## Per-fold metrics

| fold | evaluation_unit | unit_count | p0_mse_mV2 | p2_mse_mV2 | p2_minus_p0_delta_mse_mV2 | p0_RMSE_mV | p2_RMSE_mV | p2_minus_p0_delta_RMSE_mV |
|---|---|---|---|---|---|---|---|---|
| fold_1 | forecast_query_weighted_RMSE | 5592 | 478.892516 | 312.260474 | -166.632042 | 21.883613 | 17.670893 | -4.212720 |
| fold_1 | target_time_equal_weight_RMSE | 256 | 800.953234 | 1488.965841 | 688.012606 | 28.301117 | 38.587120 | 10.286003 |
| fold_2 | forecast_query_weighted_RMSE | 5880 | 568.622570 | 255.591633 | -313.030937 | 23.845808 | 15.987233 | -7.858575 |
| fold_2 | target_time_equal_weight_RMSE | 268 | 879.659442 | 850.767321 | -28.892120 | 29.659053 | 29.167916 | -0.491137 |
| fold_3 | forecast_query_weighted_RMSE | 11184 | 777.590775 | 923.039807 | 145.449032 | 27.885315 | 30.381570 | 2.496255 |
| fold_3 | target_time_equal_weight_RMSE | 558 | 1077.537258 | 1386.590554 | 309.053296 | 32.825863 | 37.236951 | 4.411089 |

## Multiplicity contributions

| fold | multiplicity | query_point_count | target_time_count | forecast_query_weighted_delta_MSE_contribution_mV2 | target_time_equal_weight_delta_MSE_contribution_mV2 |
|---|---|---|---|---|---|
| fold_1 | 1 | 2 | 2 | 49.003519 | 1070.420620 |
| fold_1 | 2 | 4 | 2 | -17.888626 | -195.377339 |
| fold_1 | 3 | 6 | 2 | 0.004182 | 0.030453 |
| fold_1 | 4 | 8 | 2 | 0.000528 | 0.002883 |
| fold_1 | 5 | 10 | 2 | 0.009805 | 0.042835 |
| fold_1 | 6 | 12 | 2 | -0.013563 | -0.049376 |
| fold_1 | 7 | 14 | 2 | -0.020601 | -0.064287 |
| fold_1 | 8 | 16 | 2 | 0.156410 | 0.427071 |
| fold_1 | 9 | 18 | 2 | 0.117087 | 0.284180 |
| fold_1 | 10 | 20 | 2 | 0.007028 | 0.015352 |
| fold_1 | 11 | 22 | 2 | -0.179327 | -0.356106 |
| fold_1 | 12 | 24 | 2 | -0.003900 | -0.007098 |
| fold_1 | 13 | 26 | 2 | -1.372069 | -2.305472 |
| fold_1 | 14 | 28 | 2 | -6.155593 | -9.604374 |
| fold_1 | 15 | 30 | 2 | -0.166191 | -0.242016 |
| fold_1 | 16 | 32 | 2 | 0.001643 | 0.002243 |
| fold_1 | 17 | 34 | 2 | -0.144299 | -0.185414 |
| fold_1 | 18 | 36 | 2 | -0.097164 | -0.117913 |
| fold_1 | 19 | 38 | 2 | -0.098927 | -0.113734 |
| fold_1 | 20 | 40 | 2 | -2.829768 | -3.090637 |
| fold_1 | 21 | 42 | 2 | -11.821059 | -12.296013 |
| fold_1 | 22 | 44 | 2 | 0.048923 | 0.048576 |
| fold_1 | 23 | 46 | 2 | -0.037476 | -0.035592 |
| fold_1 | 24 | 5040 | 210 | -175.152603 | -159.416236 |
| fold_2 | 1 | 2 | 2 | 7.258812 | 159.260509 |
| fold_2 | 2 | 4 | 2 | 10.877196 | 119.324463 |
| fold_2 | 3 | 6 | 2 | -0.019253 | -0.140808 |
| fold_2 | 4 | 8 | 2 | -0.142050 | -0.779154 |
| fold_2 | 5 | 10 | 2 | 0.001207 | 0.005296 |
| fold_2 | 6 | 12 | 2 | -0.091495 | -0.334571 |
| fold_2 | 7 | 14 | 2 | -0.006318 | -0.019804 |
| fold_2 | 8 | 16 | 2 | -1.283151 | -3.519089 |
| fold_2 | 9 | 18 | 2 | -0.090411 | -0.220404 |
| fold_2 | 10 | 20 | 2 | -0.057468 | -0.126087 |
| fold_2 | 11 | 22 | 2 | -0.002832 | -0.005648 |
| fold_2 | 12 | 24 | 2 | -0.012014 | -0.021966 |
| fold_2 | 13 | 26 | 2 | -0.581303 | -0.981073 |
| fold_2 | 14 | 28 | 2 | -0.012923 | -0.020252 |
| fold_2 | 15 | 30 | 2 | -0.307943 | -0.450424 |
| fold_2 | 16 | 32 | 2 | -0.058046 | -0.079597 |
| fold_2 | 17 | 34 | 2 | -0.038095 | -0.049166 |
| fold_2 | 18 | 36 | 2 | -0.248967 | -0.303467 |
| fold_2 | 19 | 38 | 2 | -0.019953 | -0.023040 |
| fold_2 | 20 | 40 | 2 | -0.849981 | -0.932442 |
| fold_2 | 21 | 42 | 2 | -1.516673 | -1.584584 |
| fold_2 | 22 | 44 | 2 | -0.012011 | -0.011978 |
| fold_2 | 23 | 46 | 2 | -0.591185 | -0.563947 |
| fold_2 | 24 | 5328 | 222 | -325.226080 | -297.314886 |
| fold_3 | 1 | 8 | 8 | -8.357687 | -167.513204 |
| fold_3 | 2 | 16 | 8 | 34.783365 | 348.581684 |
| fold_3 | 3 | 24 | 8 | -0.015878 | -0.106082 |
| fold_3 | 4 | 32 | 8 | -0.034212 | -0.171427 |
| fold_3 | 5 | 40 | 8 | -0.584555 | -2.343247 |
| fold_3 | 6 | 48 | 8 | -0.158064 | -0.528013 |
| fold_3 | 7 | 56 | 8 | 0.277456 | 0.794435 |
| fold_3 | 8 | 64 | 8 | 0.140907 | 0.353025 |
| fold_3 | 9 | 72 | 8 | -0.646980 | -1.440824 |
| fold_3 | 10 | 80 | 8 | 0.027539 | 0.055197 |
| fold_3 | 11 | 88 | 8 | 0.140300 | 0.255639 |
| fold_3 | 12 | 96 | 8 | -0.388211 | -0.648410 |
| fold_3 | 13 | 104 | 8 | -0.190256 | -0.293331 |
| fold_3 | 14 | 112 | 8 | 0.516336 | 0.739209 |
| fold_3 | 15 | 120 | 8 | 0.075250 | 0.100549 |
| fold_3 | 16 | 128 | 8 | -0.672766 | -0.842765 |
| fold_3 | 17 | 136 | 8 | 0.086687 | 0.102204 |
| fold_3 | 18 | 144 | 8 | -0.011373 | -0.012664 |
| fold_3 | 19 | 152 | 8 | 83.104685 | 87.666742 |
| fold_3 | 20 | 160 | 8 | 76.610921 | 76.775676 |
| fold_3 | 21 | 168 | 8 | 2.617117 | 2.497853 |
| fold_3 | 22 | 176 | 8 | 0.002758 | 0.002513 |
| fold_3 | 23 | 184 | 8 | -0.031950 | -0.027842 |
| fold_3 | 24 | 8976 | 374 | -41.842361 | -34.943620 |

## Main horizon contributions

| fold | horizon_step | query_point_count | target_time_count | forecast_query_weighted_delta_MSE_contribution_mV2 | target_time_equal_weight_delta_MSE_contribution_mV2 | abs_q | abs_t |
|---|---|---|---|---|---|---|---|
| fold_1 | 24 | 233 | 233 | 29.025062 | 958.385496 | 29.025062 | 958.385496 |
| fold_1 | 23 | 233 | 233 | -20.533520 | -104.631692 | 20.533520 | 104.631692 |
| fold_1 | 21 | 233 | 233 | -13.703876 | -13.075616 | 13.703876 | 13.075616 |
| fold_1 | 19 | 233 | 233 | -12.136144 | -11.601662 | 12.136144 | 11.601662 |
| fold_1 | 22 | 233 | 233 | -11.869894 | -11.415001 | 11.869894 | 11.415001 |
| fold_2 | 20 | 245 | 245 | -18.047837 | -16.661856 | 18.047837 | 16.661856 |
| fold_2 | 13 | 245 | 245 | -17.156557 | -15.803426 | 17.156557 | 15.803426 |
| fold_2 | 2 | 245 | 245 | -17.122821 | -16.374386 | 17.122821 | 16.374386 |
| fold_2 | 15 | 245 | 245 | -16.342604 | -15.000432 | 16.342604 | 15.000432 |
| fold_2 | 9 | 245 | 245 | -16.170994 | -14.989178 | 16.170994 | 14.989178 |
| fold_3 | 3 | 466 | 466 | 30.986544 | 27.197555 | 30.986544 | 27.197555 |
| fold_3 | 1 | 466 | 466 | 29.505720 | 25.693642 | 29.505720 | 25.693642 |
| fold_3 | 23 | 466 | 466 | 25.654024 | 163.854372 | 25.654024 | 163.854372 |
| fold_3 | 4 | 466 | 466 | 22.400936 | 20.087706 | 22.400936 | 20.087706 |
| fold_3 | 2 | 466 | 466 | 17.158639 | 15.267987 | 17.158639 | 15.267987 |

## Main pre-existing contiguous target-row segments

| fold | time_segment | query_point_count | target_time_count | forecast_query_weighted_delta_MSE_contribution_mV2 | target_time_equal_weight_delta_MSE_contribution_mV2 | abs_q | abs_t |
|---|---|---|---|---|---|---|---|
| fold_1 | fold_1_segment_0 | 5592 | 256 | -166.632042 | 688.012606 | 166.632042 | 688.012606 |
| fold_2 | fold_2_segment_0 | 5880 | 268 | -313.030937 | -28.892120 | 313.030937 | 28.892120 |
| fold_3 | fold_3_segment_3 | 1152 | 71 | 152.926085 | 159.279789 | 152.926085 | 159.279789 |
| fold_3 | fold_3_segment_1 | 1872 | 101 | -61.906870 | -52.378222 | 61.906870 | 52.378222 |
| fold_3 | fold_3_segment_2 | 6048 | 275 | 55.491943 | 44.986280 | 55.491943 | 44.986280 |
| fold_3 | fold_3_segment_0 | 2112 | 111 | -1.062126 | 157.165449 | 1.062126 | 157.165449 |

The historical H0 audit tagged `historical_full_artifact` only because coverage was 100%; this phase verifies the same condition for P0/P2. H0's old script used a strict historical-HEAD reproduction guard. This new script intentionally instead requires the specified start commit to be an ancestor of the current HEAD, while pinning every historical input SHA256.
