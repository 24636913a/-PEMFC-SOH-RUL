# P1-to-P2 paired error attribution

P1/P2 share the same TimeXer checkpoint, history, origin, target and B term. `DeltaC=C_P2-C_P1` is therefore the only forward-input-related difference. The identity `D=e_P2²-e_P1²=2e_P1DeltaC+DeltaC²` passed under the frozen tolerance. Overlapping windows are reported as prediction points; independent target timestamps are separately counted.

| fold | regime_strict | state_group_q_derived | impact | prediction_point_count | unique_target_timestamp_count | P1_RMSE_mV | P2_RMSE_mV | mean_D_mV2 | net_D_sum_mV2 | positive_D_sum_mV2 | negative_D_sum_mV2 | mean_abs_deltaC_mV | p95_abs_deltaC_mV |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| fold_1 | 0<q<1 | both_mixed | P2_improves | 783 | 53 | 37.9512 | 17.7117 | -1126.59 | -882117 | 0 | -882117 | 15.3047 | 63.544 |
| fold_1 | q=0 | both_nonmixed | P2_improves | 2267 | 199 | 21.05 | 11.2624 | -316.259 | -716959 | 0 | -716959 | 3.92693 | 12.4982 |
| fold_1 | 0<q<1 | both_mixed | P2_harms | 389 | 46 | 12.7147 | 17.3283 | 138.607 | 53918.1 | 53918.1 | 0 | 6.39125 | 15.8335 |
| fold_1 | q=0 | both_nonmixed | P2_harms | 2128 | 197 | 10.1881 | 11.4518 | 27.3464 | 58193.2 | 58193.2 | 0 | 1.59799 | 5.38452 |
| fold_1 | q=0 | pred_mixed_true_nonmixed | P2_harms | 25 | 2 | 94.6877 | 180.787 | 23718.1 | 592953 | 592953 | 0 | 69.9317 | 72.6811 |
| fold_2 | q=0 | both_nonmixed | P2_improves | 2584 | 208 | 24.0389 | 11.1236 | -454.134 | -1.17348e+06 | 0 | -1.17348e+06 | 5.57889 | 12.2692 |
| fold_2 | 0<q<1 | both_mixed | P2_improves | 700 | 56 | 43.2546 | 19.31 | -1498.08 | -1.04866e+06 | 0 | -1.04866e+06 | 21.5768 | 97.5958 |
| fold_2 | 0<q<1 | pred_nonmixed_true_mixed | P2_harms | 1 | 1 | 208.655 | 293.67 | 42704.9 | 42704.9 | 42704.9 | 0 | 85.0146 | 85.0146 |
| fold_2 | q=0 | pred_mixed_true_nonmixed | P2_harms | 2 | 1 | 236.74 | 296.694 | 31981.3 | 63962.7 | 63962.7 | 0 | 59.9462 | 64.7132 |
| fold_2 | q=0 | both_nonmixed | P2_harms | 2060 | 208 | 9.81786 | 12.3277 | 55.5817 | 114498 | 114498 | 0 | 1.80754 | 5.71846 |
| fold_2 | 0<q<1 | both_mixed | P2_harms | 533 | 47 | 18.4477 | 25.5071 | 310.293 | 165386 | 165386 | 0 | 7.59735 | 15.8243 |
| fold_3 | 0<q<1 | both_mixed | P2_improves | 1278 | 109 | 45.8168 | 21.1865 | -1650.31 | -2.1091e+06 | 0 | -2.1091e+06 | 22.6353 | 103.777 |
| fold_3 | q=0 | both_nonmixed | P2_improves | 4799 | 434 | 21.5501 | 11.8797 | -323.279 | -1.55141e+06 | 0 | -1.55141e+06 | 5.05436 | 11.9415 |
| fold_3 | q=0 | both_nonmixed | P2_harms | 3985 | 430 | 9.10405 | 10.9896 | 37.8878 | 150983 | 150983 | 0 | 1.61522 | 5.51208 |
| fold_3 | 0<q<1 | both_mixed | P2_harms | 1043 | 107 | 18.4655 | 29.1719 | 510.023 | 531954 | 531954 | 0 | 11.6947 | 31.9179 |
| fold_3 | 0<q<1 | pred_nonmixed_true_mixed | P2_harms | 19 | 1 | 192.128 | 286.524 | 45183 | 858477 | 858477 | 0 | 94.6462 | 106.483 |
| fold_3 | q=0 | pred_mixed_true_nonmixed | P2_harms | 60 | 6 | 196.953 | 319.991 | 63604 | 3.81624e+06 | 3.81624e+06 | 0 | 122.22 | 211.917 |
