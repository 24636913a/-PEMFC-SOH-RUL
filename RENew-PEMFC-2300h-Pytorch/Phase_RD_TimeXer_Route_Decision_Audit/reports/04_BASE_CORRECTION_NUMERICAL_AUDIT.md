# Base-correction numerical audit

B and C have large mV-scale values and are opposite-signed at roughly half of within-horizon prediction points; the predeclared large-amplitude cancellation flag occurs at about one tenth. This is numerical compensation within a jointly trained forward pass, not evidence that B is an independently trained history-only model. The verified squared expansion is descriptive only: `(B+C-y)^2=(B-y)^2+2(B-y)C+C^2`; it is not a physical attribution of independent error sources.

B/C numerical relationship:

| fold | opposite_sign_fraction | large_amplitude_cancellation_fraction | mean_B_C_pearson |
|---|---|---|---|
| fold_1 | 0.483369 | 0.116416 | 0.177163 |
| fold_2 | 0.495068 | 0.0993197 | 0.22668 |
| fold_3 | 0.467722 | 0.0994278 | 0.253242 |

Per-fold/horizon statistics are in `outputs/base_correction_statistics.csv`; residual matching and squared expansion checks are in `outputs/base_residual_analysis.csv`.

| fold | B_only_rmse_mV | final_rmse_mV | corr_residualB_C | squared_expansion_max_residual_mV2 |
|---|---|---|---|---|
| fold_1 | 189.224 | 16.8782 | 0.9908 | 9.07221e-11 |
| fold_2 | 189.581 | 15.4251 | 0.996905 | 8.20819e-11 |
| fold_3 | 189.658 | 30.0485 | 0.983231 | 8.57199e-11 |
