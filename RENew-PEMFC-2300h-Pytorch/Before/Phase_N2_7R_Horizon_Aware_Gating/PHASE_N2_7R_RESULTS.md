# Phase N2.7R Results — Horizon-Aware Adaptive Gating

## Development Meta-OOS comparison

| Method | Fold2 RMSE (mV) | Fold3 RMSE (mV) | Meta-OOS RMSE (mV) | MAE (mV) | R² |
|---|---|---|---|---|---|
| LSTM | 5.752 | 11.711 | 8.731 | 4.596 | 0.99533 |
| TimesNet | 5.994 | 21.445 | 13.719 | 4.962 | 0.98658 |
| Fixed .25 | 4.949 | 12.037 | 8.493 | 4.040 | 0.99534 |
| Horizon Static | 5.831 | 17.116 | 11.474 | 4.238 | 0.99114 |
| Gate_A_Horizon_Logistic | 5.007 | 14.973 | 9.990 | 3.856 | 0.99324 |
| Gate_B_Disagreement_Logistic | 5.729 | 21.033 | 13.381 | 4.397 | 0.98714 |
| Gate_C_Continuous_MLP | 5.548 | 17.211 | 11.380 | 4.083 | 0.99111 |

| Gate | AUC | Mean g | Std g | RMSE improvement vs fixed |
|---|---|---|---|---|
| Gate_A_Horizon_Logistic | 0.550 | 0.494 | 0.187 | -17.62% |
| Gate_B_Disagreement_Logistic | 0.533 | 0.477 | 0.230 | -57.55% |
| Gate_C_Continuous_MLP | 0.546 | 0.607 | 0.137 | -33.99% |

## Protocol and firewall

- All gates use frozen expert caches only. LSTM is `P2_causal_sg`; TimesNet is high-fidelity raw `T0_original`.
- Meta-OOS has exactly Fold1→Fold2 and Fold1+Fold2→Fold3 chronological evaluations, averaged over seeds 42/2024/3407. Training-fold performance is not reported as OOS.
- Final Test `[1819.997738, 2256.800000] h` was not used: target rows read = 0; Test expert predictions = false; Test gate predictions/weights = false; Test metrics = false.
- Pointwise winner means `|e_TimesNet| < |e_LSTM|`; equality is assigned to LSTM. Gate C optimizes raw-voltage fusion MSE directly and never updates either frozen expert.

## Findings

- Best main adaptive gate is **Gate_A_Horizon_Logistic**: Meta-OOS RMSE=9.990 mV, MAE=3.856 mV, R²=0.99324; relative improvements are -14.41% vs LSTM, -17.62% vs fixed .25, and 12.93% vs Horizon-Static.
- Winner AUC=0.550; mean/std gate weight=0.494/0.187; at 9.5 h mean weight=0.482; TimesNet top-10% error-tail mean weight=0.452.
- Horizon-Static is more than 1% of the adaptive gate on RMSE. State/disagreement information adds material performance beyond horizon-only weights.
- OpHrs2 is a diagnostic-only ablation: its Meta-OOS change is -1.197 mV; it does not replace the main protocol.
- Causal SG paired moving-block bootstrap (2,000 repetitions, 19 origins) gives SG−raw RMSE delta -0.478 mV, 95% CI [-1.355, 0.504] mV. Per-seed/fold rows explicitly retain any 2024/Fold3 or late-region worsening rather than averaging it away.

## Decision

Phase N2.7R status: **PASS**.

Phase N2.8 decision: **NOGO**. Pre-registered checks: adaptive vs fixed ≥2%=False; Fold3 no worse than fixed=False; adaptive vs LSTM ≥5%=False; non-degenerate std≥.01=True. No Test evaluation was started.
