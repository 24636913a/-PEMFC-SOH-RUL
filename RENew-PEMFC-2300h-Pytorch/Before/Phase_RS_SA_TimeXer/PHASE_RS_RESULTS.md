# Phase RS-SA-TimeXer final results

## Protocol and firewall

This phase starts from commit `39300ca08e7a54f8060978891650bc49920f3278`. It uses the fixed RealHyFC Development protocol: 0.5 h sampling, 48-bin (24 h) history, 24-bin (12 h) horizon, DeltaV target, F2 history (14), nine future-condition variables, and three chronological expanding folds. No Test record, feature, target, prediction, metric, distribution, or selection signal was accessed; see `outputs/test_firewall_audit.json`.

The RS rule was frozen before execution: at each horizon, `g = 1[p_mixed >= 0.5]` from the causal P2 predicted mixed-current probability, and the entire nine-variable vector is selected as `Z_RS=(1-g)Z_P1+gZ_P2`. There was no threshold tuning, no extra learned selector, no CAF, and no new backbone.

## Answers to the required questions

1. **Does the selector use both P1 and P2?** Yes. Validation P2-selection fractions were 0.214, 0.210, 0.213 in folds 1–3; therefore it is nonconstant.
2. **Prediction-state versus selected-input disagreement?** Yes, and it is explicit rather than hidden. Among `g=0` validation selections, P1 retained `q>0` at rates 0.498, 0.503, 0.522 in folds 1–3. `g=0` means “choose P1”, not “the P1 q value equals zero”.
3. **RS physical violation rate?** Zero in every fold and in the unweighted mean. P1 and P2 are also zero.
4. **P0/P1/P2 reproduction?** Passed. P0 = 24.538245 mV (R² 0.983989); P1 = 24.442880 mV (R² 0.984111); P2 = 21.346566 mV (R² 0.986918). P0 and P2 match the frozen G-STOP references within ±0.001 mV.
5. **Frozen RS performance?** 24.201794 mV, R² 0.983944.
6. **RS versus P0?** 1.371% RMSE improvement, below the required 5%.
7. **RS versus P2?** -13.376% (negative means worse): RS is 13.376% worse than P2, exceeding the allowed 3% degradation.
8. **Fold 3 overall?** RS = 31.168859 mV versus P0 = 27.885315 mV; it is worse and fails the non-worsening rule.
9. **Fold 3 continuous-nonzero?** RS = 35.798820 mV versus P2 = 32.812164 mV; it worsens the P2 negative-transfer regime rather than repairing it.
10. **Mixed-current benefit?** Preserved against P0 in all folds. RS mixed-current RMSE is 17.585370, 22.974288, 30.400431 mV versus P0 31.870978, 35.264231, 39.976872 mV.
11. **9.5 h?** RS = 23.367266 mV; better than P0 (26.074392) but worse than P2 (19.864299).
12. **12 h?** RS = 37.145374 mV, versus P0 33.527982; this is 10.789% worse and fails the 5% limit.
13. **P95/P99?** Pooled RS P95/P99 = 28.429244/106.656632 mV. P99 improves on P0 (129.651590) but is worse than P2 (68.630090).
14. **Maximum error?** RS maximum = 593.411967 mV, equal to the P2 maximum and higher than P0 (368.024498).
15. **Bootstrap?** RS−P0 using 48-origin paired blocks is -0.336452 mV, 95% CI [-1.720093, 1.129820], which crosses zero: `IMPROVEMENT_NOT_STATISTICALLY_RESOLVED`. RS−P2 is 2.855228 mV, CI [0.409411, 5.289017], conclusively unfavorable to RS in this Development audit.
16. **Gate A?** `GATE_A_FAIL`.
17. **Formal retraining?** `NOT_RUN`, prohibited by the Gate A stop rule.
18. **Retrained comparison?** `NOT_RUN`; no numbers were substituted for missing experiments.
19. **Gate B?** `NOT_RUN`, because retraining was not authorized.
20. **Three seeds?** `NOT_RUN`, because Gate B was not reached.
21. **Three-seed stability?** `NOT_RUN`.
22. **Physical contribution verified?** Yes for future-condition validity: P1/P2/RS all satisfy the audited non-negativity, current-density identity, q-range and q=1 compatibility constraints. This does not establish a new voltage mechanism.
23. **Selective-routing contribution verified?** Partially at the frozen mixed-current slice only, but not as an overall robust mechanism: it retains mixed-current benefits yet worsens Fold 3 continuous-nonzero and overall Fold 3 performance.
24. **LSTM/TimesNet gap?** `INCONCLUSIVE` in this phase. No new external-model evaluation was authorized; RS was screened solely as a causal future-condition routing change on the frozen SA-TimeXer checkpoint.
25. **New final paper candidate?** No. `NOT_ESTABLISHED` because Gate A failed and no retraining/multi-seed evidence is authorized.
26. **Test sealed?** Yes. The Test firewall is intact.

## Frozen fold result

| Model | Fold 1 RMSE (mV) | Fold 2 RMSE (mV) | Fold 3 RMSE (mV) | Mean RMSE (mV) |
|---|---:|---:|---:|---:|
| P0 | 21.883612 | 23.845808 | 27.885315 | 24.538245 |
| P1 | 21.728636 | 23.827841 | 27.772163 | 24.442880 |
| P2 | 17.670895 | 15.987232 | 30.381571 | 21.346566 |
| RS | 20.738020 | 20.698502 | 31.168859 | 24.201794 |

## Gate A and stop decision

Failed frozen criteria: mean_rmse_improves_vs_p0_at_least_5pct, mean_rmse_not_worse_than_p2_by_over_3pct, fold3_overall_rs_no_worse_than_p0, fold3_continuous_nonzero_better_than_p2, 12h_no_worse_than_p0_by_over_5pct. Because one failed criterion is sufficient, the phase stops here. No threshold change, selector change, CAF, PINN loss, new branch, retraining, multi-seed run, or Test release is authorized.
