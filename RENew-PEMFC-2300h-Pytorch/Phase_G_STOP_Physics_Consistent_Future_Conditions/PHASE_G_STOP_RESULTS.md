# Phase G-STOP Results

## Decision

**Phase status: COMPLETE — STOP.** G1 passed, P2 passed frozen-TimeXer Gate B, seed42 retraining improved the fair P0 control, but the preregistered G4 three-seed trigger failed because Fold3 worsened. The mechanism then failed on the strong causal-SG LSTM. No new unique final paper model is selected, Test remains sealed, and no further model search is authorized.

## Required questions

1. **Are zero-current events sufficient?** Yes for mixed-current analysis: 624 independent mixed segments, 624 nonzero→mixed events, and 621 mixed→nonzero events. q=1 is absent and its metrics are NOT_ESTIMABLE.
2. **What is `zero_current_fraction`?** Within-bin raw-sample fraction satisfying `abs(I.S.act)<=0.10 A`; it is not a verified shutdown label.
3. **Which physical relations are verified?** Same-bin aggregation; population std (`ddof=0`); fixed-area `J=I/A` with A=280.001183 cm²; nonnegative/range constraints; mixture mean and population variance identities. Peak-to-peak has no derived formula.
4. **Was the old 15.416839 mV Oracle physically inconsistent?** Yes: true q was combined with independently forecast current statistics. It remains a diagnostic marker, not a deployable or isolated causal result. Reproduction is 15.417 mV; the consistent current-group replacement is 14.634 mV and all-true is 14.621 mV versus P0 24.538 mV.
5. **P0 forecast quality:** q MAE/RMSE 0.024810/0.041624; mixed F1 0.5100; current mean/std/p2p MAE 4.887/2.787/14.287 A.
6. **P1 forecast quality:** q MAE/RMSE 0.017061/0.038242; mixed F1 0.5100; current mean/std/p2p MAE 4.887/2.787/14.287 A.
7. **P2 forecast quality:** q MAE/RMSE 0.001743/0.022112; mixed F1 0.9905; current mean/std/p2p MAE 2.042/2.022/14.287 A.
8. **Physical violation rate:** P0/P1/P2 trajectory rates are 100%/0%/0% under the audited range and identity rules.
9. **Frozen-TimeXer RMSE:** P0/P1/P2 = 24.538/24.443/21.347 mV.
10. **Which folds improve?** P2 improves Fold1 and Fold2; Fold3 worsens.
11. **Do zero/mixed-current regions improve?** Mixed-current RMSE improves in all folds. Continuous-nonzero improves in Folds 1–2 but worsens in Fold3.
12. **Does Fold3 improve?** No: 27.885→30.382 mV frozen; after fair retraining 27.885→28.524 mV.
13. **Does P99 improve?** Yes for frozen P2: 129.652→68.630 mV. Its maximum error nevertheless rises to 593.412 mV, so tail risk is not eliminated.
14. **Does 9.5 h worsen?** No: 26.074→19.864 mV. At 12 h, however, RMSE worsens to 35.475 mV.
15. **Gate B?** PASS for P2; FAIL for P1.
16. **Was TimeXer retraining triggered?** Yes, seed42 fair P0 and P2 were retrained with unchanged architecture/training controls.
17. **Did retraining improve?** Yes on mean seed42 RMSE: 24.538→21.553 mV (12.17%). Fold3 worsened 2.29%, so the frozen multiseed trigger did not pass.
18. **Was LSTM transfer attempted?** Yes, using chronological OOF LSTM residuals, target-completion purge, Train-only preprocessing, and no Validation residual fitting.
19. **Independent LSTM gain?** No. L0/L1/L2 mean seed42 RMSE = 7.926/21.043/21.467 mV. The OOF residual correction does not transfer to the full-data incumbent LSTM.
20. **Three seeds?** Not in G4 or G5: each preregistered seed42 trigger failed. Historical LSTM three-seed evidence remains read-only context, not a new G5 result.
21. **Unique final paper model candidate?** No new candidate. The original causal-SG LSTM remains the incumbent benchmark; P2+SA-TimeXer is promising but not multiseed-frozen.
22. **Supported paper contributions:** verified causal physical-consistency construction; P2 future-condition accuracy/violation reduction; matched-weight frozen-TimeXer Gate-B gain; explicit evidence that gains are fold- and backbone-dependent.
23. **Rejected assumptions:** q is not a shutdown label; q-only Oracle is not physically self-consistent; projection alone is insufficient; future-condition residual correction does not automatically improve a strong LSTM; performance is not uniformly improved at Fold3/12 h.
24. **Test firewall:** intact. Test target rows read=0; predictions=false; metrics=false; feature diagnostics=false; performance accessed=false.

## Stop rationale

The study answers its core question conditionally: causal physics-consistent future conditions can materially improve the frozen SA-TimeXer path, but the evidence does not establish a universal or final model improvement. The negative LSTM transfer is retained, not optimized away. The next permitted action is documentation/review or a separately authorized final protocol decision—not Test execution and not another architecture search.
