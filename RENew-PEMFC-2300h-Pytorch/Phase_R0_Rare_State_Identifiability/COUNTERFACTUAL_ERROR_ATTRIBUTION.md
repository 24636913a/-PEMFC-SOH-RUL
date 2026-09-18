# Counterfactual error attribution

On the common Oracle-estimable Validation set, CF00/CF10/CF01/CF11 RMSE is 24.476/14.769/24.519/14.952 mV. This is a frozen-input diagnostic only.

Fold1/12 h: q Shapley=1449.43 mV²; conditional-statistics Shapley=-12.64 mV²; interaction=-13.03 mV². Fold3/continuous-nonzero: 896.80, 3.63, 14.16 mV², respectively. The q factor dominates; interaction is small relative to the q attribution in the two required scopes.

Evidence: `outputs/counterfactual/counterfactual_attribution.csv`, `counterfactual_interaction.csv`, `counterfactual_event_metrics.csv`.
