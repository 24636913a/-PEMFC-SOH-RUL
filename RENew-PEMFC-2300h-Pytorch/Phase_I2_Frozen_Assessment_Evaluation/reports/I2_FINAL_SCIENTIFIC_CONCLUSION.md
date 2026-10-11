# I2 final scientific conclusion

Reused Development Assessment; n=3 frozen seeds. Not independent Test. All 9072 queries per run retained. No new training, no new model selection.

`F2_REUSED_ASSESSMENT_MULTISEED_GAIN_OBSERVED`

Additional labels: NONE

| comparison | mean_difference_mV | sample_SD_mV | improved_seed_count | maximum_degradation_mV | mean_relative_improvement_percent |
| --- | --- | --- | --- | --- | --- |
| D10_F1_minus_F0 | 1.14738503 | 17.014962 | 2 | 18.9018056 | -13.4484195 |
| D21_F2_minus_F1 | -10.695826 | 12.2450644 | 3 | 0 | 41.8145307 |
| D20_F2_minus_F0 | -9.54844101 | 10.9074723 | 3 | 0 | 38.6833174 |

## Does H Validation gain transfer?

F2 improves over F0 in 3/3 seeds, and over F1 in 3/3. F1 improves over F0 in 2/3. A stable transfer claim against both comparators requires 3/3 on both comparisons; actual counts, not H Validation ranking, determine the description.

Top5 target SSE shares span 37.54%–97.13%. Target-level contributions quantify arithmetic concentration; a unique physical or software failure mechanism is not established. Seed variation, later-time distribution change and error concentration are distinct hypotheses.

- PHASE_I_ORIGINAL_EXECUTION_BLOCKED
- PHASE_IR_H_AUTHORITY_PARITY_VERIFIED
- PHASE_IR_I_EXECUTION_SEMANTICS_DIFFERENCE_OBSERVED
- PHASE_F_ASSESSMENT_INHERITED_SELECTION_BIAS
- C1_D0_E0_REPRODUCIBILITY_UNRESOLVED
- TIMESNET_FREEZE
- BACKBONE_SEARCH_CLOSED

Historical Development Assessment has already been reused for selection. Even favourable means do not prove independent Test generalization. C1/D0/E0 reproducibility remains unresolved; no new TFT-versus-TimesNet ranking or SOH/RUL claim. Original I blocked status retained; I2 is a newly authorized protocol, not a repaired historical I run.

## Direct answer and magnitude (all targets retained)

Yes: the Phase H Validation ordering of F2 below both F0 and F1 extends to this reused Development Assessment in all three paired seeds. This is an observed directional transfer, not independent Test validation or proof of low seed variability.

| Model | Seed42 RMSE mV | Seed52 RMSE mV | Seed62 RMSE mV | Equal-seed mean ± sample SD mV |
| --- | --- | --- | --- | --- |
| F0 | 18.912760334 | 19.255237787 | 27.068481791 | 21.745493304 ± 4.613022602 |
| F1 | 18.469521813 | 38.157043369 | 12.052069812 | 22.892878331 ± 13.603013389 |
| F2 | 17.661515484 | 13.764315533 | 5.165325856 | 12.197052291 ± 6.393819231 |

F2−F0 mean paired difference is −9.548441013 mV; F2−F1 is −10.695826040 mV. F1−F0 is +1.147385028 mV and only 2/3 seeds improve: F1 alone does not show consistent gain. F1 Seed52 is especially worse (+18.901805583 mV versus F0), while F2 remains better in that seed. No new seed is selected.

F2's absolute Assessment RMSE spans 5.1653–17.6615 mV. Its SD is 6.3938 mV, so directional wins do not imply that seed variability has disappeared. F2 Seed42/52 Assessment RMSE exceed their Validation values by 11.3963/6.7297 mV, whereas Seed62 is lower by 3.2711 mV. F2 does not reverse rank against either comparator; F1-versus-F0 reverses in all three seeds.

F2 12 h endpoint RMSE, P99 and target-time-equal RMSE have equal-seed means 7.069195129, 21.748015145 and 12.298168056 mV, respectively. These diagnose different aspects of the same full cache; they do not replace the primary metric. For F2, five target times account for 37.54%–84.70% of SSE, with the maximum-SSE target at row 2952 / 1793.5 h in all three seeds. Its overlapping forecasts have multiplicity 19. The additive signed-MSE tables also identify contributions around 1529/1529.5 h and 1667/1667.5 h, depending on seed/comparator. No target deletion, additional weighting experiment or causal mechanism inference is performed.

The paired table above reports the mean of three per-seed relative improvements. It must not be conflated with the ratio of the equal-seed mean RMSEs. Likewise, pooled prediction-point RMSE is separately descriptive and is not the arithmetic mean RMSE or a prediction ensemble.

## Final stop record

All nine weights unchanged, training zero, Test zero. No automatic next stage, new seeds, architecture search or Test opening.
