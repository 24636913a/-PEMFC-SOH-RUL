# Phase H final conclusion

PHASE_H_MULTISEED_VALIDATION_COMPLETED

F2_VALIDATION_MULTISEED_GAIN_OBSERVED

## Same-seed evidence

{
  "2": {
    "RMSE_values_mV": [
      8.76148906172451,
      8.094854266394512
    ],
    "absolute_spread_mV": 0.6666347953299976,
    "relative_spread": 0.07909601535203027,
    "threshold_mV": 0.8428171664059511,
    "pass": true
  },
  "3": {
    "RMSE_values_mV": [
      8.76148906172451,
      8.094854266394512,
      8.307160723546348
    ],
    "absolute_spread_mV": 0.6666347953299976,
    "relative_spread": 0.0794763869882277,
    "threshold_mV": 0.8387834683888458,
    "pass": true
  }
}

## Development Validation results

|model|seed|RMSE_mV|MAE_mV|R2|best_epoch|stop_epoch|training_time_s|
|---|---|---|---|---|---|---|---|
|F0|42|8.307160723546348|6.413513220669751|0.9981865723007312|277|317|1378.9831558999995|
|F0|52|10.38154418028464|7.0297329478168695|0.9971678325189249|213|253|1082.095841800001|
|F0|62|8.513984865858916|3.3162695032029377|0.9980951500686074|353|393|1673.9885457|
|F1|42|9.212743228039118|4.365415924358717|0.997769650347857|204|244|1049.1641292000004|
|F1|52|8.005450730286757|4.8619562793466535|0.9983159049595827|125|165|716.9168480999997|
|F1|62|10.248600211409459|5.726064063225621|0.9972399044114829|176|216|936.5718231999999|
|F2|42|6.265169199159184|4.704243711882334|0.9989685204377853|204|244|1062.4403335999996|
|F2|52|7.034632416751345|5.03339363741288|0.9986995973767137|223|263|1161.7280554000026|
|F2|62|8.436398939225588|3.085122902042307|0.9981297087773158|324|364|1612.9249808999994|


## Paired evidence

[
  {
    "comparison": "D10_F1_minus_F0_mV",
    "mean_difference_mV": 0.08803480001514312,
    "sample_std_mV": 2.173883595612303,
    "improved_seed_count": 1,
    "max_degradation_mV": 1.7346153455505426
  },
  {
    "comparison": "D21_F2_minus_F1_mV",
    "mean_difference_mV": -1.9101978715330723,
    "sample_std_mV": 0.9920147628184847,
    "improved_seed_count": 3,
    "max_degradation_mV": 0
  },
  {
    "comparison": "D20_F2_minus_F0_mV",
    "mean_difference_mV": -1.8221630715179291,
    "sample_std_mV": 1.645711477894464,
    "improved_seed_count": 3,
    "max_degradation_mV": 0
  }
]

## Research boundary and eight answers

1. **Same Seed42: PASS under the registered material-repeatability rule, not bitwise repeatability.** A/B RMSE=8.761489/8.094854 mV, best/stop epochs=187/227 and 286/326. Their difference is 0.666635 mV (7.9096% of the mean), below T=0.842817 mV. The independent third Seed42 is 8.307161 mV, best/stop=277/317. Three-run spread=0.666635 mV (7.9476%), below T=0.838783 mV. No run was substituted or retried.
2. **CUDA strict determinism remains unsupported.** Both CUDA strict interpolation and full F0 failed at Backward on `upsample_linear1d_backward_out_cuda`; Forward/output/loss evidence survived. Both CPU strict probes passed. No strict probe performed an optimizer step. Normal CUDA F0/F1/F2 genuine Forward/Backward/clip/AdamW/reload smoke tests passed. Normal CUDA training does not claim bitwise determinism.
3. **New execution evidence exists; the unique historical cause remains unresolved.** A/B initial identity and first-batch origins/input/label/RNG agree. All common 227 epochs have identical sampling-order and post-validation RNG summaries. The first differing saved field is parameters-after SHA at epoch 16, zero-based batch 19, journal sequence 4080; prior 4079 recorded stage comparisons agree. Epoch aggregate metrics also first differ at epoch 16. Gradient tensors were not hashed: equal scalar gradient norms do not prove equal gradients or locate the originating operation. See `outputs/audits/H_REPEAT_AB_FIRST_RECORDED_DIVERGENCE.json`. This does not diagnose the unique old E2/F0 cause.
4. **F2 improves on F1 in 3/3 seeds.** F2−F1 differences are −2.947574/−0.970818/−1.812201 mV (42/52/62); mean −1.910198, sample SD 0.992015 mV. F1 improves on F0 only in 1/3 seeds; mean F1−F0=+0.088035 mV. Dual-path encoding alone has no stable RMSE gain here.
5. **F2 improves on F0 in 3/3 seeds, but Seed62 has a small margin.** F2−F0=−2.041992/−3.346912/−0.077586 mV; mean −1.822163, sample SD 1.645711 mV. The 0.077586 mV margin is not evidence of broad statistical significance.
6. **Condition-aware fusion has positive, limited Validation evidence.** Registered signal: `F2_VALIDATION_MULTISEED_GAIN_OBSERVED`. Absolute RMSE SD is 1.142630/1.122666/1.100853 mV for F0/F1/F2; relative SD is approximately 12.6%/12.3%/15.2%. F2 does not establish universally lower training variability. No architecture redesign or automatic expansion follows.
7. **A qualified paper statement is supported:** on this reused Development Validation split, F2 improved over matched F0 and F1 at three fixed seeds. n=3 does not support universal stability, significance, or pristine Test generalization. The 4920 overlapping queries are not independent samples. The scientific-figure skill informed explicit Validation/n=3 labels, sample-SD rather than confidence-interval error bars, and unit-labelled PNG/PDF exports.
8. **A separately authorized independent Test-plan review is reasonable; opening Test is not automatic.** Any future plan must acknowledge inherited selection bias and unresolved historical reproducibility. This phase stops: no new Assessment inference, Test access, Phase I, or additional training.

## Final primary table — Development Validation, n=3 seeds

All errors are mV; SD uses ddof=1. Repeat-A/B are excluded.

| Model | Seed42 | Seed52 | Seed62 | Mean | Sample SD |
|---|---:|---:|---:|---:|---:|
| F0 | 8.307161 | 10.381544 | 8.513985 | 9.067563 | 1.142630 |
| F1 | 9.212743 | 8.005451 | 10.248600 | 9.155598 | 1.122666 |
| F2 | 6.265169 | 7.034632 | 8.436399 | 7.245400 | 1.100853 |

Independent CSV-only metric audit: 11/11 PASS, 4920 rows each, exact keys/truth, tolerance 1e-10. Checkpoint reload consistency: 11/11 PASS with zero difference for the saved fixed Validation inputs. Regression tests: 8/8 PASS. Six PNG/PDF figures were generated and visually inspected; all show Development Validation, not Test.

## Frozen identities and execution

Base: `dea436c7520ca9962e7aa3005af1faaedbc786f2`.

Protocol SHA256: `88aba09c7f9a08d43f658f9c66e482713bbb60607bc88bb8059f3638d5aad9dd`.

Execution/source identity SHA256: `b45b2bdcc14c0b2ef366bc765cfe456c68f62746a9affcb0b74f6cc8fc8087f1`.

Entry points: `python -B scripts/run_h_training.py --freeze`, `python -B scripts/run_h_training.py --run`, `python -B scripts/audit_h_predictions.py`, `python -B scripts/analyze_h_stability.py`, `python -B -m unittest discover -s tests -v`. The completed formal ledger is one-shot; do not launch `--run` again. Analysis regenerates report templates; this outcome-specific section supplements them without changing frozen execution code or decision rules.

## Immutable history

TIMESNET_FREEZE; BACKBONE_SEARCH_CLOSED; PHASE_F_ASSESSMENT_INHERITED_SELECTION_BIAS; C1_D0_E0_REPRODUCIBILITY_UNRESOLVED. Historical E/F/G untouched. State-dict332955 is not trainable-parameter323483. Train/Validation only; no Assessment inference, formal Test, search, SOH or RUL. Stop, no Phase I.
