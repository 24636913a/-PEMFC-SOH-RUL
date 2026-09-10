# Phase N2.8B Results — TimeMixer++ Completion and TimeXer PEMFC Adaptation Check

## Frozen protocol and firewall

All runs used 0.5 h causal bins, 24 h / 48-step history, 12 h / 24-step horizon, exact F2 (14 features), true future DeltaV target, established chronological development folds, and train-only feature/target scaling. The loader reads only the pre-Test development prefix. `test_firewall_audit.json` records zero Test target rows, no Test predictions, metrics, plots, or feature diagnostics.

## Source conclusion

PyPOTS was fixed at `22fe29a3ef02da054c8ac6d56f4d6dc2eac2359c` and is correctly labelled **maintained public reproduction / maintained implementation**, not official author code. Its core source has concrete MRTI/TID/MCM/MRM mappings. The maintained forward creates `dec_out_list` but retains only its final scale; the PaperFaithful version therefore uses the documented paper-guided equal-weight mean of those existing scale predictions. Details are in `TIMEMIXERPP_IMPLEMENTATION_AUDIT.md`.

## Seed-42 comparison (development only)

| model | mean_r2_delta | mean_rmse_delta_mV | mean_mae_delta_mV | fold3_rmse_delta_mV | fold3_p95_mV | fold3_p99_mV | lifecycle_ratio |
|---|---|---|---|---|---|---|---|
| LSTM | 0.997446 | 9.005858 | 4.790057 | 14.651479 | 16.721832 | 53.260315 | 2.359915 |
| TimesNet HF | 0.995936 | 9.895152 | 3.546117 | 20.229832 | 14.131961 | 64.487457 | 7.879159 |
| TimeXer original | 0.965271 | 36.170404 | 18.967869 | 36.961407 | 53.015675 | 188.156769 | 1.167125 |
| TimeMixer++-PyPOTS | 0.934754 | 49.251348 | 33.081003 | 47.515156 | 90.758028 | 145.432446 | 1.153129 |
| TimeMixer++-PaperFaithful | 0.976587 | 29.212544 | 14.701398 | 30.421357 | 41.612259 | 149.422341 | 1.431153 |
| TimeXer-PEMFC-A | 0.965271 | 36.170404 | 18.967869 | 36.961407 | 53.015658 | 188.156778 | 1.167125 |
| TimeXer-PEMFC-Life | 0.961595 | 38.006646 | 20.296567 | 40.596628 | 63.631006 | 211.085715 | 1.248036 |

## Promotion and multiseed

| Model | Seed-42 promotion | Multiseed status |
|---|---:|---|
| TimeMixer++-PyPOTS | False | not_promoted_seed42_only |
| TimeMixer++-PaperFaithful | False | not_promoted_seed42_only |
| TimeXer-PEMFC-A | False | not_promoted_seed42_only |
| TimeXer-PEMFC-Life | False | not_promoted_seed42_only |

| model | multiseed_status | n_seeds | mean_r2_delta | mean_rmse_delta_mV | rmse_delta_std_mV | fold3_mean_rmse_delta_mV | fold3_p95_mV | fold3_p99_mV | lifecycle_ratio |
|---|---|---|---|---|---|---|---|---|---|
| TimesNet HF | frozen_incumbent_three_seed | 3 | 0.995593 | 9.908015 | 0.172471 | 21.444856 | 15.556553 | 75.236537 | 9.382451 |
| TimeMixer++-PyPOTS | not_promoted_seed42_only | 1 | 0.934754 | 49.251348 | 0.000000 | 47.515156 | 90.758028 | 145.432446 | 1.153129 |
| TimeMixer++-PaperFaithful | not_promoted_seed42_only | 1 | 0.976587 | 29.212544 | 0.000000 | 30.421357 | 41.612259 | 149.422341 | 1.431153 |
| TimeXer-PEMFC-A | not_promoted_seed42_only | 1 | 0.965271 | 36.170404 | 0.000000 | 36.961407 | 53.015658 | 188.156778 | 1.167125 |
| TimeXer-PEMFC-Life | not_promoted_seed42_only | 1 | 0.961595 | 38.006646 | 0.000000 | 40.596628 | 63.631006 | 211.085715 | 1.248036 |

## Diagnostic rankings

- Fold3 RMSE_delta, lower is better: LSTM > TimesNet HF > TimeMixer++-PaperFaithful > TimeXer original > TimeXer-PEMFC-A > TimeXer-PEMFC-Life > TimeMixer++-PyPOTS.
- Fold3 P99 absolute DeltaV error, lower is better: LSTM > TimesNet HF > TimeMixer++-PyPOTS > TimeMixer++-PaperFaithful > TimeXer original > TimeXer-PEMFC-A > TimeXer-PEMFC-Life.
- Lifecycle ratio Fold3/Fold1, lower is more stable: TimeMixer++-PyPOTS > TimeXer original > TimeXer-PEMFC-A > TimeXer-PEMFC-Life > TimeMixer++-PaperFaithful > LSTM > TimesNet HF.

## TimeXer verdict

The specified two adapters are fully documented in `TIMEXER_ADAPTATION_AUDIT.md`. **PERMANENT STOP: best adaptation mean RMSE_delta > 25 mV**. It is not selected merely for novelty.

## Frozen selection

- Selected backbone: **TimesNet HF**.
- Reason: Ranked only eligible final candidates by frozen order: mean R2_delta, mean RMSE_delta, Fold3 RMSE, Fold3 P99, seed stability, lifecycle ratio, then complexity.
- Recommended next phase: **Lifecycle-Aware Robust TimesNet**.
- Phase status: **PASS**. This phase completed its two TimeMixer++ variants, two permitted TimeXer adapters, fair seed-42 comparison, conditional multiseed policy, diagnostics, and decision freeze. No model was trained or selected on Test.
