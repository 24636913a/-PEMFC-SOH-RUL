# Time alignment and leakage audit

Prediction dispatch is defined as **diagnosis completion**, the final raw `DateTime` in the current curve, never its start time. Dynamic input histories end at or before dispatch. The target diagnosis completion time and its realised `OpHrs2` are output-only descriptors and are not model inputs. The only future-time covariate is the pre-specified **planned checkpoint-label gap** scenario.

The causal table is limited to its first 3,005 rows, with maximum end `1819.997738` h and no Test split rows. For dynamic pretraining, every target completes by the current dispatch; the chronological auxiliary validation set is purged until its input begins strictly after the final auxiliary training target completion. Scalers, PCA compression and curve heads are fit separately in each origin on preceding data only.

| origin_stage | target_stage | origin_diagnosis_complete | target_diagnosis_complete_output_only | planned_gap_h | origin_actual_ophrs2_output_only | target_actual_ophrs2_output_only | legal_dynamic_windows | history_available_transition_curves | protocol |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 850 | 1150 | 2025-03-20 04:45:08.005000 | 2025-04-18 07:40:56.005000 | 300.00000 | 854.50000 | 1131.00000 | 1000 | 2 | planned-checkpoint-gap scenario |
| 1150 | 1500 | 2025-04-18 07:40:56.005000 | 2025-05-27 00:58:12.005000 | 350.00000 | 1131.00000 | 1463.50000 | 1281 | 3 | planned-checkpoint-gap scenario |
| 1500 | 1750 | 2025-05-27 00:58:12.005000 | 2025-06-10 19:10:50.005000 | 250.00000 | 1463.50000 | 1719.50000 | 1614 | 4 | planned-checkpoint-gap scenario |

No 2000/2300 h raw polarization file was opened, no new Test target was read, and no Test metric was computed. Historical descriptive Test access remains truthfully marked compromised.
