# Legacy-code audit before `clean_mimo`

This is an evidence record for the decision to keep the previous experiments as
historical material only.  It is not an accusation that the old scripts were
useless: they identified useful failure modes.  It does mean their numerical
results cannot be reused as causal Phase-1 MIMO results.

## Confirmed future-information and causal-preprocessing problems

| Location | Evidence | Why it is incompatible with the new task |
|---|---|---|
| `New-PEMFC-Pytorch.py:342–371` | `delta_utot` is calculated from `y_raw`; `soh = y_raw/(v_init+...)`; both are concatenated into `X_raw`. `test_feature_leak.py:126–135` reproduces the same A/B design. | The L=1 label `Utot(t)` is encoded back into X.  Neither may appear in a causal predictor. |
| Main: `253–279`; L>1: `55–75`; future Loop50: `43–52`; FC1 RUL: `49–58` | Every script first calls `drop_duplicates`, then `ffill().bfill()`, then centre `savgol_filter(...,21,2)` on the complete life trajectory. | First-row sampling is not hourly aggregation. `bfill` accesses future observations, and a symmetric SG value uses later timestamps. Both violate online causality even if the later train/test slicing is ordered. |
| Main: `295–305` | Three L=1 lag blocks are created: `X_lag -> y_lag`, and each input is reshaped to sequence length one. | This is a multi-lag one-step state-mapping task, not `X[t-14:t] -> [t+1,...,t+4]` MIMO forecasting. |
| Main: `383–395` | The split is applied after creating/augmenting labelled samples. | The three lag copies contain a target-derived feature.  The split is not a protection against the leakage that has already occurred. |
| Main and related scripts | The ΔV loss is formed by `np.diff(y_train, ...)` after concatenating the three lag blocks. | At block junctions, adjacent array entries are not adjacent physical timestamps; the auxiliary derivative target is not consistently defined. |
| `test_L_gt_1.py:33–145` | FC2 is `0.5 h`, `SEQ_LEN=20` (10 h), knee point is estimated from the full filtered series, then only knee ±50 h is retained; `80/10/10` occurs after windows are built. | This is not a common FC1/FC2 1 h, N=15, H=4 protocol. Knee/segment selection sees the entire target trajectory, the test set is very small, and adjacent windows share raw observations across sample-split boundaries. |
| `test_future_loop50.py:24–74` | FC2 only, `0.5 h`, `L=10`, `TAU=1`, `80/10/10`, one-step label. | Its Loop50 probes a different task from MIMO 15 h→4 h. It also inherits first-row sampling, bfill, and centred SG. It reports only R² rather than horizon-wise accuracy and baseline skill. |
| `test_fc1_rul.py:137–167` | `build_features(..., y_target)` creates `du` and `soh` from `y_raw` and appends them to training X. | The same label-to-input leakage recurs in the RUL training stage. |
| `test_fc1_rul.py:231–260` | After `Sp`, `ts_data = data_values[time_h > Sp]`; every forecast builds `x_sens = ts_data[i:i+1]`, which includes observed future `Utot`; SOH is calculated from it. | This is teacher-forcing online monitoring, not future RUL. It must never be described as a no-future-information forecast. |

## Evaluation-interpretation problems

- The old `L>1` and future-loop experiments do not report a common h=1…4
  MAE/RMSE/R² table, an overall four-horizon mean, Persistence skill, or Linear
  trend skill.  A high one-step R² by itself cannot establish a useful
  multi-horizon forecast.
- The old 50-run summaries measure seed variation for the legacy task.  They
  cannot justify choosing best test seeds, top-K models, SWA combinations, or
  ensembles as a main causal result.
- FC1/FC2 sampling intervals, window lengths, segment selection, and split
  ratios differ across old test scripts.  Numbers are therefore not directly
  comparable across the two operating conditions.
- RUL is a separate task from four-hour SOH forecasting.  A model that reads
  post-`Sp` measurements is useful for monitoring but cannot support a claim of
  true future RUL without explicit teacher-forcing qualification.

`clean_mimo` addresses these issues by hourly aggregation, forward-only missing
value handling, 50/10/40 row splitting before window construction, train-only
scalers, split-contained N=15/H=4 windows, timestamp/scaler/source audits, and
predeclared baselines.
