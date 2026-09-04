# clean_mimo — Phase 1

This is a new, self-contained baseline experiment.  It does not import, edit,
or reuse the legacy Adaptive GRN + MCLSTM + L=1 TFT preprocessing or outputs.

## Why the old line is historical evidence only

The earlier route is not a valid starting point for an online causal MIMO
experiment because its implementation includes all of the following:

- `ΔUtot` is calculated from `y_raw`, then appended to `X`; `SOH=y_raw/v_init`
  is appended similarly.  Both make the target available in the input.
- A centre-window Savitzky–Golay filter is applied to the complete trajectory
  before splitting.  Each value can therefore use future samples.  `bfill()`
  can also fill a missing earlier value from a future measurement.
- The hourly reduction uses `drop_duplicates`, retaining the first sample of
  each bin rather than representing the whole hour.  It discards FC2
  within-hour current variability.
- The previous L>1 test uses FC2 `0.5 h × L=20`, a knee-centred short segment,
  `80/10/10` splitting, one seed, and one-step labels.  It is not equivalent to
  Zhang (2021)'s unified `1 h, N=15, H=4` MIMO protocol.
- The old FC1 RUL script reads `ts_data` after the starting point and feeds the
  resulting true future sensors (including current-time `Utot`) into each
  prediction.  It is teacher forcing / monitoring, not true future RUL.

Accordingly, no legacy R², RUL, feature-importance, SG, or Loop-50 result
should be compared as if it came from this protocol.

## Fixed Phase-1 definition

For both FC1 and FC2, raw `Ageing` CSV records are pooled by hourly interval
`[k,k+1)`, and the aggregate is labelled at the end of that interval.  The
input list is identical for both stacks:

`Utot_mean`, `U1_mean`–`U5_mean`, `J_mean`, `I_mean`, `I_std`, `I_p2p`, and
the mean values of the recorded temperature, pressure, flow, and relative
humidity channels.  Thus FC1 legitimately has almost-zero `I_std`/`I_p2p` and
FC2 retains available **hour-level** current variability.  This does not claim
to reconstruct the high-frequency ripple waveform.

There is no SG filtering, no `bfill`, no zero fill, no target-derived feature,
no attention/Transformer/GRN/MCLSTM/physics loss, and no hyperparameter search.
Missing observations are forward-filled only from recorded earlier values;
unresolvable leading rows are dropped.

The supervised task is exactly:

```text
X[t-14 : t] -> [Utot(t+1), Utot(t+2), Utot(t+3), Utot(t+4)]
```

All windows are contained in one chronological split: 50% train / 10%
validation / 40% test.  The feature and target scalers are fitted after the
split, using training windows only.

The initial three models use the same input and the same evaluation protocol:

1. Persistence: all four estimates equal observed `Utot(t)`.
2. Linear trend: a least-squares trend of the previous 15 observed hourly
   `Utot` values is extrapolated 1–4 h.
3. MIMO-LSTM: one unidirectional LSTM layer (`hidden=64`) and one linear
   four-output head; MSE + Adam + validation early stopping.

The optional fourth model is `--include-residual`, which predicts
`Utot(t+h)-Utot(t)` and then adds back the observed `Utot(t)`.  This is legal,
but it is deliberately not part of the first vanilla-baseline run.

## Running it

The default data root is the existing local IEEE PHM2014 directory.  On another
machine, set `PEMFC_IEEE2014_PATH` or pass `--data-root`.

```powershell
cd "E:\沈农信电\ PEMFC-SOH+RUL\数据\New-PEMFC-Pytorch\clean_mimo"
python leakage_audit.py --fc FC1 --output-dir outputs\FC1\audit_only
python run_fc1.py --seeds 42
python run_fc2.py --seeds 42
```

Only after inspecting the single-seed output and audit should the fixed,
pre-declared ten-seed protocol be used:

```powershell
python run_fc1.py --seeds 42,52,62,72,82,92,102,112,122,132
python run_fc2.py --seeds 42,52,62,72,82,92,102,112,122,132
```

The ten-seed report is descriptive (`min/mean/max/std/median`); no test R²
ranking, Top-K selection, or ensemble is implemented.  The only model
checkpoint selection is validation-loss early stopping.

## Deliverables and acceptance criteria

Each run writes an immutable audit log and pipeline metadata, per-horizon CSV
metrics (MAE, RMSE, R² for h=1…4 and arithmetic mean), persistence-relative
RMSE/MSE skill, raw test predictions, a h=4 curve, history, checkpoint, and
JSON result.  The audit fails closed if timestamps, scaler fit scope, feature
provenance, or split isolation is violated.

Read `RESULTS.md` after each completed FC1/FC2 run.  Phase 1 succeeds only if
both audits pass and the three first models run reproducibly.  Whether MIMO
actually beats Persistence and Linear must be concluded from the saved
per-horizon results—not from a single aggregate R².

## Later work — explicitly not implemented now

If Phase 1 establishes a clean and reproducible baseline, the next method is
**Operating-Condition-Modulated MIMO-LSTM (OCM-MIMO-LSTM)**: a condition
encoder `z_t=phi(c_t)` directly modulates the LSTM input and forget gates.  Its
claim must be an operating-condition-modulated degradation-memory mechanism,
not a generic replacement model.  Only then consider horizon sensitivity
`H=1/4/8/12/24`, true future RUL without post-`Sp` measurements, and external
validation.  OCM-LSTM is intentionally absent from Phase 1.
