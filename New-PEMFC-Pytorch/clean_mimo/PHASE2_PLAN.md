# Phase 2 — Persistence-Anchored Residual MIMO

## Frozen purpose

Phase 2 tests **target expression**, not model complexity.  It retains the
strictly causal hourly Phase-1 pipeline and its fixed `N=15`, `H=4`,
chronological `50/10/40` protocol, while changing the neural target from an
absolute future voltage to

```text
delta_y[h] = Utot(t+h) - Utot(t), h = 1..4
Uhat(t+h) = observed Utot(t) + delta_Uhat(t+h)
```

`Utot(t)` is the last already observed historical measurement and is used only
as a persistence anchor / output skip connection.  It is never a future
measurement.

## Predeclared seed-42 comparison

For both FC1 and FC2, this round runs exactly four models with unchanged model
configuration: Persistence, 15-hour Linear trend, absolute MIMO-LSTM, and
Residual MIMO-LSTM.  The neural models both use a one-layer LSTM (`hidden=64`),
MSE, Adam, batch size 64, learning rate `1e-3`, weight decay `1e-5`, and
validation early stopping.

The residual has its own `StandardScaler`, fitted only to flattened training
residual labels.  Its final layer is unconstrained `Linear(64, 4)`; no sigmoid
or tanh constrains the predicted residual range.

Results are written only under `outputs_phase2/`.  Existing `outputs/` is a
read-only Phase-1 baseline.

## Supported formulations, not automatically run this round

The Phase-2 runner supports two input modes and four **predeclared** feature
sets.  Future selections must be made from validation artifacts, never from
test ranking.

| Feature set | Inputs |
|---|---|
| R1 `utot_only` | `Utot_mean` |
| R2 `voltage_state` | `Utot_mean`, `U1_mean`–`U5_mean` |
| R3 `state_load` | R2 plus `J_mean`, `I_mean`, `I_std`, `I_p2p` |
| R4 `full_26` | all 26 existing causal hourly features |

`absolute_inputs` keeps the legal observed history unchanged.
`relative_voltage_inputs` transforms each selected voltage channel within each
window as `V(t-k)-V(t)` using the final historical row only.  It leaves
operating-condition features unchanged before their train-only scaler.  Thus
no target-time row or future operating condition enters the transformation.

Known elapsed operating time is deliberately absent from all Phase-2 feature
sets.  It remains a later, standalone ablation rather than a shortcut for
lifecycle level drift.

## Required audit checks

Phase-2 audit fails closed if any Phase-1 causal check fails, if the anchor is
not exactly the origin Utot, if a residual is not `y_future-Utot_origin`, if
the residual scaler is not fitted only on training residuals, if relative
references are not the final historical row, if a scaler is refitted on
validation/test, or if an output path is not under `outputs_phase2/`.

## Validation policy

Seed-42 Phase 2 records ordinary validation metrics as diagnostics.  It does
not select an architecture from test results.  Before choosing among R1–R4 or
absolute/relative inputs, Phase 2b must use predeclared expanding chronological
validation folds (35/40, 45/50, and 55/60 percent train/validation boundaries)
with train-only scalers and split-contained windows.  Phase 2b is not executed
in this round.

## FC2 event diagnostic

`fc2_event_diagnostics.py` is descriptive only: it analyzes held-out FC2
one-hour voltage-change events and their observed-at-origin condition context.
It never supplies a model input and never removes samples.  It tests whether
large raw-voltage changes coincide with operating-condition transitions or
recovery-like events, rather than attributing FC2 failure to LSTM capacity.

## Stop boundary

This phase excludes feature-ladder runs, multiple seeds, Loop50, horizons
beyond 4 h, OCM, RUL, external data, hyperparameter search, attention,
Transformer/TFT, GRN, MCLSTM, PINN, xLSTM, Mamba, and physics losses.  After
the predeclared seed-42 comparison and the branch decision, work stops for
user review.
