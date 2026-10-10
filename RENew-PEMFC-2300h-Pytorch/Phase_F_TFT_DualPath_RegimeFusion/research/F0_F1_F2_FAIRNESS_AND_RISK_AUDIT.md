# F0–F2 Fairness, Data Firewall and Risk Audit

## Frozen common experiment

### Inherited selection and reproducibility limitations

`PHASE_F_ASSESSMENT_INHERITED_SELECTION_BIAS`: Phase E chose the Top10+EWMA(alpha=0.3) input condition after Development Assessment had been exposed. E2's Assessment RMSE was 7.785821 mV, while its Validation RMSE was 6.064596 mV and E0 Top10 Raw's Validation RMSE was 4.797507 mV. Re-running F0–F2 with that inherited input can test the frozen architecture contrast, but cannot turn this preprocessing choice into independent validation evidence.

`C1_D0_E0_REPRODUCIBILITY_UNRESOLVED`: existing same-key, same-truth Top10-Raw records are 11.445854 mV (C1), 13.169241 mV (D0), and 19.318765 mV (E0). The available code/configuration/log evidence does not verify one causal explanation. In particular, it is not established that a scaler, CUDA nondeterminism, or environment difference explains any pair. No F0–F2 report may use cross-stage numerical differences as a scientific effect.

`PHASE_F_TFT_ARCHITECTURE_EXPLORATION_ONLY`: TimesNet's frozen main route and backbone-search closure remain in force. F0–F2 is an authorized TFT architecture branch with no automatic replacement or cross-protocol ranking consequence.

All three stages must independently train only after a future implementation authorization, using:

- the physical mixed causal table read only as `pd.read_csv(nrows=3007)`, then `bin_end_ophrs2_h < 1819.997738`, yielding exactly 3005 Development rows;
- chronological row partitions Train `[0,2103)`, Validation `[2103,2404)`, Assessment `[2404,3005)`;
- legal window counts Train 1380, Validation 205, Assessment 378;
- 0.5 h bins, 48-step history, 24-step DeltaV horizon, all 14 Raw F2 schema channels;
- the Phase-E Train-only common scaler and target scale procedure; Spearman Top10 must be a Train-derived standardized-space zero mask;
- causal EWMA alpha 0.3 of historical `cell_voltage_mean`, initialized at the first value of each 48-step history;
- the Phase-E/T1-R Fold-2 optimizer and training settings, Seed42, MSE, and identical early-stopping protocol.

F0 must be retrained. Phase-E E2 (7.785821 mV) is historical exploratory evidence, not an F0 cache and not a new baseline result. RMSE in mV is the sole formal ranking metric; MAE and R² are secondary diagnostics. Assessment is an already reused Development region, not independent Test data.

## Information firewall

- No future measured covariate enters F0, F1 or F2. Decoder contains only deterministic relative forecast step.
- No formal project Test row, Test label, 2000/2300 late-life data, raw full-file hash, or unbounded causal CSV read is allowed.
- No new feature selection, EWMA alpha, kernel/dilation, gate size, loss, head, lookback, horizon, Optuna run, or model search may be selected from Assessment results.
- Each model persists origin, target-row and horizon keys with predictions for independent recomputation and target-time overlap diagnostics.

## Parameter and compute reporting

For each F0–F2 model record actual trainable parameter count, estimated forward FLOPs for one `[1,48,14]` input (with estimator/version and unsupported operations disclosed), peak allocated CUDA memory during the first fixed training batch, mean training seconds/epoch, total wall time, and batch-32 inference milliseconds after warm-up. No claim that a score difference is caused solely by architecture is permitted when parameter and compute changes are not reported.

## Deterministic execution audit

Before every stage record Python, NumPy and PyTorch seeds; Torch/CUDA/cuDNN versions; deterministic-algorithm status; `cudnn.deterministic`; `cudnn.benchmark`; device name; DataLoader generator seed/order; initial state-dict SHA-256; scaler/label-development-prefix fingerprint; selected Top10 indices; and exact package-source hashes.

For a fixed synthetic tensor in `eval()` mode, record an input hash and output hash before training and after checkpoint reload. A fixed first training batch must have a persisted index/origin list and pre-update forward checksum. Resume/reload must reproduce the saved checkpoint output to a declared numerical tolerance. This audit is required because historical C1/D0/E0 raw-condition values were not strictly reproducible across differing scaler/provenance records; it does not retroactively resolve those historical values.

## Risks and fixed interpretation

| Risk | Mitigation / interpretation boundary |
|---|---|
| Extra F1/F2 parameters improve fit | Report count/compute; do not attribute all gain to multi-scale or regime mechanism |
| Small single split / reused Assessment | Call all results Development exploratory; do not select a final model for Test |
| F2 gate collapse | Report it as a diagnostic, do not tune it |
| Future leakage through convolution | Enforce left-only padding and branch perturbation tests |
| Decoder future-covariate leakage | Keep native strict-history input adapter unchanged and test placeholders |
| VSN/gate functional duplication | Gate sees only five fixed summary variables and selects between paths, never raw-variable weights |

## Preregistered run-ending record

Each stage shall emit a source-access audit, checkpoint, complete assessment prediction cache, independent metric audit, parameter/compute ledger, gate diagnostics (F2), and a single report. A missing cache or an identifier/true-value mismatch is a technical failure, not an “insufficient gain” scientific conclusion.
