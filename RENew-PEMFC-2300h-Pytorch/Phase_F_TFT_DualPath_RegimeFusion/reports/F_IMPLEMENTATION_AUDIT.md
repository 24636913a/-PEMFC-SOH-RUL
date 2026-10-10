# Phase F Engineering Implementation Audit

## Result

`PHASE_F_ENABLED_FORWARD_PARITY_PASS`

`READY_FOR_F0_F1_F2_TRAINING_AUTHORIZATION_REVIEW` is engineering review readiness, not training authorization.

Actual suite: 15 tests passed (13 original + 2 new). Full execution log: `outputs/implementation_audit/F_SYNTHETIC_TEST_LOG.txt`. No RealHyFC table, Assessment target, or formal Test source was opened. The original suite exercised disabled native parity, F1 shapes/causal convolution, F2 local gate/fusion gradients, decoder firewall, native state handoff, synthetic backward and checkpoint reload; it did not establish enabled full-forward parity. The two new tests close that coverage gap.

## Three distinct forward comparisons

Disabled parity delegates to native forward and is not evidence for custom forward. Enabled neutral-fusion parity uses a TEST-ONLY subclass returning the LSTM representation at fusion; the formal history branch still executes once per case. F1/F2 full parity runs both formal enabled models, including the gate once per case, with exact initial g=0.5. A patched inherited forward raises if either enabled candidate attempts native fallback. Each enabled comparison uses input seeds 219/731 and batch sizes 1/3.

All shared named parameters and buffers are checked for identical initial values, names and shapes, explicitly copied and checked again. F0-to-F1 permits only history_branch extra state; F1-to-F2 permits only regime_gate extra state. VSN, LSTM encoder/decoder, GRN/GLU, attention, output head and shared convolution state are included. Complete key lists and hook counts are in `F_ENABLED_FORWARD_PARITY_AUDIT.json`. Formal model code and frozen architecture were unchanged.

Predeclared float tolerances: atol=1e-06, rtol=1e-06; exact equality checked first, integer/length fields require exact equality. All measured full outputs are exactly equal; empty static_variables tensors are structurally equal with recorded max difference 0. Eval-only: additional dropout RNG consumption precludes a claim of training-mode bitwise parity.

|Comparison (eval)|Input shape|prediction|encoder_attention|decoder_attention|encoder_variables|decoder_variables|static_variables|encoder_lengths|decoder_lengths|Result|
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
|Disabled path|[2, 48, 14]|0.0|0.0|0.0|0.0|0.0|0.0|0.0|0.0|PASS|
|Enabled neutral fusion|[1, 48, 14]|0.0|0.0|0.0|0.0|0.0|0.0|0.0|0.0|PASS|
|Enabled neutral fusion|[3, 48, 14]|0.0|0.0|0.0|0.0|0.0|0.0|0.0|0.0|PASS|
|F1/F2 initial full forward|[1, 48, 14]|0.0|0.0|0.0|0.0|0.0|0.0|0.0|0.0|PASS|
|F1/F2 initial full forward|[3, 48, 14]|0.0|0.0|0.0|0.0|0.0|0.0|0.0|0.0|PASS|

## Runtime

- pytorch-forecasting: 1.8.0
- PyTorch: 2.6.0+cu124
- CUDA: 12.4
- `_tft.py` SHA-256: `6ad5f6c806d314cfec53851ce0914c6d3476c4b05bd136a2764f4a562f1b7343`
- `sub_modules.py` SHA-256: `9ef892c565f6891ddf6d21c52c4d7f90393ec731924851aee872468cffe2cf9e`

## Measured parameters

|Model|Trainable parameters|
|---|---:|
|F0|323483|
|F1|332379|
|F2|332436|

F1 adds a causal multi-scale branch; F2 adds 57 gate parameters. The upstream `forward()` is monolithic, so the project-local F1 subclass mirrors the audited 1.8.0 flow and changes only the post-LSTM encoder representation prior to concatenation with the untouched decoder representation. This copy is version-sensitive and is protected by the dependency manifest.

## Engineering risks

The main risk is upstream-source drift. This generator fails closed on tests, dependency identity, parity or parameter-count failure and replaces its success status with BLOCKED. F1/F2 add parameters, so future differences cannot be assigned solely to a mechanism without the frozen parameter/compute ledger.

Historical limits remain: `PHASE_F_ASSESSMENT_INHERITED_SELECTION_BIAS`, `C1_D0_E0_REPRODUCIBILITY_UNRESOLVED`, and `PHASE_F_TFT_ARCHITECTURE_EXPLORATION_ONLY`. TimesNet remains frozen and backbone search closed. Synthetic parity does not resolve historical numerical reproducibility or authorize formal training.

## Reproduction

Run `python -B scripts/audit_f_model_implementation.py` from this phase directory. It executes the entire unittest suite, repeats all parity comparisons, and regenerates this report and JSON evidence. No dataset/cache loaders are called.
