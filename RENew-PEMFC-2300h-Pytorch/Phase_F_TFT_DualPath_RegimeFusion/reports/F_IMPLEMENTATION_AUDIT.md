# Phase F Engineering Implementation Audit

## Result

`PHASE_F_IMPLEMENTATION_READY_FOR_TRAINING_REVIEW`

All synthetic-only unit tests passed. No RealHyFC table, Assessment target, or formal Test source was opened. The suite exercised native parity, F1 shapes and causal convolution, F2 gate initialization/gradients, decoder placeholder firewall, native encoder-to-decoder state handoff, synthetic backward and synthetic checkpoint reload.

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

The main risk is upstream-source drift. A future run must fail closed if the recorded PTF hashes differ. F1/F2 also add parameters, so future RMSE differences cannot be assigned solely to a mechanism without the frozen parameter/compute ledger. READY means engineering review may be requested; it is not training authorization.
