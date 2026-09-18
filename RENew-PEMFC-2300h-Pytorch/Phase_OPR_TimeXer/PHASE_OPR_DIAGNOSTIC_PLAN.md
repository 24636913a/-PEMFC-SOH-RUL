# Phase OPR-D diagnostic plan

## Frozen protocol

- Start commit: `7c4ca89fdabd2f104a2700cddc1ad1ea6a380d50`.
- Development-only RealHyFC protocol: 0.5 h bins, 48-step history, 24-step horizon, DeltaV target, F2 14 historical inputs and nine future-condition inputs.
- Existing N2.8F M0 checkpoints and the fixed three chronological Development folds are reused unchanged.
- P0, P1, P2 and RS must reproduce their frozen mean RMSE values within 0.001 mV before any diagnostic conclusion.

## Authorized analysis

The diagnostic recomputes inference only, with dropout disabled, and splits frozen M0 output in normalized target space into its history-only base term `B_h(X)` and FiLM condition correction `C_h(X,Z_h)`. It then converts both terms to physical DeltaV mV using the original fold target scaler.

P1/P2 error attribution and reference-condition comparisons are offline Development diagnostics. Actual future condition data is never fed to a formal inference path and never used as a selector or training target.

## Forbidden actions

No neural-network training, architecture modification, seed sweep, Test access, threshold tuning, data removal, anomaly correction, or implementation of OPR-TimeXer. Historical phase directories are read-only.

## Decision logic

Recommend an OPR training phase only if baselines reproduce, decomposition and RS identities hold, Fold 3 / 12 h degradation is localized without an audit failure, and the reference-condition difference offers a falsifiable structural motivation. This diagnosis cannot assert that OPR will improve metrics.
