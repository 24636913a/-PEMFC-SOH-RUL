# Phase R0 — Rare-State Error Identifiability and Robustness Feasibility

## Fixed contract

- Fixed start commit: `e4b66b0ccd37623f1b4f0eb510e002483389ed7a`.
- Development-only task: 0.5 h bins, 48 historical steps, 24 future steps, DeltaV target, 14 historical features, 9 future-condition features, and three chronological folds.
- Test is sealed. No Test feature, target, raw current, distribution, prediction, or metric may be read.
- Historical phase directories are read-only. All new work is confined to this directory.

## Fail-closed sequence

1. Verify fixed commit, frozen feature/order/fold/checkpoint contract and reuse frozen prediction files only after numerical alignment checks.
2. Establish a Test firewall before any data-table read. A source containing Test rows may not be read wholesale and then filtered.
3. Reconstruct strictly causal P2 training OOF using the unchanged G-STOP protocol only if the legal development prefix and target-completion purge can be proven.
4. Measure independent state-error events and probability reliability from train OOF separately from validation.
5. Reconstruct physical P2 inputs and development-only true conditional current statistics; halt if exact reconstruction is unavailable.
6. Run only frozen-M0, physical-consistent oracle counterfactuals CF00/CF10/CF01/CF11 on the common valid sample set.
7. Separate train support from frozen-model response evidence. Produce exactly one preregistration direction or `NO_MODEL_EXPERIMENT_AUTHORIZED`.

## Prohibited work

No neural retraining, no P2 hyperparameter change, no threshold tuning, no new network module, no multi-seed experiment, no Test access, and no automatic entry into a subsequent phase.
