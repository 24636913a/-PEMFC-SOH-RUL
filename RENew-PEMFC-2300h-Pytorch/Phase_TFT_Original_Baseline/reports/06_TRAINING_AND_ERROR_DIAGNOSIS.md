# Training and error diagnosis

Each fold first used the final chronological 15% of its outer-train origins as an inner stopper. The selected epoch then governed a same-seed reinitialised refit on the entire outer train; outer validation was not used for epoch selection. The maximum available epochs were 300 with patience 30, AdamW (0.001, weight decay 1e-5), batch size 64, and gradient clipping 1.0.

| Fold | selected epoch | best inner-stopper RMSE (mV) | final evaluated inner epoch |
|---|---:|---:|---:|
| fold_1 | 246 | 64.057 | 276 |
| fold_2 | 297 | 29.318 | 300 |
| fold_3 | 289 | 18.122 | 300 |

Fold 2 and Fold 3 reached the 300-epoch limit with best epochs 297 and 289, respectively; this is evidence that the fixed Stage-T0 training budget, rather than a demonstrated convergence plateau, may be a follow-up tuning question. It does **not** invalidate the frozen baseline or license in-stage retuning.

| Fold | prediction SD (mV) | target SD (mV) | median per-origin prediction/target SD ratio | near-constant origin fraction |
|---|---:|---:|---:|---:|
| fold_1 | 196.148 | 194.692 | 0.9994 | 0.0000 |
| fold_2 | 196.473 | 195.135 | 1.0090 | 0.0000 |
| fold_3 | 196.157 | 194.917 | 1.0050 | 0.0000 |

The prediction dispersion closely matches target dispersion and no origin has a near-constant 24-step prediction under the stated 0.01 mV criterion; T0 therefore provides no evidence that its poorer RMSE is explained by a trivial constant or excessively smoothed output. No abrupt-load subgroup was pre-registered or exported in the T0 prediction artifact, so this stage makes no unsupported claim about transition-specific failure. Stage 0 verified the actual PTF architecture, finite GPU forward/backward, gradient finiteness, extractable VSN/attention tensors, and prediction invariance to unavailable future measurements. The completed model has 86,883 trainable parameters. The formal overall RMSE is 26.158 mV.
