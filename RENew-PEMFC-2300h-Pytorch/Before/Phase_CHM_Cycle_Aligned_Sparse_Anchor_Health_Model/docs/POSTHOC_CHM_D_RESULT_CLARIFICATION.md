# CHM-D posthoc result clarification

This document is an interpretation and qualification erratum only. It does not alter the execution-time record in `final/FINAL_DECISION.yaml`, retrain a model, rerun LOAO or baselines, perform bootstrap, access Test, or define SOH or RUL.

## Frozen Development result

The original primary status remains `DEVELOPMENT_HEALTH_MODEL_NOT_QUALIFIED`. The recorded results are CHM MAE 0.0015171364900874822, B0 MAE 0.012389576604708038, B1 MAE 0.0012968694357638848, and B2 MAE 0.0004768699068203319. Gate 1 passed, Gate 2 passed, Gate 3 failed, and Gate 4 passed.

CHM substantially improved on the uncalibrated P1S functional baseline B0, but it did not exceed simple prespecified calibration. In particular, CHM MAE is approximately 3.18 times B2 MAE. Gate 3 is an explicit failure, not a near miss.

B2 was a prespecified baseline, not a model added after observing CHM's result. Within the frozen Development LOAO experiment, the prespecified nonnegative affine calibration substantially outperformed the neural latent-state CHM. This does not validate B2 as SOH, establish Test performance, support RUL, or prove lifetime extrapolation.

## Gate 5 qualification

The original serialized `Gate_5: FAIL` remains preserved as an execution-time artifact. The code triggers bootstrap only after Gates 1--4 all pass. Because Gate 3 failed, bootstrap was not triggered: no outer LOAO intervals were generated, no held-out coverage was observed, and no 5/6 coverage statement was evaluated. The scientific status of Gate 5 is therefore `NOT_EVALUATED_GATE_3_STOP`, not an observed coverage failure. Gate 3 alone already fixes the primary result as not qualified.

## Retrospective anchor-prefix diagnostic

`CHM_D_CAUSAL_ONLINE_DIAGNOSTIC.csv` is preserved unchanged as historical output. Its qualified name is `ANCHOR_PREFIX_FULL_HOU_RETROSPECTIVE_DIAGNOSTIC`. The diagnostic restricts sparse polarization anchors to the prefix available before each nominal checkpoint, but its CHM fitting still uses the full frozen Development HOU sequence. It is therefore retrospective and not a strict causal-online deployment evaluation. The historical MAE/RMSE are 0.002789163059406574 and 0.002881061921034611, respectively.

The frozen HOU table confirms that the nominal 1150 h, 1500 h, and 1750 h fits included 53, 28, and 5 future HOU rows, respectively. No HOU-truncated strict-online experiment is performed here; it would require a separate preregistered protocol.

## S1 and negative-result freeze

`S1_ROBUSTNESS_SUPPORTED` means only that the all-six-anchor and the 300/850/1750 reduced-anchor fits yielded similar structurally monotonic latent trajectories. It does not validate health, independently validate sparse supervision, establish Test generalization, or establish causal tracking.

The final status is `CHM_D_NEGATIVE_DEVELOPMENT_RESULT_FROZEN`. The evidence supports a simple calibrated P1S mapping over more complex latent-state dynamics in this Development sparse-anchor regime. It does not mean the P1S health route failed: B2's strict-LOAO MAE is far below B0's. It means CHM's latent dynamics supplied no additional generalization gain.

`Sparse-Anchor Affine Health Calibration` is recorded only as a `POSTHOC_NEXT_HYPOTHESIS`. No new training, calibration, Test opening, or CHM optimization is authorized. Any future study must be separately preregistered.
