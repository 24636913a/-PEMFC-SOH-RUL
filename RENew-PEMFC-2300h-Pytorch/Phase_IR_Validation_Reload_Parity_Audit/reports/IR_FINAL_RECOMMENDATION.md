# Phase I-R final recommendation

Only fixed 32 Validation origins are inferred. No Assessment, Test, training or model changes. Exact parity remains np.array_equal; diagnostic tolerances do not change the PASS rule.

## Status

`PHASE_IR_H_AUTHORITY_PARITY_VERIFIED`

PHASE_IR_I_EXECUTION_SEMANTICS_DIFFERENCE_OBSERVED

| model | seed | A | B | C | D | E | max_abs_difference_mV |
| --- | --- | --- | --- | --- | --- | --- | --- |
| F0 | 42 | True | True | False | True | False | 9.784926448e-05 |
| F0 | 52 | True | True | False | True | False | 6.523284299e-05 |
| F0 | 62 | True | True | False | True | False | 6.523284299e-05 |
| F1 | 42 | True | True | False | True | False | 6.523284299e-05 |
| F1 | 52 | True | True | False | True | False | 9.784926448e-05 |
| F1 | 62 | True | True | False | True | False | 9.784926448e-05 |
| F2 | 42 | True | True | False | True | False | 8.154105374e-05 |
| F2 | 52 | True | True | False | True | False | 9.784926448e-05 |
| F2 | 62 | True | True | False | True | False | 6.523284299e-05 |

## Cause and evidence level

CONTROLLED_FACTOR_IDENTIFIED_REQUIRES_GRAD_FLAGS: disabling parameter gradient flags alone reproduces the full Path E numerical output in 9/9 checkpoints; no_grad/inference_mode alone and CPU-to-CUDA alone do not change outputs. This identifies an execution-context cause, not the unique internal CUDA/cuDNN kernel.

- no_grad -> inference_mode alone: 0/9 outputs differ from Path A (exact comparison).
- requires_grad=False alone: 9/9 outputs differ from Path A (exact comparison).
- CPU construct/load -> CUDA alone: 0/9 outputs differ from Path A (exact comparison).

## Unified interface and mathematical scope

unmodified frozen H e.predict(model, fixed_preprocessed_inputs, 32, cuda); eval plus no_grad; preserve original requires_grad flags. This restores the frozen historical implementation rather than altering model equations, weights, input, hyperparameters, scaler or target. Numerical equivalence is demonstrated only for these fixed Validation inputs in the frozen environment, not every possible input or GPU.

## Any future Assessment prerequisites

1. Retain original Phase I protocol, BLOCKED ledger and reports unchanged.
2. A separately authorized revised Assessment protocol is required; do not weaken the exact parity gate.
3. Freeze a new execution-code identity using the demonstrated historical interface, and distinguish requires_grad flags from actual parameter updates.
4. Reverify nine immutable checkpoints, environment, Development prefix and original Assessment keys/truth under the new authority.
5. Reproduce all nine fixed Validation references exactly before any newly authorized Assessment forward.
6. User authorization for that revised execution remains necessary; this audit does not unblock or restart I.

## State and operation proof

Every successful path saves per-entry parameter/buffer hashes, equality, before/after gradient flags, grad=None, and operation guards. requires_grad=True is not training. No backward, optimizer creation, optimizer step or state change occurred on successful audited paths.

## Limits and stop

New training NO; new Assessment inference NO; formal Test accessed NO; original I ledger modified NO; original checkpoints modified NO. Historical selection bias and E2/F0 reproducibility uncertainty are unchanged. Stop after the finite audit; no performance experiment, model selection, SOH/RUL or automatic I rerun.
