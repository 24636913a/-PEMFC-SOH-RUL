# I2 execution contract

Raw subprocess logs retain their original Windows CRLF bytes. Phase-local Git attributes mark `.log` as binary to preserve exact execution evidence without rewriting logs; this changes neither prediction bytes nor frozen execution code.

Reused Development Assessment; n=3 frozen seeds. Not independent Test. All 9072 queries per run retained. No new training, no new model selection.

Protocol SHA: `d9d5dd2ad31e7fe8a6a28aaa19d706bea0a7ad16092b6de449171172bb239307`

Complete code identity: `e1945efba676202247bf0600de00be7f3b11f9487ac6ed8827f1ec175398f2c3`

CUDA construction -> strict load -> eval -> original e.predict(model,x,32,device) -> inherited torch.no_grad. Original requires_grad=True preserved; it does not imply training. Every parameter/buffer compared bytewise before/after; all gradients None, backward/optimizer disabled during inference. Original float32 inverse scaling retained and independently checked against CSV. Gate NOT_COLLECTED to avoid any additional Assessment path. Only nrows=3007 then Development filter ->3005; no whole-N1 checksum. Saved E scaler consumed, never fit. All source/code fingerprints reverified at startup and after evaluation.

- PHASE_I_ORIGINAL_EXECUTION_BLOCKED
- PHASE_IR_H_AUTHORITY_PARITY_VERIFIED
- PHASE_IR_I_EXECUTION_SEMANTICS_DIFFERENCE_OBSERVED
- PHASE_F_ASSESSMENT_INHERITED_SELECTION_BIAS
- C1_D0_E0_REPRODUCIBILITY_UNRESOLVED
- TIMESNET_FREEZE
- BACKBONE_SEARCH_CLOSED
