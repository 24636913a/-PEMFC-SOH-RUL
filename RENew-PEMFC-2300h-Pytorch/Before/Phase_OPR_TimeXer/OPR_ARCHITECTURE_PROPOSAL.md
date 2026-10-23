# Candidate OPR-TimeXer proposal — not implemented

The sole pre-registered candidate is `Y_pred(h)=B_h(X)+C_h(X,Z_P2,h)-C_h(X,Z_ref)`. It reuses the frozen TimeXer history encoder, the same Step-Aligned FiLM correction network, shared correction weights, and the same horizon state for future and reference branches. It adds no Transformer, attention, recurrent module, future-condition forecaster, learned gate, or PINN loss.

When `Z_P2,h=Z_ref`, only the additional condition correction is constrained to zero. The history term `B_h(X)` remains, so this does not claim zero DeltaV or a physical law. This phase neither implements nor trains the candidate.
