# Paper-to-inspired mapping

`PAPER_VERIFIED` names a formula or declared component. `PAPER_EQUATION_INTERPRETATION` means the formula is used literally when prose/shape language conflicts. `EXPLORATORY_BRIDGE` closes an unspecified connection. `ENGINEERING_DEFAULT` is a fixed non-core implementation choice.

| Phase PI component | Classification | Frozen decision |
|---|---|---|
| Per-variable embedding and embedding rFFT | PAPER_VERIFIED | `48→128→128`, rFFT along `D`, top two distinct periods. |
| Integer period rule | PAPER_EQUATION_INTERPRETATION | Half-up `clip(floor(D/f+0.5),2,D)`. |
| Inner period scale | EXPLORATORY_BRIDGE | Use outer `p_i`, not `D`, to make Eq. (6) close. |
| Variable-axis Eq. (8) attention | PAPER_EQUATION_INTERPRETATION | `softmax(QKᵀ/√D)` is `[B,14,14]`; it is not cross-period Transformer attention. |
| Patch-to-GPRU connection | CORE_EXPLORATORY_BRIDGE | Right-pad patch to `D`, apply `W_V`, then `E @ patch_value`. |
| Base graph | CORE_EXPLORATORY_BRIDGE | `A=E` for each sample/branch. |
| GPRU | PAPER_VERIFIED | Eq. (9), with paper-specified variable-specific biases. |
| A/Aᵀ paths and equal mix | CORE_EXPLORATORY_BRIDGE | Shared mix-hop weights, equal averaging. |
| Hop depth/two-period fusion | EXPLORATORY_BRIDGE / CORE_EXPLORATORY_BRIDGE | Two hops; two zero-initialized scalar fusion logits. |
| Branch summary and forecast MLP | ENGINEERING_DEFAULT | Fixed small shared MLPs, no dropout. |
| Voltage-only loss | RealHyFC target-specific training adaptation | Other future channels are never supervised. |
