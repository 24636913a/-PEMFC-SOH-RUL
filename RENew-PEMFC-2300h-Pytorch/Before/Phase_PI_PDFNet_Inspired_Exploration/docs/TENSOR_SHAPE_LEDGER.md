# Tensor shape ledger

| Symbol | Shape | Meaning |
|---|---|---|
| `x` | `[B,48,14]` | Frozen Raw F2 input. |
| `X_embed` | `[B,14,D]` | Shared per-variable embedding; `D=128` unless pre-score resource fallback is recorded. |
| `P_i,Q_i,K_i,V_i` | `[B,14,D]` | Branch representation and Eq. (8) projections. |
| `E_i` / `A_jl` | `[B,14,14]` | Variable-axis attention / directed dynamic adjacency. |
| `S_jl,H_jl,O_jl` | `[B,14,D]` | Patch state, hidden state, and graph output. |
| `O_intra_j,O_inter_j,H_i` | `[B,14,D]` | Masked patch mean, period-level output, branch summary. |
| `H_fused` | `[B,14,D]` | Two-period scalar-logit fusion. |
| `output` | `[B,24,14]` | Standardized all-variable absolute-voltage output. |
| `pred_delta` | `[B,24]` | `output[:,:,0] - x[:,-1,0]`; sole optimized target. |
