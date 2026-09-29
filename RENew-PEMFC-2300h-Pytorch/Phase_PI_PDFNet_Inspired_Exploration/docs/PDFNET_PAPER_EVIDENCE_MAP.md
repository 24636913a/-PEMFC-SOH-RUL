# PDFNet paper evidence map

Source: `参考论文/PDFNet.pdf`, SHA-256 `b9f457f5…ee252867`, verified before
any Development model performance.

| Paper evidence | Page | Classification | Use in Phase PI |
|---|---:|---|---|
| MLP maps each variable from history to `X_embed ∈ R^(N×D)` | 4, §4.1 | PAPER_VERIFIED | Shared 48→128 per-variable embedding. |
| rFFT each variable, mean amplitude, exclude selected frequencies and `p_i=D/f_i` | 4, Eq. (3)–(4) | PAPER_VERIFIED | Per-sample embedding-domain period selection; integer mapping is specified separately. |
| Period and patch segmentation with zero padding | 4, Eq. (5)–(6) | PAPER_VERIFIED | Right zero-padding with masks. The inner-scale closure is a bridge. |
| `Conv1D(Padding(X^j))` | 5, Eq. (7) | PAPER_VERIFIED | One shared 1D kernel. |
| `P`, `Q/K/V`, and attention | 5, Eq. (8) | PAPER_VERIFIED / EQUATION_TEXT_CONFLICT | Equation-literal variable-axis attention. |
| GPRU gates and `H^{j,0}=EV` | 5, Eq. (9), text | PAPER_VERIFIED | Exact gate algebra and fresh hidden state per outer segment. |
| `b_r,b_u,b_o ∈ R^(N×D)` | 5, Eq. (9) explanation | PAPER_VERIFIED | Variable-specific GPRU bias `[14,D]`. |
| Dynamic graph gate and adjacency | 6, Eq. (10)–(11) | PAPER_VERIFIED | Literal gated graph equations; base `A=E` is bridged. |
| DGCN propagation/selection | 6, Eq. (12)–(13) | PAPER_VERIFIED | Two-hop, bidirectional details are preregistered bridges. |
| Period-level `GatedDG` | 6, Eq. (14) | PAPER_VERIFIED | One shared GatedDG at period level after masked intra aggregation. |
| Forecast MLP and all-variable MSE | 6, Eq. (15)–(16) | PAPER_VERIFIED | Shared all-variable forecast head; RealHyFC loss is target-specific adaptation. |

The paper contains insufficient executable detail to identify a unique code
implementation. The source-audit decision in Phase BG remains unchanged:
this is not an Official PDFNet claim.
