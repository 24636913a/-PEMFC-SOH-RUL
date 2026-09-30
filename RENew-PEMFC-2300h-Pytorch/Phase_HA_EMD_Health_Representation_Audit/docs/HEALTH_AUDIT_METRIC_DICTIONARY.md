# Health-audit metric dictionary

| Item | Definition | Evidence label |
|---|---|---|
| Online endpoint | Cache feature `[47, :]` released at that origin from its own 48-point past-only history | PROJECT_EVIDENCE |
| `D_X` | Median late 24 h block endpoint minus median early 24 h block endpoint in one common-support stratum | PROJECT_ANALYSIS_DESIGN |
| ASR | `abs(D_X) / median(abs(adjacent online endpoint difference))` in the same stratum | PROJECT_ANALYSIS_DESIGN |
| Online roughness | Distribution and variation of adjacent, same-segment endpoint differences | PROJECT_ANALYSIS_DESIGN |
| Retrospective revision `U_k` | Endpoint at origin `t+k`, index `47-k`, minus the online endpoint of the same physical time at `t`, for k=1/2/4/8 | PROJECT_ANALYSIS_DESIGN |
| RDR95 | `P95(abs(U_1)) / abs(D_EMD)` where the denominator is a nonzero common-support EMD age contrast; otherwise undefined | PROJECT_ANALYSIS_DESIGN |
| Condition-adjusted association | Fixed OLS model with age and frozen operating-condition covariates. It is not a causal effect. | PROJECT_ANALYSIS_DESIGN |
| 24 h / 48 h blocks | Consecutive 0.5 h origins, retained only inside segment and chronological-third bounds | PROJECT_ANALYSIS_DESIGN |

No multi-unit trendability or prognosability metric is calculated: the three
fold views belong to one stack trajectory and are not independent fuel-cell
units.

