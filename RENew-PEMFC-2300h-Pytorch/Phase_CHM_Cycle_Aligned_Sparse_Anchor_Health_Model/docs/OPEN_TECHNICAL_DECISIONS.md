# Technical Decisions Resolved Before Execution

CHM Technical Review Resolution V1 closes the former candidate-level technical
decisions. The candidate is `TECHNICALLY_RESOLVED_AWAITING_FINAL_FREEZE`, not
`EXECUTION_AUTHORIZED`; final review may accept or reject the complete protocol
but no fit may reopen a definition after seeing results.

| Former decision | Resolved candidate definition |
| --- | --- |
| Anchor mapping | Median of exactly three frozen HOU latent states |
| BoL / initial state | BoL normalization only; learned `H_1=sigmoid(eta)` |
| Latent transition scale | Multiplicative softplus decay with `1e-3` numerical constant |
| Input schema | `R-1` plus seven reference-standardized conditions; constants audited/excluded |
| Model / residual head | `Linear(d,16) → SiLU → Linear(16,1)`; residual head excluded |
| Huber loss | Delta `0.001` |
| Lambda selection | 27-point grid; nested LOAO mean MAE; deterministic `1e-8` tie rule |
| Seeds / early stop | 42, 2024, 3407; training composite loss only |
| B1 / B2 | Isotonic (`increasing=True`, clip) / nonnegative affine |
| Anchor uncertainty | Polarization SD + 3-interval MAD + support-median component |
| LOAO metrics | No per-anchor Spearman; aggregate descriptive metrics use six predictions |
| Coverage / S1 | 5/6 small-sample coverage gate; three numeric S1 thresholds |
| Bootstrap | Original time axis, 4-cycle/500 primary, 8-cycle/200 sensitivity weights |
| Online diagnostic | 1150/1500/1750 only; no future anchors |
| Test wording | CHM nonaccess is future-stage specific; historical records retained |

## No remaining blocking technical decision

`remaining_blocking_technical_issues: []` is recorded in
`configs/CHM_TECHNICAL_REVIEW_RESOLUTION_V1.yaml`. The following remain frozen
and are not revisitable in execution: P1S operation point/token, RCR cycle
identity, PAA 48 h / three-interval / 2-of-3 alignment, all-six primary anchor
set, S1 membership, age prohibition, Test firewall, backbone closure, and RUL
prohibition.
