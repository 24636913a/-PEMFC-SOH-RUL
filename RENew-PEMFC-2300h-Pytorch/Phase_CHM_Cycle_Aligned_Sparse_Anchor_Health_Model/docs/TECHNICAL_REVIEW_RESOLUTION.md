# Technical Review Resolution V1

## Resolution outcome

The first CHM draft had several implementation-blocking ambiguities. This V1
resolution converts them into a technically defined candidate protocol while
keeping `execution_authorized=false`. It neither trains nor evaluates a model.

| Former ambiguity | V1 resolution |
| --- | --- |
| Anchor-to-cycle mapping | Median of three frozen HOU latent states; no nearest-cycle replacement |
| Complete and partial interval mixture | 146 `COMPLETE_CYCLE_HOU` plus exactly six frozen `PREANCHOR_PARTIAL_HOU` |
| BoL versus first state | BoL is target normalization only; `H_1=sigmoid(eta)` |
| Latent step scale | Fixed multiplicative decay with numerical constant `1e-3` |
| Redundant condition inputs | `R-1` plus reference-standardized seven conditions only |
| Capacity / residual head | `Linear(d,16) → SiLU → Linear(16,1)`; residual head excluded |
| Anchor weighting | Frozen polarization, interval-MAD, and support-median variance components |
| Model selection | 27-point grid, nested LOAO, three frozen seeds, deterministic tie break |
| Baselines | B0 same median functional; B1 isotonic; B2 nonnegative affine |
| Per-anchor correlation | Prohibited; aggregate descriptive correlation has n=6 |
| Coverage / S1 | 5/6 small-sample coverage gate; three quantitative S1 thresholds |
| Bootstrap chronology | Original HOU time axis preserved; block multiplicity weights only |
| Online diagnostic | 1150/1500/1750 aggregate only; no future anchors |
| Test wording | CHM-specific nonaccess statement; historical project records retained |

## No remaining execution-blocking mathematical contradiction

The candidate has no remaining `BLOCKING_TECHNICAL_ISSUE` identified by this
static review. Its state is therefore
`TECHNICALLY_RESOLVED_AWAITING_FINAL_FREEZE`, not
`EXECUTION_AUTHORIZED`.

Final protocol review must inspect the candidate, source hashes, and static
test record before any implementation or training request. It may approve or
reject the candidate; it may not silently change its frozen definitions after
model results exist.

## Historical integrity

This resolution does not replace RCR/RHA/P1/PAA/PAV results. In particular,
PAA remains posthoc exploratory and PAV's 6/6 temporal qualification plus 3/6
median-source joint-support label remain descriptive constraints on CHM sparse
supervision, not proof of a validated SOH.
