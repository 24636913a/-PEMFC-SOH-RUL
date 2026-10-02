# Open Technical Decisions — CHM Protocol Review Only

These are the only CHM design points requiring technical review before an
executable protocol is frozen. They are deliberately unresolved; no model fit
may choose them empirically in this phase.

| Decision | Required frozen choice before first fit | Prohibited resolution route |
| --- | --- | --- |
| Bootstrap block length | One cycle-block length justified from the 9.52 h frozen cycle scale, plus a prespecified sensitivity if retained | Test coverage or Test error |
| Ensemble size | A fixed number of legal bootstrap replicates and invalid-replicate handling | Stopping once a preferred interval appears |
| Lambda grid and selection | A finite all-Development grid and nested/LOAO-compatible selection rule | Held-out anchor, Test, or post-hoc residual tuning |
| B1/B2 definitions | Exact isotonic implementation and exact constrained low-order monotone alternative | Selecting the baseline after seeing Test results |
| Latent-step scale | A positive fixed scale or a mathematically equivalent, pre-specified parameterization | Clipping trajectories after fitting |
| Anchor-noise formula | Robust estimator for polarization repeat dispersion, three-interval spread, epsilon, and units | Reweighting from model residuals |
| Robust observation loss | Exact `ρ` and its scale | Adapting it separately to favorable anchors |
| Early stopping | A held-out-anchor-safe criterion, epoch cap, seed policy, and failure rule | Monitoring the held-out anchor or Test |
| Condition residual head | Explicitly include or exclude a small residual head before execution | Adding it after primary-model results |
| Causal-online scoring | Exact prefix-training rule, BoL-only 300 h treatment, and aggregation | Folding future anchors into an online score |

The following are **not** open: P1S operation point/token, RCR cycle identity,
PAA 48 h / three-interval / 2-of-3 alignment, the all-six primary anchor set,
S1 anchor membership, age prohibition, Test firewall, backbone closure, and
RUL prohibition.
