# Next-phase RealHyFC-specific mechanism contract

## Status and boundary

This is a forward-looking interface contract, not authorization to train. The targeted TFT route terminated as `TFT_NOT_VIABLE`; no new general forecasting backbone may be searched. Any separately approved next phase must examine a falsifiable **RealHyFC-specific mechanism** on TimesNet or on an independently specified latent-state carrier, not add Mamba, ModernTCN, PatchTST, Informer, Autoformer, or another backbone.

The present Raw F2-only test deliberately excluded cycle/health mechanisms to preserve clean attribution of the local recurrent-cell question. That exclusion does not imply that RealHyFC operating structure is unimportant.

## Required mechanism interface

| Contract item | Frozen requirement | Leakage boundary |
| --- | --- | --- |
| A. Repeated duty cycle | Treat the approximately 9.5 h operating-age repeat as a hypothesis to be operationalized before training. | Never infer an origin's phase from future actual current. |
| B. Repeated operation points | Define repeated current/current-density states and their support using Development-only data. | Do not select states by Test error or future target voltage. |
| C. Cycle phase | Any phase feature must be causal and origin-known at forecast issuance. | No future actual current, target, or post-origin schedule reconstruction. |
| D. Rich operating conditions | Reconsider N1-approved online temperature, pressure, humidity and flow variables separately from the present Raw F2 baseline. | Explicitly classify each as past-known, origin-known, future-unknown, or protocol-known; future actual values remain unavailable in the strict scenario. |
| E. Fast state | Preserve a causal 0.5 h dynamic operating state. | Input history must obey the frozen split, segment and continuity rules. |
| F. Slow health state | Prefer a cycle-level update rule, specified before observing model scores. | No dense interpolation of sparse health anchors into 0.5 h ground truth. |
| G. Sparse polarization anchors | Use `j*=1.0 A/cm²` as primary and `j*=1.6 A/cm²` as sensitivity anchors only for a slow-health state. | No Test anchors for development choices; no SOH/RUL claim without a separately validated target construction. |

## Minimum preregistration requirements for any future mechanism study

1. Define the causal availability of every proposed variable and independently test its future-perturbation invariance.
2. Freeze a Development-only split, origin universe, task semantics, unit of evaluation and baseline before scores are examined.
3. State one mechanism-specific, falsifiable hypothesis and an ablation in which all non-mechanism components are held fixed.
4. Keep the target as dynamic voltage forecasting unless a separate phase independently establishes an SOH target; do not imply SOH or RUL from forecast quality.
5. Preserve `TIMESNET_FREEZE`, `BACKBONE_SEARCH_CLOSED`, `EMD_SPECIFIC_ADVANCE`, and Phase HA `FORECAST_ONLY_REPRESENTATION — HEALTH_EVIDENCE_NOT_IDENTIFIABLE` until an authorized study can change them.

No training, cycle-aware module, fast/slow model, sparse-anchor loss, SOH construction, or RUL model is created by this contract.
