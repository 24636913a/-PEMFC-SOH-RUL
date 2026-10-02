# Uncertainty Protocol

## Primary uncertainty method

The proposed primary uncertainty method is a **Block Bootstrap Ensemble** of
the small monotone latent-state network. It is not a Bayesian network.

The resampling unit is a contiguous block of RCR-frozen complete cycles. A
replicate jointly resamples the cycle observation, all seven conditions,
condition deviations, and any anchor record attached to that cycle. It must
not independently resample individual cycle rows or reorder information within
a block. A replicate that cannot form its predeclared legal sparse-supervision
set is invalid and must be counted and reported, not silently repaired.

For each legal replicate, anchor targets may receive a Gaussian perturbation
drawn from the frozen `σ_pol,k² + σ_cycle,k²` rule. The perturbation is based
only on Development measurements and is independent of Test coverage and model
residuals.

The ensemble returns, for every original cycle time coordinate, the median,
2.5th percentile, and 97.5th percentile of `H_c`.

## Causality boundary

For causal-online evaluation at a cycle or held-out checkpoint, training,
resampling, and anchor perturbation may use only data permitted at that time.
Later anchors cannot appear in a bootstrap replicate for an earlier online
prediction. Offline retrospective bootstrap results must be labeled as such.

## Pending freeze

The exact contiguous block length, number of replicates, robust spread
estimator, epsilon, polarization-repeat estimator, and invalid-replicate rule
must be decided before the first model fit. Candidate block lengths should be
chosen from cycle-scale quantities rather than from Test performance; no value
is frozen by this design-only phase.

Bootstrap coverage is a calibration diagnostic with only six non-BoL anchors;
it is not a basis for statistical-significance claims.
