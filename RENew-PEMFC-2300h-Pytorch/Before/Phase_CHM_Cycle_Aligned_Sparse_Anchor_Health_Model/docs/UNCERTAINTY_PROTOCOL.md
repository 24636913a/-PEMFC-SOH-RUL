# Uncertainty Protocol

## Primary uncertainty method

The primary method is a **time-axis-preserving block-weight bootstrap** of the
small monotone latent-state network, not a Bayesian network and not a
concatenated resampled time sequence. Every replicate keeps the original HOU
timestamp order, frozen anchor positions, and frozen partial-HOU positions.

Within each legal contiguous complete-cycle segment, sample contiguous four-
cycle blocks with replacement and convert draws to integer multiplicity weights
on the original complete-cycle HOU. These weights apply to complete-cycle
observation loss and to smoothness through the minimum weight of every legal
complete-cycle triple. Partial HOU are not resampled and retain observation
weight one. Anchor losses and locations are never deleted because a block
weight is zero.

The primary analysis schedules 500 replicates with block length four and needs
at least 475 valid replicates. The predeclared sensitivity schedules 200 with
block length eight and needs at least 190 valid. Numerical optimizer failure is
recorded as `INVALID_REPLICATE` without replacement.

## Causality boundary

For causal-online evaluation at a cycle or held-out checkpoint, training,
block weights, and anchor perturbation may use only data permitted at that time.
Later anchors cannot appear in a bootstrap replicate for an earlier online
prediction. Offline retrospective bootstrap results must be labeled as such.

## Pending freeze

Anchor perturbations use the resolved formula
`sigma_pol² + sigma_int² + sigma_support²`, with weights normalized across the
training anchors of each fold. For each permitted training anchor,
`Y_k^(b) ~ Normal(Y_k, sigma_k²)`. The held-out outer target never appears in
training or perturbation. Bootstrap seed is 90210.

For every outer held-out anchor, the replicate distribution of its frozen
median-of-three-HOU functional supplies the 2.5%, 50%, and 97.5% values.
Coverage is a small-sample gate, not a statistical-significance claim.
