# Frozen Uncertainty Protocol

## Time-axis-preserving block-weight bootstrap

CHM uses a **time-axis-preserving block-weight bootstrap**, not a synthetic
resampled time sequence. Every replicate retains the original strict HOU
timestamp order, original anchor positions, and original partial-HOU positions.

Within every legal contiguous complete-cycle segment, form contiguous blocks of
four complete cycles for the primary analysis and eight for sensitivity. Sample
those blocks with replacement and convert the draw to non-negative integer
multiplicity weights on the **original** complete-cycle HOU. No sampled block is
concatenated into a new chronology.

Complete-cycle multiplicities weight observation loss. For a legal consecutive
complete-cycle smoothness triple, use the minimum of its three multiplicities.
Partial HOU are never block-resampled and have observation weight one. Anchor
loss retains its frozen uncertainty weight; anchors are not deleted when an
associated complete-cycle block has weight zero.

## Anchor uncertainty and perturbation

For each training anchor,

\[
\sigma_{pol,k}=SD(V_{pol,k,j^*=1.0})/V_{pol,BoL},
\]

with at least two frozen polarization repeats; otherwise the run fails closed.

\[
\sigma_{int,k}=1.4826\,MAD(R_{k,1},R_{k,2},R_{k,3}),
\]

\[
\sigma_{support,k}=|\operatorname{median}(R_{all3})-
\operatorname{median}(R_{joint-support})|,
\]

\[
\sigma_k^2=\sigma_{pol,k}^2+\sigma_{int,k}^2+\sigma_{support,k}^2,
\qquad w_k=(\sigma_k^2+10^{-8})^{-1}.
\]

Weights are normalized to mean one only across the training anchors of that
fold. For each replicate and each permitted training anchor,
`Y_k^(b) ~ Normal(Y_k, σ_k²)`. Held-out outer anchors are never perturbed for
training, and model residuals never estimate or update sigma.

## Replicate accounting

Use 500 scheduled primary replicate indices (block length 4) and 200 scheduled
sensitivity indices (block length 8), based on bootstrap seed 90210. An
optimizer numerical failure is an `INVALID_REPLICATE`, recorded without a
replacement draw. Primary analysis requires at least 475 valid replicates;
sensitivity requires at least 190. Otherwise uncertainty is invalid and CHM
cannot receive Development qualification.

For each outer LOAO held-out anchor, primary replicates train on the original
time axis without its target. The replicate distribution of the frozen
median-of-three-HOU functional gives 2.5%, 50%, and 97.5% values for coverage.
