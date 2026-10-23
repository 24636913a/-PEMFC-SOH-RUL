# Frozen CHM uncertainty protocol — Final V2

The sparse-anchor reliability term is `LOCAL_WINDOW_SCATTER_PROXY`. It is not
independent-repeat variance and not a probabilistic label variance. For each
anchor, `q² = sigma_pol_local² + sigma_int² + sigma_support²`; its inverse is a
training-anchor mean-normalized `reliability_weight`.

Anchor-target Gaussian perturbation is `DISABLED`; MC Dropout is prohibited.
Uncertainty therefore represents time-axis-preserving block-weight resampling
and optimization/data variation, not a complete measurement-error posterior.

Bootstrap never selects profiles. It runs only after deterministic Gates 1–4
pass: 200 block-4 replicates per outer fold (minimum 190 legal), then 500
all-six block-4 replicates (minimum 475) and 100 all-six block-8 sensitivity
replicates (minimum 95). Partial HOU are not resampled, anchor positions remain
fixed, and the original HOU chronology is never reordered.
