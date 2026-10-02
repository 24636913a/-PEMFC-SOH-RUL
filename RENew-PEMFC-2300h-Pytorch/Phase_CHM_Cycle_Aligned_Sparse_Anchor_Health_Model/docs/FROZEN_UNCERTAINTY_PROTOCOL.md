# Frozen CHM uncertainty protocol

Final V1 uses `TIME_AXIS_PRESERVING_BLOCK_WEIGHT_BOOTSTRAP` only after
deterministic Development Gates 1–4 pass. Bootstrap is never used to select a
loss profile.

The seven allowed Development polarization CSV files contain no explicit
replicate, trial, or sweep identifier. Consequently, the j*=1.0 ± 0.01 local
scatter is named `sigma_pol_local_proxy`. It is a within-file local
polarization-window scatter proxy, not independent-repeat measurement variance
and not probabilistic label variance. The anchor reliability scale is
`q² = sigma_pol_local² + sigma_int² + sigma_support²`; its normalized inverse
is a `reliability_weight`, not an inverse-variance statistical weight.

`anchor_target_gaussian_perturbation` is `DISABLED`. Primary uncertainty is
cycle-block/optimization-data variation only: six outer 200-replicate
block-4 runs (minimum 190 legal each), followed only after qualification by a
500-replicate all-six block-4 run (minimum 475) and a 100-replicate all-six
block-8 sensitivity run (minimum 95). Partial HOU are never resampled, anchor
positions stay fixed, and the original time order is never reordered.
