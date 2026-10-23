# Feature dictionary

The 14 Raw F2 columns remain in the exact frozen Phase BS order. All augmented
routes append two `float32` columns at positions 14 and 15.

| Route | Position 14 | Position 15 | Availability |
| --- | --- | --- | --- |
| SimpleTrend | `causal_mean8_V` | `causal_slope8_V_per_h` | per-position causal within the observed block |
| VMDFeature | `vmd_low_V` | `vmd_low_slope8_V_per_h` | origin-conditioned causal block |
| EMDFeature | `emd_remainder2_V` | `emd_remainder2_slope8_V_per_h` | origin-conditioned causal block |

VMD low mode, EMD remainder, center frequencies, reconstruction residuals, IMF
counts and IMF energy are not global row-level features and are never targets.
