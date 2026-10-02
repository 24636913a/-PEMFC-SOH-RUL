# Test Firewall Plan

## Current state

All Test artifacts remain sealed during CHM protocol design and all later
Development model-selection work. This includes 2000 h and 2300 h
polarization, 2300 h after-Air-Air-start polarization, Test EIS, Test raw
operating data, Test anchor targets, Test model metrics, and Test-derived
thresholds.

No path in CHM may read a Test file. The current phase creates neither a data
loader nor any model code, so no indirect Test access is possible.

## Future one-time opening

Only a separate `Phase_THV — Test Health Validation` may open Test after all of
the following have been committed and hash-frozen:

1. architecture and latent-step scale;
2. final lambda grid and Development-only selection rule;
3. baseline implementations;
4. anchor-noise and bootstrap protocol;
5. preprocessing and condition schema;
6. causal-online evaluation rule; and
7. source and executable code hashes.

2000 h and 2300 h are an external temporal holdout for late-life extrapolation,
not a tuning resource. The after-Air-Air-start artifact is a separately
reported intervention/recovery sensitivity and must never be blended into
normal degradation Test scoring.

Test validation, even if positive, does not automatically authorize RUL.
