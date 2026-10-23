# Phase PSDM — Probabilistic Degradation State Model Feasibility

This formal Development-only phase evaluates a causal filtered latent
degradation health state based solely on historically observed P1S retention
and physical raw-cycle identities.  It has two intentionally separated stages:

1. Freeze a 12-row, Development-only polarization anchor snapshot using the
   Git-pinned `polarization_points_development.csv` source.
2. Fit and evaluate Persistence, ULLT, and PSDM state-space models on the
   frozen CMRF H3/H6 authorities.

The first attempt at this phase was stopped before training because it opened a
mixed N1 polarization table.  That historical access remains recorded; the
current run uses only the allowlisted source in Stage A and does not access
Test or late-life targets.
