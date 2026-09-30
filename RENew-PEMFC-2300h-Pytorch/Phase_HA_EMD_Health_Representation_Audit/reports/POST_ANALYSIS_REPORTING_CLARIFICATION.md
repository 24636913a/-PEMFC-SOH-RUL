# Post-analysis reporting clarification

This clarification changes no Phase HA calculation, data access, protocol,
threshold, matching rule, bootstrap configuration, figure, CSV, or final
research state.  It corrects wording in the final report only.

`CONDITION_ADJUSTED_EFFECTS.csv` contains fitted point estimates but every
registered residual moving-block bootstrap has `bootstrap_success_rate = 0`,
`uncertainty_status = UNCERTAINTY_NOT_IDENTIFIABLE`, and infinite condition
number.  Therefore those coefficients cannot be described as established
condition-adjusted age associations.  The corrected report calls them
non-evidential point estimates and retains the exact final state:
`FORECAST_ONLY_REPRESENTATION — HEALTH_EVIDENCE_NOT_IDENTIFIABLE`.
