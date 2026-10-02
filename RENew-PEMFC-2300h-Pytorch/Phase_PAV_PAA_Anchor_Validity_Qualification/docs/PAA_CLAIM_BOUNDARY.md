# PAA Claim Boundary after PAV Qualification

Phase PAV is a `POSTHOC_QUALIFICATION_AUDIT`.  It preserves the frozen Phase
PAA result `P1S_PREANCHOR_COMPATIBILITY_EXPLORATORY_PASS`; it does not validate
P1S as SOH, create a continuous health truth, or make a causal claim.

## Permitted statement

> The fixed P1S operating-voltage representation showed exploratory compatibility with sparse Development polarization anchors under the registered pre-characterization alignment procedure.

## Required qualifications

- Internal `DateTime` fields in the six Development polarization CSVs establish
  temporal order only.  The nominal `300h` … `1750h` filename labels were not
  used as exact times.
- The registered PAA 2-of-3 complete-cycle joint-common-support rule remains
  unchanged.  The frozen median source itself is inside that support for
  **3/6** anchors.  At the remaining anchors, the registered
  rule passed but the final median can be contributed by an interval outside
  the complete-cycle joint-support region.
- PAA condition matching preserved a negative direction, but it had no
  preregistered distance cutoff and does not prove all operating-condition
  confounding was eliminated.  The appropriate phrase is
  `condition-adjusted directional robustness`, not complete condition matching.

## Prohibited statements

Do not call P1S a validated SOH, an independent SOH validation, continuous
health truth, a causal degradation indicator, or evidence that all anchor
estimates are fully condition matched.

## Public-context sources

The [Zenodo underlying-data record](https://zenodo.org/records/20715542) and
the [associated RealHyFC paper record](https://zenodo.org/records/20742676)
describe a >2000 h durability test with regular electrochemical
characterizations and nominal polarization datasets.  They do **not** provide
the exact DateTime of each of the six characterization runs; the internal CSV
timestamps, not those public pages, are the time provenance used here.
