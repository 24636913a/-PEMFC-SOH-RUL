# Final P1S pre-anchor alignment audit

## Scope

This is a **POSTHOC_EXPLORATORY** audit of the fixed state token
`221||221.200`. It does not change Phase RCR, dual-point RHA, or P1 results.
The representation is **P1S Operating-Voltage Retention**, not SOH.

## Firewall

Test raw rows accessed: **False**. Rows accessed
after the latest required pre-characterization anchor cutoff:
**0**. Test polarization
and EIS accessed: **False**. No source file after the frozen
`Loadcycle_134-150.csv` cutoff was opened.

## Final result

**`P1S_PREANCHOR_COMPATIBILITY_EXPLORATORY_PASS`**

P1S support: **True**; long-term drift:
**True**; joint condition support:
**True**; matched direction:
**True**; anchor coverage:
**True**.

| Quantity | Value |
| --- | ---: |
| Raw EARLY–LATE contrast | -0.00705327 |
| Local variation | 0.00019139 |
| 4-cycle CI upper | -0.00628141 |
| Matched contrast | -0.00658831 |

| checkpoint_h | selected_cycle_interval_ids | joint_support_interval_count | status | P1S_anchor_retention |
| --- | --- | --- | --- | --- |
| 300 | PARTIAL_PRE_300|COMPLETE_024|COMPLETE_023 | 3.0 | ALIGNED | 0.9956275785669164 |
| 600 | PARTIAL_PRE_600|COMPLETE_049|COMPLETE_048 | 2.0 | ALIGNED | 0.9915913989148296 |
| 850 | PARTIAL_PRE_850|COMPLETE_074|COMPLETE_073 | 3.0 | ALIGNED | 0.990112305187449 |
| 1150 | PARTIAL_PRE_1150|COMPLETE_100|COMPLETE_099 | 2.0 | ALIGNED | 0.985142214444108 |
| 1500 | PARTIAL_PRE_1500|COMPLETE_125|COMPLETE_124 | 2.0 | ALIGNED | 0.9865916156775026 |
| 1750 | PARTIAL_PRE_1750|COMPLETE_150|COMPLETE_149 | 2.0 | ALIGNED | 0.9856019202793188 |

No neural model, continuous SOH, RUL model, or automatic next phase was created.
