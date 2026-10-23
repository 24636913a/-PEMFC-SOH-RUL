# Late-Life Operating-Condition Support Drift Audit

## 1. Objective

This target-blind audit identifies why frozen Development joint support rejected the 2000 h latest-three P1S intervals while accepting two of three at 2300 h. It evaluates operating-condition support only, not health-prediction performance.

## 2. Data boundary / target-blind statement

Only frozen Development support, Development P1S checkpoint metadata, and existing target-blind Stage-P interval artifacts were read. No polarization target, Stage T, one-shot runner, ledger write, Y, AE, MAE, or Rule A/B/C calculation was performed.

## 3. Frozen Development support

| variable | support_lower_bound | support_upper_bound | support_width | late_life_IQR | late_life_typical_variation_over_support_width | support_box_assessment |
|---|---|---|---|---|---|---|
| anode_stoichiometry | 1.51162 | 1.51246 | 0.00084 | 0.00208 | 2.48527 | POTENTIALLY_OVERRESTRICTIVE_SUPPORT_DIMENSION |
| cathode_stoichiometry | 1.71098 | 1.71236 | 0.00138 | 0.00348 | 2.52193 | POTENTIALLY_OVERRESTRICTIVE_SUPPORT_DIMENSION |
| anode_inlet_RH_pct | 40.07957 | 40.48869 | 0.40912 | 0.21548 | 0.52667 | NOT_FLAGGED_BY_DESCRIPTIVE_RULE |
| anode_inlet_temperature_C | 75.01313 | 75.16282 | 0.14968 | 0.14968 | 1.00000 | NOT_FLAGGED_BY_DESCRIPTIVE_RULE |
| anode_hydrogen_flow_nl_min | 23.29630 | 23.31462 | 0.01831 | 0.03662 | 2.00010 | POTENTIALLY_OVERRESTRICTIVE_SUPPORT_DIMENSION |
| anode_inlet_pressure_kPag | 164.38960 | 164.49068 | 0.10107 | 0.00000 | 0.00000 | NOT_FLAGGED_BY_DESCRIPTIVE_RULE |
| current_density | 0.78923 | 0.78923 | 0.00000 | 0.00000 |  | NOT_FLAGGED_BY_DESCRIPTIVE_RULE |

The support bounds above are reported exactly as frozen. They were not refit, widened, or otherwise modified.

## 4. Late-life support timeline

The timeline contains `69` structurally legal, pre-characterization P1S intervals from `1504.310` to `2256.770` OpHrs2. It is exported in `outputs/LATE_LIFE_P1S_SUPPORT_TIMELINE.csv`.

## 5. 2000 h failure anatomy

All latest-three intervals were present within 48 h, but frozen support was `0/3`; G4 was the only failed alignment gate. The variables failing all three selected intervals were: `anode_stoichiometry, cathode_stoichiometry`.

Q1 — sudden or persistent: `PERSISTENT`. The final contiguous joint-support failure run before 2000 h contained `10` interval(s) in the last-ten audit, beginning at `1915.47` OpHrs2. In the full pre-2000 interval history, the persistent anode/cathode-stoichiometry violations both began at `1772.62` OpHrs2 and lasted `25` interval(s).

Q2 — same stable new condition: `YES_FOR_THE_PRIMARY_STOICHIOMETRY_SHIFT`. The three selected intervals share the same above-support anode-stoichiometry level, while cathode stoichiometry remains above support in all three; RH and temperature contribute smaller interval-specific differences.

Q3 — earliest sustained violation: `anode_stoichiometry, cathode_stoichiometry`. Q4 — variables causing all selected intervals to fail: `anode_stoichiometry, cathode_stoichiometry`.

The ten preceding structural-legal intervals are exported without target information in `outputs/CHECKPOINT_LATEST10_P1S_INTERVALS.csv`; their interval IDs, age gaps, all seven conditions, and per-variable support flags are retained there.

| recency_rank | cycle_interval_id | age_gap_h | joint_support | support_fail_variables |
|---|---|---|---|---|
| 1 | POSTMORTEM_PARTIAL_PRE_2000 | 0.030 | False | anode_inlet_temperature_C;anode_inlet_RH_pct;anode_stoichiometry;cathode_stoichiometry |
| 2 | COMPLETE_175 | 6.510 | False | anode_inlet_RH_pct;anode_stoichiometry;cathode_stoichiometry |
| 3 | COMPLETE_174 | 16.030 | False | anode_stoichiometry;cathode_stoichiometry |
| 4 | COMPLETE_173 | 25.560 | False | anode_inlet_temperature_C;anode_stoichiometry;cathode_stoichiometry |
| 5 | COMPLETE_172 | 35.090 | False | anode_inlet_temperature_C;anode_inlet_RH_pct;anode_hydrogen_flow_nl_min;anode_stoichiometry;cathode_stoichiometry |
| 6 | COMPLETE_171 | 44.610 | False | anode_inlet_temperature_C;anode_hydrogen_flow_nl_min;anode_stoichiometry;cathode_stoichiometry |
| 7 | COMPLETE_170 | 54.130 | False | anode_inlet_temperature_C;anode_inlet_RH_pct;anode_hydrogen_flow_nl_min;anode_stoichiometry;cathode_stoichiometry |
| 8 | COMPLETE_169 | 63.660 | False | anode_inlet_temperature_C;anode_inlet_RH_pct;anode_hydrogen_flow_nl_min;anode_stoichiometry;cathode_stoichiometry |
| 9 | COMPLETE_168 | 73.180 | False | anode_inlet_temperature_C;anode_inlet_RH_pct;anode_hydrogen_flow_nl_min;anode_stoichiometry;cathode_stoichiometry |
| 10 | COMPLETE_167 | 82.700 | False | anode_inlet_RH_pct;anode_hydrogen_flow_nl_min;anode_stoichiometry;cathode_stoichiometry |

## 6. 2300 h support recovery

Q5 — why 2300 h recovered: the frozen latest-three selection had `2/3` support because ranks 1–2 returned inside every frozen support dimension; rank 3 remained outside `anode_inlet_temperature_C, anode_stoichiometry`. The observed outcome is therefore a partial operating-condition recovery, not a change to the frozen support rule.

| recency_rank | cycle_interval_id | age_gap_h | joint_support | support_fail_variables |
|---|---|---|---|---|
| 1 | POSTMORTEM_PARTIAL_PRE_2300 | 0.030 | True |  |
| 2 | COMPLETE_200 | 6.520 | True |  |
| 3 | COMPLETE_199 | 16.040 | False | anode_inlet_temperature_C;anode_stoichiometry |
| 4 | COMPLETE_198 | 25.570 | False | anode_inlet_RH_pct;cathode_stoichiometry |
| 5 | COMPLETE_197 | 35.090 | False | cathode_stoichiometry |
| 6 | COMPLETE_196 | 44.620 | False | anode_inlet_temperature_C;anode_stoichiometry;cathode_stoichiometry |
| 7 | COMPLETE_194 | 62.980 | False | anode_stoichiometry;cathode_stoichiometry |
| 8 | COMPLETE_193 | 72.510 | False | anode_inlet_temperature_C;anode_inlet_RH_pct;anode_stoichiometry;cathode_stoichiometry |
| 9 | COMPLETE_192 | 82.410 | False | anode_stoichiometry;cathode_stoichiometry |
| 10 | COMPLETE_190 | 103.420 | False | anode_stoichiometry;cathode_stoichiometry |

## 7. Variable-level drift ranking

| variable | violation_count | violation_rate | below_count | above_count | longest_consecutive_violation_run | support_box_assessment |
|---|---|---|---|---|---|---|
| anode_stoichiometry | 51 | 0.73913 | 0 | 51 | 24 | POTENTIALLY_OVERRESTRICTIVE_SUPPORT_DIMENSION |
| cathode_stoichiometry | 44 | 0.63768 | 0 | 44 | 24 | POTENTIALLY_OVERRESTRICTIVE_SUPPORT_DIMENSION |
| anode_inlet_RH_pct | 31 | 0.44928 | 31 | 0 | 13 | NOT_FLAGGED_BY_DESCRIPTIVE_RULE |
| anode_inlet_temperature_C | 28 | 0.40580 | 0 | 28 | 10 | NOT_FLAGGED_BY_DESCRIPTIVE_RULE |

## 8. Regime analysis

The sequential support-state decomposition found `14` pass/fail regime segment(s); the complete chronology is in `outputs/JOINT_SUPPORT_REGIME_SEGMENTS.csv`. The 2000 h failure and 2300 h partial recovery show that support status is not a monotone late-life loss of P1S availability.

## 9. Is the support box overly restrictive?

Assessment: `POTENTIALLY_OVERRESTRICTIVE`. This is a descriptive statement based on frozen-width versus late-life P1S-condition variability; it does not authorize changing the frozen box or rerunning the Test.

## 10. Implications for P1S

P1S should be abandoned: `NO`. Principal weakness: `HARD_SUPPORT_GATE`. The evidence supports an observability/support conclusion only; it does not validate P1S health-prediction performance because targets remain blinded.

## 11. Recommended next protocol

Recommended route: `Route A - Condition-Normalized P1S Health Representation`. Any future protocol would need a separate authorization and must not reinterpret or rerun the consumed AHC-TV Test.

## Target-blind boundary

`TARGET_BLIND_ONLY`; one-shot rerun `NO`; ledger modified `NO`; polarization targets remain blinded `YES`.
