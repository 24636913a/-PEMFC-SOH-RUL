# Posthoc Gate Status Clarification

## Scope and preserved history

This is a documentation-only clarification issued after the original Phase RHA
execution. It does not change the frozen protocol, any original output, the
operation-point selection, or the final dual-point status:

`CONDITION_COMMON_SUPPORT_INSUFFICIENT`.

The original `final/FINAL_DECISION.yaml` remains preserved as an historical
execution record. Its serialized `C: false` and `D: false` fields must not be
read as negative empirical results.

## Correct gate statuses

| Gate | Correct status | Basis |
| --- | --- | --- |
| Gate A: point support | PASS | P1 and P2 met the frozen evaluation-support requirements. |
| Gate B: condition completeness | PASS | Frozen condition-completeness requirements were met. |
| Joint common support | FAIL | P2 had 13 / 22 / 11 EARLY / MIDDLE / LATE joint-support cycles; the frozen minimum is 15 in every stratum. |
| Gate C: voltage visibility | NOT_RUN | The preregistered stop followed the joint-support failure. |
| Gate D: sparse-anchor compatibility | NOT_RUN | Development polarization access is allowed only after Gates A--C pass. |

No formal Gate-C early--late voltage contrast, condition-matched voltage
contrast, or Gate-C bootstrap was calculated. Development polarization was not
read, and Gate-D anchor compatibility was not calculated. Consequently, the
historical values `C: false` and `D: false` do **not** mean
`VOLTAGE_DRIFT_NOT_IDENTIFIABLE` or
`SPARSE_ANCHOR_COMPATIBILITY_NOT_SUPPORTED`.

## Raw Test access clarification

The boundary-aware stream stopped in the crossing file
`Loadcycle_151-162.csv`. Its frozen raw column set included `OpHrs2`,
`DateTime`, operation-point/current fields, active-cell voltages, and the seven
condition variables. Therefore the precise record is:

- `raw_post_boundary_rows_accessed = 31316`
- `raw_post_boundary_rows_used = 0`
- `test_raw_rows_accessed = true`
- `test_raw_rows_used_for_analysis = false`
- `test_raw_rows_used_for_threshold_selection = false`
- `test_raw_rows_used_for_decision = false`
- `test_polarization_accessed = false`
- `test_EIS_accessed = false`

In Chinese: 跨界 chunk 中有 31,316 条边界后原始行被读取，读取列包含冻结的电压和工况字段；这些行随后在进入稳定驻留窗分配、周期测量、工况支持、匹配、绘图、门槛或决策前被过滤。

Thus this phase must not be described as “Test accessed = false.” The accurate
statement is: **Test raw rows were accessed in the boundary-crossing chunk but
not used.**
