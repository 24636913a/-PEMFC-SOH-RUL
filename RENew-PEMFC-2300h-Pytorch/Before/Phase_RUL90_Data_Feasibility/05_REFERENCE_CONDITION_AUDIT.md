# 参考工况审计（Reference-condition audit）

## 冻结规则

候选只来自 Development 可见的名义极化曲线：BoL、300、600、850、1150、1500、1750 h；排除同龄 `afterAirAirStart` 文件。候选 `j*` 为 [0.8, 1.0, 1.2, 1.4, 1.6] A/cm²，容差 ±0.01 A/cm²。硬门限要求诊断覆盖率≥90%、至少 5 个 Development checkpoint、距 500 mV lower voltage limit 至少 50 mV，并比较重复离散度、单调性、年龄相关性与 1500 h Air-Air shift。规则在查看任何 Test 电压前保存为 `outputs/METHOD_FREEZE.yaml`。

## 候选证据

| target_jstar_A_cm2 | development_checkpoint_count | development_checkpoint_total | coverage | voltage_start_V | voltage_end_V | relative_loss | median_repeat_std_V | spearman_age_voltage | monotonicity | non_increasing_fraction | air_air_shift_1500h_V | minimum_margin_to_500mV_V | hard_gate_pass | selection_score |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1.0 | 7 | 7 | 1.0 | 0.6989533 | 0.6792323 | 0.0282151 | 0.0005537 | -0.9642857 | 0.9596071 | 0.8333333 | 0.0039156 | 0.1792323 | True | 0.7518452 |
| 0.8 | 7 | 7 | 1.0 | 0.7181279 | 0.7040559 | 0.0195955 | 0.0035051 | -0.9642857 | 0.9562735 | 0.8333333 | 0.0027865 | 0.2040559 | True | 0.6666667 |
| 1.4 | 7 | 7 | 1.0 | 0.6575884 | 0.6314462 | 0.0397547 | 0.0020307 | -0.9642857 | 0.9974405 | 0.8333333 | 0.0056338 | 0.1314462 | True | 0.6519984 |
| 1.2 | 7 | 7 | 1.0 | 0.6769584 | 0.6546221 | 0.0329951 | 0.0025668 | -0.9642857 | 0.9793092 | 0.8333333 | 0.0046814 | 0.1546221 | True | 0.6359389 |
| 1.6 | 7 | 7 | 1.0 | 0.637379 | 0.6075992 | 0.0467223 | 0.0013662 | -1.0 | 1.0 | 1.0 | 0.0064545 | 0.1075992 | True | 0.4541188 |

## 配置可比性

Development 名义 polarization 文件的 `Number of cells` 与 `Membrane active area [cm2]` 统计见 `outputs/polarization_configuration.csv`。本指标使用 active 10-cell per-cell mean；`U6` 不进入计算。BoL 文件为源文件标注的 BoL checkpoint，但其内部 `OpHrs` 非零，因此称为“source-labelled BoL reference”，而非未经限定的零使用小时状态。
