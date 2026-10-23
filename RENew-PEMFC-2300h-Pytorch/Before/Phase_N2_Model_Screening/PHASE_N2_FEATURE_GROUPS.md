# Phase N2 Feature Groups（特征组）

所有特征均来自 prediction origin（预测起点）及以前的 Phase N1 causal table（因果表）。未来真实工况不输入模型。每个 fold 的 median imputer（中位数填补器）和 standard scaler（标准缩放器）只在该 fold Train 上拟合。

## F1 — Voltage State（电压状态，5列）

| 精确列名 | 物理含义 |
| --- | --- |
| `cell_voltage_mean` | 10 个有效单体平均电压 |
| `cell_voltage_std` | 同一 bin 的跨单体电压标准差 |
| `cell_voltage_range` | 跨单体电压极差 |
| `cell_voltage_min` | 有效单体最低电压 |
| `cell_voltage_max` | 有效单体最高电压 |

## F2 — Voltage + Load（电压+负载，14列）

F1 加：

| 精确列名 | 物理含义 |
| --- | --- |
| `actual_current_A_mean` | 实际电流均值 |
| `actual_current_A_std` | 实际电流 bin 内标准差 |
| `actual_current_A_peak_to_peak` | 实际电流峰峰值 |
| `actual_current_density_A_cm2_mean` | 实际电流密度均值 |
| `actual_current_density_A_cm2_std` | 实际电流密度标准差 |
| `actual_current_density_A_cm2_peak_to_peak` | 实际电流密度峰峰值 |
| `load_transition_count` | 负载转换次数 |
| `operation_point_transitions` | 运行点转换次数 |
| `zero_current_fraction` | 零电流样本占比 |

## F3 — Condition-Aware（运行条件感知，27列）

F2 加以下 actual means（实际均值）：

| 精确列名 | 物理含义 |
| --- | --- |
| `actual_anode_inlet_temperature_C_mean` | 阳极入口温度 |
| `actual_cathode_inlet_temperature_C_mean` | 阴极入口温度 |
| `actual_coolant_inlet_temperature_C_mean` | 冷却液入口温度 |
| `actual_coolant_outlet_temperature_C_mean` | 冷却液出口温度 |
| `actual_anode_inlet_pressure_kPag_mean` | 阳极入口表压 |
| `actual_cathode_inlet_pressure_kPag_mean` | 阴极入口表压 |
| `actual_anode_inlet_rh_pct_mean` | 阳极入口相对湿度 |
| `actual_cathode_inlet_rh_pct_mean` | 阴极入口相对湿度 |
| `actual_anode_hydrogen_flow_nl_min_mean` | 阳极氢气流量 |
| `actual_cathode_air_flow_nl_min_mean` | 阴极空气流量 |
| `actual_coolant_flow_l_min_mean` | 冷却液流量 |
| `actual_anode_stoichiometry_mean` | 阳极化学计量比 |
| `actual_cathode_stoichiometry_mean` | 阴极化学计量比 |

未加入 exported voltage、set points、重复的 min/max/range、metadata、coverage diagnostic、EIS、polarization 或 spatial EIS。F3 的目的不是穷举列，而是检验核心实际运行条件是否带来稳定增益。
