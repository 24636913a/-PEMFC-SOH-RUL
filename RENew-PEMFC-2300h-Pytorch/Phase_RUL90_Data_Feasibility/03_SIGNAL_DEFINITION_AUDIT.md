# 信号定义审计（Signal definition audit）

`U.S.AveCell [V]` 是导出的单体平均电压；同时重算 active channels `U1-U5,U7-U11` 的平均值作为交叉检查。`U6 [V]` 按原始源统计表现为非物理/flatline 候选，因此不进入 active-cell SOH 或 dispersion。`MIN(U.S [V])` 仅被标为 stack-voltage candidate；其 MIN 统计语义未由原始字段定义证实。

所有异常仅标记：由 `|ΔU.S.AveCell|≥0.05 V` 产生的跳变根据同秒 `ΔI` 被区分为低负载/停机、正常负载响应或原因不明；未删除任何原始样本。示例记录在 `outputs/voltage_jump_events.csv`。

## 核心字段统计

| original_field | category | unit | physical_meaning | raw_count | missing_count | missing_percent | zero_percent | min | max | mean | std | p1 | p50 | p99 | abnormal_percent | sampled_quantile_count |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| U.S.AveCell [V] | average cell voltage | V | exported average single-cell voltage | 8604654 | 0 | 0.0 | 0.0 | -0.019947 | 0.943248 | 0.5766344 | 0.2828701 | 0.0010937 | 0.706766 | 0.8425213 | 0.0 | 8613 |
| I.S.act [A] | current | A | actual stack current | 8604654 | 0 | 0.0 | 19.7741013 | 0.0 | 505.2644348 | 168.3246084 | 116.375271 | 0.0 | 156.2852936 | 504.2878418 | 0.0 | 8613 |
| j.S.act [A/cm2] | current density | A/cm2 | actual stack current density | 8604654 | 0 | 0.0 | 19.7745894 | 0.0 | 1.804516 | 0.6011587 | 0.4156261 | 0.0 | 0.558162 | 1.801028 | 0.0 | 8613 |
| U1 [V] | cell voltage | V | individual voltage channel | 8604654 | 0 | 0.0 | 0.0 | -0.039369 | 0.950896 | 0.5760511 | 0.2824734 | 0.001343 | 0.705649 | 0.8423084 | 0.0 | 8613 |
| U2 [V] | cell voltage | V | individual voltage channel | 8604654 | 0 | 0.0 | 3.49e-05 | -0.045167 | 0.933622 | 0.5758268 | 0.282636 | 0.001099 | 0.706015 | 0.8413828 | 0.0 | 8613 |
| U3 [V] | cell voltage | V | individual voltage channel | 8604654 | 0 | 0.0 | 0.4030028 | -0.022462 | 0.947844 | 0.5767595 | 0.2837188 | -0.000427 | 0.70748 | 0.8413901 | 0.0 | 8613 |
| U4 [V] | cell voltage | V | individual voltage channel | 8604654 | 0 | 0.0 | 0.2630786 | -0.021058 | 0.938017 | 0.5768156 | 0.2834454 | 0.000122 | 0.707419 | 0.8411973 | 0.0 | 8613 |
| U5 [V] | cell voltage | V | individual voltage channel | 8604654 | 0 | 0.0 | 0.0001511 | -0.020447 | 0.945952 | 0.5751538 | 0.2822244 | 0.001587 | 0.704733 | 0.8437661 | 0.0 | 8613 |
| U7 [V] | cell voltage | V | individual voltage channel | 8604654 | 0 | 0.0 | 3.49e-05 | -0.020569 | 0.940703 | 0.5770515 | 0.2827626 | 0.00116 | 0.70687 | 0.8436051 | 0.0 | 8613 |
| U8 [V] | cell voltage | V | individual voltage channel | 8604654 | 0 | 0.0 | 1.16e-05 | -0.017518 | 0.947111 | 0.5784133 | 0.2831404 | 0.001709 | 0.708823 | 0.8436807 | 0.0 | 8613 |
| U9 [V] | cell voltage | V | individual voltage channel | 8604654 | 0 | 0.0 | 2.32e-05 | -0.017884 | 0.943571 | 0.5784222 | 0.2830601 | 0.002014 | 0.708457 | 0.8440371 | 0.0 | 8613 |
| U10 [V] | cell voltage | V | individual voltage channel | 8604654 | 0 | 0.0 | 4.65e-05 | -0.018677 | 0.942717 | 0.5778769 | 0.2838961 | 0.000916 | 0.708884 | 0.8417936 | 0.0 | 8613 |
| U11 [V] | cell voltage | V | individual voltage channel | 8604654 | 0 | 0.0 | 6.97e-05 | -0.019227 | 0.943693 | 0.5739725 | 0.2818036 | 0.000977 | 0.703452 | 0.8433391 | 0.0 | 8613 |
| MIN(`U.S [V]) | stack voltage candidate | V | MIN semantic unresolved; not accepted as dedicated stack voltage | 0 | 0 |  |  |  |  |  |  |  |  |  |  | 0 |
| T.Si.A [°C] | temperature | °C | anode inlet temperature | 8604654 | 0 | 0.0 | 0.0 | 23.7710075 | 84.9422302 | 71.4503823 | 10.0159483 | 34.8476906 | 75.1628113 | 79.9527283 | 0.0 | 8613 |
| T.Si.C [°C] | temperature | °C | cathode inlet temperature | 8604654 | 0 | 0.0 | 0.0 | 23.5856991 | 3273.964355 | 71.207835 | 32.3179448 | 34.3743439 | 74.6319656 | 81.9242935 | 0.0 | 8613 |
| p.So.A [kPag] | pressure | kPag | anode outlet pressure | 8604654 | 0 | 0.0 | 0.0 | -0.252848 | 171.215744 | 125.1782946 | 64.6799839 | -0.252848 | 159.8881683 | 164.1360168 | 0.0 | 8613 |
| p.So.C [kPag] | pressure | kPag | cathode outlet pressure | 8604654 | 0 | 0.0 | 18.3455721 | 0.0 | 151.944458 | 117.3877074 | 60.7895993 | 0.0 | 149.8218384 | 150.7315216 | 0.0 | 8613 |
| RH.Si.A  [%] | humidity | % | anode inlet relative humidity | 8604654 | 0 | 0.0 | 0.0 | 24.5974159 | 212.7523651 | 49.5038745 | 24.8212494 | 34.2767363 | 40.235466 | 144.7374268 | 0.0 | 8613 |
| RH.Si.C [%] | humidity | % | cathode inlet relative humidity | 8604654 | 0 | 0.0 | 0.0 | 0.000163 | 143.4680634 | 36.9321931 | 21.9487541 | 22.3644543 | 28.9947376 | 129.4364178 | 0.0 | 8613 |
| F.Si.H.A [nl/min] | flow | nl/min | anode hydrogen flow | 8604654 | 0 | 0.0 | 0.0 | 0.109867 | 53.1136513 | 18.9520496 | 11.0439677 | 0.128178 | 16.5921516 | 52.9488525 | 0.0 | 8613 |
| F.Si.Air.C [nl/min] | flow | nl/min | cathode air flow | 8604654 | 0 | 0.0 | 0.0 | 0.319834 | 143.3176117 | 48.9786883 | 32.429317 | 0.447768 | 44.7847519 | 142.8698425 | 0.0 | 8613 |

## Stack voltage candidate

| original_field | category | unit | physical_meaning | raw_count | missing_count | missing_percent | zero_percent | min | max | mean | std | p1 | p50 | p99 | abnormal_percent | sampled_quantile_count |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| MIN(`U.S [V]) | stack voltage candidate | V | MIN semantic unresolved; not accepted as dedicated stack voltage | 0 | 0 |  |  |  |  |  |  |  |  |  |  | 0 |
