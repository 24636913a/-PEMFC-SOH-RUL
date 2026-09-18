# SOH 构造（State of Health construction）

定义：`SOH_U(t_i) = V_ref(t_i) / V_ref,0`。其中 `V_ref(t_i)` 是已冻结 `j*=1 A/cm²` 的名义极化曲线 active-cell median voltage，`V_ref,0=0.698953307 V` 是最早合法的 source-labelled BoL reference。

这不是由运行工况瞬时电压构造的 SOH；动态电压的工作点混杂、停机和 load transition 均不被删除或重写。Development 可见原始锚点在 `outputs/reference_voltage_development.csv`，标准化轨迹在 `outputs/soh_trajectory_development.csv`。

| stage | ageing_hour | reference_voltage_V | SOH_U | valid_sample_count | active_voltage_std_V |
| --- | --- | --- | --- | --- | --- |
| BoL | 0.0 | 0.69895331 | 1.0 | 5550 | 0.00093007 |
| 300h | 300.0 | 0.6882167 | 0.98463902 | 1816 | 0.00052837 |
| 600h | 600.0 | 0.68474991 | 0.97967904 | 1146 | 0.00048909 |
| 850h | 850.0 | 0.6833948 | 0.97774028 | 970 | 0.00051111 |
| 1150h | 1150.0 | 0.6798183 | 0.97262334 | 91 | 0.00133166 |
| 1500h | 1500.0 | 0.68068489 | 0.97386319 | 89 | 0.00135097 |
| 1750h | 1750.0 | 0.6792323 | 0.97178494 | 651 | 0.00055365 |
