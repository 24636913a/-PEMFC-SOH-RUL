# 可以计算什么，不能计算什么

认证的 `HI_V=100*V_pred/V_BoL` 合格样本为0，输出HI_BoL_percent全部NA。本轮从未把任意动态电压除以0.6989533305168152 V。

在145个具有统计共同支持的目标上，计算明确改名的 `CR_V=100*V_pred/V_early`，即相对早期Development的条件电压保持率。它只是一项回顾性探索性健康代理，未证明测量等价，不能称标准SOH。每个种子有2871条重叠查询；完整逐点电压、分母与比值见outputs/CONDITIONAL_RETENTION_DIAGNOSTIC.csv。

误差定义为对同一早期分母的 `100*(V_pred-V_actual)/V_early`，是预测电压误差向比值的传播，不是对真实SOH标签的误差。

| seed | n | MAE | RMSE | P95 | MAX |
| --- | --- | --- | --- | --- | --- |
| 42 | 2871 | 0.639581 | 0.863525 | 1.786491 | 4.408943 |
| 52 | 2871 | 0.687828 | 0.936775 | 1.982542 | 4.958311 |
| 62 | 2871 | 0.391800 | 0.554497 | 1.114232 | 4.139649 |

上述单位均为条件保持率百分点。Seed42比值范围95.538359%–102.503253%；189条预测超过100%，不裁剪、不强制单调。固定12h终点有117个支持查询，相邻支持点出现59次上升，既可能来自恢复/工况/参考分母变化，也可能来自预测误差，不能当成不可逆损伤被修复的证据。原始动态电压全查询Seed62仍有307.985mV最大误差，条件支持子集较好的尾部不能掩盖全缓存异常。

参考分母在主分析0.690754–0.736441V，参考周期数3–17。周期SD中位数2.667346mV，最大8.813080mV。Seed42采用周期均值或周期中位数的最大比值差0.545787百分点；参考周期最小/最大值敏感性范围宽度中位数1.088362、最大3.070101百分点。这是分母敏感性，不是置信区间。

误差传播导数：dCR/dV_pred=100/V_early（百分点/V），dCR/dV_early=−100*V_pred/V_early²。以周期SD作局部扰动尺度，Seed42线性敏感性中位数0.365649百分点；不是新的物理损失或有效训练梯度宣称。

完整敏感性、负载分组、24步误差和原始电压保持基线分别见REFERENCE_SENSITIVITY_POINTWISE.csv、CONDITIONAL_RETENTION_LOAD_DIAGNOSTIC.csv、CONDITIONAL_RETENTION_HORIZON_ERRORS.csv、CONDITIONAL_RETENTION_BASELINE_METRICS.csv。原始电压保持基线跨负载极差，不能把胜过它解释为健康识别成功。

固定12h终点的时间/电流秩相关仅作描述，不检验显著性，不把相关性当退化真值：

| outcome | exposure | n | descriptive_Spearman |
| --- | --- | --- | --- |
| actual_voltage_V | target_ophrs_h | 117 | 0.064028 |
| actual_voltage_V | j | 117 | -0.997025 |
| conditional_voltage_retention_observed_percent | target_ophrs_h | 117 | -0.045752 |
| conditional_voltage_retention_observed_percent | j | 117 | -0.738495 |
| CR_mean_percent | target_ophrs_h | 117 | 0.174040 |
| CR_mean_percent | j | 117 | 0.231414 |

原始电压与电流的秩相关约−0.997；采用早期同统计工况参照后，实测条件保持率与电流仍约−0.738，与运行时间约−0.046。预测比值与时间约+0.174。这说明归一化没有排除明显的负载混淆，也没有给出清晰的退化轨迹证据。B判定仅支持条件电压诊断的可计算性和探索价值，不表示健康趋势已被验证。

严格在线SOH为0：未来实际工况仅作事后匹配，无可认证的提前给定工况计划；且冻结缓存checkpoint选择曾使用Validation标签至2025-05-26 17:29:26.005000，晚于全部缓存起点。回顾性缓存误差仍保留其原Development Validation身份，不伪装成前瞻部署。
