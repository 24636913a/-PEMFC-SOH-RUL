# Phase N2.6R 结果报告

## 1. 直接结论

| 模型 | RMSE（mV） | MAE（mV） | R² | 相对 LSTM RMSE 改善 |
|---|---:|---:|---:|---:|
| LSTM | 9.006 | 4.790 | 0.994762 | +0.00% |
| TimesNet | 9.895 | 3.546 | 0.991663 | -9.87% |
| ModernTCN | 15.373 | 10.343 | 0.987236 | -70.70% |

以上为 `seed=42`、三个完全相同 Development folds 的宏平均，便于 LSTM、TimesNet、ModernTCN 三模型公平横向比较。RMSE 与 MAE 越低越好，R² 越接近 1 越好。**Phase N2.6R：PASS；最终冻结 LSTM；通用骨干模型搜索终止；Test 继续冻结。**

## 2. 固定协议与 Test firewall

- 数据：0.5 h 因果聚合，F2 精确 14 特征。
- 任务：24 h（48 steps）历史预测未来 12 h（24 steps）绝对平均单体电压；预测增量由 `V_hat(t+h)-V(t)` 等价重建。
- 验证：Phase N2 原三折 expanding-window、Matched-Origin、fold-Train-only scaler。
- Test：`[1819.997738, 2256.800000] h`；目标读取 0 行、预测 0 个、RMSE/R²/MAE 均未计算。
- 本阶段没有第四个模型、没有结构创新、没有 Phase N3。

## 3. 实现保真结论

TimesNet 固定自 [THUML Time-Series-Library commit 4e938a1](https://github.com/thuml/Time-Series-Library/tree/4e938a1767106324dd753b2a44832bf870a0252e)，保留 DataEmbedding、FFT 周期发现、48→72 时间扩展、TimesBlock、Inception_Block_V1、残差、projection、未来 24 步切片以及官方样本级归一化/反归一化；外部仍保留每折 Train-only scaler。主损失只监督 `cell_voltage_mean`，没有未来真实负载输入。

ModernTCN 固定自 [官方 commit 56a9a2c](https://github.com/luodhhh/ModernTCN/tree/56a9a2c018385cd5acef015378cae7f084d1b11c)，保留 RevIN、逐变量 patching、大/小核、多阶段层次、两类 FFN、Flatten_Head。该固定版本虽声明多尺度层，但原 `forward_feature` 未调用；本地透明 adapter 仅激活已声明的 top-down multi-scale fusion，vendor 源码未修改。52/52 项实现及 CPU/GPU 数值审计通过，详见 `OFFICIAL_IMPLEMENTATION_AUDIT.md`。

## 4. seed=42 三折明细

| 模型 | Fold | RMSE（mV） | MAE（mV） | R² |
|---|---:|---:|---:|---:|
| LSTM | fold_1 | 6.208 | 3.916 | 0.997921 |
| LSTM | fold_2 | 6.158 | 4.169 | 0.997952 |
| LSTM | fold_3 | 14.651 | 6.284 | 0.988412 |
| TimesNet | fold_1 | 2.568 | 1.595 | 0.999644 |
| TimesNet | fold_2 | 6.888 | 3.155 | 0.997437 |
| TimesNet | fold_3 | 20.230 | 5.889 | 0.977909 |
| ModernTCN | fold_1 | 15.401 | 10.205 | 0.987207 |
| ModernTCN | fold_2 | 15.789 | 10.594 | 0.986532 |
| ModernTCN | fold_3 | 14.930 | 10.230 | 0.987967 |

平均 RMSE 最低与平均 R² 最高均为 **LSTM**。Fold3 最低 RMSE 为 **LSTM**：14.651 mV。

## 5. 与 Phase N2.6 轻量适配版比较

- TimesNet：轻量版 15.132 mV → 高保真版 9.895 mV，RMSE 改善 **34.61%**。
- ModernTCN：轻量版 16.394 mV → 高保真版 15.373 mV，RMSE 改善 **6.23%**。

高保真预测结构确实显著修正了 TimesNet 的轻量 head 偏差，但这不等于稳定超过 LSTM。

## 6. 多随机种子替换门

TimesNet-T1 因 seed=42 三折均值仅比 LSTM 差 9.87%，满足“差距不超过 10%”条件，依法进入 `[42, 2024, 3407]` 多种子复验。结果：

| 模型 | 3-seed RMSE（mV） | 3-seed MAE（mV） | 3-seed R² |
|---|---:|---:|---:|
| LSTM | 8.206 | 4.516 | 0.995847 |
| TimesNet | 9.908 | 3.811 | 0.990960 |

TimesNet 三种子宏平均相对 LSTM 的 RMSE 改善为 **-20.75%**；仅 **1/3 folds** 平均 RMSE 更低，且 Fold3 三种子均明显变差。替换五项规则：

1. 3-seed Mean RMSE 至少降低 2%：**FAIL**；
2. 至少 2/3 folds RMSE 更低：**FAIL**；
3. Mean MAE 不高于 LSTM：**PASS**；
4. Mean R² 不低于 LSTM：**FAIL**；
5. Late Development RMSE 恶化不超过 3%：**FAIL**。

因此 **不允许替换 LSTM**。

## 7. 逐时距与 9.5 h 邻域

| 模型 | 8.5 h | 9.0 h | 9.5 h | 10.0 h | 10.5 h |
|---|---:|---:|---:|---:|---:|
| LSTM | 6.559 | 6.368 | 3.846 | 7.460 | 8.661 |
| TimesNet | 11.343 | 11.079 | 11.048 | 9.775 | 8.179 |
| ModernTCN | 14.453 | 14.332 | 14.763 | 13.946 | 14.611 |

表中为 RMSE（mV）；对应 R² 与 MAE 已写入 `cycle_region_rmse_r2.csv`。h=9.5 h 最低 RMSE 为 **LSTM**：3.846 mV。TimesNet 的 FFT 各配置/折首位周期为：timesnet_hf_t1/fold_1: 8 steps (4.00 h), timesnet_hf_t1/fold_2: 8 steps (4.00 h), timesnet_hf_t1/fold_3: 8 steps (4.00 h), timesnet_hf_t2/fold_1: 4 steps (2.12 h), timesnet_hf_t2/fold_2: 4 steps (2.12 h), timesnet_hf_t2/fold_3: 4 steps (2.12 h)。周期发现只使用各折 Train 输入，未据 Validation/Test 人工强塞 19 steps。

## 8. 老化阶段，重点 Late Development

Development Validation origin 范围按物理老化时长三等分：`1002.500 / 1270.833 / 1539.167 / 1807.500 h`。Late Development 最低 RMSE 为 **LSTM**：12.261 mV。多种子下 TimesNet Late RMSE 为 22.956 mV，LSTM 为 10.420 mV，候选恶化 120.30%。

## 9. 配对移动块自助法

TimesNet 对 LSTM 的三种子配对 ΔRMSE（candidate−LSTM）95% CI 为 **[2.522, 8.369] mV**。该检验使用 19-step moving blocks、2000 repetitions；统计结果不改变预注册替换结论。

## 10. Q1–Q12 明确回答

### Q1 — 高保真 TimesNet 的 RMSE、MAE、R²？

seed=42 三折宏平均：RMSE **9.895 mV**（0.009895 V），MAE **3.546 mV**（0.003546 V），R² **0.991663**。

### Q2 — 高保真 ModernTCN 的 RMSE、MAE、R²？

seed=42 三折宏平均：RMSE **15.373 mV**（0.015373 V），MAE **10.343 mV**（0.010343 V），R² **0.987236**。

### Q3 — 相比 Phase N2.6 轻量版改善多少？

TimesNet RMSE 改善 **34.61%**；ModernTCN 改善 **6.23%**。

### Q4 — 平均 RMSE 最低？

**LSTM**。

### Q5 — 平均 R² 最高？

**LSTM**。

### Q6 — Fold3 谁最好？

**LSTM**，RMSE 14.651 mV。

### Q7 — Late Development 谁最好？

**LSTM**，seed=42 RMSE 12.261 mV。

### Q8 — 9.5 h 附近谁最好？

**LSTM**，h=9.5 h RMSE 3.846 mV。

### Q9 — 多 seed 是否支持替换 LSTM？

**NO。** TimesNet 仅通过 5 项规则中的 **1 项**，关键的平均 RMSE、R²和 Late Development 门槛均失败；ModernTCN 未达到多种子入围门。

### Q10 — 最终冻结哪个 backbone？

冻结 **LSTM**。

### Q11 — 是否正式停止继续搜索通用模型？

**YES。General backbone search STOP。** 官方高保真 TimesNet 虽修复轻量版偏差，但多种子 Fold3/晚期泛化不稳；完整 ModernTCN 仍明显落后。继续横向堆叠 Transformer/Mamba/TCN/Mixer 的证据价值已低。

### Q12 — 下一阶段围绕哪个 backbone 创新？

围绕 **LSTM** 做 PEMFC 任务机制创新；不重复无边界的通用骨干枚举。本阶段不实施创新，也不授权 Phase N3。

## 11. 结论边界

本结论只适用于冻结的 0.5 h/F2/24 h→12 h/三个 Development folds 协议。Test performance 从未打开；最终 Test 仍留给后续一次性评估。所有详细数值、图和冻结决定均位于本阶段目录。
