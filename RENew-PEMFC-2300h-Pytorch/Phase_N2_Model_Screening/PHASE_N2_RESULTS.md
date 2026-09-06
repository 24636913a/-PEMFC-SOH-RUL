# Phase N2 — 强基线、历史长度诊断与现代时序模型筛选

英文名称：Phase N2 — Strong Baselines, Temporal-Diagnostic Analysis and Modern Time-Series Model Screening

## 阶段结论

**Phase N2：PASS**

**Phase N3 Final Frozen Evaluation（阶段 N3 最终冻结评估）：GO**

Phase N2 已完成强基线、Feature Group（特征组）、Matched-Origin（匹配预测起点）Lookback（历史长度）、五模型筛选、Top-2 多随机种子、Horizon sensitivity（预测时距敏感性）、1 h robustness（稳健性）和 moving-block bootstrap（移动块自助法）。最终冻结的 deterministic backbone（确定性预测骨干模型）为 **LSTM（长短期记忆网络）**。

整个阶段只使用前 80% development region（开发区）。没有构造最终 Test 的预测标签、预测结果、误差、曲线或模型排名。

## 1. 冻结任务

| 项目 | Phase N2 冻结结果 |
| --- | --- |
| 主分辨率 | 0.5 h |
| 1 h 用途 | 最终候选稳健性检查，不参与模型选择 |
| Target（目标） | 直接多步电压增量 `ΔV(t+h)=V(t+h)−V(t)` |
| 电压定义 | 10 个有效单体的 `cell_voltage_mean` |
| Feature Group | F2 — Voltage + Load（电压+负载），14列 |
| Lookback | 24 h，48 steps |
| Primary Horizon | 12 h，24 steps |
| Diagnostic Horizons | 4/8/9.5/12/24 h |
| Backbone | LSTM，2层、hidden=64、dropout=0.1 |
| 参数量 | 55,448 |
| 强基线 | Ridge Lag Regression（岭回归滞后模型） |
| Test | `[1819.997738,2256.800000] h`，Phase N2 完全冻结 |

## 2. Test firewall（测试集防火墙）

Phase N2 loader 先只读取全文件的结构和 split metadata（划分元数据），定位第一行 Test；随后使用 `nrows` 只读取该位置之前的 feature/target rows。模型窗口构造前再次要求 `bin_end_ophrs2_h <= 1819.997738 h`。

- Test metadata rows 可计数，但 Test target rows 读取数为 0。
- Test predictions constructed（已构造 Test 预测）为 `False`。
- 所有 training/validation origins 均来自三个 development folds。
- 所有图均为 Validation-only（仅验证集）或聚合验证结果。

## 3. Expanding-window validation（扩展窗口验证）

| Fold | Train | Validation | 24 h→12 h Train windows | Validation windows |
| --- | --- | --- | ---: | ---: |
| Fold 1 | `[72.788690,946.393214)` | `[946.393214,1164.794345)` | 1,048 | 233 |
| Fold 2 | `[72.788690,1164.794345)` | `[1164.794345,1383.195476)` | 1,281 | 245 |
| Fold 3 | `[72.788690,1383.195476)` | `[1383.195476,1819.997738)` | 1,526 | 466 |

每个 fold 的 median imputer（中位数填补器）、input StandardScaler（输入标准缩放器）和 per-horizon target scaler（逐时距目标缩放器）仅使用对应 fold Train 拟合。

## 4. 强基线结果

第一轮 F1、24 h→12 h 强基线结果：

| Baseline | 三折平均 RMSE | MAE | 相对 Persistence 改善 | 相对 Cycle Repeat 改善 |
| --- | ---: | ---: | ---: | ---: |
| Persistence | 194.92 mV | 119.14 mV | 0% | −317.3% |
| Local Linear Trend | 138.03 mV | 99.33 mV | 29.2% | −195.6% |
| Cycle Repeat | 49.84 mV | 9.40 mV | 74.4% | 0% |
| Ridge（F1） | 37.58 mV | 19.40 mV | 80.7% | 22.3% |

Cycle Repeat 是非常强的物理基线：它将 Persistence 的 RMSE 从约 195 mV 降到约 50 mV，证明 9.5 h 周期必须被认真控制。但它不是最终最强简单基线。采用冻结 F2 后，Ridge 进一步达到 **26.34 mV**，成为后续 `Skill_Best` 分母。

## 5. Feature Group 诊断

| 参考模型 | F1 RMSE | F2 RMSE | F3 RMSE | 冻结判断 |
| --- | ---: | ---: | ---: | --- |
| Ridge | 37.58 mV | 26.34 mV | 25.83 mV | F2 相对 F1 通过；F3 相对 F2 未通过 |
| LSTM | 12.51 mV | 9.01 mV | 8.71 mV | F2 相对 F1 通过；F3 相对 F2 未通过 |

F2 相对 F1：

- Ridge 平均改善 29.9%，3/3 folds 改善。
- LSTM 平均改善 28.0%，3/3 folds 改善。
- 两者的 19-step paired bootstrap 均支持 F2。

F3 相对 F2：

- Ridge 平均改善仅 1.95%，低于 2%门槛，并有 fold 恶化 6.49%。
- LSTM 平均点估计改善 3.31%，但只有 1/3 folds 改善，最差 fold 恶化 66.6%。
- 虽然 LSTM 聚合 bootstrap 偏向 F3，但预冻结的 fold stability（折稳定性）规则明确不通过。

因此冻结 **F2**。结论是：历史负载信息具有稳定增益；增加实际温压湿流量和化学计量比后没有稳定的跨生命周期增益，不应为了平均值的小改善扩大输入维度。

## 6. Matched-Origin Lookback 诊断

`matched_origin_audit.csv` 中所有 Train/Validation origin、target、anchor 和 target-end 一致性检查均通过。

| 参考模型 | 12 h | 24 h | 48 h |
| --- | ---: | ---: | ---: |
| Ridge | 26.23 mV | **25.93 mV** | 26.93 mV |
| LSTM | 9.96 mV | 9.39 mV | **9.35 mV** |

48 h 对 LSTM 仅比 24 h 改善约 0.44%，低于 2%门槛；Ridge 则恶化约 3.86%。两种模型的 paired bootstrap 均未给出 48 h 的稳定优势。12 h 对 LSTM 和 Ridge 也未通过替换规则。

因此继续冻结 **24 h Lookback**。这不是声称 24 h 为全局最优，而是说明在严格 Matched-Origin 下，没有足够证据偏离物理上覆盖约 2.5 个 duty cycles（负载循环）的默认候选。

## 7. 五模型筛选

统一使用 F2、24 h→12 h、seed=42、相同 folds/origins/targets/scaler/early stopping：

| 模型 | RMSE | MAE | Mean Skill_Best | Worst-fold Skill_Best | 正 Skill folds | 参数量 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| **LSTM** | **9.01 mV** | **4.79 mV** | **0.666** | **0.514** | 3/3 | 55,448 |
| PatchTST | 14.32 mV | 7.97 mV | 0.481 | 0.181 | 3/3 | 126,232 |
| iTransformer | 16.89 mV | 9.47 mV | 0.375 | 0.191 | 3/3 | 105,688 |
| TimeMixer | 22.68 mV | 12.51 mV | 0.143 | 0.108 | 3/3 | 89,368 |
| TCN | 26.78 mV | 16.05 mV | −0.026 | −0.112 | 1/3 | 39,048 |

Top-2 modern candidates（现代候选）是 **PatchTST** 和 **iTransformer**。但总体最优 learned model（学习模型）是 LSTM。TCN 未超过最强简单基线，不满足 learned-value GO 条件。

## 8. Top-2 与 LSTM 多随机种子

种子为 `[42,2024,3407]`：

| 模型 | 三 seed 平均 RMSE | Seed std | Seed range | Mean Skill_Best | Skill std | Worst fold-seed Skill | 正 Skill fold-seed |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **LSTM** | **8.21 mV** | 0.87 mV | 7.27–9.01 mV | **0.692** | 0.031 | **0.514** | 9/9 |
| PatchTST | 14.35 mV | 0.39 mV | 13.97–14.75 mV | 0.479 | 0.012 | 0.126 | 9/9 |
| iTransformer | 16.03 mV | 1.21 mV | 14.64–16.89 mV | 0.402 | 0.045 | 0.191 | 9/9 |

PatchTST 的 seed-level mean 较稳定，但 Fold 3 误差持续偏高。LSTM 在所有 9 个 fold-seed combinations（折-随机种子组合）中均保持正 `Skill_Best`，且平均误差显著最低。

## 9. 统计稳健性

以一个经验周期 19 steps 为 block length，进行 2000 次 paired moving-block bootstrap。最终 LSTM 相对 Ridge：

- Mean ΔRMSE（LSTM−Ridge）：**−16.43 mV**。
- Median ΔRMSE：约 −16.40 mV。
- 95% CI：**[−20.79,−12.24] mV**。
- LSTM 更优的 bootstrap probability：**1.000**。

置信区间完全低于 0，说明 learned gain（学习增益）不是单一 fold 或少量 origin 的偶然结果。

## 10. Horizon sensitivity 与 9.5 h 周期

Top-2 modern candidates 的聚合结果：

| Horizon | PatchTST Skill_Best | PatchTST Skill_Cycle | iTransformer Skill_Best | iTransformer Skill_Cycle |
| --- | ---: | ---: | ---: | ---: |
| 4 h | 0.552 | 0.751 | 0.475 | 0.699 |
| **8 h** | **0.602** | **0.770** | **0.490** | **0.700** |
| 9.5 h | 0.538 | 0.729 | 0.407 | 0.646 |
| 12 h | 0.481 | 0.719 | 0.375 | 0.663 |
| 24 h | 0.409 | 0.725 | 0.181 | 0.613 |

Top-2 的最佳 aggregate learned skill（聚合学习技能）出现在 **8 h**。24 h 时误差和 fold 风险增加，但 PatchTST 仍保持正 Skill。

对最终 LSTM 的 12 h 主任务逐 step 检查显示，**h=9.5 h** 的 `Skill_Best=0.859`、`Skill_Cycle=0.909`、RMSE 约 **3.85 mV**，是最清晰的 temporal skill（时序技能）位置。模型并非只复制周期，因为它在一个完整循环处显著超过 Cycle Repeat。

## 11. 1 h robustness

保持相同 24 h history 和12 h future：

| 分辨率 | Ridge RMSE | LSTM RMSE | LSTM Skill_Best | LSTM Skill_Cycle |
| --- | ---: | ---: | ---: | ---: |
| **0.5 h** | 26.34 mV | **9.01 mV** | **0.666** | 0.816 |
| 1 h | 31.43 mV | 24.93 mV | 0.290 | 0.839 |

1 h 下 LSTM 仍超过最强简单基线，因此 learned value 没有消失；但绝对 RMSE 明显高于 0.5 h。Phase N1 的 0.5 h 主分辨率优势得到模型层面的支持。1 h 不用于重新选择模型。

## 12. Accuracy–efficiency（精度–效率）

| 模型 | 参数量 | Peak GPU memory | 三折训练总时间 | RMSE |
| --- | ---: | ---: | ---: | ---: |
| LSTM | 55,448 | 146.6 MB | 36.9 s | **9.01 mV** |
| PatchTST | 126,232 | 31.9 MB | 43.5 s | 14.32 mV |
| iTransformer | 105,688 | 35.5 MB | 49.8 s | 16.89 mV |
| TimeMixer | 89,368 | 21.1 MB | 34.9 s | 22.68 mV |
| TCN | 39,048 | 0 MB（CPU） | 229.0 s | 26.78 mV |

确定性 cuDNN 下，小型 TCN Conv1d 在本机 GPU 有严重 kernel-launch overhead，因此使用数学等价的 CPU 执行并如实记录。LSTM 不仅精度最高，而且参数量低于三个 modern Transformer/mixer candidates；没有证据支持“更现代、参数更多”本身带来更好性能。

## 13. Q1–Q11 明确回答

### Q1 — Cycle Repeat 基线有多强？

**很强，但不是最强。** 它比 Persistence 降低约 74.4% RMSE，达到 49.84 mV；冻结 F2 后的 Ridge 达到 26.34 mV，成为最强简单基线。

### Q2 — 模型能否真正超过最强简单基线？

**YES。** LSTM seed=42 Mean Skill_Best=0.666，3/3 folds 为正；多 seed mean=0.692；bootstrap 95% CI 完全支持 LSTM 优于 Ridge。

### Q3 — 负载和运行条件是否提高预测能力？

**负载信息 YES，额外运行条件 NO（未证明稳定增益）。** F2 相对 F1 在 Ridge/LSTM 上均通过；F3 相对 F2 未通过跨 fold 稳定性规则。因此冻结 F2。

### Q4 — 12/24/48 h 哪个更合理？

**24 h。** 48 h 对 LSTM 仅有 0.44%点估计改善且 Ridge 恶化；bootstrap 不支持改变。12 h 也未通过替换规则。

### Q5 — L>1 的价值是否得到模型结果支持？

**YES。** 具有历史序列的 LSTM 显著超过 Persistence、Cycle Repeat 和 Ridge；Matched-Origin 中 24 h LSTM 比 12 h 更好约 5.8%，表明跨多个 duty cycles 的历史有实际作用。48 h 的额外历史则没有稳定增益。

### Q6 — 五个模型谁最好？

**LSTM 最好。** 顺序为 LSTM、PatchTST、iTransformer、TimeMixer、TCN。Top-2 modern candidates 为 PatchTST 与 iTransformer。

### Q7 — 现代模型是否只是参数更多？

PatchTST、iTransformer 和 TimeMixer 都产生一定正 Skill，但均未超过更小的 LSTM。现代架构并非无效，却没有为本任务提供足以抵消复杂度的优势。

### Q8 — 哪个 horizon 最体现 learned temporal skill？

Top-2 的 aggregate optimum（聚合最佳）是 8 h；最终 LSTM 的逐 step 最强位置是 **9.5 h**，正好对应完整经验周期，并显著超过 Cycle Repeat。

### Q9 — 0.5 h 优势在1 h robustness 中是否仍成立？

**YES。** LSTM 在 0.5 h 为9.01 mV，在1 h 为24.93 mV。1 h 下仍有正 Skill，但保留动态分辨率的0.5 h 明显更好。

### Q10 — 冻结哪个 deterministic backbone？

冻结 **LSTM，F2，24 h Lookback，12 h Horizon，0.5 h sampling**。具体配置见 `PHASE_N2_FROZEN_CONFIG.yaml`。

### Q11 — 是否进入 Phase N3 Final Frozen Evaluation？

**GO。** 数据、特征、Lookback、Horizon、模型、超参数、scaler policy 和 seeds 已冻结。Phase N3 原则上不得修改这些配置，只允许执行最终冻结 Test evaluation。

## 14. 重要限制

1. 所有结论来自 development folds，尚未访问最终 Test performance。
2. Phase N2 只研究短中期电压增量预测；没有 RUL、EOL、dense SOH 或 uncertainty 结论。
3. F3 的聚合平均值略好不等于稳定贡献；其生命周期 fold instability 是拒绝 F3 的原因。
4. 24 h 是经规则保留的合理候选，不宣称为所有任务的全局最优历史长度。
5. 模型实现是本地可审计 PyTorch screening implementation，不声称逐行复现各论文官方仓库。

## 15. 核心输出

- `outputs/baseline_results/baseline_summary.csv`
- `outputs/baseline_results/baseline_per_horizon_metrics.csv`
- `outputs/feature_group_diagnostics/feature_group_comparison.csv`
- `outputs/lookback_diagnostics/lookback_matched_origin_comparison.csv`
- `outputs/lookback_diagnostics/matched_origin_audit.csv`
- `outputs/model_screening/modern_model_screening_summary.csv`
- `outputs/model_screening/modern_model_per_horizon_metrics.csv`
- `outputs/model_screening/per_horizon_metrics.csv`
- `outputs/robustness_analysis/multi_seed_robustness.csv`
- `outputs/robustness_analysis/horizon_sensitivity.csv`
- `outputs/robustness_analysis/1h_robustness_comparison.csv`
- `outputs/statistical_tests/bootstrap_pairwise_comparison.csv`
- `outputs/model_screening/model_efficiency_summary.csv`
- `outputs/frozen_config/PHASE_N2_FROZEN_CONFIG.yaml`

Phase N2 到此停止。未执行 Final Test、Phase N3、RUL、uncertainty、ensemble、PINN、future-known duty-cycle、EIS 或 spatial EIS 融合。
