# Phase N2.6 结果报告

## 1. 执行结论

- Phase N2.6：**PASS**。
- LSTM reproduction：**PASS**，seed=42 三折数值与 Phase N2/N2.5 完全一致。
- 新模型多 seed 入围：**0 个**；四类候选均比 LSTM 差超过预注册的 10% 门槛。
- 最终冻结 backbone：**LSTM**，结论为 `LSTM remains incumbent`。
- 下一阶段结构创新建议：继续围绕 **LSTM**，但本阶段不实施创新。
- Test firewall：Test target 读取 0 行、预测 0 个、指标 0 项；继续禁止打开 Test。

## 2. 冻结协议

所有模型使用 0.5 h、F2 精确 14 特征、24 h（48 steps）历史、12 h（24 steps）direct residual target、三个 Phase N2 expanding-window folds、Matched-Origin 和 fold-Train-only scaler。训练统一使用 AdamW、1e-3、weight decay 1e-5、batch size 64、最多 300 epochs、patience 30、gradient clip 1.0 和 MSE。

## 3. 实现与 sanity audit

7 个预注册配置均通过 CPU/GPU 前向、反向、有限梯度、48×14→24 shape、参数规模和 inverse-transform 检查。TimesNet 与 ModernTCN 是本地可审计适配；PatchTSMixer 使用 Transformers 4.53.0 稳定公开实现；Compact Mamba 使用 Transformers 4.53.0 + mambapy 1.2.0 parallel scan。官方 Mamba CUDA selective-scan kernel 在当前 Windows 环境不可用。

实现来源和边界详见 `MODEL_IMPLEMENTATION_NOTES.md`。官方参考：

- https://github.com/thuml/Time-Series-Library
- https://github.com/luodhhh/ModernTCN
- https://huggingface.co/docs/transformers/v4.53.0/model_doc/patchtsmixer
- https://github.com/state-spaces/mamba

## 4. seed=42 三折筛选

| 模型 | 最佳配置 | RMSE（mV） | 相对 LSTM 改善 | Mean Skill_LSTM | Worst-fold Skill_LSTM | 参数量 |
|---|---|---:|---:|---:|---:|---:|
| LSTM | LSTM | 9.006 | reference | 0.000 | 0.000 | 55,448 |
| TimesNet | timesnet_d48_k3 | 15.132 | -68.03% | -0.701 | -0.795 | 94,728 |
| ModernTCN | moderntcn_d64_k13_b3 | 16.394 | -82.04% | -0.934 | -1.176 | 82,904 |
| PatchTSMixer | patchtsmixer_p6_d64 | 17.183 | -90.80% | -1.077 | -1.413 | 72,159 |
| Compact Mamba | compact_mamba_d64_l2 | 18.413 | -104.45% | -1.144 | -1.485 | 68,184 |

“相对 LSTM 改善”为 `1 - RMSE_candidate/RMSE_LSTM`，负值表示候选更差。筛选排名前两名新模型是 TimesNet 和 ModernTCN，但二者均未达到多 seed 入围门槛，因此它们不是合格 Top-2。

## 5. TimesNet 周期发现

TimesNet 的 FFT 周期发现只使用各折 Train windows。各折首要整数周期为：fold_1: 2 steps, fold_2: 2 steps, fold_3: 2 steps。首要周期没有接近 Phase N1 的 19 steps；在当前 48-step 输入与 F2 特征上，FFT 更强调 duty cycle 内的短周期变化。这一结果未用 Validation 或 Test 调整模型频率。

## 6. 多随机种子

由于没有新模型满足“相对 LSTM 劣化不超过 10%”，预注册规则禁止继续消耗多 seed。LSTM 复用 Phase N2 的 seeds `[42, 2024, 3407]`，三 seed RMSE 为 **8.206 ± 0.873 mV**，范围 `7.275–9.006 mV`。不存在支持替换 LSTM 的多 seed 证据。

## 7. Paired moving-block bootstrap

| 比较 | Mean ΔRMSE（mV） | 95% CI（mV） | Candidate better probability |
|---|---:|---:|---:|
| TimesNet vs LSTM | 7.255 | [3.941, 10.331] | 0.000 |
| ModernTCN vs LSTM | 7.463 | [4.183, 10.549] | 0.000 |
| PatchTSMixer vs LSTM | 7.753 | [4.915, 10.440] | 0.000 |
| Compact Mamba vs LSTM | 10.135 | [6.606, 14.058] | 0.000 |

ΔRMSE 定义为 candidate − LSTM；负值才有利于候选。四个区间均未支持替换 LSTM。

## 8. 逐时距、周期邻域和老化阶段

24 个 horizon step 已全部输出。h=9.5 h 时，LSTM RMSE 为 3.846 mV。新模型按三折算术平均 RMSE 排名最好的是 **CompactMamba**（8.556 mV）；按“先逐折计算、再取均值”的 Mean Skill_LSTM 排名最好的是 **TimesNet**（-0.971）。两种聚合口径均没有任何候选超过 LSTM，也不能改变完整 12 h 任务的筛选结论。

Late Development 中 RMSE 最低的是 **LSTM**，数值为 12.261 mV。阶段边界为 `1002.500 / 1270.833 / 1539.167 / 1807.500 h`，只来自 Development Validation origins。

## 9. 精度与效率

按精度优先、参数量次优的审计规则，Accuracy–Efficiency 最佳仍是 **LSTM**。新模型参数量受到 50k–200k 限制，但没有以合理额外成本换来更低 RMSE。详细显存、epoch 时间和总训练时间见 `model_accuracy_efficiency_comparison.csv`。

## 10. Q1–Q12 明确回答

### Q1 — TimesNet 是否优于 LSTM？

**NO。** 最佳 TimesNet 为 15.132 mV，相对 LSTM 改善 -68.03%。

### Q2 — ModernTCN 是否优于 LSTM？

**NO。** 最佳 ModernTCN 为 16.394 mV，相对 LSTM 改善 -82.04%。

### Q3 — PatchTSMixer 是否优于 LSTM？

**NO。** 最佳 PatchTSMixer 为 17.183 mV，相对 LSTM 改善 -90.80%。

### Q4 — Compact Mamba 是否优于 LSTM？

**NO。** Compact Mamba 为 18.413 mV，相对 LSTM 改善 -104.45%。

### Q5 — 哪一个新模型在 9.5 h 周期附近最有优势？

没有新模型在 h=9.5 h 超过 LSTM。若只在新模型内部比较，**CompactMamba** 的三折平均 RMSE 最低（8.556 mV）；**TimesNet** 的 Mean Skill_LSTM 最高（-0.971）。这一口径差异来自各折 LSTM 误差尺度不同，不构成替换证据。

### Q6 — 哪一个模型在 Late Development 最稳定？

**LSTM** 的 Late Development RMSE 最低。

### Q7 — 哪个模型的 Accuracy–Efficiency 最佳？

**LSTM**。它同时保持最低 RMSE 和较低训练成本。

### Q8 — 多 seed 是否支持替换 LSTM？

**NO。** 没有新模型通过 10% 入围门，因此没有合法的新模型多 seed 替换证据。

### Q9 — Bootstrap 是否支持替换 LSTM？

**NO。** 四个候选对 LSTM 的 ΔRMSE 95% CI 均不支持候选更优。

### Q10 — 最终冻结哪个 backbone？

冻结 **LSTM**，即 `LSTM remains incumbent`。

### Q11 — 下一阶段围绕哪个模型做结构创新？

围绕 **LSTM** 做有针对性的结构创新；不重复 Phase N2.5 已失败的 xLSTM/CA 组合，也不在本阶段实施新结构。

### Q12 — 是否仍然禁止打开 Test？

**YES。继续禁止打开 Test。** Phase N2.6 不授权 Phase N3；最终 Test 继续保留。

## 11. 限制与停止声明

1. TimesNet 和 ModernTCN 是轻量适配，不等同于官方 benchmark 配置；结论只约束本任务、预注册规模与统一训练协议。
2. PatchTSMixer 和 Compact Mamba 使用稳定公开实现适配统一 head；Mamba 未使用官方 CUDA selective-scan kernel。
3. 新模型未达到多 seed 入围门，未强行运行或挑选最佳 seed。
4. 本阶段未访问 Test performance，未进入 Phase N3，未进行结构创新、SOH、RUL、EIS、ensemble 或 uncertainty。
