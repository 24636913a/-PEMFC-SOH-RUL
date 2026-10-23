# Phase N2.5 结果报告

## 1. 执行结论

- Phase N2.5：**PASS**。
- Architecture innovation（结构创新）：**NO-GO**。
- 最终冻结模型：**LSTM**。
- Phase N3：**GO**，但只能原样读取冻结配置后执行最终 Test evaluation（测试评估）。
- Test 防火墙：目标读取 0 行、预测 0 个、性能指标 0 项。

## 2. 数据与协议

本阶段严格继承 Phase N2：0.5 h sampling、F2 的 14 个精确特征、24 h lookback、12 h direct multi-horizon residual target、三个 expanding-window development folds、Train-only scaler。经验周期固定 P=19 steps（9.5 h），错误周期在训练前固定 P=15 steps（7.5 h）。

普通 LSTM seed 42 的三折复现与 Phase N2 差异为 0.000%，通过 2% reproduction gate（复现门）。

## 3. xLSTM 实现来源与边界

xLSTM 结构依据为 Beck 等人的 xLSTM 论文（https://arxiv.org/abs/2405.04517）与 NX-AI 官方仓库（https://github.com/NX-AI/xlstm）。检查的上游 release 为 v2.0.4。当前实现是面向 48 步连续时序的 `local auditable implementation（本地可审计实现）`，没有复制或修改上游代码，不声明逐行复现官方语言模型 block stack。

## 4. 三个 xLSTM 变体

| 模型 | seed 42 三折 RMSE | 相对 LSTM | 参数量 |
|---|---:|---:|---:|
| LSTM | 9.006 mV | reference | 55,448 |
| sLSTM | 17.689 mV | 未超过 | 69,592 |
| mLSTM | 19.057 mV | 未超过 | 78,952 |
| Hybrid xLSTM | 18.231 mV | 未超过 | 74,272 |

最低 xLSTM RMSE 来自 **sLSTM**，因此它是后续唯一 xLSTM backbone；但它相对普通 LSTM 的改善为 -112.55%，没有通过 Condition A。

## 5. 核心消融与多 seed

| 模型 | 三 seed 平均 RMSE | seed 标准差 | Mean Skill_Best | Worst-fold Skill | 参数量 |
|---|---:|---:|---:|---:|---:|
| LSTM | 8.206 mV | 0.873 | 0.692 | 0.514 | 55,448 |
| xLSTM (sLSTM) | 17.441 mV | 0.962 | 0.334 | 0.220 | 69,592 |
| CA-LSTM | 9.086 mV | 0.705 | 0.667 | 0.427 | 65,817 |
| CA-xLSTM | 17.615 mV | 0.712 | 0.334 | 0.121 | 79,961 |

CA-LSTM 相对 LSTM 改善为 -10.73%，仅 1/3 folds、1/3 seeds 改善。CA-xLSTM 相对 xLSTM 改善为 -1.00%；相对普通 LSTM 改善为 -114.68%，0/3 folds、0/3 seeds 改善。

## 6. 真实周期与错误周期

seed 42 三折下，CA-xLSTM P=19 的 RMSE 为 17.927 mV，P=15 为 18.679 mV。P=19 相对 P=15 改善 4.03%。该对照支持经验周期位置具有结构意义，但不足以证明 CA-xLSTM 可替换普通 LSTM。

## 7. 配对移动块自助法

| 比较 | Mean ΔRMSE | 95% CI | Candidate better probability |
|---|---:|---:|---:|
| xLSTM_vs_LSTM | 8.966 mV | [7.210, 10.798] mV | 0.000 |
| CA-LSTM_vs_LSTM | 2.037 mV | [1.061, 3.072] mV | 0.000 |
| CA-xLSTM_vs_xLSTM | 1.462 mV | [-0.067, 3.203] mV | 0.033 |
| CA-xLSTM_vs_LSTM | 10.394 mV | [7.972, 12.844] mV | 0.000 |
| P19_vs_P15 | 0.660 mV | [-1.168, 2.493] mV | 0.248 |

ΔRMSE 定义为 candidate-reference；负值有利于 candidate。CA-xLSTM vs LSTM 的 95% CI 未完整低于 0，因此不满足最终 GO 标准。

## 8. 9.5 h 与周期邻域

| 模型 | h=9.5 h RMSE | MAE | Skill_Best | Skill_Cycle |
|---|---:|---:|---:|---:|
| LSTM | 2.912 mV | 1.573 mV | 0.893 | 0.931 |
| xLSTM | 7.547 mV | 2.677 mV | 0.730 | 0.826 |
| CA-LSTM | 2.747 mV | 1.756 mV | 0.903 | 0.939 |
| CA-xLSTM | 11.933 mV | 3.407 mV | 0.598 | 0.745 |
| Ridge | 27.647 mV | 17.380 mV | 0.000 | 0.377 |
| Cycle Repeat | 45.395 mV | 7.857 mV | -0.625 | 0.000 |

8.5–10.5 h 曲线用于判断局部稳定性。结果没有显示 CA-xLSTM 在该邻域形成足以反转总体排名的优势。

## 9. 周期权重解释

CA-LSTM 的 Development Validation 平均权重为：当前状态 0.348、上一周期 0.353、上上周期 0.299。这些权重只用于解释，没有反馈到结构或超参数选择。

## 10. 老化阶段稳定性

| 阶段 | LSTM RMSE | CA-LSTM RMSE | CA-xLSTM RMSE |
|---|---:|---:|---:|
| Early Development | 6.597 mV | 5.882 mV | 16.325 mV |
| Middle Development | 10.953 mV | 14.297 mV | 20.047 mV |
| Late Development | 10.420 mV | 12.401 mV | 23.119 mV |

阶段边界按所有合法 Development Validation origins 的实际 ageing 范围预先等分。周期模型在晚期没有保持稳定优势。

## 11. 参数与效率

| 模型 | 参数量 | 峰值 GPU 显存 | 平均 epoch 时间 | 三 seed 总训练时间 | RMSE |
|---|---:|---:|---:|---:|---:|
| LSTM | 55,448 | 146.6 MB | 0.049 s | 119.0 s | 8.206 mV |
| xLSTM | 69,592 | 862.1 MB | 0.212 s | 543.9 s | 17.441 mV |
| CA-LSTM | 65,817 | 146.1 MB | 0.071 s | 155.6 s | 9.086 mV |
| CA-xLSTM | 79,961 | 861.2 MB | 0.227 s | 558.6 s | 17.615 mV |

新增结构提高了参数量和训练成本，但没有提高三 seed 总体精度，因此效率证据同样不支持替换 LSTM。

## 12. Q1–Q12 明确回答

### Q1 — sLSTM / mLSTM / Hybrid xLSTM 中哪个最好？

**sLSTM** 的 seed 42 三折平均 RMSE 最低，是预注册规则下的最佳 xLSTM variant。

### Q2 — xLSTM 本身是否真正优于普通 LSTM？

**NO。** 相对 LSTM 改善为 -112.55%，0/3 seed-42 folds 改善。

### Q3 — Cycle-Aware Memory 是否产生独立增益？

**不稳定。** CA-LSTM 相对 LSTM 为 -10.73%；CA-xLSTM 相对 xLSTM 为 -1.00%。局部 fold 有改善，但三 seed 总体没有稳定增益。

### Q4 — CA-LSTM 是否优于 LSTM？

**NO。** 三 seed 平均 RMSE 9.086 vs 8.206 mV，仅 1/3 folds、1/3 seeds 改善。

### Q5 — CA-xLSTM 是否优于 xLSTM？

**NO。** 三 seed 平均改善为 -1.00%。

### Q6 — CA-xLSTM 是否最终优于普通 LSTM？

**NO。** 相对 LSTM 改善为 -114.68%，未通过 fold、seed 或 bootstrap 门槛。

### Q7 — 真实 19-step 周期是否优于错误 15-step 周期？

**YES（seed 42 三折平均）。** P=19 改善 4.03%，但该证据只支持周期位置，不足以支持整体架构替换。

### Q8 — 模型优势是否集中在 9.5 h 附近？

**没有形成可改变最终排名的优势。** 9.5 h 及 8.5–10.5 h 邻域均已单独报告。

### Q9 — 周期模块在不同老化阶段是否稳定？

**NO。** 早中期的局部改善没有稳定延伸到 Late Development。

### Q10 — 代价是多少？

CA-LSTM 参数量为 65,817，CA-xLSTM 为 79,961，均高于 LSTM 的 55,448；训练时间与显存也增加，但总体 RMSE 未改善。

### Q11 — 最终冻结哪个模型？

冻结 **LSTM**。结论为 `Architecture innovation NO-GO`。

### Q12 — 是否允许进入 Phase N3？

**GO。** Phase N3 只允许读取 `PHASE_N2_5_FROZEN_CONFIG.yaml`，按 Fold 3 的 Train 0–60%、Validation 60–80%、Test 80–100% 和三个 seeds 执行一次最终测试；不得根据 Test 选择 seed、模型或超参数。

## 13. 限制与停止声明

1. 所有比较均来自 Development Validation，未访问 Test target、预测或性能。
2. 本地 xLSTM 实现适配连续短序列任务，不等同于官方语言模型 block stack；负结果只约束当前实现和任务。
3. P=15 只使用预注册的 seed 42 对照；没有搜索错误周期。
4. 本阶段没有执行 SOH、RUL、EIS、uncertainty、ensemble、Mamba 或外部验证。
5. Phase N2.5 到此停止，未进入 Phase N3。
