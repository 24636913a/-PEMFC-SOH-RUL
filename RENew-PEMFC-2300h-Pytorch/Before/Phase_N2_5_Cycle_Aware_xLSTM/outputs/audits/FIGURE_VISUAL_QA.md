# Phase N2.5 图像视觉质量审计

审计日期：2026-09-06  
审计范围：`outputs/figures` 中 Phase N2.5 的全部 12 张 PNG。  
审计结论：**PASS**。

## 审计标准

- 图像能够正常解码，分辨率与文件大小足以支持报告阅读；
- 中文标题、英文模型名、坐标轴、图例和数值无乱码；
- 关键元素未被画布边界裁切，标注不存在影响理解的重叠；
- 图中只使用 Development/Validation 结果，不包含最终 Test 预测、Test 曲线或 Test 指标；
- 图意与对应 CSV 和 Phase N2.5 报告一致。

## 逐图结论

| 图像 | 结论 | 备注 |
|---|---|---|
| `xlstm_variant_rmse.png` | PASS | 四个筛选模型、三折结果及均值可辨识。 |
| `cycle_aware_ablation_rmse.png` | PASS | A0–A3 消融对比清晰。 |
| `lstm_vs_ca_xlstm_skill.png` | PASS | Skill 指标和模型标签清晰。 |
| `per_horizon_rmse_curves.png` | PASS | 0.5–12 h 逐时距曲线可读。 |
| `cycle_region_8_5_to_10_5h.png` | PASS | 8.5–10.5 h 周期邻域标注完整。 |
| `true_period_vs_wrong_period.png` | PASS | P=19 与预冻结 P=15 使用同一 seed=42 协议。 |
| `multi_seed_variability.png` | PASS | 三随机种子均值与波动范围可读。 |
| `bootstrap_delta_rmse.png` | PASS | ΔRMSE、零线和 95% CI 可读。 |
| `cycle_attention_weights.png` | PASS | 三个同相位状态的总体权重可读。 |
| `cycle_attention_weights_vs_ageing.png` | PASS | 不同老化阶段的权重变化可读。 |
| `accuracy_vs_parameter_count.png` | PASS | 参数量—精度关系及标签未被裁切。 |
| `representative_validation_prediction.png` | PASS | 仅为 Validation 代表性轨迹，无 Test 内容。 |

## 说明

周期注意权重图只用于事后解释，没有用于模型选择或重新调参。`true_period_vs_wrong_period.png` 的 P=19 与 P=15 比较限定为共同可用的 seed=42；统计显著性以配对移动块 bootstrap 结果为准。
