# Phase N2.7 Results — LSTM–TimesNet误差互补性、TimesNet稳健调优与因果预处理筛选

## 主要结果

| 模型/方案 | RMSE (mV) | MAE (mV) | R² |
|---|---|---|---|
| Frozen LSTM | 8.206 | 4.516 | 0.99585 |
| Original TimesNet | 9.908 | 3.811 | 0.99096 |
| Robust-Tuned TimesNet (T0_original) | 9.908 | 3.811 | 0.99096 |
| Best Preprocessed LSTM (P2_causal_sg) | 7.786 | 4.409 | 0.99626 |
| Best Preprocessed TimesNet (P0_raw) | 9.908 | 3.811 | 0.99096 |
| Best Fixed Fusion (α=0.25) | 7.180 | 3.729 | 0.99652 |
| Origin-wise Oracle (upper bound) | 5.261 | 3.038 | 0.99796 |

以上均为 Development 三个 expanding folds、三个 seeds 的 `(seed, fold)` 宏平均；Test `[1819.997738, 2256.800000] h` 未读取目标、未构造预测、未计算指标、未做 shift 诊断。

## 冻结结论

- Phase N2.7：**PASS**。
- 最终 LSTM 专家：Frozen LSTM 架构 + `P2_causal_sg`（仅历史 5 点、二次拟合，仅处理 voltage-state，预测 raw target）。
- 最终 TimesNet 专家：N2.6R 高保真 `T0_original` + `P0_raw`。
- TimesNet Huber 在 seed=42 将 Fold3 改善 11.10%，但未通过三 seed 复核，因此不冻结为最终版本。
- Phase N2.8：**NOGO**。Gate 1/2/3/4 = `True/True/False/True`。

## Q1–Q7：误差互补性与融合上限

**Q1：LSTM和TimesNet是否具有明显误差互补性？**  YES。最终两专家绝对误差 Pearson/Spearman 为 0.419/0.080，Top10% 大误差 Jaccard 为 0.123，Origin-wise Oracle 相对最终 LSTM 改善 32.43%。

**Q2：TimesNet整体有多少比例预测点优于LSTM？** 58.14%。

**Q3：Fold1/Fold2/Fold3 win rate？** 76.25% / 58.36% / 48.96%。

**Q4：误差相关系数？** signed error Pearson/Spearman = 0.394/0.143；absolute error Pearson/Spearman = 0.419/0.080。

**Q5：Top10%大误差重合率？** Jaccard = 0.123。

**Q6：Origin-wise Oracle改善？** RMSE = 5.261 mV，相对最终 LSTM 改善 32.43%；Pointwise Oracle RMSE = 4.132 mV。二者均仅为不可部署的理论上限。

**Q7：固定融合是否优于LSTM？** YES。三组预注册权重中最佳为 α=0.25，RMSE=7.180 mV，相对 LSTM 改善 7.78%。

## Q8–Q10：Distribution shift 与 winner 可预测性

**Q8：TimesNet Fold3失稳是否与 distribution shift 相关？** TimesNet origin RMSE 与 Train-defined Mahalanobis shift score 的 Pearson/Spearman 为 0.034/0.026，接近 0，不支持强单调关系。High Shift 分层中 TimesNet 明显弱于 LSTM，说明存在相对脆弱的 shift regime，但单一 Mahalanobis score 不能解释 Fold3 失稳。

**Q9：High Shift区域RMSE？** LSTM=7.540 mV，TimesNet=11.208 mV。Late Development：LSTM=10.092 mV，TimesNet=22.956 mV。

**Q10：Winner能否被因果特征预测？** 最大 chronological ROC-AUC=0.614；阈值 0.65 未达到。Fold1 的 233 个 origin 全部由 TimesNet 获胜，因训练标签只有一个类别，Fold1→Fold2 诊断不可估计；Fold1+Fold2→Fold3 可估计但未过门槛。分类器只使用过去 48 steps 的因果摘要，不使用 future target/load、fold、ageing stage 或时间。

| 时间顺序诊断 | ROC-AUC | Accuracy | Precision(T) | Recall(T) |
|---|---|---|---|---|
| Fold1 Validation → Fold2 Validation | NA | NA | NA | NA |
| Fold1+Fold2 Validation → Fold3 Validation | 0.614 | 0.655 | 0.196 | 0.441 |

## Q11–Q16：TimesNet调优与因果预处理

**Q11：Huber/LR/dropout/weight decay 是否改善 Fold3？** seed=42 下只有 Huber 满足三道初筛门槛；lower LR 与 more dropout 使 Fold3恶化，stronger regularization 基本无变化，robust combination 未达到 10% Fold3 改善。Huber 未在三 seed 复核中维持全部门槛。

**Q12：最佳稳健 TimesNet？** 最终保留 `T0_original`：RMSE=9.908 mV，MAE=3.811 mV，R²=0.99096；不宣称调优成功。

**Q13：EMA是否有效？** LSTM seed=42 mean RMSE=8.532 mV，初筛有益，但弱于 causal SG，未进入多 seed；TimesNet EMA 未通过门槛。

**Q14：Causal SG是否有效？** 对 LSTM 有效并通过三 seed：RMSE=7.786 mV，Fold3=11.711 mV；对 TimesNet seed=42 的 mean improvement 仅 0.37%，未达到 2%。

**Q15：LSTM最终 Raw 还是 Filtered？** `P2_causal_sg`。

**Q16：TimesNet最终 Raw 还是 Filtered？** `P0_raw`。

## Q17–Q18：最终专家与下一阶段判定

**Q17：最终双路径候选专家？** `LSTM + causal SG voltage-state inputs` 与 `official high-fidelity TimesNet T0 + Raw inputs`。优势区域的稳定性按三 seed 检查；满足“全部三 seed 且 RMSE 至少改善 5%”的区域包括：Fold=fold_1, ageing_stage=Early Development。

**Q18：是否进入 Phase N2.8？** **NOGO**。

- Gate 1 Origin-wise Oracle ≥5%：True（32.43%）。
- Gate 2 存在稳定 TimesNet 优势区域：True。
- Gate 3 至少一个 chronological AUC ≥0.65：False（max=0.614）。
- Gate 4 High Shift 与 Late Development 均由 LSTM 更安全：True。

## 完整性与可复现性

- 预测成对完整性：9/9 `(seed, fold)` PASS。
- 因果滤波：future sample=false、centered=false、backfill=false、target filtered=false、load transition filtered=false。
- Shift：每个 fold 的 scaler、Ledoit-Wolf covariance、PSI bins、shift tertiles 均只由该 fold Train 拟合。
- Winner label：对每个 origin 使用三 seed 的 24-step RMSE均值确定；Logistic Regression 按 Fold1→Fold2、Fold1+2→Fold3 顺序评估。
- Test firewall：`test_target_rows_read=0`、`test_predictions_constructed=false`、`test_metrics_computed=false`、`test_shift_diagnostics_computed=false`。
- 本阶段没有训练 Adaptive Dual-Path Gate，也没有进入 Phase N2.8 实施。
