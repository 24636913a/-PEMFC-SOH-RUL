# Phase N2.7 执行计划

阶段名称：`Phase N2.7 — LSTM–TimesNet Error Complementarity, TimesNet Robustness Tuning and Causal Preprocessing Screening`

## 1. 冻结范围

- 数据分辨率：0.5 h。
- 特征：Phase N2 冻结的 F2 精确 14 列。
- Lookback：24 h（48 steps）。
- Horizon：12 h（24 steps）。
- Validation：完全继承 Phase N2 的三个 expanding-window folds。
- 随机种子：`[42, 2024, 3407]`。
- Test：`OpHrs2 ∈ [1819.997738, 2256.800000] h`，禁止读取 target、构造预测、计算指标、漂移或 gate 诊断。
- 本阶段不训练正式 Adaptive Dual-Path Model，不进入 Phase N2.8，不执行 Phase N3。

## 2. 历史预测复用

LSTM 使用 Phase N2 的冻结多随机种子预测，seed=42 以 Phase N2.6R 复现结果进行一致性核验；TimesNet 使用 Phase N2.6R 的 `timesnet_hf_t1` 三随机种子预测。每个 `(seed, fold)` 必须满足 origin、target、horizon 和真实标签逐元素一致，否则 fail-closed。

## 3. 指标聚合口径

- 模型整体 RMSE/MAE/R²：先对每个 `(seed, fold)` 计算，再对 9 个配对单元取宏平均，避免较长 Fold3 因样本数较多而改变预注册三折权重。
- Win rate、误差相关和大误差重合：对指定 scope 内的全部合法 prediction point 计算。
- Ageing stage：使用 Development Validation origin 的固定三等分边界，预计为 `1002.5 / 1270.8333 / 1539.1667 / 1807.5 h`，不使用 Test。
- Oracle：Pointwise Oracle 逐点选误差较小专家；Origin-wise Oracle 在每个 `(seed, origin)` 上用完整 24-step RMSE 选择一个专家。Oracle 仅表示理论上限。
- Fixed fusion：只评估 `alpha ∈ {0.25, 0.50, 0.75}`，公式为 `(1-alpha)LSTM + alpha TimesNet`。

## 4. Shift 诊断冻结方法

- Feature SMD/PSI：对应 fold 的 Train 与 Validation 历史窗口比较；PSI 十分位边界只由 Train 定义。
- Window summaries：仅用过去 48 steps，针对 9 个重点变量计算 mean、std、last 和线性 slope。
- Mahalanobis：先以 fold Train 的中位数、均值和标准差处理 36 个窗口摘要，再用 Train-only Ledoit–Wolf shrinkage covariance 拟合。
- Low/Medium/High shift：由对应 fold Train Mahalanobis score 的 1/3 与 2/3 分位数定义。
- 运行工况分组：电流、电流波动、零电流比例、电压离散度使用 Train tertiles；load transition 与 operation-point transition 使用 Train median。Validation 不参与边界拟合。

## 5. Winner predictability 冻结方法

- 每个 origin 的 winner 标签由三个种子的 24-step RMSE先取种子均值后比较：TimesNet 更低则标记 1，否则标记 0。
- 输入只含因果窗口摘要与 Train-only shift score，不含 fold、ageing stage、未来 target、未来 load 或未来时间。
- 分类器固定为 `StandardScaler + LogisticRegression(C=1.0, class_weight='balanced', max_iter=2000)`。
- Diagnostic 1：Fold1 Validation 训练，Fold2 Validation评估。
- Diagnostic 2：Fold1+Fold2 Validation训练，Fold3 Validation评估。
- 只报告 ROC-AUC、Accuracy、Precision、Recall；AUC≥0.65 视为存在可用专家选择信号。

## 6. TimesNet 有限调优

核心模型保持 Phase N2.6R 官方高保真 forecasting path。T0 为原 `timesnet_hf_t1`，只新增：

| 配置 | Loss | LR | Dropout | Weight decay |
|---|---|---:|---:|---:|
| T1 Huber | Huber(delta=1.0) | 1e-3 | 0.1 | 1e-5 |
| T2 Lower LR | MSE | 5e-4 | 0.1 | 1e-5 |
| T3 More Dropout | MSE | 1e-3 | 0.2 | 1e-5 |
| T4 Stronger Regularization | MSE | 1e-3 | 0.1 | 1e-4 |
| T5 Robust Combination | Huber(delta=1.0) | 5e-4 | 0.2 | 1e-5 |

第一轮仅 seed=42。候选必须同时满足：Fold3 RMSE 相对 T0 改善≥10%、三折 Mean RMSE 不恶化超过2%、Late Development RMSE降低。通过者按 Fold3、Late、Mean RMSE 顺序选一个进入 `[42, 2024, 3407]`；多种子仍需满足相同方向约束，否则最终保留 T0。

## 7. 因果预处理

- P0 Raw。
- P1 Causal EMA：只处理五个 voltage-state 特征，`alpha=0.5`，每个 temporal segment 独立初始化。
- P2 Causal one-sided SG：只使用 `[t-4,t]`，window=5，polyorder=2；segment 前四点用已有历史及可识别的最高阶数。
- Load/current/transition 特征保持原始；未来 raw target、当前 raw anchor 均不平滑。
- seed=42 比较 `LSTM × {Raw,EMA,SG}` 与 `最终稳健 TimesNet architecture × {Raw,EMA,SG}`。
- 相对同模型 Raw，只有 Mean RMSE降低≥2%、Fold3恶化≤5%、Late不恶化才进入多随机种子；多种子复验仍满足同一门槛才冻结为有效预处理。

## 8. 最终 N2.8 判定

用最终 LSTM 与最终 TimesNet 专家重新计算全部互补性、融合与 shift 结果。N2.8 建议 GO 需要核心证据同时成立：Origin-wise Oracle 改善≥5%；TimesNet 至少存在一个稳定优势区域；至少一个 chronological winner AUC≥0.65；High Shift/Late 中 LSTM 明显更稳。否则 NOGO。

## 9. 停止条件

完成附件要求的 15 份核心 CSV、12 张 Development-only 图片、因果/Test/配对完整性审计、结果报告和冻结 YAML 后停止。禁止正式 Gate、Attention、xLSTM、Mamba、额外 Transformer/TCN、EIS、SOH、RUL、PHM2014 和 Test。
