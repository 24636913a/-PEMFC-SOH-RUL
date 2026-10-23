# Phase N2.5 预注册实验计划

## 1. 研究目标

本阶段只研究重载动态工况下 PEMFC 多时距运行状态预测。输入为过去 24 h 的 F2（电压+负载）因果特征，输出为未来 0.5–12 h 相对预测起点的 24 步电压增量，之后重建绝对电压。

本阶段不研究 SOH（健康状态）、RUL（剩余使用寿命）、EIS（电化学阻抗谱）、不确定性、集成学习或外部数据迁移。

## 2. Test 防火墙

- 最终 Test 边界固定为 `OpHrs2 ∈ [1819.997738, 2256.800000] h`。
- 只允许读取 Test 的 schema、行数、时间范围和合法窗口数量。
- 特征、目标、预测和任何性能指标都只在 Development 区完成。
- 数据加载器必须在首个 Test 行之前停止目标与特征读取；Test 目标读取行数必须为 0。

## 3. 完全继承 Phase N2 的协议

- Sampling resolution（采样分辨率）：0.5 h。
- Feature Group（特征组）：F2，精确 14 列从 `PHASE_N2_FROZEN_CONFIG.yaml` 读取。
- Lookback（历史长度）：24 h，`L=48`。
- Primary Horizon（主预测时距）：12 h，`H=24`。
- Target（目标）：`ΔV(t+h)=V(t+h)-V(t)`。
- Empirical period（经验周期）：`P=19 steps=9.5 h`。
- 三个 expanding-window development folds、Matched-Origin、Train-only median/scaler、AdamW、MSE 和 early stopping 均不改变。
- 普通 LSTM 固定为 hidden dimension 64、2 layers、dropout 0.1，不调参。

## 4. xLSTM 变体预注册

仅筛选三个固定候选，不进行结构搜索：

1. `sLSTM`：两层、隐藏维度 64，指数输入/遗忘门、稳定化状态与标量归一化记忆。
2. `mLSTM`：两层、隐藏维度 64、4 个 memory heads，矩阵记忆及 covariance-style update。
3. `Hybrid xLSTM`：一层 sLSTM 后接一层 mLSTM，隐藏维度 64。

三者均为 `local auditable implementation（本地可审计实现）`，以 xLSTM 论文和 NX-AI 官方仓库为结构依据，但不声明逐行复现官方语言模型代码。官方 release v2.0.4 依赖面向语言模型的 block stack 与可选 CUDA kernel；当前 Windows/PyTorch 环境未安装该 package，因此采用与本任务规模匹配的本地实现并完整记录方程、参数和代码。

参考资料：

- https://arxiv.org/abs/2405.04517
- https://github.com/NX-AI/xlstm
- https://github.com/NX-AI/xlstm/releases/tag/v2.0.4

## 5. 周期模块预注册

周期状态固定取编码序列的 `t`、`t−P`、`t−2P`。共享两层 scorer 产生三个 softmax 权重，再用逐维 sigmoid gate 融合当前状态和周期上下文。真实周期固定 `P=19`；错误周期在任何 Validation 结果产生前固定为 `P_wrong=15 steps=7.5 h`，不搜索其他错误周期。

## 6. 实验顺序与选择规则

1. 复现检查：同一代码流水线复现普通 LSTM seed 42；三折平均 RMSE 与 Phase N2 差异超过 2% 即停止。
2. xLSTM 筛选：LSTM、sLSTM、mLSTM、Hybrid xLSTM，seed 42。
3. 选择 seed 42 三折平均 RMSE 最低的 xLSTM 作为后续唯一 xLSTM backbone。
4. 核心消融：A0 LSTM、A1 最佳 xLSTM、A2 CA-LSTM(P=19)、A3 CA-xLSTM(P=19)、A4 CA-xLSTM(P=15)。
5. A0–A3 使用 seeds `[42, 2024, 3407]`；A4 作为预注册错误周期对照使用 seed 42。
6. moving-block bootstrap（移动块自助法）固定 block length 19 steps、2,000 次。

CA-xLSTM 只有在相对普通 LSTM 三 seed 平均 RMSE 改善至少 2%、至少 2/3 folds 改善、三 seed 平均改善、95% bootstrap CI 优先完整低于 0，且 P=19 平均优于 P=15 时才能冻结。若 CA-LSTM 更好则冻结 CA-LSTM；若创新结构都没有稳定超过 LSTM，则冻结 LSTM 并给出 `Architecture innovation NO-GO（结构创新未通过）`。

## 7. 分析范围

- 单独报告 h=9.5 h 及 8.5–10.5 h 周期邻域。
- Development 按固定三等分 ageing coordinate 分为 Early、Middle、Late；该分层只用于事后解释，不用于调参。
- 若周期模型有效，保存三个周期 attention 权重；权重不反馈到模型选择。
- Optional LSTM-Mamba 对照不执行，以避免延迟核心预注册实验和扩张模型空间。

## 8. 停止条件

完成变体筛选、周期消融、错误周期、多 seed、bootstrap、周期邻域、老化阶段、效率比较与最终冻结后立即停止。不得打开 Test performance，不得进入 Phase N3。

