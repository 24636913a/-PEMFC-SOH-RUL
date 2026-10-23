# Phase N2.6R 执行计划

## 1. 研究边界

本阶段只复验 LSTM、TimesNet（时间二维变化建模网络）和 ModernTCN（现代时间卷积网络）。不增加模型，不改变数据、特征、目标、历史长度、预测时距、Development folds 或 scaler 规则，不进入 Phase N3。

最终 Test 固定为 `OpHrs2 ∈ [1819.997738, 2256.800000] h`。本阶段只允许读取 Test 的 schema、行数与时间边界；Test target 读取、Test 预测以及 Test RMSE/R²/MAE 均必须保持为零。

## 2. 冻结数据协议

- Sampling：0.5 h。
- Feature group：F2，精确 14 个特征，继承 Phase N2.5 frozen config。
- Lookback：24 h = 48 steps。
- Horizon：12 h = 24 steps。
- 监督目标：未来 `cell_voltage_mean`；绝对电压预测与 `ΔV(t+h)=V(t+h)-V(t)` 数学互换。
- Validation：继承 Phase N2 三个 expanding-window folds；所有模型使用完全相同的 prediction origins 与 target timestamps。
- Scaling：每折只用该折 Train 拟合外部 feature scaler；模型内部官方归一化按各架构定义保留。
- 主评价：绝对电压 RMSE、MAE、R²；Bias 作为辅助。模型排序优先级为 RMSE、MAE、R²、fold 稳定性。

## 3. 冻结模型与训练协议

### LSTM reference

直接复现冻结配置：hidden dimension 64、2 layers、dropout 0.1、AdamW、learning rate 1e-3、weight decay 1e-5、batch size 64、MSE、最多 300 epochs、patience 30。seed=42 三折相对 Phase N2.6 差异必须小于 2%，否则停止。

### TimesNet high-fidelity

源码固定为 THUML Time-Series-Library commit `4e938a1767106324dd753b2a44832bf870a0252e`，vendoring 至 `vendor/TimesNet_official` 并保留 LICENSE。Wrapper 只适配 `[B,48,14] → [B,24,14]`、目标通道与 scaler，不改变官方 TimesBlock。

- T1：`d_model=32, d_ff=64, e_layers=1, top_k=2, num_kernels=3, dropout=0.1`。
- T2：`d_model=64, d_ff=128, e_layers=2, top_k=3, num_kernels=2, dropout=0.1`。

必须保留 DataEmbedding、FFT period discovery、48→72 temporal prediction extension、extended-sequence TimesBlock、Inception_Block_V1、official residual、projection、future 24-step slice 和官方 sample-wise normalization/de-normalization。主 loss 只作用于未来 `cell_voltage_mean`，不输入未来真实负载，不对其他未来变量施加辅助监督。

### ModernTCN high-fidelity

源码固定为 luodhhh/ModernTCN commit `56a9a2c018385cd5acef015378cae7f084d1b11c` 的 `ModernTCN-Long-term-forecasting`，vendoring 至 `vendor/ModernTCN_official` 并保留 LICENSE。Wrapper 只做 shape、目标通道与 scaler 适配。

- M1：4 stages，`dims=[16,24,32,48]`，`num_blocks=[1,1,1,1]`，FFN ratio 2，patch size 4，stride 2，downsample ratio 2。
- M2：4 stages，预检后冻结为 `dims=[18,26,34,50]`，`num_blocks=[1,1,1,1]`，FFN ratio 2，patch size 4，stride 2，downsample ratio 2。

每阶段 large/small kernels 按 48-step 输入的有效长度在配置中显式冻结。必须保留 RevIN、per-variable patching、large/small kernel、variable-independent FFN、variable-mixing FFN、multi-stage hierarchy、multi-scale fusion、Flatten_Head 和官方 forecasting output path；禁止 temporal mean pooling。

## 4. 实现保真闸门

训练前生成 `implementation_fidelity_audit.csv`。任何必需核心项失败，对应架构不得标记为 high-fidelity，也不得进入正式比较。还必须完成 CPU/GPU forward/backward、shape、有限输出/梯度、参数量、inverse-transform 和 Test firewall sanity audit。允许参数量与 LSTM 不同，但每个高保真配置应小于 500k；若官方完整结构因短序列约束无法满足，将记录为 implementation unavailable，而不是删减核心模块。

## 5. 筛选、多 seed 与统计规则

第一轮固定 seed=42，运行 LSTM、T1、T2、M1、M2 的三个 Development folds。每个 family 只按 Development mean RMSE 选择最佳配置。

候选满足任一条件才进入 seeds `[42,2024,3407]`：

1. Mean RMSE ≤ 1.10 × LSTM；或
2. 至少 2/3 folds RMSE 低于 LSTM。

若候选 Mean RMSE > 1.10 × LSTM 且 0/3 folds 改善，则停止其多 seed。替换 LSTM 必须同时满足：三 seed mean RMSE 至少改善 2%、至少 2/3 folds 改善、mean MAE 不高于 LSTM、mean R² 不低于 LSTM、Late Development RMSE 恶化不超过 3%，并由 paired moving-block bootstrap 支持。

Bootstrap 固定 block 19 steps、2000 次，只使用相同 Development origins。逐时距输出 0.5–12 h 的 RMSE/MAE/R²；单独输出 8.5–10.5 h；老化阶段按 inherited Early/Middle/Late Development 边界分析。

## 6. 停止条件

完成官方源码固定、保真审计、LSTM 复现、seed=42 三折筛选、条件式多 seed、bootstrap、per-horizon、cycle-region、ageing-stage、效率、结果报告与 backbone 冻结后立即停止。若 TimesNet 与 ModernTCN 均明显差于 LSTM且 0/3 folds 改善，冻结 `General backbone search STOP`，后续不再扩充通用模型池。

## 7. 训练前配置修订记录

第一次保真审计显示，建议的 ModernTCN M2 `dims=[24,32,48,64]` 在保留四级 hierarchy、两类 FFN、large/small kernel、multi-scale fusion 与 Flatten_Head 后共有 767,848 个参数，超过预注册的 `<500k` 上限。该检查发生在任何模型训练之前。M2 因此收缩为逐级仍宽于 M1 的 `[18,26,34,50]`，参数量为 488,902；其阶段数、block 数、FFN ratio、patching、kernel、RevIN、融合与预测头全部不变。除该参数上限修订外，计划不变。
