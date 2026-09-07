# Phase N2.6 执行计划

英文名称：`Phase N2.6 — Modern Temporal Architecture Expansion and Backbone Re-Screening`

冻结日期：2026-09-06。本文档在模型训练前冻结。

## 1. 研究边界

本阶段只进行 Backbone Re-Screening（骨干模型再筛选），候选限定为 TimesNet、ModernTCN、PatchTSMixer 和 Compact Mamba。LSTM 是不调参的 incumbent baseline（现任基准）。不增加其他模型，不进行模型创新，不访问最终 Test 性能，不开展 SOH、RUL、EIS、uncertainty、ensemble 或外部数据验证。

最终 Test 的 `OpHrs2` 范围固定为 `[1819.997738, 2256.800000] h`。只允许读取 schema、行数和时间边界；Test target、预测、指标和曲线均禁止。

## 2. 冻结数据协议

- 数据：Phase N1 的 `causal_features_30min.csv`；
- Sampling：0.5 h；
- Feature Group：F2，精确 14 列从 Phase N2.5 冻结配置读取；
- Target：`delta_V(t+h) = V(t+h) - V(t)`，重建 `V_hat(t+h) = V(t) + delta_V_hat(t+h)`；
- Lookback：24 h，即 48 steps；
- Horizon：12 h，即 24 steps；
- Development validation：继承 Phase N2 的三个 expanding-window folds；
- Scaler：每折只在该折 Train 上拟合 feature median、StandardScaler 和每时距 target mean/std；
- Matched-Origin：所有候选必须与 Phase N2 LSTM 使用完全相同的 Validation origins 和 targets；
- 训练：AdamW，learning rate 1e-3，weight decay 1e-5，batch size 64，最多 300 epochs，early stopping patience 30，gradient clip 1.0，MSE loss。

## 3. 预注册候选配置

每类配置数均不超过 3，且不根据 Validation 临时扩展搜索。

### TimesNet

1. `timesnet_d32_k2`：d_model=32，d_ff=48，e_layers=2，top_k=2，2 个 Inception kernel；
2. `timesnet_d48_k3`：d_model=48，d_ff=48，e_layers=2，top_k=3，2 个 Inception kernel。

周期通过每个 TimesBlock 的 Train-input FFT 自动发现，不把 P=19 硬编码为唯一周期。另以每折 Train windows 统计主要频率，生成 `timesnet_period_discovery.csv`。Validation 和 Test 不参与周期发现。

### ModernTCN

1. `moderntcn_d80_k7_b2`：patch=4，stride=2，d_model=80，2 blocks，large kernel=7，expansion=2；
2. `moderntcn_d64_k13_b3`：patch=4，stride=2，d_model=64，3 blocks，large kernel=13，expansion=3。

使用 patch embedding、large-kernel depthwise temporal convolution、pointwise channel mixing 和 residual block 的轻量适配。

### PatchTSMixer

1. `patchtsmixer_p4_d64`：patch=4，stride=4，d_model=64，3 layers，expansion=2；
2. `patchtsmixer_p6_d64`：patch=6，stride=3，d_model=64，3 layers，expansion=2。

使用 Hugging Face Transformers 4.53.0 的 PatchTSMixer backbone，采用 mix-channel、gated attention、无 self-attention，并接统一 LayerNorm + Linear(24) head。

### Compact Mamba

1. `compact_mamba_d64_l2`：d_model=64，2 Mamba blocks，state_size=16，conv_kernel=4，expand=2。

优先使用 Hugging Face Transformers 4.53.0 的标准 MambaModel。官方 `mamba_ssm`/`causal_conv1d` Windows CUDA kernel 当前环境不可用时，使用该模型明确支持的 `mambapy 1.2.0` 纯 PyTorch parallel-scan backend；如果导入、CPU/GPU 前向、反向或数值检查仍失败，则标记 `implementation unavailable / unstable`，不编写差异过大的自制 Mamba。

训练前环境预检补充：Transformers 顺序 fallback 已通过前向和反向，但训练效率不满足本阶段筛选需求；`mambapy 1.2.0` 是公开、操作系统无关的纯 PyTorch Mamba 实现，并已被 Transformers 作为 `use_mambapy=True` 后端支持。因此本计划在任何训练开始前固定使用该后端，不在训练结果产生后切换实现。

## 4. 筛选与多随机种子规则

第一轮全部使用 seed=42、相同三折。每个模型家族先按三折平均 RMSE 选择本家族最佳配置，再与 LSTM 比较。

只有三折 Mean RMSE 不比 LSTM 差超过 10% 的新模型，才有资格进入 Top-2 和多随机种子。如果合格模型不足两个，只训练实际合格数量；若全部不合格，停止多 seed 扩展，不强行选 Top-2。

合格 Top-1/Top-2 与 LSTM 使用 seeds `[42, 2024, 3407]`。候选只有同时满足以下条件才能替换 LSTM：

1. 三 seed Mean RMSE 改善至少 2%；
2. 至少 2/3 folds 的三 seed 平均优于 LSTM；
3. 至少 2/3 seeds 的三折平均优于 LSTM；
4. Worst-fold 相对 LSTM 恶化不超过 3%；
5. paired moving-block bootstrap 支持，block=19 steps，2000 repetitions。

## 5. 统计与解释分析

- 主要指标：RMSE、MAE、Bias、R²、Skill_Best 和 Skill_LSTM；
- Per-horizon：0.5–12 h 共 24 个 step；
- 周期邻域：8.5、9.0、9.5、10.0、10.5 h；
- 老化阶段：只对 Development Validation origins 按统一 Early/Middle/Late ageing 边界比较；
- 复杂度：参数量、峰值 GPU 显存、平均 epoch 时间、总训练时间；
- bootstrap：四个家族最佳配置分别对 LSTM；若存在两名多 seed 合格候选，再做 Top1 vs Top2。

## 6. 停止条件

完成 LSTM 复现、四类候选筛选、合法 Top candidates 多 seed、bootstrap、per-horizon、周期邻域、ageing-stage、efficiency、报告和冻结 backbone 后立即停止。Phase N2.6 无论结果如何都继续禁止打开 Test，不进入 Phase N3。
