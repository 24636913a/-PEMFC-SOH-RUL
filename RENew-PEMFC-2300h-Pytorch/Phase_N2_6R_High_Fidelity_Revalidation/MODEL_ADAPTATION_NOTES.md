# 模型适配说明

## 1. 共同输入输出

- 输入：`[batch, 48, 14]`，仅包含过去 24 h 的 F2 特征。
- 输出监督：未来 24 steps 的 `cell_voltage_mean`。
- 评价：将输出恢复为绝对电压后计算 RMSE、MAE、R² 与 Bias。
- 未来真实 current、current density、transition 或其他工况变量从未作为 decoder input。
- 所有 missing-value median 与外部 StandardScaler 只用对应 fold Train 拟合。

## 2. LSTM

LSTM 完全沿用 Phase N2/N2.6：两层、hidden dimension 64、dropout 0.1、last-state LayerNorm + Linear(24)。训练目标仍为每个 horizon 的标准化电压增量，预测后加回 origin 电压。该路径只用于冻结参考，不重新调参。

## 3. TimesNet

TimesNet 主体直接调用固定 commit 的官方 `Model`。输入先经过项目的 fold-Train-only feature scaling，再进入官方 sample-wise normalization、DataEmbedding、48→72 prediction extension、TimesBlock、projection 和官方 de-normalization。官方输出为 `[batch,24,14]`，Wrapper 只取第 0 通道得到 `[batch,24]`。

训练 loss 只作用于外部 feature-scaling 坐标中的未来绝对 `cell_voltage_mean`。预测完成后使用 fold-Train 电压 feature mean/scale 恢复为 V。其他 13 个输出没有辅助监督；这种处理保留官方多变量 forecasting path，同时不改变原 PEMFC 电压任务。

## 4. ModernTCN

ModernTCN 使用固定 commit 的 long-term forecasting modules。Wrapper 将 `[batch,48,14]` 转换为官方逐变量格式 `[batch,14,48]`。模型内部依次执行 RevIN、per-variable patching、四级 downsampling 与 blocks、top-down multi-scale fusion、Flatten_Head 和 RevIN de-normalization；Wrapper 最后只取第 0 变量的 24-step forecast。

由于 frozen commit 的原 `forward_feature` 未调用已声明的 multi-scale modules，本阶段在 vendor 外使用最小兼容 adapter 激活这些模块。核心 Stage/Block、kernel、FFN、RevIN 和 head 均直接来自官方类，vendor 源码保持不变。该 adapter 是实现保真修复，不是新的模型创新。

## 5. 训练一致性

所有模型使用 AdamW、learning rate 1e-3、weight decay 1e-5、batch size 64、MSE、gradient clip 1.0、最多 300 epochs 和 patience 30。early stopping 依据相同 Development Validation origins 的绝对电压 RMSE；不使用 Test，也不根据 R² 单独选模。
