# 官方实现与保真审计

## 1. 审计结论

四个预注册配置全部通过实现保真与数值稳定性闸门，共 52/52 项通过。只有通过该闸门的结果才允许进入 N2.6R 正式比较。

| 配置 | 架构 | 参数量 | 保真状态 |
|---|---|---:|---|
| timesnet_hf_t1 | TimesNet | 149,046 | PASS |
| timesnet_hf_t2 | TimesNet | 335,702 | PASS |
| moderntcn_hf_m1 | ModernTCN | 445,456 | PASS |
| moderntcn_hf_m2 | ModernTCN | 488,902 | PASS |

全部配置通过 `[4,48,14] → [4,24]` CPU/GPU forward-backward、有限输出、有限 loss、有限输入梯度和有限参数梯度检查。正式训练仍只使用 Development folds。

## 2. TimesNet

- 官方仓库：https://github.com/thuml/Time-Series-Library
- 固定 commit：`4e938a1767106324dd753b2a44832bf870a0252e`
- vendored source：`vendor/TimesNet_official`
- source archive SHA-256：`8AD00C38D00BAA6B4A17DF2A688A96C945A0C005B6CACECA0E932E6F5AC6EDAA`
- `models/TimesNet.py` SHA-256：`F64C4BED1FD7347090044A0163BD4209C9F9EC1C1B19CEFF47842DF36B64BBA7`
- LICENSE：已保留。

运行时确认 DataEmbedding、`Linear(48,72)` 时间扩展、FFT period discovery、extended-sequence TimesBlock、Inception_Block_V1、official residual、projection、sample-wise normalization/de-normalization 和 future 24-step slice 均实际参与 forward。Wrapper 只选择 `cell_voltage_mean` 输出通道，不包含 last-hidden-state head 或 temporal mean pooling。

## 3. ModernTCN

- 官方仓库：https://github.com/luodhhh/ModernTCN
- 固定 commit：`56a9a2c018385cd5acef015378cae7f084d1b11c`
- 使用目录：`ModernTCN-Long-term-forecasting`
- vendored source：`vendor/ModernTCN_official`
- source archive SHA-256：`441CFF323292CDADC6FD6872218743B64098ABB3322FCDAEE3B8704C9A234EB7`
- `models/ModernTCN.py` SHA-256：`DAF74722E40D928E5ED6BC871B528A5D5D2C9090F30CAFB89110CC88DC60FA30`
- LICENSE：已保留。

运行时确认 RevIN、逐变量 stem patching、large/small kernel re-parameterization branches、变量独立 FFN、变量混合 FFN、四级 temporal hierarchy、top-down multi-scale fusion、Flatten_Head 和 RevIN de-normalization 均实际参与 forward；不存在 temporal mean pooling。

### 固定 commit 的 multi-scale 兼容说明

该官方 long-term 文件构造了 `lat_layer`、`smooth_layer`、`up_sample_conv`，并在 `use_multi_scale=True` 时把 Flatten_Head 输入长度设置为 stage-0 patch length；但原 `forward_feature` 在融合前直接返回最后一级特征，因此原路径既没有调用上述模块，也无法与其 multi-scale head shape 对齐。

本阶段不修改 vendor 文件。独立 adapter 保留官方 stem、downsampling、Stage、Block、large/small convolution、两类 FFN、RevIN 和 Flatten_Head，只按官方已声明模块执行四级 top-down lateral-addition fusion。运行时 hooks 证实每次 forward 调用 4 个 lateral、4 个 smooth 和 3 个 upsample 模块。该差异必须在论文或后续报告中描述为：

`official frozen commit + audited multi-scale activation adapter`

不得描述为“官方源码零修改直接运行”。

## 4. 审计文件

逐项证据位于 `outputs/implementation_audit/implementation_fidelity_audit.csv` 和同名 JSON。任何后续代码变动都必须重新运行该审计。
