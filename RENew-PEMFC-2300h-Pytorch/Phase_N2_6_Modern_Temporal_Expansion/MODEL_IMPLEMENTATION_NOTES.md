# Phase N2.6 模型实现说明

## 1. 实现来源

| 模型 | 实现来源 | 版本或 commit | 是否官方 | 是否修改 | 备注 |
|---|---|---|---|---|---|
| LSTM | Phase N2 本地可审计 PyTorch 实现 | PyTorch 2.6.0+cu124 | 项目基准 | 否 | 不调参，只做必要复现并复用已冻结多 seed 结果。 |
| TimesNet | THUML Time-Series-Library 架构参考 | `4e938a1767106324dd753b2a44832bf870a0252e` | 上游仓库官方 | 是 | `local auditable adaptation（本地可审计适配实现）`；保留 FFT period discovery、1D→2D period reshape、intra/inter-period 2D convolution 和频率加权聚合，改为轻量 kernel 集合及统一 24-step residual head。未复制上游代码，不声明完全复现官方 benchmark。 |
| ModernTCN | `luodhhh/ModernTCN` 架构参考 | `56a9a2c018385cd5acef015378cae7f084d1b11c` | 上游仓库官方 | 是 | `local auditable adaptation（本地可审计适配实现）`；保留 patch embedding、large-kernel depthwise temporal convolution、channel mixing 与 residual block，使用统一 24-step residual head。未复制上游代码。 |
| PatchTSMixer | Hugging Face Transformers 稳定公开实现；IBM Granite TSFM 为作者生态参考 | Transformers 4.53.0；Granite TSFM `fe7a35697723e2a2f5246ae979474bfc554e26c0` | 稳定公开实现 | 是 | 使用标准 PatchTSMixerModel backbone，外接 LayerNorm + Linear(24)；关闭 self-attention，保留 patch/channel mixing 与 gated attention。 |
| Compact Mamba | Hugging Face Transformers 稳定 MambaModel；`mambapy` parallel-scan backend；`state-spaces/mamba` 为作者官方参考 | Transformers 4.53.0；mambapy 1.2.0 (`05f05fe`)；官方仓库 `e9594ce1c732d97440f0332fdc43170a2294dbfa` | 稳定公开实现，非作者 CUDA 包 | 是 | 输入经 Linear 投影后作为 `inputs_embeds`，使用标准 Mamba blocks 和统一 head。官方 `mamba_ssm`/`causal_conv1d` 未安装；Windows 下无匹配官方预编译 CUDA selective-scan wheel，故固定使用 Transformers 支持的 `use_mambapy=True` 纯 PyTorch parallel scan。若数值或反向不稳定则跳过。 |

## 2. 官方链接

- TimesNet / Time-Series-Library：https://github.com/thuml/Time-Series-Library
- ModernTCN：https://github.com/luodhhh/ModernTCN
- PatchTSMixer 文档：https://huggingface.co/docs/transformers/v4.53.0/model_doc/patchtsmixer
- IBM Granite TSFM：https://github.com/ibm-granite/granite-tsfm
- Mamba：https://github.com/state-spaces/mamba
- mambapy：https://github.com/alxndrTL/mamba.py

## 3. 当前运行环境

- Windows 11；Python 3.11.9；
- PyTorch 2.6.0+cu124，CUDA 可用；
- 计划冻结时 `transformers`、`mamba_ssm`、`causal_conv1d` 均未安装；
- Transformers 4.53.0 将安装到本阶段 `vendor` 子目录，不修改旧项目或 Phase N2/N2.5；
- 官方 Mamba CUDA selective scan 不作为本阶段可用实现；计划在训练前固定使用 mambapy 1.2.0 parallel scan，实际状态以 `implementation_sanity_audit.json` 为准。

## 4. 输入输出适配

所有候选输入均为 `[batch, 48, 14]`，输出均为 `[batch, 24]` 的标准化 residual voltage。训练后使用每折 Train target mean/std 逆变换，并加回 prediction origin 的平均单体电压，得到绝对电压预测。所有 scaler 只在对应 Train 拟合。
