# 代码、checkpoint与环境

权威文件路径/本地SHA/base blob SHA/last commit 均在 `L_REPRODUCIBILITY_MANIFEST.json` / `L_SOURCE_MANIFEST.json`，不是仅列模型名称。九个checkpoint分别与 J 与 K 身份交叉核验并前后两次 byte hash；不加载，不改写。F0/F1/F2 × Seed42/52/62，无 Repeat-A/B，无最佳Seed。

## 冻结来源

Canonical TFT = TFT；F1 = F1 + F_CONV；F2 = F2 + F_SCHEMA；任务训练配置 = H_CONFIG/H_COMMON/H_TRAIN；预处理 = E_CODE/E_SCALER/C_CODE；推理经验 = IR_COMMON/IR_REPORT；指标 = I_METRIC/C_CODE；Assessment运行契约 = I2_COMMON/I2_RUN；架构/限制 = J_REPORT。

## 原H Path A，不能替换

CUDA构建模型，strict load，`model.eval()`，`torch.no_grad()`；**保持原 requires_grad flags**；原 `e.predict -> c.predict`，batch32、原有顺序与末batch划分。不能替换为原I `predict_frozen`、`inference_mode + requires_grad_(False)`。I-R发现执行语义影响数值，不把它解释成唯一已识别 kernel 根因。本轮不进行 parity forward；历史抽样/缓存核验不等于任意新机器 full reproduction。

## 历史环境权威

H：Python3.11.9、PyTorch2.6.0+cu124、pytorch-forecasting1.8.0、CUDA12.4、cuDNN90100、RTX3070LaptopGPU、threads1、cudnn deterministic=true/benchmark=false、strict deterministic algorithms=false、TF32 matmul=false/cudnn=true、CUBLAS_WORKSPACE_CONFIG=null。H_CONFIG还保留历史 upstream `_tft.py` 与 sub_modules SHA。

本轮只用 importlib.metadata 记录当前安装版本，不 import torch、不测试当前GPU kernels、不宣称 runtime全一致。未来执行需核验实际 device/driver/upstream代码版本和权重加载契约；环境差异失败时不得 silently改推理路径。

严格 CUDA bitwise determinism 不支持：H 的 upsample_linear1d_backward strict probe失败，normal CUDA允许有限 repeat variation。此限制与 C1/D0/E0 历史复现 unresolved 分开。

E Train-only common scaler被冻结，target scale = 0.13680318714138442 V。只消费该artifact；不重新fit、不给新Test参与scaler。E preprocessing先历史电压EWMA，再common transform，再Top10 standardized-space zero mask；14slot固定。
