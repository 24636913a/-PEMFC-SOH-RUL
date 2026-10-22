# 最小GPU严格确定性运行验证

**判定：STRICT_GPU_DETERMINISM_NOT_QUALIFIED。** 当前PyTorch2.6.0+cu124 / CUDA12.4 / pytorch-forecasting1.8.0环境中，原始F2不改结构和运算逻辑时，不能完成严格确定性的CUDA反向传播。工作区设置合法，但这不是整个模型可严格确定性训练的充分条件。

## 实际执行范围

预先固定Seed202、Top10、两次各6轮短程诊断（1750个Train窗口、218个Validation窗口，每轮55次更新）。原正式450轮上限、patience40、48→24、14通道、Top10/Top11掩码、7:1:2边界、Train scaler、EWMA、AdamW及超参数和指标均保持不变。短程预算是运行资格诊断，不是新预测精度实验。

第一进程已启动，但在Epoch1 / batch1的反向阶段报错：**完成Epoch为0，优化器更新为0**。按预先冻结的停止规则，没有启动第二次短程训练；仅做了一次无优化器、无权重更新的错误定位，记录准确的前向来源。未进行完整训练、六种子训练或特征实验，未读取后留数值或计算新的后留成绩。

计划中的速度对照只允许在两次短程逐项一致后执行（各5步预热+20步计时）。该条件未达到，所以两种速度对照均未启动，速度代价记为**NA**，不是0%。不能用失败进程启动耗时或单次前向耗时冒充训练速度比较。

## 环境设置已正确执行

- 独立入口`run_validation.py`只导入标准库，用新Python子进程运行，不复用已初始化CUDA的VS Code内核。
- 子进程启动环境先设置`CUBLAS_WORKSPACE_CONFIG=:4096:8`；worker在导入torch之前再次明确设置。导入torch后确认CUDA尚未初始化。
- 在模型初始化前启用`torch.use_deterministic_algorithms(True, warn_only=False)`，保持`cudnn.deterministic=True`、`cudnn.benchmark=False`及原TF32设置。
- 软件：PyTorch 2.6.0+cu124、CUDA 12.4、cuDNN 90100、pytorch-forecasting 1.8.0；GPU NVIDIA GeForce RTX 3070 Laptop GPU，驱动552.22。
- 前向计算通过，先前缺少工作区配置时的CuBLAS拒绝已被跨过；本轮没有全局修改系统环境变量。

合法工作区设置依据：[CUDA12.4 cuBLAS重复性说明](https://docs.nvidia.com/cuda/archive/12.4.1/cublas/index.html#results-reproducibility)、[PyTorch2.6确定性接口](https://docs.pytorch.org/docs/2.6/generated/torch.use_deterministic_algorithms.html)。这些文档不保证所有CUDA算子都有确定性实现。

## 准确阻断算子与位置

原始错误：

```text
upsample_linear1d_backward_out_cuda does not have a deterministic implementation
```

零更新定位使用autograd anomaly tracing，仅增加错误来源记录，没有替换任何模型运算。它确认本次先被拒绝的调用链为：

```text
RealHyFC F2.forward（正式源码第479行）
  decoder_variable_selection
  single_variable_grns.relative_forecast_step
  resample_norm
  TimeDistributedInterpolation.interpolate
  F.interpolate(mode="linear", align_corners=True)
  UpsampleLinear1DBackward0
  upsample_linear1d_backward_out_cuda（严格反向被拒绝）
```

具体模块：`decoder_variable_selection.single_variable_grns.relative_forecast_step.resample_norm.resample`。

依赖源码：`C:\Users\1\AppData\Roaming\Python\Python311\site-packages\pytorch_forecasting\models\temporal_fusion_transformer\sub_modules.py`，实际CUDA分支调用位于第60—61行，函数起点第52行。源码SHA256为`9ef892c565f6891ddf6d21c52c4d7f90393ec731924851aee872468cffe2cf9e`。完整安装路径、模块清单、异常前向栈和反向栈保存在`operator_location/OPERATOR_LOCATION.json`。

该线性重采样用于原生TFT隐藏特征维度的匹配，**不是RealHyFC时序数据插值，也不是新增的模型分支**。模型中共有17个此类模块；本次准确定位到decoder的相对预测时间坐标GRN，不能据此假定仅改掉一处就能解决全部问题。PyTorch2.6文档明确将CUDA `interpolate` 的linear反向列为严格模式可能报错的操作。

## 两次重复性比较与速度结果

| 项目 | 结果 |
|---|---|
| 当前初始化与原Seed202初始化 | SHA256完全一致 |
| Train/Validation预处理输入与原协议 | SHA256完全一致 |
| F2参数量 | 332436，原模型类身份通过 |
| CUBLAS设置时机与合法值 | 通过 |
| 第一批次严格前向 | 通过 |
| 第一批次严格反向 | 不合格，原算子缺少确定性实现 |
| 两次初始化、逐轮Loss、梯度、权重一致性 | NA，未完成第一次短程训练 |
| 完整严格训练速度及额外代价 | NA，未完成任何严格更新 |
| 安全直接启用到正式程序 | 不可以，会在首批次反向报错 |

没有用旧缓存制造一致结果，也没有把失败或不存在的指标写成0。定位探针的权重哈希前后相同，未构造优化器。

## 是否值得现在进行完整训练

**不值得直接执行当前配置的完整严格训练**：阻断发生在第一批次，延长Epoch没有帮助。正式`F2_Train_Interactive.py`保持原样；不能直接合并严格开关并宣称问题已解决。两次完整旧运行的28.466/33.837 mV及全部历史权重、结果仍原样保留。

以后若继续，需要先单独评估软件栈是否支持该原生算子的严格确定性反向，或另行审查确定性实现的数学等价性与checkpoint兼容性；这超出本轮授权。本轮不升级依赖、不换算子、不迁移单个算子到CPU、不局部关闭确定性、不使用warn_only=True，也不发起新的模型研究。通过前述算子资格后，才重新进行两次短程逐项检查，再评估是否批准完整重复。

## 独立入口与证据

在本文件夹的父目录终端使用原Python3.11/CUDA解释器：

```powershell
& 'D:/ruanjian/python3.11.9/python.exe' -B F2_Deterministic_Validation/run_validation.py
```

每次调用都会新建输出目录；该入口专供固定短程资格验证，不提供all或完整训练模式。当前环境下会重现已记录的拒绝，不会自动寻找替代算法。不要直接在已有torch/CUDA的交互内核执行worker；launcher会建立新进程来保证环境变量时机。

本次运行：`outputs/20261019T082402_605729Z_seed202_top10_strict`。主证据为`PLAN.json`、`strict_a/ENVIRONMENT.json`、`INITIALIZATION.json`、`FAILURE.json`及`operator_location/OPERATOR_LOCATION.json`；汇总为`VALIDATION_SUMMARY.json`、`QUALIFICATION_TABLE.csv`。

原主程序、审计、旧checkpoint、预测和训练结果全部保留。320个受保护文件SHA256复核不变。新的资格检查成功完成，但**严格训练本身没有通过**，两者不得混淆。
