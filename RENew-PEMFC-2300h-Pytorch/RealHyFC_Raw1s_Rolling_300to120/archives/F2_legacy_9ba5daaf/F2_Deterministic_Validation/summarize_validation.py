"""Read-only qualification summary. No training or fallback is executed here."""
from pathlib import Path
from datetime import datetime,timezone
import csv
import hashlib
import json

HERE=Path(__file__).resolve().parent
FOLDER=HERE.parent
REPO=FOLDER.parents[2]


def read(path):return json.loads(path.read_text(encoding='utf-8'))
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def save(path,value):path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8',newline='\n')


if __name__=='__main__':
    run=HERE/read(HERE/'LATEST_RUN.json')['run']
    plan=read(run/'PLAN.json')
    assert plan['status']=='STRICT_PREFIX_FAILED_STOPPED'
    env=read(run/'strict_a/ENVIRONMENT.json');failure=read(run/'strict_a/FAILURE.json')
    init=read(run/'strict_a/INITIALIZATION.json')
    location=read(run/'operator_location/OPERATOR_LOCATION.json')
    assert env['actual_workspace']==':4096:8' and env['workspace_configured_before_torch_import'] and env['cuda_uninitialized_after_torch_import']
    assert env['deterministic_algorithms'] and not env['warn_only']
    assert failure['epoch']==1 and failure['batch']==1 and failure['phase']=='backward' and failure['optimizer_updates']==0
    assert 'upsample_linear1d_backward_out_cuda' in failure['error']
    assert 'upsample_linear1d_backward_out_cuda' in location['error'] and location['weights_unchanged']
    assert location['optimizer_updates']==0 and not location['optimizer_constructed']
    assert not any((run/name).exists() for name in ['strict_b','benchmark_baseline','benchmark_strict','PAIR_COMPARISON.json'])
    assert sha(HERE/'deterministic_prefix.py')==plan['worker_sha256']
    assert sha(HERE/'run_validation.py')==plan['launcher_sha256']
    protected=read(run/'HISTORY_BEFORE.json')
    for name,digest in protected.items():assert sha(REPO/name)==digest,name
    oldrun=FOLDER/'F2_Training_Outputs/20261019T052754_499042Z_seed202_compare'
    oldinit=read(oldrun/'seed202/top10/TRAINING_LEDGER.json')
    assert init['initial_state_sha256']==oldinit['initialization_sha256']
    with (oldrun/'MASK_APPLICATION_AUDIT.csv').open(encoding='utf-8') as f:oldmask=list(csv.DictReader(f))
    for partition,key in [('TRAIN','prepared_train_sha256'),('VALIDATION','prepared_validation_sha256')]:
        reference=next(r for r in oldmask if r['feature_mode']=='top10' and r['partition']==partition)
        assert init[key]==reference['prepared_sha256']
    stack='\n'.join(location['anomaly_warnings'])
    assert 'decoder_variable_selection' in stack and 'resample_norm' in stack and 'F.interpolate' in stack
    decoder=next(m for m in location['interpolation_modules'] if m['module'].startswith('decoder_variable_selection'))
    summary=dict(status='STRICT_GPU_DETERMINISM_NOT_QUALIFIED',qualification_completed=True,
        planned_repetitions=2,started_repetitions=1,completed_short_repetitions=0,completed_epochs=0,
        optimizer_updates=0,additional_error_location_probes=1,error_location_probe_updates=0,
        environment_setup_valid=True,workspace=':4096:8',workspace_before_torch_import=True,
        strict_warn_only=False,first_batch_forward_succeeded=True,strict_backward_supported=False,
        model_and_preprocessing_identity_verified=True,
        first_blocking_operator='upsample_linear1d_backward_out_cuda',
        forward_autograd_node='UpsampleLinear1DBackward0',
        source_module=decoder['module'],dependency_source=decoder['source_file'],
        dependency_function_first_line=decoder['function_first_line'],dependency_interpolate_call_lines=decoder['interpolate_call_lines'],
        active_forward_interpolate_line=60,formal_model_call_line=479,
        pair_initialization_comparison=None,pair_epoch_loss_comparison=None,pair_gradient_comparison=None,pair_weight_comparison=None,
        training_speed_overhead_percent=None,training_speed_status='NA: no strict training step completed; conditional benchmark not started',
        safe_to_enable_in_formal_program=False,full_training_recommended_now=False,full_training_started=False,
        six_seed_training_started=False,network_operator_replacement=False,local_determinism_disabled=False,
        old_files_unchanged=len(protected),formal_program_sha256=sha(FOLDER/'F2_Train_Interactive.py'),
        no_holdout_scoring=True,no_new_accuracy_results=True,finished_utc=datetime.now(timezone.utc).isoformat())
    save(HERE/'VALIDATION_SUMMARY.json',summary)
    rows=[dict(check='Workspace configured before PyTorch/CUDA',result='PASS',evidence=':4096:8'),
          dict(check='Strict deterministic algorithms / warn_only=False',result='PASS',evidence='ENVIRONMENT.json'),
          dict(check='Original model / initialization / Train+Validation inputs',result='PASS',evidence='SHA256 exact matches'),
          dict(check='First batch forward',result='PASS',evidence='Failure phase is backward'),
          dict(check='Strict CUDA backward',result='NOT_QUALIFIED',evidence='upsample_linear1d_backward_out_cuda'),
          dict(check='Two-repeat loss/gradient/weight comparison',result='NA',evidence='Stopped on first unsupported operation'),
          dict(check='Speed overhead',result='NA',evidence='No complete strict update, no benchmark'),
          dict(check='Formal program applicability',result='NOT_QUALIFIED',evidence='Would fail on first training batch')]
    with (HERE/'QUALIFICATION_TABLE.csv').open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=['check','result','evidence']);w.writeheader();w.writerows(rows)
    text=f'''# 最小GPU严格确定性运行验证

**判定：STRICT_GPU_DETERMINISM_NOT_QUALIFIED。** 当前PyTorch2.6.0+cu124 / CUDA12.4 / pytorch-forecasting1.8.0环境中，原始F2不改结构和运算逻辑时，不能完成严格确定性的CUDA反向传播。工作区设置合法，但这不是整个模型可严格确定性训练的充分条件。

## 实际执行范围

预先固定Seed202、Top10、两次各6轮短程诊断（1750个Train窗口、218个Validation窗口，每轮55次更新）。原正式450轮上限、patience40、48→24、14通道、Top10/Top11掩码、7:1:2边界、Train scaler、EWMA、AdamW及超参数和指标均保持不变。短程预算是运行资格诊断，不是新预测精度实验。

第一进程已启动，但在Epoch1 / batch1的反向阶段报错：**完成Epoch为0，优化器更新为0**。按预先冻结的停止规则，没有启动第二次短程训练；仅做了一次无优化器、无权重更新的错误定位，记录准确的前向来源。未进行完整训练、六种子训练或特征实验，未读取后留数值或计算新的后留成绩。

计划中的速度对照只允许在两次短程逐项一致后执行（各5步预热+20步计时）。该条件未达到，所以两种速度对照均未启动，速度代价记为**NA**，不是0%。不能用失败进程启动耗时或单次前向耗时冒充训练速度比较。

## 环境设置已正确执行

- 独立入口`run_validation.py`只导入标准库，用新Python子进程运行，不复用已初始化CUDA的VS Code内核。
- 子进程启动环境先设置`CUBLAS_WORKSPACE_CONFIG=:4096:8`；worker在导入torch之前再次明确设置。导入torch后确认CUDA尚未初始化。
- 在模型初始化前启用`torch.use_deterministic_algorithms(True, warn_only=False)`，保持`cudnn.deterministic=True`、`cudnn.benchmark=False`及原TF32设置。
- 软件：PyTorch {env['torch']}、CUDA {env['cuda']}、cuDNN {env['cudnn']}、pytorch-forecasting {env['packages']['pytorch-forecasting']}；GPU {env['GPU']}，驱动{env['driver']}。
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

具体模块：`{decoder['module']}`。

依赖源码：`{decoder['source_file']}`，实际CUDA分支调用位于第60—61行，函数起点第52行。源码SHA256为`{decoder['source_file_sha256']}`。完整安装路径、模块清单、异常前向栈和反向栈保存在`operator_location/OPERATOR_LOCATION.json`。

该线性重采样用于原生TFT隐藏特征维度的匹配，**不是RealHyFC时序数据插值，也不是新增的模型分支**。模型中共有{len(location['interpolation_modules'])}个此类模块；本次准确定位到decoder的相对预测时间坐标GRN，不能据此假定仅改掉一处就能解决全部问题。PyTorch2.6文档明确将CUDA `interpolate` 的linear反向列为严格模式可能报错的操作。

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

本次运行：`{run.relative_to(HERE).as_posix()}`。主证据为`PLAN.json`、`strict_a/ENVIRONMENT.json`、`INITIALIZATION.json`、`FAILURE.json`及`operator_location/OPERATOR_LOCATION.json`；汇总为`VALIDATION_SUMMARY.json`、`QUALIFICATION_TABLE.csv`。

原主程序、审计、旧checkpoint、预测和训练结果全部保留。{len(protected)}个受保护文件SHA256复核不变。新的资格检查成功完成，但**严格训练本身没有通过**，两者不得混淆。
'''
    (HERE/'README.md').write_text(text,encoding='utf-8',newline='\n')
    print('QUALIFICATION_COMPLETED_STRICT_TRAINING_NOT_SUPPORTED_NO_FULL_TRAINING')
