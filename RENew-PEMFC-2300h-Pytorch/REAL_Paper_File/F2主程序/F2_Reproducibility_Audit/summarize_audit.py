"""Summarize actual traces, reconcile the preserved I/O failure, protect old assets."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import math

import numpy as np
import pandas as pd
import torch

from audit_saved_evidence import main as audit_saved, read, save

HERE=Path(__file__).resolve().parent
FOLDER=HERE.parent
REPO=FOLDER.parents[2]
OUT=HERE/'outputs'


def first_diff(a,b,key):
    indices=[i for i,(u,v) in enumerate(zip(a,b)) if u[key]!=v[key]]
    i=indices[0] if indices else None
    return dict(field=key,unequal_batches=len(indices),first_step=i+1 if i is not None else None,
                first_epoch=a[i]['epoch'] if i is not None else None,first_batch=a[i]['batch'] if i is not None else None)


def main():
    audit_saved()
    evidence=read(OUT/'saved_evidence/SAVED_EVIDENCE_SUMMARY.json')
    traces=[[json.loads(line) for line in (OUT/label/'BATCH_TRACE.jsonl').read_text(encoding='utf-8').splitlines()]
            for label in ['original_a','original_b']]
    a,b=traces
    assert len(a)==len(b)==330
    initial=[read(OUT/label/'INITIALIZATION.json') for label in ['original_a','original_b']]
    assert initial[0]==initial[1]
    historical_run=FOLDER/'F2_Training_Outputs'/evidence['metrics'][0]['run']
    old_masks=pd.read_csv(historical_run/'MASK_APPLICATION_AUDIT.csv')
    for partition,key in [('TRAIN','prepared_train_sha256'),('VALIDATION','prepared_validation_sha256')]:
        expected=old_masks.loc[(old_masks.feature_mode=='top10')&(old_masks.partition==partition),'prepared_sha256'].iloc[0]
        assert initial[0][key]==expected
    old_ledger=read(historical_run/'seed202/top10/TRAINING_LEDGER.json')
    assert initial[0]['initialization_sha256']==old_ledger['initialization_sha256']
    old_history=pd.read_csv(historical_run/'seed202/top10/training_history.csv')
    for label in ['original_a','original_b']:
        epochs=read(OUT/label/'EPOCH_TRACE.json')
        assert [r['sampler_order_sha256'] for r in epochs]==old_history.sampler_order_sha256.iloc[:6].tolist()
    keys=['indices','input_sha','target_sha','rng_before','rng_after_forward','rng_after_step',
          'weights_before','output_sha','loss','gradient_norm','gradients_before_clip','gradients_after_clip','weights_after']
    differences=[first_diff(a,b,key) for key in keys]
    for key in keys[:6]:assert not any(u[key]!=v[key] for u,v in zip(a,b)),key
    mask_keys=['module','p','eligible_sha','kept_sha']
    for u,v in zip(a,b):
        assert [{k:d[k] for k in mask_keys} for d in u['dropouts']]==[{k:d[k] for k in mask_keys} for d in v['dropouts']]
    pd.DataFrame(differences).to_csv(OUT/'PREFIX_FIRST_DIVERGENCE.csv',index=False)
    # Numeric first-batch backward comparison; tensors were obtained with zero optimizer updates.
    receipts=[read(OUT/('backward_'+label)/'RECEIPT.json') for label in ['a','b']]
    assert receipts[0]==receipts[1]
    assert receipts[0]['optimizer_updates']==0 and receipts[0]['initialization']==receipts[0]['weights_after']
    gradients=[torch.load(OUT/('backward_'+label)/'gradients_before_clip.pt',map_location='cpu',weights_only=True) for label in ['a','b']]
    gradient_rows=[]
    for name,x in gradients[0].items():
        y=gradients[1][name];dx=x.double().numpy();dy=y.double().numpy();difference=dy-dx
        gradient_rows.append(dict(parameter=name,values=x.numel(),numerically_unequal=int(np.count_nonzero(difference)),
            max_abs_difference=float(np.max(np.abs(difference))),mean_abs_difference=float(np.mean(np.abs(difference))),
            L2_difference=float(np.sqrt(np.sum(difference**2)))))
    grad=pd.DataFrame(gradient_rows)
    grad.to_csv(OUT/'FIRST_BATCH_GRADIENT_NUMERICAL_DIFFERENCES.csv',index=False,float_format='%.17g')
    changed=grad.loc[grad.numerically_unequal>0]
    strict=read(OUT/'strict_check/STRICT_RESULT.json')
    assert strict['optimizer_updates']==0
    assert strict['status']=='STRICT_DIAGNOSTIC_REJECTED_OPERATION' and strict['phase']=='forward' and 'CuBLAS' in strict['error']
    totals=[read(OUT/label/'COMPLETE.json') for label in ['original_a','original_b']]
    assert all(r['epochs']==6 and r['optimizer_updates']==330 and not r['holdout_scored'] for r in totals)
    protection=read(HERE/'HISTORY_BEFORE.json')
    for name,digest in protection.items():assert hashlib.sha256((REPO/name).read_bytes()).hexdigest()==digest,name
    execution=read(HERE/'PROBE_EXECUTION.json')
    worker=HERE/'probe_training_prefix.py'
    assert hashlib.sha256(worker.read_bytes()).hexdigest()==execution['worker_sha256']
    assert 'UnicodeEncodeError' in execution.get('error','') and execution['completed']==['original_a','original_b']
    reconciliation=dict(status='PASS_EXISTING_ARTIFACTS_RECONCILED_NO_RERUN',
        original_controller_receipt='PROBE_EXECUTION.json',original_controller_status=execution['status'],
        preserved_failure='Console GBK/UTF-8 relay only; strict result saved before forwarding failed.',
        completed_original_prefixes=2,epochs_per_prefix=6,total_optimizer_updates=660,
        no_update_backward_checks=2,strict_forward_rejected=True,new_formal_performance_results=0,
        protected_old_files_unchanged=len(protection),training_source_unchanged=True,
        finished_utc=datetime.now(timezone.utc).isoformat())
    save(HERE/'RECONCILIATION.json',reconciliation)
    summary=dict(status='AUDIT_COMPLETED',prefix_first_differences=differences,
        exact_input_and_batch_order_all_330=True,exact_CPU_CUDA_Python_NumPy_RNG_all_330=True,
        exact_observable_dropout_masks_all_330=True,dropout_modules_in_first_batch=len(a[0]['dropouts']),
        first_batch_numeric_gradient_differences=changed.to_dict('records'),
        max_numeric_gradient_difference=float(grad.max_abs_difference.max()),
        strict_determinism_status=strict['status'],strict_rejected_phase=strict.get('phase'),
        historical_exact_CUDA_kernel_not_identifiable=True,
        historical_Dropout_state_not_retroactively_verified=True,
        no_optimizer_updates_for_numeric_backward_pair=True,source_and_legacy_files_unchanged=True)
    save(OUT/'PROBE_SUMMARY.json',summary)
    lookup={r['field']:r for r in differences}
    metrics=evidence['metrics'];stats=evidence['rmse_summary']
    rows=''.join(f"| {i+1} | {r['RMSE_mV']:.6f} | {r['MAE_mV']:.6f} | {r['R2']:.9f} | {r['best_epoch']} | {r['stop_epoch']} | {r['optimizer_updates']} |\n" for i,r in enumerate(metrics))
    gd='\n'.join(f"- `{r.parameter}`：{r.numerically_unequal}个非零数值差，最大绝对差{r.max_abs_difference:.12g}。" for r in changed.itertuples())
    olddiff=pd.read_csv(OUT/'saved_evidence/EPOCH_NUMERICAL_DIFFERENCES.csv',float_precision='round_trip')
    fifth=olddiff.loc[(olddiff.epoch==5)&(olddiff.field=='validation_standardized_MSE')].iloc[0]
    text=f'''# Seed202 / Top10 重复性专项检查

结论：两次完整训练的初始化、数据、预处理和批次顺序相同；不是用了不同权重热启动，也没有发现Early Stopping错误。旧日志中第5轮开始出现数值差异，第8轮开始影响最佳轮次判断。最小诊断在完全相同的随机状态和Dropout有效掩码下复现了反向梯度差异，随后才出现参数分歧。这支持CUDA反向数值非确定性经迭代放大的解释，不能承诺当前正式程序同种子逐位一致。

不能唯一确定原两次完整训练的具体CUDA内核：旧实验未保存逐批RNG、完整梯度、每步参数和cuBLAS工作区环境。因此“具体历史算子已查明”“历史Dropout掩码已逐位追溯”均不能宣称。当前严格确定性诊断暴露的CuBLAS配置缺口，也不能单独证明它就是旧第5轮的唯一来源。

## 两次完整运行的真实波动

| 运行 | RMSE（mV） | MAE（mV） | R²（ΔV） | 最佳Epoch | 停止Epoch | 更新次数 |
|---|---:|---:|---:|---:|---:|---:|
{rows}
RMSE极差{stats['range_mV']:.6f} mV，第二次比第一次高{stats['later_vs_first_percent']:.4f}%。两次均值{stats['mean_mV']:.6f} mV，样本标准差（ddof=1）{stats['sample_sd_mV']:.6f} mV。只有2次完整重复，不是置信区间、无偏总体估计或显著性证据。

两次在同一12648个查询上的预测差RMS为{evidence['repeated_run_prediction_difference']['RMS_mV']:.6f} mV，最大绝对预测差{evidence['repeated_run_prediction_difference']['max_abs_mV']:.6f} mV。最终最佳参数来自不同Epoch，不能把其参数差当作同一步的局部误差。两次后留结果全部保留，不挑低RMSE的一次替换另一份。

## 旧证据逐项检查

- 原源码SHA、数据SHA、48→24、14通道、7:1:2边界、Top10掩码、超参数、Train scaler、标准化目标尺度一致；窗口和实际预处理输入身份一致。
- 初始化SHA及首次AdamW更新后的SHA均一致；共同{evidence['common_epochs']}轮的完整采样顺序哈希一致。未加载旧权重，优化器初始为空。
- 前4轮的损失与梯度范数统计严格一致。第5轮Validation标准化MSE差为{fifth.difference:.17g}；第6轮最大梯度范数也开始不同。第8轮最佳Epoch判断首次不同。
- 独立逐轮复算Validation改善阈值1e-12及patience40，两次均准确执行；首次157→197，第二次122→162。Early Stopping放大了轨迹差异，但不是起始分歧来源。
- 日历时间、运行范围和物理样本身份未移动。两份协议的请求种子列表、UTC时间及约1e-16量级的描述性KMeans中心差异，与训练输入无关。
- 源码明确设置`cudnn.deterministic=True`、`cudnn.benchmark=False`，但同时`use_deterministic_algorithms(False)`；仅前两项不等于整个CUDA计算图确定性。

## 最小范围诊断：不评价后留区间

预先固定两次独立进程、各6轮的Seed202/Top10前缀；所有结构、超参数、Train/Validation数据和采样方式保持正式程序设置。前缀结束是诊断预算截断，不是假装完成了450轮/40轮耐心的正式训练；不产生新的后留成绩，不用于选模型。只新建诊断目录，660次真实更新。

两次共330个批次：输入、目标、采样顺序、前后CPU/CUDA/Python/NumPy随机状态全部相同，所有Dropout层在非零输入位置的可观测有效掩码相同。零输入上潜在随机掩码不可观察，不作额外断言。

- 梯度哈希首次不同：第{lookup['gradients_before_clip']['first_step']}批。
- 更新后参数哈希首次不同：第{lookup['weights_after']['first_step']}批（Epoch {lookup['weights_after']['first_epoch']}，batch {lookup['weights_after']['first_batch']}）。
- 前向输出哈希首次不同：第{lookup['output_sha']['first_step']}批。
- 标量Loss首次不同：第{lookup['loss']['first_step']}批。

微小梯度差最初可能被参数更新的浮点舍入掩盖，因此“首次更新参数相同”不能证明梯度完全相同。AdamW动量状态也可能先于参数出现分歧；本轮未逐批保存完整优化器状态，不能把参数哈希相同解释成全部训练状态相同。诊断插入了只读hook、哈希和同步，改变执行时序；首次分歧批次不能机械等同于旧完整运行的第5轮。

为排除只有正负零或哈希差异，又固定补充两次首批次forward/backward，**优化器更新0次**。两次初始化、输入、前向输出及随机状态相同，捕获真实非零梯度差：

{gd}

最大{summary['max_numeric_gradient_difference']:.12g}，属于很小的反向数值差异；不能直接把这个幅度等同于最终RMSE差。它为“相同随机数仍可出现数值分歧”提供了直接证据。

## CUDA、Dropout与严格确定性

当前实际环境为PyTorch2.6.0+cu124、CUDA12.4、cuDNN90100、RTX3070 Laptop、驱动552.22。当前进程的CUBLAS_WORKSPACE_CONFIG未设置；旧实验没有记录该变量，历史值标为未知。TF32设置保持原程序值，不能仅凭启用cuDNN TF32就认定是非确定性的根因。

单批次严格诊断只将诊断进程的`use_deterministic_algorithms`设为True，未修改正式程序；在encoder VSN的变量预缩放Linear前向阶段即抛出CuBLAS确定性配置错误，更新次数0。说明当前环境不能只靠翻转一个布尔开关宣称确定性已完成；本轮没有修改永久环境变量，也没有用新配置替换旧结果。

PyTorch2.6官方文档说明，随机种子和cuDNN确定性标志并不覆盖全部算子；严格模式会对部分未满足条件的CuBLAS计算报错，CUDA≥10.2需要按文档在CUDA初始化前设置适当工作区配置。该资料用于解释资格边界，不代替本地实测：[重复性说明](https://docs.pytorch.org/docs/2.6/notes/randomness.html)、[确定性算子接口](https://docs.pytorch.org/docs/2.6/generated/torch.use_deterministic_algorithms.html)。

## 可信的后续记录方案（本轮不自动执行新完整训练）

1. 每次独立运行绑定`run_id + seed + repeat_id + feature_mode`，训练前固定全部重复预算；同种子重复与不同种子分别汇总，保留全部结果，不选最好的一次。
2. 保存数据/源码/掩码/样本/scaler哈希、依赖版本、GPU/驱动/cuDNN、确定性与TF32标志、环境变量及启动命令。缺失信息标NA，不能事后补成已记录。
3. 初始化及首次更新存参数哈希；按批次记录输入/采样身份、RNG前后状态摘要、梯度范数/摘要、参数更新摘要及优化器状态摘要。为定位数值差异，预先固定保存初始与少量里程碑的完整模型、优化器及CPU/CUDA/Python/NumPy RNG状态；这些是审计快照，不用于违反从头训练规则的热启动。
4. 每轮保存全部Train/Validation损失、改善规则、最佳Epoch、stale和更新次数，最佳checkpoint与Validation决策一一绑定；全部权重冻结后再评价后留区间。
5. 如以后要求更强的逐位重复性，先独立登记运行环境协议：进程启动时设置合法CUBLAS_WORKSPACE_CONFIG，并启用严格确定性，遇到不支持算子即报错；先做短前缀配对验证，再决定是否批准完整重复。不得暗改当前程序配置或把新运行替换历史结果。跨版本/硬件仍不保证逐位一致。
6. 当前两次完整结果报告均值、样本SD和范围；更可靠的稳定性评价须预先固定更多重复（例如至少3次），本轮不自动追加，不按后留成绩调整Top10/Top11或超参数。

机器可读字段模板见`EXPERIMENT_RECORD_SCHEMA.json`。这里只交付记录方案，不宣称正式主程序已自动保存了全部RNG/中间梯度。

## 完整性与失败证据

两次前缀和严格诊断都已落盘；控制台转发严格错误文本时发生GBK/UTF-8 UnicodeEncodeError，控制器原FAILURE_PRESERVED凭据原样保留。仅修复诊断控制器终端UTF-8转发，未重跑完成的训练。`RECONCILIATION.json`基于已存在的COMPLETE、STRICT_RESULT及历史哈希重新核验，没有把原错误改写为成功。

本轮正式主程序、模型结构、Top10/Top11、数据划分、超参数、旧训练结果和权重全部未修改；{len(protection)}个受保护文件SHA256保持不变。全部新增证据位于本目录，不归档或删除旧成果。
'''
    (HERE/'README.md').write_text(text,encoding='utf-8',newline='\n')
    schema=dict(status='PROPOSED_RECORDING_SCHEMA_NOT_YET_INSTRUMENTED_IN_FORMAL_ENTRY',
        identity=['run_id','seed','repeat_id','feature_mode','source_data_sha256','source_code_sha256','protocol_sha256','window_identity_sha256','scaler_sha256'],
        runtime=['python_package_versions','torch_cuda_cudnn','GPU_driver','deterministic_algorithms','cudnn_deterministic','cudnn_benchmark','TF32_flags','CUBLAS_WORKSPACE_CONFIG','launch_command'],
        before_training=['initial_parameter_sha256','initial_optimizer_empty','all_RNG_states','sampler_generator_state'],
        per_batch=['indices_sha256','input_target_sha256','RNG_before_after_sha256','gradient_norm','gradient_sha256','weights_before_after_sha256','optimizer_state_sha256'],
        per_epoch=['train_loss','validation_loss','best_epoch','stale_epochs','optimizer_update_count','checkpoint_sha256'],
        predeclared_snapshots=['initial_full_model_optimizer_RNG','fixed_milestones_full_model_optimizer_RNG'],
        reporting=['all_repeats_kept','n','mean','sample_sd','range','no_best_repeat_selection','historically_exposed_holdout_label'])
    save(HERE/'EXPERIMENT_RECORD_SCHEMA.json',schema)
    print('AUDIT_REPORT_WRITTEN_ORIGINAL_MODEL_AND_RESULTS_UNCHANGED')


if __name__=='__main__':main()
