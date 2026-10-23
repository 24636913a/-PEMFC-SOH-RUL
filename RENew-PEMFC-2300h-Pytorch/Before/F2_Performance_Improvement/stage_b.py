"""Six fresh B instances. Registration requires completed, independently audited A."""
from datetime import datetime,timezone
import math
import numpy as np
import pandas as pd
import frozen_f2 as f
import settings as s
from train import train_one
from selection import collect,qualify
from io_utils import js,csv,text

B_FILES=['residual_fusion.py','stage_b.py','tests_b.py']


def register(run):
    f.require(f.read_json(run/'A_RESULT.json')['status']=='A_COMPLETE','A_REQUIRED')
    f.require(f.read_json(run/'audits/A_INDEPENDENT_AUDIT.json')['status']=='PASS','A_INDEPENDENT_ACCEPTANCE_REQUIRED')
    a=f.read_json(run/'A_RESULT.json')
    f.require(a['B_motivation_qualified'],'B_MOTIVATION_NOT_QUALIFIED')
    tests=f.read_json(f.HERE/'preflight/B_CONTRACT_TESTS.json')
    f.require(tests['status']=='PASS','B_CONTRACT_TESTS_REQUIRED')
    base=a['selected_configuration'].copy()
    configs=dict(B_control=base,B_v2=dict(base,fusion='residual',residual_cap=.25,normalization_epsilon=1e-6))
    before={name:f.sha(f.HERE/name) for name in B_FILES}
    f.require(before==tests['source_sha256'],'B_TESTED_SOURCE_CHANGED')
    path=run/'B_REGISTRATION.json'
    expected=dict(A_result_sha256=f.sha(run/'A_RESULT.json'),A_audit_sha256=f.sha(run/'audits/A_INDEPENDENT_AUDIT.json'),
        configurations=configs,seeds=list(s.INITIAL_SEEDS),source_sha256=before,
        formula='H = HL + 0.25*(1-g)*RMS(HL)/max(RMS(HC),1e-6)*HC',
        norm_scope='All 48 historical latent positions and 64 channels of this example; available by origin',
        gradient_through_normalizers=True,extra_trainable_parameters=0,parameters_each=332436,
        budget=6,loss='ordinary unweighted MSE',
        qualification='Frozen P0 generic gates against BOTH fresh B_control and original A0; CNN-off Validation RMSE worse in >=2/3 seeds; actual CNN gradients/parameter changes/forecast changes required',
        CNN_off_required_seeds=2,holdout_used_for_design_or_selection=False)
    if path.exists():
        old=f.read_json(path)
        f.require(all(old[key]==value for key,value in expected.items()),'B_REGISTRATION_CHANGED')
        return old
    result=dict(registered_UTC=datetime.now(timezone.utc).isoformat(),**expected)
    js(path,result)
    text(run/'B_预注册.md','''# 阶段B唯一融合候选（训练前登记）

A九次训练及独立审计完成后才登记本阶段。A1/A2没有稳定升级资格，因此采用A0学习率/Dropout；不是事后取平均值最好但资格未通过的配置。A门控/路径动机检查只允许测试，不代表已经证实原融合有害。

原式：H=g HL+(1-g) HC。

唯一新式：H=HL+0.25(1-g) RMS(HL)/max(RMS(HC),1e-6) HC。

RMS对每个样本已知的48步历史隐序列及64维计算，所有梯度均正常通过。CNN补充的RMS不超过HL的0.25倍；0.25与1e-6只有这一组固定值，不搜索。使用vector_norm避免零向量sqrt反向的未定义导数。

相同的历史5→8→1门控决定补充幅度；g在新式中不再是LSTM占比。LSTM主体系数为1，CNN是幅值有界的可学习方向补充。这里的“有界”是隐空间设计约束，不是电化学物理损失，也不是输出电压误差的上界。

仅覆盖fuse_encoder，其他原生TFT、VSN、LSTM encoder/decoder、CNN、Attention、decoder残差及24步头完全不变。无额外参数，两模型均332436；相同种子初始化的全部权重与state_dict键严格一致。不是把普通LSTM换名为F2。

Seed42/202/212，每个种子先新的原融合对照、再新的v2，各从头训练，450轮/patience40等政策与A完全相同；共6个新模型，不复用A权重。Validation逐模型保留全部5232查询。先满足P0三种子一致性、平均RMSE至少下降5%、MAE及尾部守门条件，且对本轮A0也有一致改善；再要求关闭CNN补充的Validation RMSE在至少2/3种子恶化，以及CNN确有训练梯度、参数变化和预测影响。没有新增配方或额外重复次数来挽救失败候选。

普通CUDA仍不保证逐位重复。B_control是计划内的同配置重复，可以量化波动，不能替换A0来美化基准。只有B_v2是新候选；不能选表现偶然较好的对照重复作为“改进模型”。后留窗口、成绩均不参与本阶段。
''')
    return result


def run_stage(run,frame,windows,scaler,prepared):
    registered=register(run)
    if (run/'B_RESULT.json').exists():return f.read_json(run/'B_RESULT.json')
    done=[]
    for seed in s.INITIAL_SEEDS:
        for candidate,config in registered['configurations'].items():
            for name,digest in registered['source_sha256'].items():
                f.require(f.sha(f.HERE/name)==digest,'FROZEN_B_SOURCE_CHANGED:'+name)
            folder=run/f'models/B/{candidate}_seed{seed}'
            if folder.exists():
                f.require((folder/'EVALUATION_AUDIT.json').exists() and f.read_json(folder/'EVALUATION_AUDIT.json')['status']=='PASS','B_PARTIAL_REQUIRES_REVIEW')
                ledger=f.read_json(folder/'TRAINING_LEDGER.json')
                f.require(ledger['status']=='TRAINED' and ledger['configuration']==config,'B_RESUME_IDENTITY')
            else:ledger=train_one(run,'B',candidate,seed,config,frame,windows,scaler,prepared)
            done.append(ledger);js(run/'B_PROGRESS.json',dict(completed=len(done),maximum=6,models=done))
        folders=[run/f'models/B/{name}_seed{seed}' for name in registered['configurations']]
        initial=[f.read_json(p/'BEFORE_TRAINING.json')['initialization_sha256'] for p in folders]
        f.require(len(set(initial))==1,'B_INITIALIZATION_MISMATCH')
        histories=[f.read_csv(p/'training_history.csv') for p in folders];n=min(map(len,histories))
        f.require(np.array_equal(histories[0].sampler_order_sha256.iloc[:n],histories[1].sampler_order_sha256.iloc[:n]),'B_SAMPLER_MISMATCH')
    table=collect(run,'B');f.require(len(table)==6,'ALL_SIX_B_REQUIRED')
    c=table.loc[table.candidate=='B_v2'];control=table.loc[table.candidate=='B_control']
    original=f.read_csv(run/'tables/A_VALIDATION_PER_SEED.csv').query("candidate=='A0'")
    matched=qualify(c,control);against_A0=qualify(c,original)
    participation=[]
    for seed in s.INITIAL_SEEDS:
        folder=run/f'models/B/B_v2_seed{seed}'
        h=f.read_csv(folder/'training_history.csv');paths=f.read_csv(folder/'paths_VALIDATION.csv')
        cf=f.read_csv(folder/'COUNTERFACTUAL_VALIDATION.csv').query("mode=='cnn_off'").iloc[0]
        normal=float(c.loc[c.seed==seed,'RMSE_mV'].iloc[0])
        participation.append(dict(seed=seed,normal_RMSE_mV=normal,CNN_off_RMSE_mV=float(cf.RMSE_mV),
            off_worse=bool(cf.RMSE_mV>normal),prediction_change_RMS_mV=float(cf.prediction_change_RMS_mV),
            CNN_train_gradient_nonzero=bool(h.cnn_parameter_grad_norm.gt(0).any()),
            gate_train_gradient_nonzero=bool(h.gate_parameter_grad_norm.gt(0).any()),
            supplement_ratio_max=float(paths.cnn_to_lstm_contribution_ratio.max()),
            normalization_floor_windows=int(paths.cnn_rms.lt(1e-6).sum())))
    actual=all(row['CNN_train_gradient_nonzero'] and row['gate_train_gradient_nonzero'] and row['prediction_change_RMS_mV']>0 for row in participation)
    bound=all(row['supplement_ratio_max']<=.25+1e-6 for row in participation)
    wins=sum(row['off_worse'] for row in participation)
    qualified=matched['qualified'] and against_A0['qualified'] and actual and bound and wins>=2
    result=dict(status='B_COMPLETE',fresh_instances=6,qualified=bool(qualified),candidate='B_v2',
        configuration=registered['configurations']['B_v2'],matched_control_decision=matched,A0_reference_decision=against_A0,
        actual_CNN_and_gate_participation=actual,residual_bound_verified=bound,CNN_off_worse_seeds=wins,
        parameter_changes_require_independent_audit=True,holdout_metrics_used=False)
    csv(run/'tables/B_VALIDATION_PER_SEED.csv',table);csv(run/'tables/B_CNN_PARTICIPATION.csv',participation)
    means=[]
    for name,g in table.groupby('candidate'):
        means.append(dict(candidate=name,n=3,**{key+'_mean':math.fsum(g[key])/3 for key in ['RMSE_mV','MAE_mV','R2','P99_mV']},
            **{key+'_SD':float(g[key].std(ddof=1)) for key in ['RMSE_mV','MAE_mV','R2','P99_mV']}))
    csv(run/'tables/B_VALIDATION_MEAN_SD.csv',means);js(run/'B_RESULT.json',result)
    print('B_COMPLETE',result,flush=True)
    from report_b import write
    write(run,table,pd.DataFrame(means),result)
    return result
