"""Apply the P0 rules after complete A/B acceptance. No training at import."""
from datetime import datetime,timezone
import pandas as pd
import frozen_f2 as f
import settings as s
from selection import qualify,choose
from io_utils import js,csv,text


def review_ab(run):
    for stage,expected in [('A',9),('B',6)]:
        audit=f.read_json(run/f'audits/{stage}_INDEPENDENT_AUDIT.json')
        result=f.read_json(run/f'{stage}_RESULT.json')
        f.require(audit['status']=='PASS' and audit['instances']==expected and result['status']==stage+'_COMPLETE','COMPLETE_AB_ACCEPTANCE_REQUIRED')
    a=f.read_csv(run/'tables/A_VALIDATION_PER_SEED.csv');b=f.read_csv(run/'tables/B_VALIDATION_PER_SEED.csv')
    original=a.loc[a.candidate=='A0'];control=b.loc[b.candidate=='B_control'];v2=b.loc[b.candidate=='B_v2']
    a_result=f.read_json(run/'A_RESULT.json');b_result=f.read_json(run/'B_RESULT.json')
    decisions={name:qualify(a.loc[a.candidate==name],original) for name in ['A1','A2']}
    f.require(decisions==a_result['decisions'],'A_RULE_RECALCULATION_CHANGED')
    matched=qualify(v2,control);against_original=qualify(v2,original)
    f.require(matched==b_result['matched_control_decision'] and against_original==b_result['A0_reference_decision'],'B_RULE_RECALCULATION_CHANGED')
    p=f.read_csv(run/'tables/B_CNN_PARTICIPATION.csv')
    changes=f.read_csv(run/'audits/B_PATH_PARAMETER_CHANGES.csv')
    relevant=changes.loc[(changes.candidate=='B_v2') & changes.group.isin(['CNN','LSTM_encoder','gate'])]
    f.require(len(relevant)==9,'B_THREE_SEEDS_THREE_PATHS')
    actual_parameters=bool(relevant.parameter_change_L2.gt(0).all())
    v2_qualified=bool(b_result['qualified'] and actual_parameters)
    names=[name for name,value in decisions.items() if value['qualified']]
    if v2_qualified:names.append('B_v2')
    table=pd.concat([a,b],ignore_index=True)
    selected=choose(table,names)
    configs={**s.CONFIGS_A,**f.read_json(run/'B_REGISTRATION.json')['configurations']}
    summary=dict(status='AB_ACCEPTED',reviewed_UTC=datetime.now(timezone.utc).isoformat(),
        A_result_sha256=f.sha(run/'A_RESULT.json'),B_result_sha256=f.sha(run/'B_RESULT.json'),
        A_audit_sha256=f.sha(run/'audits/A_INDEPENDENT_AUDIT.json'),B_audit_sha256=f.sha(run/'audits/B_INDEPENDENT_AUDIT.json'),
        eligible_candidates=names,selected_preC_candidate=selected,
        selected_preC_configuration=configs[selected] if selected else None,
        B_CNN_LSTM_gate_best_parameter_changes_nonzero=actual_parameters,
        B_v2_final_eligible=v2_qualified,
        candidate_table_is_Validation_only=True,controls_are_not_upgrade_candidates=True,
        holdout_metrics_used=False,flow_source_sha256=f.sha(f.HERE/'phase_flow.py'))
    path=run/'AB_REVIEW.json'
    if path.exists():
        existing=f.read_json(path)
        # A repeat read of an accepted decision cannot change its scientific content.
        f.require(all(existing[key]==value for key,value in summary.items() if key!='reviewed_UTC'),'AB_REVIEW_CHANGED')
        return existing
    csv(run/'tables/AB_QUALIFIED_CANDIDATE_POOL.csv',table.loc[table.candidate.isin(names)] if names else
        pd.DataFrame(columns=table.columns))
    js(path,summary)
    return summary


def stop_if_no_upgrade(run):
    review=review_ab(run)
    if review['eligible_candidates']:return None
    f.require(not (run/'models/C').exists() and not (run/'models/D').exists(),'UNQUALIFIED_EXTRA_TRAINING_EXISTS')
    decision=dict(status='NO_QUALIFIED_UPGRADE',frozen_UTC=datetime.now(timezone.utc).isoformat(),
        A_fresh_instances=9,B_fresh_instances=6,C_fresh_instances=0,D_fresh_instances=0,
        total_fresh_instances=15,C_status='NOT_TRIGGERED_NO_QUALIFIED_PREDECESSOR',
        D_status='NOT_TRIGGERED_NO_QUALIFIED_UPGRADE',selected_upgrade=None,
        new_holdout_evaluation=False,new_holdout_metrics=None,
        reason='P0 consistent paired Validation and tail guards not met; no forced winner and no rescue repeats',
        AB_review_sha256=f.sha(run/'AB_REVIEW.json'),selection_partitions=['TRAIN','VALIDATION'])
    target=run/'FINAL_DECISION.json'
    if target.exists():
        existing=f.read_json(target)
        f.require(all(existing[key]==value for key,value in decision.items() if key!='frozen_UTC'),'FINAL_DECISION_CHANGED')
        return existing
    js(target,decision)
    text(run/'03_阶段C_D_停止决定.md','''# 阶段C/D未触发，不能写成已执行

A九次、B六次真实训练均已完成并通过独立核验。按照训练前P0判据，没有候选取得稳定升级资格。因此没有合格的事件加权前置方案，阶段C不启动；没有唯一合格方案，阶段D六种子不启动。

这不是“训练失败”，也不是证明所有F2改进均不可能有效，而是本轮有限候选未达到预先定义的保守升级标准。不能为了凑齐六种子，在看到结果后放宽门槛、改残差系数、添加新损失配方或选择更好的对照重复。

本轮总计15次全新GPU训练；C=0、D=0。没有构造或计算新后留预测/误差，新六种子均值、标准差、后留曲线及是否超过LSTM的答案均为NA。旧F2与旧LSTM后留成绩只保留为历史背景，不能拿新的Validation误差替代后留成绩比较。

原生F2、旧LSTM结果及全部历史文件保持不变。真实负面结果、混合的验证收益、CNN梯度/预测影响及重复性波动均保留，已经完成本轮的停止规则；不自动转入下一阶段训练。
''')
    return decision
