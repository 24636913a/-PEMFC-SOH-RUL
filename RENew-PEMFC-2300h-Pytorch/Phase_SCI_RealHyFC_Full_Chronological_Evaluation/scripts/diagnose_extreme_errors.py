"""Post-hoc error explanation only: never changes windows, predictions or ranking.

Added after the inference protocol freeze. Selection is explicitly diagnostic:
ten largest absolute query errors per frozen model/seed in the late partition.
Target conditions are observed retrospectively, not forecast inputs.
"""
from common import *
def generate():
    protocol();src=rd(P/'outputs/audits/FULL_ROW_IDENTITY.csv');rows=[]
    fields=['bin_end_ophrs2_h','first_datetime','last_datetime','cell_voltage_mean','actual_current_A_mean','actual_current_A_std','actual_current_density_A_cm2_mean','zero_current_fraction','load_transition_count','temporal_segment_id','bin_valid_for_windows','gap_before_flag']
    for model in MODELS:
        for seed in SEEDS:
            f=rd(P/f'outputs/predictions/LATE_LIFE_{model}_seed{seed}.csv');f['signed_error_mV']=(f.y_pred_delta_V-f.y_true_delta_V)*1000;f['abs_error_mV']=abs(f.signed_error_mV)
            for r in f.nlargest(10,'abs_error_mV').to_dict('records'):
                for prefix,col in [('origin','origin_row'),('target','target_row')]:
                    raw=src.iloc[int(r[col])]
                    for field in fields:r[prefix+'_'+field]=raw[field]
                rows.append(r)
    f=pd.DataFrame(rows);cs('outputs/audits/POSTHOC_LATE_EXTREME_QUERIES.csv',f)
    top=f.loc[f.model=='F2'].sort_values('abs_error_mV',ascending=False).iloc[0]
    text=f'''# 后段极端误差：事后解释补充

本文件属于预测后描述诊断，不参与筛选、剔除、调参或模型排名。规则为每个模型/种子列出后段绝对误差最大的10条查询；完整60套原预测不变。未来实际工况仅在这里解释，未输入模型。

F2最大误差出现在Seed{int(top.seed)}、起点行{int(top.origin_row)}（{top.origin_age_h:.1f}运行小时），目标行{int(top.target_row)}（{top.target_age_h:.1f}运行小时），第{int(top.horizon_step)}步。真实目标电压{top.true_raw_target_voltage_V:.9f} V，预测{top.predicted_voltage_V:.9f} V，绝对误差{top.abs_error_mV:.6f} mV。起点原始电压{top.origin_raw_voltage_V:.9f} V。

对应起点平均电流{top.origin_actual_current_A_mean:.6f} A，目标平均电流{top.target_actual_current_A_mean:.6f} A，两行zero_current_fraction分别{top.origin_zero_current_fraction:.6f}、{top.target_zero_current_fraction:.6f}；segment均为{int(top.origin_temporal_segment_id)}，两行均被历史规则标为有效。全部合法窗口的时钟连续性另由LEGAL_CLOCK_CONTINUITY.csv审计，跨行缺口标志为0，最大相邻原始采样间隔1.001秒。不能把这一大误差归为非法窗口并删除。

该例显示模型预测明显偏低而真实电压保持较高。工况变化及未来负载未知是解释预测困难的可能因素，但本诊断不能单独证明误差的物理或架构因果来源。不能将此误差当作不可逆退化或寿命标签。本轮保留这些失败案例，F2后段不占优的结论不作优化。

完整起点/目标时间、真实工况、各模型及种子极端查询见outputs/audits/POSTHOC_LATE_EXTREME_QUERIES.csv。
'''
    write('reports/07_后段极端误差事后诊断.md',text)
    js('outputs/audits/POSTHOC_DIAGNOSTIC_IDENTITY.json',dict(status='POSTHOC_DESCRIPTIVE_ONLY',script_sha256=sha(Path(__file__)),protocol_sha256=sha(P/'configs/FROZEN_CHRONOLOGICAL_PROTOCOL.yaml'),source_identity_sha256=sha(P/'outputs/audits/FULL_ROW_IDENTITY.csv'),selection='largest ten late absolute query errors per model and seed; no exclusion',model_or_window_changes=False,future_conditions='retrospective explanation only; never model inputs'))
if __name__=='__main__':generate()
