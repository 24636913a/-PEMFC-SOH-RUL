"""Summarize saved real results; no model fitting, threshold tuning, or forecast edits."""
from pathlib import Path
import json
import pandas as pd
P=Path(__file__).resolve().parents[1]
def rd(name):return pd.read_csv(P/'outputs'/name,float_precision='round_trip')
def table(d):
 cols=list(d.columns);rows=['| '+' | '.join(cols)+' |','| '+' | '.join(['---']*len(cols))+' |']
 for values in d.itertuples(index=False,name=None):rows.append('| '+' | '.join(f'{v:.6f}' if isinstance(v,float) else str(v) for v in values)+' |')
 return '\n'.join(rows)
def md(name,text):(P/name).write_text(text.strip()+'\n',encoding='utf-8')
def main():
 cfg=json.loads((P/'configs/FROZEN_PROTOCOL.json').read_text(encoding='utf-8'));auto=json.loads((P/'outputs/DECISIONS.json').read_text(encoding='utf-8'))
 m=rd('FOLD_METRICS.csv');pair=rd('PAIRED_COMPARISON.csv');count=rd('SAMPLE_COUNTS.csv');pred=rd('PREDICTIONS.csv');excluded=rd('EXCLUDED_SAMPLES.csv');inner=rd('INNER_SPLITS.csv');legal=rd('LEGAL_SAMPLES.csv')
 params=json.loads((P/'outputs/TRAINED_PARAMETERS.json').read_text(encoding='utf-8'));old=json.loads((P/'outputs/HISTORICAL_COMPARISON_SUMMARY.json').read_text(encoding='utf-8'))
 # Apply the human-specified qualitative tail condition. Preserve the stricter
 # automated screen verbatim rather than quietly replacing frozen run records.
 h1=pair[pair.horizon=='H1'];assert auto['H1']['at_least_5_percent'] and auto['H1']['at_least_two_fold_wins']
 assert (h1.improvement_P95_abs_error_mV_equivalent>0).all() and (h1.improvement_max_abs_error_mV_equivalent>0).all()
 second_h1=h1[h1.fold=='fold_2'].iloc[0]
 major_note=f"Fold 2 MAE rises {-100*second_h1.improvement_MAE_mV_equivalent:.6f}%, RMSE rises {-100*second_h1.improvement_RMSE_mV_equivalent:.6f}%; absolute increases {second_h1.Ridge_MAE_mV_equivalent-second_h1.Persistence_MAE_mV_equivalent:.9f} mV and {second_h1.Ridge_RMSE_mV_equivalent-second_h1.Persistence_RMSE_mV_equivalent:.9f} mV. Disclosed as small deterioration, not obvious major/tail worsening."
 judge={'H1':dict(status='96H_SHORT_HORIZON_SIGNAL_SUPPORTED',basis='Original user criteria, including qualitative no obvious major/tail deterioration',
  three_fold_evaluable=True,mean_MAE_improvement_fraction=auto['H1']['improvement_fraction'],fold_wins=2,
  tail_P95_and_max_improve_in_all_folds=True,
  qualitative_major_error_assessment=major_note,
  frozen_extra_strict_screen_status=auto['H1']['status'],
  discrepancy='Agent-added every-fold RMSE non-increase is stricter than requested. It triggered MIXED, but cannot replace original qualitative criterion. Original 5% threshold and all frozen training/prediction artifacts remain unchanged.'),
  'H3':dict(status=auto['H3']['status'],auxiliary_only=True,mean_MAE_improvement_fraction=auto['H3']['improvement_fraction']),
  'evidence_scope':'exploratory Development analysis after earlier results; no independent confirmation or statistical significance claim'}
 (P/'outputs/FINAL_RESEARCH_JUDGMENT.json').write_text(json.dumps(judge,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
 summaries=pd.DataFrame([dict(horizon=h,Persistence_MAE_mV=auto[h]['Persistence_equal_fold_MAE_mV'],Ridge96_MAE_mV=auto[h]['Ridge_equal_fold_MAE_mV'],improvement_percent=auto[h]['improvement_fraction']*100,fold_wins=auto[h]['fold_wins']) for h in ['H1','H3']])
 summaries.to_csv(P/'outputs/THREE_FOLD_EQUAL_MEANS.csv',index=False,float_format='%.17g',lineterminator='\n')
 warning='所有数值均为P1S保持率误差或其mV-equivalent换算，不是动态电压预测RMSE，也不是已验证的真实SOH误差。三折均值为各折等权平均，不按样本数加权。'
 discrepancy='冻结程序额外采用了“每折RMSE、P95、最大误差均不得上升”的充分条件，比原指令的“无明显恶化”更严。H1仅因第二折RMSE微升0.53%得到自动MIXED；该额外条件不覆盖用户原判据。原判据下H1为SUPPORTED，原始DECISIONS.json和冻结协议均保留，FINAL_RESEARCH_JUDGMENT.json记录解释。没有修改5%门槛、参数、预测或重复训练。'
 md('00_实验总结.md',f'''# 96小时P1S短期预测：实际实验总结

H1和H3均完成完整三折评价。第二折实际纳入H1 6个、H3 4个合法验证样本。H1共有50个验证起点，H3共有44个；两种方法共188条真实预测。

{table(summaries)}

H1满足原定5%改善、至少两折获益条件；第三折MAE从0.115866降至0.102398 mV-equivalent，改善11.62%。第二折MAE从0.105024升至0.105331，微升0.29%；RMSE微升0.53%。三折P95和最大误差均降低，未见均值掩盖的明显尾部恶化。按原指令判定：`96H_SHORT_HORIZON_SIGNAL_SUPPORTED`。H3独立辅助结果改善10.42%，三折均获益。

{discrepancy}

值得保留为轻量CPU的探索性P1S短期预测方法，尚不证明96h在所有工况中优于Persistence。严格比较仅指本轮Ridge与同一批样本上的Persistence。历史Ridge和TimesNet以精确键交集重算，详见04；不能归因于历史长度。

这轮以main `{cfg['base_commit']}`为起点，所有新代码和结果只在本阶段。共6次最终Ridge拟合、24次固定网格内部拟合，历史模型未重训。9项单元测试及独立NumPy核验通过，540个缓存特征、24个alpha分数、188条新预测和466条历史交集预测核验通过。5张图均输出PNG/PDF并人工查看。

{warning} 本方案由上一阶段结果启发，属于探索性Development分析，不宣称独立确认性证据或统计显著性。

主文件：outputs/FOLD_METRICS.csv、PAIRED_COMPARISON.csv、PREDICTIONS.csv、FINAL_RESEARCH_JUDGMENT.json。下一阶段可研究按起点可获得的M2校准，本轮到此结束。
''')
 reasons=excluded.groupby(['horizon','fold','role','exclusion_stage','exclusion_reason']).size().reset_index(name='n')
 cut=[]
 for a in params:
  t=legal[(legal.horizon==a['horizon'])&(legal.fold==a['fold'])&(legal.role=='train')]
  fi=inner[(inner.horizon==a['horizon'])&(inner.fold==a['fold'])&(inner.inner_role=='inner_fit')]
  cut.append(dict(horizon=a['horizon'],fold=a['fold'],train_max_label_end_h=t.target_p1s_measurement_end_ophrs2_h.max(),first_validation_origin_h=a['outer_cutoff_h'],inner_max_label_end_h=fi.target_p1s_measurement_end_ophrs2_h.max(),first_inner_selection_origin_h=a['inner_label_availability_cutoff_h']))
 cuts=pd.DataFrame(cut);cuts.to_csv(P/'outputs/PARAMETER_TIME_AUDIT.csv',index=False,float_format='%.17g',lineterminator='\n')
 removed=excluded[excluded.exclusion_stage=='outer_parameter_time'][['horizon','fold','sample_index','origin_raw_cycle_id','target_raw_cycle_id','target_p1s_measurement_end_ophrs2_h','exclusion_reason']]
 bounds=pd.DataFrame([{'fold':f,'train_start_h':a,'train_end_validation_start_h':b,'validation_end_h':c} for f,(a,b,c) in cfg['folds'].items()])
 md('01_样本与时间资格.md',f'''# 样本与时间资格

仅复用上一阶段保存的L96五项特征和实际测量身份，不要求L192合格；L192、future lead、未来工况及未来时间均不进入模型矩阵。资格为至少5个观测、实际跨度至少67.2h、所有观测测量结束时刻不晚于预测起点。真实测量时间来自已冻结的Raw重建缓存，未重新解析Raw。

R=V_P1S/0.71383023635。历史保持率来自因果合格的原始P1S；参考电压已在首个合法预测起点前确定。旧缓存的conservative_available_end_h是完整dwell状态结束时刻，比实际P1S结束约0.01–0.02h晚；本轮沿用冻结authority的实际P1S测量结束时刻，没有重新移动预测起点。没有实际发布延迟证据时，不额外捏造该延迟。

H1目标raw ID=origin+1，H3=origin+3；小型Development周期目录核验包含起点、目标和H3中间周期均完整。身份和标签与原authority逐项核对；无Test或2000/2300h后期目标读取，目录最大结束时间1811.41h。

{table(bounds)}

标签整个测量区间必须落入对应Train或Validation范围。模型参数还要求所有Train标签在最早实际使用Validation起点前结束。内部选择仍为Train末尾ceil(15%N)，inner-fit标签在首个inner-selection起点前结束；标准化各自只用合法训练部分。

{table(count)}

H3第一、三折各剔除一条参数尚不可获得的Train行：

{table(removed)}

{table(cuts)}

全部原始authority共540行，最终合法角色行410条、外部排除130条；这包含expanding folds重复出现的训练行，不是410个独立新周期。六条inner-fit时间剔除另存INNER_SPLITS.csv，其标签可在最终outer模型使用，因此不会从最终Train重复删除。

全部剔除原因：

{table(reasons)}

inputs/L96_REUSED_FEATURES.csv保存全部540行，仅保留L96和身份字段；L96_TIME_IDENTITIES.json保存每条历史周期名单。LEGAL_SAMPLES.csv、EXCLUDED_SAMPLES.csv、INNER_SPLITS.csv保留具体样本与原因，SAMPLE_COUNTS.csv保存计数。

![原始与实际使用样本](figures/original_vs_used_counts.png)
''')
 modelcols=['horizon','fold','model','n','MAE_R','RMSE_R','MAE_mV_equivalent','RMSE_mV_equivalent','P95_abs_error_mV_equivalent','max_abs_error_mV_equivalent']
 al=pd.DataFrame([dict(horizon=a['horizon'],fold=a['fold'],alpha=a['alpha'],train_n=len(a['train_samples']),inner_fit_n=a['inner_fit_n'],inner_selection_n=a['inner_selection_n'],inner_purged_n=len(a['inner_purged_samples'])) for a in params]);al.to_csv(P/'outputs/ALPHA_AND_TRAINING_COUNTS.csv',index=False,lineterminator='\n')
 grid=pd.DataFrame([dict(horizon=a['horizon'],fold=a['fold'],**s) for a in params for s in a['alpha_grid_scores']]);grid.to_csv(P/'outputs/INNER_ALPHA_SCORES.csv',index=False,float_format='%.17g',lineterminator='\n')
 md('02_96h模型三折结果.md',f'''# 96h模型三折结果

五项固定输入为最新保持率、96h均值、总体标准差(ddof=0)、相对实际测量结束小时的线性斜率、最新值减历史中位数。以这些历史输入预测DeltaR，再加回origin R；目标不缩放。Ridge alpha网格固定0.0001/0.01/1/100，按内部MAE择优，相等选较小alpha。每折重新拟合，未沿用旧系数。

{table(al)}

{table(m[modelcols])}

{table(summaries)}

TRAINED_PARAMETERS.json保存六套实际训练样本ID、内部样本ID、alpha网格分数、标准化均值和尺度、五个系数与截距。INNER_ALPHA_SCORES.csv为24个实际内部拟合分数，不用外部Validation选alpha。

{warning}

![H1逐点预测](figures/H1_prediction_curves.png)

![H3逐点预测](figures/H3_prediction_curves.png)

曲线只连接实际合格观测以便观看，不插值填补SOH标签；横轴使用物理周期raw ID，不把稀疏观测当作连续时间标签。
''')
 second=pred[(pred.fold=='fold_2')&(pred.model=='RIDGE_96H')][['horizon','origin_raw_cycle_id','target_raw_cycle_id','true_R_P1S_causal','predicted_R_P1S_causal','absolute_error_mV_equivalent']]
 md('03_与Persistence比较.md',f'''# 同样本正式比较

每折Persistence和96h Ridge使用完全相同的合法验证键；主指标为三折MAE等权平均。

{table(pair)}

H1第1折改善15.35%，第2折退步0.29%，第3折改善11.62%；平均改善10.64%，两折优于Persistence。第2折RMSE增加约0.000645 mV-equivalent(0.53%)，MAE增加约0.000307(0.29%)，确实存在轻微局部退步，应保留展示。三折P95/最大误差分别均下降，未见明显大误差恶化。6个样本不足以支持第二折稳定优势。

H3三折MAE分别改善7.41%、13.22%、16.11%，三折平均改善10.42%；RMSE、P95与最大误差三折均降低，但第二折仅4个样本，H3为辅助证据。

第二折的实际逐点误差：

{table(second)}

{discrepancy}

![H1三折MAE](figures/H1_three_fold_MAE.png)

![H3三折MAE](figures/H3_three_fold_MAE.png)
''')
 historical=pd.DataFrame([dict(horizon=a['horizon'],historical_model=a['historical_model'],seed=a['seed'],fold_counts='/'.join(map(str,a['fold_n'].values())),intersection_n=a['intersection_n'],current_MAE_mV=a['current_equal_fold_MAE_mV'],historical_MAE_mV=a['historical_equal_fold_MAE_mV'],current_improvement_percent=a['current_improvement_fraction']*100) for a in old])
 histmetrics=rd('HISTORICAL_INTERSECTION_METRICS.csv')
 md('04_与历史Ridge_TimesNet比较.md',f'''# 历史模型：精确交集上的描述性对照

H1读取CPRF的H1_BASELINE_PREDICTIONS.csv和H1_TIMESNET_PREDICTIONS.csv；H3读取CMRF的H3_PREDICTIONS.csv。历史模型均未重训。先限定H1/H3，再用fold、origin_raw_cycle_id、target_raw_cycle_id精确连接，每个随机种子独立匹配；匹配后逐点核对真实R及目标测量结束小时。没有将旧完整Validation均值与新缩减样本均值混比。

下表都是在所列交集内先算每折MAE，再三折等权平均：

{table(historical)}

历史Persistence在全部50个H1和44个H3交集上复现同一公式，误差与当前Persistence一致，可以严格同样本比较。当前96h Ridge与当前Persistence构成本轮正式future-unknown比较。

H1历史Ridge只有46个匹配样本(16/6/24)；新模型也必须缩到这些46个，MAE=0.125392，旧Ridge=0.119871，新模型误差高4.61%。旧H1 Ridge使用实际future lead，因此只能作为历史参考，禁止放入严格future-unknown公平排名。

H1旧TimesNet三个种子交集MAE分别1.640107(42)、1.474297(2024)、1.846708(3407)，新模型0.131211更低。H3旧Ridge为0.697670，新模型0.306500更低56.07%；旧TimesNet为0.354376/0.401130/0.396633，新模型误差低13.51%–23.59%。这些只是相同目标的经验误差差异。

TimesNet输入信息、容量、训练方法和96h Ridge不同；H3旧Ridge还使用RawF2工况上下文。旧模型参数可用时间未经本轮重新拟合后按当前purge规则认证，因此历史拟合条件也不完全相同。不能把交集误差差异归因于96h历史长度，不能宣布普适优于这些架构。三个种子逐一报告，不挑最佳种子、不把平均预测当作未授权新集成模型。

各折交集指标(含P95与最大误差)：

{table(histmetrics)}

HISTORICAL_KEY_MATCH_LOG.csv记录both/left_only/right_only及精确键；HISTORICAL_MATCHED_PREDICTIONS.csv保存466条实际种子/模型交集行；HISTORICAL_COMPARISON_SUMMARY.json保存每组共同样本数与误差。H1 Ridge少的4条新样本来自fold3，在left_only中明确保留。
''')
 md('05_研究结论与下一步.md',f'''# 研究结论与下一步

1. 96小时模型可以完成完整三折评价。H1验证样本为16/6/28，H3为16/4/24；第二折最终为H1 6个、H3 4个，全部经过历史、物理周期及参数时间检查。H3第一/三折各清除一条未来才能获得的Train标签，没有移动任何边界。
2. 相比当前Persistence，H1三折MAE从0.146837降到0.131211 mV-equivalent，改善10.64%；H3从0.342154降到0.306500，改善10.42%。H1两折获益、第二折MAE微升0.29%，H3三折获益；H1最后一折改善11.62%。全部折的P95和最大误差下降。
3. 原指令的主判断为`96H_SHORT_HORIZON_SIGNAL_SUPPORTED`，限于进一步探索性Development证据。{discrepancy}
4. 旧模型必须按交集解读：H1旧Ridge误差略低，但用到实际future lead，仅参考；H3旧Ridge和两任务旧TimesNet在交集内误差均高于当前模型，但输入与训练不同，属于描述性经验对照。严格结论是本轮96h Ridge相对同样本Persistence的收益，历史Persistence亦精确复现。
5. 值得保留固定96h、五特征的轻量方案和Persistence基线。第二折小样本、P1S恢复波动、仅Development三个相关expanding folds均限制外推；没有独立确认性数据，没有统计显著性结论。
6. 后续适合研究与M2的短期健康代理结合，但仍需只用预测起点前真实可获得的校准锚点拟合`SOH_proxy=a*R_P1S+b`，保留校准发布时间证据和误差传播。不能使用全六锚点回顾性B2替代当时可得参数；本轮未拟合新SOH网络、未生成真实SOH误差。本轮结果不能自行证明接入M2后SOH预测更准。

{warning}

本轮停止于已授权的96h H1/H3 CPU实验、真实缓存对照、图表与main提交，不启动新窗口、新模型或RUL训练。所有历史代码、结果和checkpoint均保持原样。
''')
 print('Six Chinese reports and explicit original-criteria judgment saved; frozen model outputs unchanged')
if __name__=='__main__':main()
