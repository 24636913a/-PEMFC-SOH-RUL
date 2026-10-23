"""Report fixed matched data and, after all nine runs, real paired results."""
import sys,json
from pathlib import Path
import pandas as pd
import numpy as np
P=Path(__file__).resolve().parents[1]
MODELS=['F2_MATCHED','F2_CAPACITY_CONTROL','F2_P1S96_HEALTH_AWARE']
def table(d):
 rows=['| '+' | '.join(d.columns)+' |','| '+' | '.join(['---']*len(d.columns))+' |']
 for vals in d.itertuples(index=False,name=None):rows.append('| '+' | '.join(f'{v:.6f}' if isinstance(v,float) else str(v) for v in vals)+' |')
 return '\n'.join(rows)
def md(name,text):(P/name).write_bytes((text.strip()+'\n').encode('utf-8'))
def js(rel,data):(P/rel).write_bytes((json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False)+'\n').encode('utf-8'))
def preparation():
 q=json.loads((P/'outputs/CPU_QUALIFICATION.json').read_text());d=pd.read_csv(P/'inputs/P1S_TFT_ALIGNMENT.csv');cfg=json.loads((P/'configs/FROZEN_PROTOCOL.json').read_text());param=pd.read_csv(P/'outputs/MODEL_PARAMETER_COUNTS.csv').drop_duplicates('model')
 counts=[]
 for role in ['train','validation']:
  a=q[role];row=dict(role=role,original_origins=a['original_n'],qualified_origins=a['eligible_n'],excluded_origins=a['excluded_n'],distinct_target_rows=a['unique_target_rows'])
  for reason in ['COUNT_BELOW_5','SPAN_BELOW_67_2_H','REFERENCE_NOT_YET_AVAILABLE','TIME_IDENTITY_AMBIGUOUS']:row[reason]=a['reason_counts'].get(reason,0)
  counts.append(row)
 counts=pd.DataFrame(counts);counts.to_csv(P/'outputs/COVERAGE_COUNTS.csv',index=False,lineterminator='\n')
 coverage=pd.DataFrame([dict(role=r,origin_min_h=q[r]['origin_min_h'],origin_max_h=q[r]['origin_max_h'],target_min_h=q[r]['target_min_h'],target_max_h=q[r]['target_max_h'],latest_P1S_age_min_h=d[(d.role==r)&d.eligible].latest_P1S_age_h.min(),latest_P1S_age_median_h=d[(d.role==r)&d.eligible].latest_P1S_age_h.median(),latest_P1S_age_max_h=d[(d.role==r)&d.eligible].latest_P1S_age_h.max()) for r in ['train','validation']])
 coverage.to_csv(P/'outputs/ACTUAL_TIME_COVERAGE.csv',index=False,float_format='%.17g',lineterminator='\n')
 md('01_P1S与TFT时间对齐.md',f'''# P1S与TFT时间对齐

P1S身份来自冻结的131条Development实际测量缓存，不重新解析Raw。保持率R=V_P1S/0.71383023635，参考可获得时刻171.82091h；完整物理周期1–10仅作参考，输入测量从周期11开始。每条保留raw_cycle_id、middle_50测量起止时间、保守可获得时间、完整周期资格。

TFT起点取原始bin_end_ophrs2_h，不使用row index×0.5。测量的实际结束时刻须在[t-96,t]内，且conservative_available_end_h<=t；同一周期不重复计数。五项特征为最新R、均值、总体标准差、按实际测量结束小时的线性斜率、最新值减中位数。没有未来P1S补齐、全序列平滑或回顾性SOH映射。

{table(counts)}

{table(coverage)}

时间字段区分：给定缓存的保守P1S可获得时刻与完整周期终点不同，目录中的周期终点还晚约0.69–0.70h。本轮按用户指定的P1S保守可获得字段判断测量值已完成，完整周期身份沿用冻结事后目录，不声称完成在线周期完整性判定；没有把后续周期尾段的测量值输入模型。两种时刻逐条对照另存P1S_AVAILABILITY_VS_CYCLE_END.csv。

这里只统计同规则允许的预测起点，不按预测结果筛样本。P1S_TFT_ALIGNMENT.csv逐条保留全部1585个原起点的资格及原因、测量周期名单、最新测量距离起点的小时差；P1S_HISTORY_TIME_IDENTITIES.json保留真实测量结束/可获得时刻。

3007行限定读取后以年龄边界筛到3005行，构造Train [0,2103)和Validation [2103,2404)，复现1380/205原窗口。没有Assessment窗口或Test推理。匹配后Train末端标签1130.5h，首个Validation起点1324h，满足模型参数时间隔离。

![实际时间覆盖](figures/train_validation_coverage.png)
''')
 md('02_健康感知融合结构.md',f'''# 健康感知融合结构

只在本阶段继承原F2，直接只读导入原Canonical TFT、双路径和因果多尺度卷积类；未复制修改编码器forward。48×14历史输入、24步DeltaV输出、hidden64/continuous32、LSTM一层、四头注意力、原decoder和head保持不变。decoder仍只有确定性相对预测步，不加入未来实测工况或健康特征。

|模型|门控输入|MLP|用途|
|---|---|---|---|
|F2_MATCHED|历史工况5项|5→8→1|同样本、从头训练的原F2|
|F2_CAPACITY_CONTROL|历史工况5项+5维零|10→8→1|与健康模型等参数对照|
|F2_P1S96_HEALTH_AWARE|历史工况5项+历史健康5项|10→8→1|检查真实P1S信息贡献|

原工况按冻结的scaled/masked encoder槽位历史均值取得：load_transition_count、zero_current_fraction、actual_current_density_A_cm2_std、actual_current_A_std、actual_current_density_A_cm2_mean。ELU及sigmoid沿用原门控。融合仍为gate×LSTM+(1-gate)×Conv，标量扩展广播到编码器，不复制健康值到未来decoder。

{table(param[['model','total_trainable_parameters','gate_parameters','additional_parameters']])}

同Seed以新建原F2作为共同参数模板；B/C共用同一扩展门控，原5列权重、bias和output层沿用A，新5列从独立seed+100000初始化。原output零初始化使三组初始gate完全为0.5，CPU初始预测完全一致。模型构造后重置训练RNG，DataLoader单独按Seed设置，逐共同epoch核验采样顺序。

新增仅40个名义参数。B的新增列始终乘零，对数据损失的梯度为零，因此不会增加有效健康信息或有效表达能力；AdamW仍按同一协议处理全部参数。B与A后续可能因不同矩阵形状/CUDA非严格确定性产生数值训练差异，不能把这种差异自动解释成额外健康收益。C与B参数和初始化相同，只有真实P1S进入门控的有效信息不同。

Raw F2 scaler沿用原全部1380个合法未滤波Train窗口，和历史E/F hash逐项一致；健康scaler只在匹配后的870个Train健康向量拟合。Top10与逐窗口EWMA .3不变。原96h Ridge的H1/H3未来预测没有作为TFT输入。

![结构](figures/health_fusion_architecture.png)
''')
 md('03_数据覆盖与泄漏测试.md',f'''# 数据覆盖与泄漏测试

{table(counts)}

最低条件固定为Train>=100、Validation>=30、去重目标行>=24及时间隔离全部通过；本轮870/178/224满足。排除510个Train及27个Validation起点，不放宽96h、5观测或67.2h条件。所有A/B/C和Seed共用同一870/178身份及输入fingerprint。

实际执行15项单元测试全部通过：未来P1S扰动的特征/预测不变、未来目标与decoder输入隔离、观测不足、跨度不足、参考可获得性、保守结束时间、重复周期拒绝、三个Seed共享参数、三组初始gate/预测一致、B不接受有效健康信息、Train-only健康scaler、角色与标签时间隔离、原核心方法未覆盖、B新增列零数据梯度、Windows原子checkpoint保存重载。

GPU三模型均完成真实合成前向、反向、clip、AdamW step和权重落盘重载。GPU环境继承H normal_runtime；CUDA严格bitwise determinism不宣称。最终冻结协议SHA：`{(P/'configs/FROZEN_PROTOCOL.sha256').read_text().strip()}`。

合成GPU预检曾发现Windows对rb只读句柄执行fsync会报EBADF，正式运行前修复为rb+。原冻结协议、执行文件、失败日志及修复原因保存在configs/PREFLIGHT_REVISION_1。没有正式训练失败重试、没有通过Validation成绩调参；修复仅影响落盘句柄，不改模型、数据、5%判据或训练预算。

inputs/MATCHED_INPUTS.npz及INPUT_FINGERPRINTS.json绑定全部历史张量、健康张量、标签、raw row目标身份和实际时间；不对完整N1做哈希。正式workers只读本阶段冻结数组，仓库防火墙禁止历史写入及未允许的数据读取。
''')
def results(endpoint_assessment='unreviewed',endpoint_note=''):
 m=pd.read_csv(P/'outputs/ALL_RUN_METRICS.csv',float_precision='round_trip');assert len(m)==9
 cols=['RMSE_mV','MAE_mV','endpoint_12h_RMSE_mV','target_time_equal_RMSE_mV','P95_abs_error_mV','P99_abs_error_mV','max_abs_error_mV']
 mean=m.groupby('model')[cols].mean().reindex(MODELS);mean['RMSE_sample_SD_mV']=m.groupby('model').RMSE_mV.std(ddof=1).reindex(MODELS)
 mean.reset_index().to_csv(P/'outputs/THREE_SEED_MEANS.csv',index=False,float_format='%.17g',lineterminator='\n')
 pairs=[]
 for seed in [42,52,62]:
  d=m[m.seed==seed].set_index('model');row=dict(seed=seed)
  for i,g in enumerate(MODELS):
   for col in cols:row[f'{["A","B","C"][i]}_{col}']=float(d.loc[g,col])
  for label in ['A','B']:
   row[f'C_vs_{label}_RMSE_improvement_percent']=100*(1-row['C_RMSE_mV']/row[label+'_RMSE_mV'])
   row[f'C_vs_{label}_endpoint_improvement_percent']=100*(1-row['C_endpoint_12h_RMSE_mV']/row[label+'_endpoint_12h_RMSE_mV'])
  row['C_beats_both_RMSE']=row['C_RMSE_mV']<min(row['A_RMSE_mV'],row['B_RMSE_mV']);pairs.append(row)
 pair=pd.DataFrame(pairs);pair.to_csv(P/'outputs/PAIRED_SEED_COMPARISONS.csv',index=False,float_format='%.17g',lineterminator='\n')
 reductions={label:float(1-mean.loc[MODELS[2],'RMSE_mV']/mean.loc[MODELS[i],'RMSE_mV']) for i,label in enumerate(['A','B'])}
 primary=all(x>=.05 for x in reductions.values()) and int(pair.C_beats_both_RMSE.sum())>=2
 if primary and endpoint_assessment=='unreviewed':status='QUALITATIVE_ENDPOINT_REVIEW_PENDING'
 elif primary and endpoint_assessment=='pass':status='HEALTH_CONTEXT_ADDED_VALUE_SUPPORTED'
 elif any(x>0 for x in reductions.values()) or pair.C_beats_both_RMSE.any():status='HEALTH_CONTEXT_MIXED_SIGNAL'
 else:status='HEALTH_CONTEXT_NO_CLEAR_GAIN'
 decision=dict(status=status,mean_RMSE_reduction_vs_controls=reductions,paired_seed_wins_over_both=int(pair.C_beats_both_RMSE.sum()),
  fixed_5_percent_and_2_seeds_pass=bool(primary),endpoint_assessment=endpoint_assessment,endpoint_note=endpoint_note,
  endpoint_mean_improvement_vs_controls={label:float(1-mean.loc[MODELS[2],'endpoint_12h_RMSE_mV']/mean.loc[MODELS[i],'endpoint_12h_RMSE_mV']) for i,label in enumerate(['A','B'])},
  exploratory_Development_only=True,independent_Test_claim=False,no_statistical_significance_claim=True)
 js('outputs/RESEARCH_DECISION.json',decision)
 quick=m.pivot(index='model',columns='seed',values='RMSE_mV').reindex(MODELS).reset_index();quick.columns=['model','Seed42_RMSE_mV','Seed52_RMSE_mV','Seed62_RMSE_mV'];quick['mean_RMSE_mV']=mean.RMSE_mV.to_numpy();quick['sample_SD_mV']=mean.RMSE_sample_SD_mV.to_numpy()
 readable='健康上下文达到本轮探索性实用门槛，值得有条件保留。' if status=='HEALTH_CONTEXT_ADDED_VALUE_SUPPORTED' else '健康上下文只有部分或不稳定收益，不能升级为已证明有效的正式创新。' if status=='HEALTH_CONTEXT_MIXED_SIGNAL' else '健康上下文未超过两个正式对照，不支持将其升级为有效创新。' if status=='HEALTH_CONTEXT_NO_CLEAR_GAIN' else '主门槛通过，尚需完整评价末端与尾部误差后才能作最终判断。'
 note='所有本轮误差均为真实动态DeltaV电压预测mV，不是P1S mV-equivalent，也不是SOH/RUL误差。主指标按178×24个预测查询等权；三Seed均值为三次独立运行RMSE的算术平均。重叠查询与去重时间行不意味着统计独立。'
 md('04_九组正式训练结果.md',f'''# 九组正式训练结果

九组全部完成，A/B/C各Seed42/52/62均为从头训练，没有失败组反复重启或替换成绩。每组870个Train、178个Validation起点，4272个预测查询和224个去重目标时间行。

{table(quick)}

完整逐运行指标：

{table(m[['model','seed',*cols,'best_epoch','stop_epoch','training_seconds','peak_GPU_bytes']])}

三Seed平均与样本SD(ddof=1)：

{table(mean.reset_index())}

{note}

每组逐点CSV、24步长CSV、目标时刻SSE均值CSV、门控权重CSV、逐epoch日志及best checkpoint都在outputs。checkpoint保存协议SHA、输入fingerprint、初始模型身份、最佳epoch与固定Validation预测；正式worker重载同设备预测逐点一致。独立核验不使用新的训练。

![三模型主指标](figures/three_models_RMSE.png)

![全部步长](figures/all_horizon_errors.png)

![固定Seed42全起点末端预测](figures/three_model_predictions_seed42.png)

![逐轮训练](figures/training_curves.png)
''')
 md('05_消融对照与独立增益.md',f'''# 消融对照与独立增益

最终判断：`{status}`。{readable}

相对原F2_MATCHED的三Seed平均RMSE改善{100*reductions['A']:.4f}%；相对同参数零输入对照改善{100*reductions['B']:.4f}%。正数是改善，负数是恶化。配对Seed同时胜过两个对照：{int(pair.C_beats_both_RMSE.sum())}/3。预设要求相对两对照均至少5%，至少2/3配对Seed同时获益，不因结果修改。

{table(pair[['seed','A_RMSE_mV','B_RMSE_mV','C_RMSE_mV','C_vs_A_RMSE_improvement_percent','C_vs_B_RMSE_improvement_percent','C_beats_both_RMSE']])}

12h末端逐Seed完整比较：

{table(pair[['seed','A_endpoint_12h_RMSE_mV','B_endpoint_12h_RMSE_mV','C_endpoint_12h_RMSE_mV','C_vs_A_endpoint_improvement_percent','C_vs_B_endpoint_improvement_percent']])}

末端评价：{endpoint_assessment}。{endpoint_note}

目标时刻等权误差为每个target row先对其所有查询取误差平方均值，再跨224个目标行平均并开方；不是先集成预测，也没有根据模型误差移除任何目标。各Seed的TTE、P95/P99/最大误差原样公开：

{table(m[['model','seed','target_time_equal_RMSE_mV','P95_abs_error_mV','P99_abs_error_mV','max_abs_error_mV']])}

本轮实际表现：健康模型相对原F2的三个Seed均退步；相对参数对照虽然Seed42/52分别改善约4.58%/5.94%，Seed62却恶化约124.25%，三Seed平均仍更差。参数对照相对原F2均值低约5.43%，但只在Seed62优于原F2，不能归因于真实健康信息或稳定的容量收益。健康组RMSE样本SD为3.905145mV，约为两对照的两倍。

B/C具有相同参数数目及初始化，B新增健康输入固定为零，只有C有真实健康上下文。因此C相对B的配对差异是本轮识别健康信息价值的主要对照。B比A的变化不可直接归因于新增健康信息：B没有健康输入，且新增列没有数据梯度；不同形状的CUDA数值路径及正常训练非严格确定性可能带来训练差异。

原Phase H全量Validation的F2三SeedRMSE为6.265169/7.034632/8.436399，平均7.245400mV，只作历史附录；1380/205原窗口与本轮870/178不同，禁止混合排名或计算跨阶段改善。本轮不会修复或覆盖P2/P3工况门控独立稳定收益未获支持的结论。

![真实门控权重分布](figures/gate_weight_distributions.png)
''')
 md('06_下一阶段SOH与物理约束计划.md',f'''# 下一阶段SOH与物理约束计划

{readable} 当前决定为`{status}`，只适用于重复使用的Development Validation探索性分析；没有独立Test证据、统计显著性证明或电化学因果识别。未来工作不能据单个Seed或个别指标替换本轮主判据。

本轮仅改历史健康上下文门控，未加入M2、SOH预测head或物理loss，未完成SOH/RUL预测。可保留本阶段历史健康特征接口、门控输出和动态DeltaV输出，供下一阶段单独研究使用。

如本轮未满足全部门槛，保持原TFT与既有96h Ridge成果；不为追求联合模型反复改门控。若未来研究M2，必须依据起点前真实可获得的校准锚点建立SOH_proxy=a*R_P1S+b，禁止全六锚点回顾性B2冒充当时可得参数，分别评估P1S预测及校准误差传播。

物理约束另行验证适用工况、实际激活比例和训练梯度；已证明无激活的旧非负电压惩罚不能直接当成有效创新。本轮不启动上述后续研究。既有Ridge 10.64%/10.42%健康代理改善不变，不把它们与本轮动态电压RMSE直接比较。
''')
 md('00_最终实验总结.md',f'''# 96h健康感知TFT：最终实际结果

{readable} 最终状态：`{status}`。

1. 合法匹配870个Train、178个Validation起点，Validation覆盖224个去重目标时间行；原1380/205窗口严格复现。保守P1S可获得时刻、96h跨度和模型参数时间隔离均通过。
2. 健康模型和两个对照均实际完成三个Seed训练，共九次，无正式失败重试。H原450上限/pat40/AdamW/学习率/batch32及全部核心结构保持不变。
3. 原F2_MATCHED三Seed平均RMSE：{mean.loc[MODELS[0],'RMSE_mV']:.6f}mV。
4. 加入健康上下文后三Seed平均RMSE：{mean.loc[MODELS[2],'RMSE_mV']:.6f}mV；零输入同参数对照为{mean.loc[MODELS[1],'RMSE_mV']:.6f}mV。
5. 相对原F2改善{100*reductions['A']:.4f}%，相对参数对照改善{100*reductions['B']:.4f}%。
6. 相同B/C参数与初始化用于区分真实P1S信息贡献；不能把仅B/A数值差异说成健康收益。完整配对结果见05。
7. 同时优于两个对照的Seed为{int(pair.C_beats_both_RMSE.sum())}/3，逐Seed与末端真实数据均保留。末端评价：{endpoint_assessment}；{endpoint_note}
8. {readable}不宣称独立Test优越或统计显著性，历史P2/P3负面消融结论保留。
9. 后续M2/软物理需要分别认证，当前仅保留接口，本轮不加入、不自动扩展。

{table(quick)}

{note}

15项测试、GPU合成前后向/optimizer/保存重载预检及正式checkpoint重载通过。正式前Windows保存句柄修复留档，未改变模型或门槛。全部源文件与训练代码按最终冻结协议校验；真实预测/指标CSV由独立脚本复算，所有文件只在新阶段。

代码和输入身份：configs/FROZEN_PROTOCOL.json及SHA、inputs/INPUT_FINGERPRINTS.json；真实成绩：outputs/ALL_RUN_METRICS.csv、PAIRED_SEED_COMPARISONS.csv；checkpoint与训练记录：outputs/checkpoints和training_logs；判定：outputs/RESEARCH_DECISION.json。

![本轮三模型主指标](figures/three_models_RMSE.png)

本轮到此停止，不追加窗口、模型、Seed、M2训练、物理loss或RUL实验。历史目录只读，完成后仅提交本阶段并安全推送main。
''')
 print(json.dumps(decision,ensure_ascii=False,indent=2))
 print(quick.to_string(index=False));print(pair[['seed','A_endpoint_12h_RMSE_mV','B_endpoint_12h_RMSE_mV','C_endpoint_12h_RMSE_mV']].to_string(index=False))
if __name__=='__main__':
 preparation()
 if (P/'outputs/ALL_RUN_METRICS.csv').exists():results(sys.argv[1] if len(sys.argv)>1 else 'unreviewed',sys.argv[2] if len(sys.argv)>2 else '')
