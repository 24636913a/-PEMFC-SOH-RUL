"""Analysis-only rendering and independent saved-cache checks."""
import json,sys,unittest
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from run_bridge_audit import P,ROOT,sha,js,cs,md,overlaps

def verify():
 b=pd.read_csv(P/'inputs/DEVELOPMENT_BIN_IDENTITY_AND_CONDITIONS.csv',float_precision='round_trip')
 records=0;keys=set()
 for file in ['USED_DYNAMIC_PREDICTIONS_AUDIT_ONLY.csv','F_ORIGINAL_USED_PREDICTIONS_AUDIT_ONLY.csv']:
  d=pd.read_csv(P/'outputs'/file,float_precision='round_trip');records+=len(d)
  o=d.origin_row.to_numpy(int);t=d.target_row.to_numpy(int)
  assert np.max(abs((b.cell_voltage_mean.to_numpy()[t].astype(np.float32)-b.cell_voltage_mean.to_numpy()[o].astype(np.float32))-d.y_true_delta_V.to_numpy()))<1e-7
  assert np.array_equal(b.bin_end_ophrs2_h.to_numpy()[o],d.origin_actual_h.to_numpy())
  assert np.array_equal(b.bin_end_ophrs2_h.to_numpy()[t],d.target_actual_h.to_numpy())
  assert np.allclose(d.origin_voltage_V+d.y_pred_delta_V,d.predicted_dynamic_voltage_V,rtol=0,atol=2e-16)
  assert not d.duplicated(['cache_source','origin_row','target_row','horizon_step']).any()
  keys.update(t)
 targets=b.iloc[sorted(keys)].copy();starts=pd.to_datetime(targets.first_datetime);ends=pd.to_datetime(targets.last_datetime)
 events=pd.read_csv(P/'outputs/ALL_REFERENCE_EVENT_MATCH_AND_REJECTIONS.csv')
 for _,r in events.loc[events.nominal_checkpoint_h>0].iterrows():
  n=sum((starts<=pd.Timestamp(r.reference_last_calendar))&(ends>=pd.Timestamp(r.reference_first_calendar)))
  assert n==r.overlapping_target_bins==0
 final=json.loads((P/'outputs/FINAL_DECISION.json').read_text())
 assert records==final['prediction_records_audited']==41568 and len(keys)==698
 assert final['SOH_mapping_fits']==0 and final['improvement_percent'] is None
 for config in ['FROZEN_PROTOCOL.json','F_CACHE_EXTENSION.json']:
  cfg=json.loads((P/'configs'/config).read_text());assert all(sha(ROOT/f)==h for f,h in cfg['source_sha256'].items())
 js('outputs/INDEPENDENT_VERIFICATION.json',dict(status='PASS',verified_prediction_records=records,distinct_target_bins=len(keys),
  reference_calendar_intersections_recomputed=6,reconstruction_and_identity_PASS=True,all_historical_sources_unchanged=True))

def render():
 final=json.loads((P/'outputs/FINAL_DECISION.json').read_text());cond=json.loads((P/'outputs/CONDITION_SUPPORT_SUMMARY.json').read_text())
 event=pd.read_csv(P/'outputs/ALL_REFERENCE_EVENT_MATCH_AND_REJECTIONS.csv');metrics=pd.read_csv(P/'outputs/ALL_DYNAMIC_CACHE_METRICS.csv')
 targets=pd.read_csv(P/'outputs/ALL_DISTINCT_TARGET_CONDITION_AUDIT.csv')
 plt.rcParams.update({'font.family':'Arial','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'savefig.dpi':300})
 def save(fig,name):
  fig.savefig(P/'figures'/f'{name}.png',bbox_inches='tight');fig.savefig(P/'figures'/f'{name}.pdf',bbox_inches='tight');plt.close(fig)
 fig,ax=plt.subplots(figsize=(9,3.3))
 for _,r in targets.iterrows():
  a=mdates.date2num(pd.Timestamp(r.first_datetime));z=mdates.date2num(pd.Timestamp(r.last_datetime))
  ax.plot([a,z],[0,0],color='#4C78A8',lw=5,solid_capstyle='butt')
 for _,r in event.loc[event.nominal_checkpoint_h>0].iterrows():
  x=mdates.date2num(pd.Timestamp(r.reference_first_calendar));y=mdates.date2num(pd.Timestamp(r.reference_last_calendar))
  ax.plot([x,y],[1,1],color='#F58518',lw=7,solid_capstyle='butt');ax.annotate(f'{int(r.nominal_checkpoint_h)}h\nnominal',(x,1),xytext=(0,12),textcoords='offset points',ha='center',fontsize=8)
 ax.set_yticks([0,1],['Saved F2 target bins','Polarization measurements']);ax.set_ylim(-.5,1.7);ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'));ax.set_xlabel('Actual calendar date in 2025');ax.set_title('Observed calendar intervals: zero overlap with reference events')
 fig.tight_layout();save(fig,'reference_and_prediction_calendar')
 fig,axs=plt.subplots(1,2,figsize=(8.2,3.2))
 axs[0].hist(targets.actual_current_density_A_cm2_mean,bins=24,color='#4C78A8',edgecolor='white');axs[0].axvline(1,color='#F58518',ls='--',label='Reference j*=1.0');axs[0].axvline(.789229989,color='#777777',ls=':',label='P1S j');axs[0].legend(fontsize=8);axs[0].set_xlabel('Target-bin mean current density (A/cm²)');axs[0].set_ylabel('Distinct bins')
 axs[1].bar(['All target bins','Near j*=1.0','Condition certified'],[698,148,0],color=['#4C78A8','#F58518','#777777']);axs[1].set_ylabel('Distinct bins');axs[1].tick_params(axis='x',labelsize=8)
 for i,n in enumerate([698,148,0]):axs[1].text(i,n+15,str(n),ha='center')
 axs[1].set_ylim(0,820);fig.tight_layout();save(fig,'condition_support')
 fig,ax=plt.subplots(figsize=(8,3.2));labels=[]
 for _,r in metrics.iterrows():
  src=r.cache_source;role='H Val' if 'Phase_H_' in src else ('Matched Val' if 'HealthAware' in src else ('F Assess' if 'assessment_' in src else 'F Val'))
  labels.append(f'{role}\nseed{int(r.seed)}')
 bars=ax.bar(np.arange(len(metrics)),metrics.RMSE_dynamic_mV,color=['#4C78A8']*3+['#F58518']*3+['#777777']*2)
 ax.set_xticks(np.arange(len(metrics)),labels,fontsize=8);ax.set_ylabel('Dynamic voltage RMSE (mV)');ax.set_title('Existing voltage error, separately audited; no SOH accuracy claim')
 for bar,n in zip(bars,metrics.RMSE_dynamic_mV):ax.text(bar.get_x()+bar.get_width()/2,n+.2,f'{n:.2f}',ha='center',fontsize=8)
 ax.set_ylim(0,max(metrics.RMSE_dynamic_mV)*1.18);fig.tight_layout();save(fig,'existing_dynamic_error')
 def table(frame):
  return '\n'.join(['| '+' | '.join(frame.columns)+' |','| '+' | '.join(['---']*len(frame.columns))+' |']+[
   '| '+' | '.join('NA' if pd.isna(v) else f'{v:.6f}' if isinstance(v,float) else str(v) for v in row)+' |' for row in frame.itertuples(index=False,name=None)])
 tab=table(event[['nominal_checkpoint_h','reference_SOH','nearest_gap_wallclock_h','overlapping_target_bins']])
 mt=table(metrics[['cache_source','seed','records','distinct_targets','RMSE_dynamic_mV']])
 texts={
 '00_通俗研究结论.md':f'''# 实际结论
 TFT动态预测目前不能被本数据证实为参考SOH预测。状态为 `{final['status']}`。
 已实际核对8份冻结缓存、41,568条预测、698个不同目标窗。六次极化测量与目标窗日历区间重合数为0；过去锚点结果发布时间、极化精确运行小时也未记录。参考SOH并非不存在，而是没有合法的共同未来参考事件。
 未拟合SOH映射，未生成新的SOH成绩；相对Persistence和P1S/M2的改善百分比为不可评价。148个目标窗电流接近参考电流，不代表温湿压及内部状态等价。33个P1S事件落入预测窗，也不等于两种电压测量定义相同。
 保留TFT作为动态电压预测研究基础；暂不作为已验证SOH预测基础。下一阶段先审查电化学物理资格，不能在不存在的SOH输出上加约束。
 ''',
 '01_TFT与SOH关系.md':'''# 输出和参考定义
 原F2用过去24h、48×14槽位，经固定Top10置零及历史EWMA，预测未来24个0.5h窗的ΔV。恢复动态电压使用未滤波起点电压 Vhat=Vorigin+ΔVhat，缓存ΔV单位已核验为V，不能重复乘尺度。
 参考SOH定义 Vpol(j*=1.0)/0.6989533305168152，P1S定义 VP1S/0.71383023635。两种分母及测量条件不同。禁止直接动态V除极化BoL。
 低频动态电压变化也混有负载、温湿压、恢复和边界效应。K/HA保留FORECAST_ONLY_REPRESENTATION、HEALTH_EVIDENCE_NOT_IDENTIFIABLE：共同工况支持及年龄效应可辨识未成立。这项负面证据没有改写。
 已有96h健康门控HEALTH_CONTEXT_NO_CLEAR_GAIN保持；本轮没有重启门控、改变双路径或生成连续SOH标签。
 ''',
 '02_时间对齐与工况支持.md':f'''# 实测匹配
 {tab}
 表中差距是最近缓存目标区间与极化区间之间的日历小时，仅为核对证据，绝不是未来运行lead。nominal数字只用于检查点身份。pretest_ophrs来自极化前durability记录，不是极化测量运行小时。BoL仅为归一化原点。
 Development边界采用真实运行小时<1819.997738；仅读取混合N1前3005记录，不读Test，不重解析Raw。完整日历区间、目标行ID、起点电压和历史条件见逐点核对表；未来实际工况字段明确标为target_audit_only，不进入任何拟合。
 698个不同目标窗中148个平均j在固定描述带[0.95,1.05]，物理认证数0；描述带不作为资格caliper。P1S电流固定0.789229989，与j*=1.0不同。入口温湿压/气体流量有历史统计，但膜水合、界面分压、准稳态和传感器校准不完整。整0.5h动态均值不等于P1S middle-dwell子集。
 P1S actual结束运行小时与保守可获得记录分别保留；未直接保存actual结束日历时刻就记缺失，不按秒差补造。冻结完整周期身份不证明预测起点已有在线周期分类。
 原六缓存及新增原F缓存分别有冻结身份；F扩展只补齐已暴露Development Assessment覆盖，没有调整任何匹配规则或模型选择，也没有新Assessment推理。
 ''',
 '03_映射方法与误差分析.md':f'''# 资格在先
 先冻结事件区间、时间可获得性、工况与测量等价、评价单位，再运行核对。原F额外两缓存扩展沿用同规则。所有共同合法SOH事件数0，因此条件性past-only affine映射没有进入拟合阶段。
 下表只核验已有动态电压预测误差，绝不是SOH误差。
 {mt}
 ΔV与N1 float32目标减起点逐条核对误差<1e-7V。起点/目标真实运行小时、日历区间及Vhat恢复独立核验。没有SOH映射残差就不能量化其误差，也不能把信息缺口归因于网络预测误差。
 输出SOH方法资格CSV保留NA，不用0表示不存在的准确率。重叠窗/不同seed的记录不当作独立健康测量。
 ''',
 '04_与P1S_M2方法比较.md':'''# 共同目标资格
 参考Persistence、过去P1S/M2、TFT动态映射均没有可评价共同未来参考事件；本轮三者SOH百分点MAE和改善比例都是NA。
 既有AHC chronological prefix缓存已保存用于审查，但其R_current是本检查点当前代理估计；未来检查点R不可提前给模型。全六锚点B2是回顾性校准，未使用作早期预测参数。PHI全六参数与单调轨迹也没有作为当前输入。
 既有长时SOH四路线只是nominal顺序回放，精确目标运行小时和实际发布时间未知；不复制其成绩作为本轮TFT公平排名。
 96h Ridge原H1三折P1S MAE改善10.641466%、H3改善10.420448%，仅为已有健康代理预测探索性成果，不换算为本轮SOH改善。
 33个P1S末测量落在已预测时间窗，只通过时间落窗，不通过测量算子等价；因此未计算具有误导性的“P1S预测精度”。
 ''',
 '05_是否具备SOH预测资格.md':'''# 科研判定
 TFT_TO_SOH_INSUFFICIENT_SUPPORT。主要失败是事件时间身份与参考条件/测量映射未闭合；真实参考标签有7个（包括BoL），不能说完全没有SOH参考。
 无合法事件意味着不能检验是否超过简单基线，也不能认定“动态预测误差过大”或“映射误差过大”；这两种性能原因仍未知。没有否定TFT理论上能够辅助健康建模。
 最有用补证：同步保存参考测试精确durability时钟、完整测试区间与结果发布；提供预测目标覆盖的标准状态测量与工况计划；建立整窗动态电压到规定工况电压的可核验观测模型及独立参考事件。极化参考本身是参考性能保持率，也未独立分离不可逆损伤和可逆恢复。
 ''',
 '06_后续物理约束接口.md':'''# 进入阶段B
 不在未建立的SOH输出上增加物理残差。保留原F2动态ΔV接口、起点V恢复、历史信息边界。
 下一阶段审查两种不同依据：Nernst状态相关热力学上界，以及包含活化/欧姆/传质损失的定量电化学幅值残差（训练期教师可能接口）。不重启旧非负电压或未认证极化方向约束。
 必须先核验状态与参数身份、真正Train覆盖、有效梯度及额外信息公平性。缺变量时物理损失为不可计算NA，而不是0；合成梯度不代表真实收益。阶段A独立提交后立即继续阶段B。
 '''}
 for name,text in texts.items():md(name,text)
 md('README.md','# TFT动态电压到参考SOH\n\n实际执行资格核对，未形成合法映射实验。先运行 scripts/run_bridge_audit.py freeze/run，再 F_cache_extension freeze/run；完成后用 deliver_bridge.py verify/render。已完成run拒绝重复。最终状态见outputs/FINAL_DECISION.json。')
 md('reports/README.md','报告位于阶段根目录00–06，实测证据及独立核验位于outputs。所有图为描述性核对，不含虚构SOH误差。')

if __name__=='__main__':{'verify':verify,'render':render}[sys.argv[1]]()
