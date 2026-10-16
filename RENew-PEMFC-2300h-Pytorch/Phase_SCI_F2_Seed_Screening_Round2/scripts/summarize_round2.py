"""Reports use real completed caches only; selection is never unbiased performance."""
from screen_common import *
from reliable_io import js

def table(frame):
 if not len(frame):return 'NA：没有相应完整结果。'
 def fmt(v):
  if pd.isna(v):return 'NA'
  if isinstance(v,(float,np.floating)):return f'{v:.6f}'
  return str(v)
 return '| '+' | '.join(map(str,frame.columns))+' |\n| '+' | '.join(['---']*len(frame.columns))+' |\n'+'\n'.join('| '+' | '.join(fmt(v) for v in row)+' |' for row in frame.itertuples(index=False,name=None))

def report(name,text):
 (P/'reports'/name).write_text(text.strip()+'\n',encoding='utf-8',newline='\n')

def main():
 c=cfg();state=json.loads((P/'outputs/execution_ledger/PROGRESS.json').read_text());assert state['status']!='RUNNING'
 rank=ranking();hist=rd(P/'outputs/HISTORICAL_THREE_CANDIDATES.csv');selected=rank[rank.qualified].sort_values('seed') if len(rank) else pd.DataFrame()
 rows=[]
 for s in SEEDS:
  row=dict(state['seeds'][str(s)]);row['qualified']=bool(row.get('qualified',False)) if row['status']=='COMPLETED' else None
  rows.append(row)
 cs('outputs/ALL_50_CANDIDATE_STATES.csv',pd.DataFrame(rows))
 hist['origin_round']='HISTORICAL_SELECTED';new=selected.copy();new['origin_round']='ROUND2_SELECTED'
 fields=['seed','origin_round','RMSE_mV','MAE_mV','R2','step1_RMSE_mV','endpoint_RMSE_mV','max_abs_error_mV','P95_mV','P99_mV','A_pass','B_pass','qualified','selection_status','best_epoch','checkpoint','checkpoint_sha256','predictions','prediction_sha256','protocol_sha256']
 combined=pd.concat([hist,new],ignore_index=True).reindex(columns=fields);cs('outputs/F2_SIX_CANDIDATE_SEEDS.csv',combined)
 horizon=rd(P/'outputs/ROUND2_HORIZON_INDEPENDENT.csv') if len(rank) else pd.DataFrame()
 historical_h=rd(P/'outputs/HISTORICAL_HORIZON_METRICS.csv')
 combined_h=pd.concat([historical_h,horizon[horizon.seed.isin(selected.seed)] if len(selected) else pd.DataFrame()],ignore_index=True)
 cs('outputs/SIX_CANDIDATE_HORIZON_METRICS.csv',combined_h)
 stats=[]
 for name,frame in [('ALL_COMPLETED_NEW_SEEDS',rank),('QUALIFIED_NEW_SEEDS_SELECTED',selected),('COMBINED_CANDIDATES_SELECTED',combined)]:
  for k in ['RMSE_mV','MAE_mV','R2','endpoint_RMSE_mV','max_abs_error_mV']:
   x=frame[k].to_numpy('float64') if len(frame) else np.array([])
   stats.append(dict(group=name,metric=k,n=len(x),mean=float(x.mean()) if len(x) else None,sample_std=float(x.std(ddof=1)) if len(x)>1 else None,median=float(np.median(x)) if len(x) else None,minimum=float(x.min()) if len(x) else None,maximum=float(x.max()) if len(x) else None))
 cs('outputs/FULL_AND_SELECTED_STATISTICS.csv',pd.DataFrame(stats))
 diag=[]
 for row in combined.to_dict('records'):
  h=combined_h[combined_h.seed.eq(row['seed'])];maxrow=h.loc[h.RMSE_mV.idxmax()]
  diag.append(dict(seed=row['seed'],step1_RMSE_mV=row['step1_RMSE_mV'],step24_RMSE_mV=row['endpoint_RMSE_mV'],step24_to_step1_ratio=row['endpoint_RMSE_mV']/row['step1_RMSE_mV'],maximum_horizon_RMSE_mV=maxrow.RMSE_mV,maximum_at_horizon_step=int(maxrow.horizon_step),max_abs_error_mV=row['max_abs_error_mV']))
 cs('outputs/CANDIDATE_HORIZON_DIAGNOSTICS.csv',pd.DataFrame(diag))
 tail=[]
 for row in rank.to_dict('records'):
  frame=rd(P/row['predictions']);e=(frame.y_pred_delta_V-frame.y_true_delta_V).to_numpy()*1000;at=int(np.argmax(abs(e)));r=frame.iloc[at];total=float(np.dot(e,e))
  tail.append(dict(seed=row['seed'],origin_row=int(r.origin_row),target_row=int(r.target_row),horizon_step=int(r.horizon_step),target_age_h=float(r.target_age_h),signed_error_mV=float(e[at]),max_abs_error_mV=float(abs(e[at])),worst_point_SSE_percent=float(100*e[at]**2/total),same_target_SSE_percent=float(100*np.dot(e[frame.target_row.eq(r.target_row)],e[frame.target_row.eq(r.target_row)])/total),endpoint_RMSE_mV=row['endpoint_RMSE_mV'],A_pass=row['A_pass'],B_pass=row['B_pass'],selection_status=row['selection_status']))
 if tail:cs('outputs/ALL_NEW_SEED_TAIL_DIAGNOSTICS.csv',pd.DataFrame(tail))
 failures=[r['seed'] for r in rows if r['status'] in ['FAILED','TIMEOUT']];notstarted=[r['seed'] for r in rows if r['status']=='NOT_STARTED'];attempted=len(SEEDS)-len(notstarted)
 a_count=int(rank.A_pass.sum()) if len(rank) else 0;both=len(selected);best=float(rank.RMSE_mV.min()) if len(rank) else None
 old_summary=json.loads((OLD/'outputs/FINAL_SUMMARY.json').read_text());old_progress=json.loads((OLD/'outputs/execution_ledger/PROGRESS_IO_REPAIR.json').read_text())
 h_metrics=rd(H/'outputs/metrics/H_VALIDATION_METRICS.csv');h_f2=h_metrics[h_metrics.model.eq('F2')]
 provenance=pd.DataFrame([
  dict(stage='Historical Phase H, F2 only',registered_seed_count=3,attempted_seed_count=len(h_f2),completed_seed_count=int(h_f2.status.eq('COMPLETED').sum()),failed_seed_count=0,registered_seeds='42;52;62',completed_training_time_s=float(h_f2.training_time_s.sum()),wall_budget_s=None,elapsed_wall_s=None,retained_candidates='42',source_sha256=sha(H/'outputs/metrics/H_VALIDATION_METRICS.csv')),
  dict(stage='Screening Round1',registered_seed_count=old_summary['registered'],attempted_seed_count=old_summary['registered']-len(old_summary['not_started']),completed_seed_count=old_summary['completed'],failed_seed_count=len(old_summary['failed']),registered_seeds=';'.join(map(str,json.loads((OLD/'configs/frozen_seed_screening.json').read_text())['seeds'])),completed_training_time_s=old_summary['training_time_sum_s'],wall_budget_s=28800,elapsed_wall_s=old_progress['elapsed_s'],retained_candidates='103;193',source_sha256=sha(OLD/'outputs/FINAL_SUMMARY.json')),
  dict(stage='Screening Round2',registered_seed_count=50,attempted_seed_count=attempted,completed_seed_count=len(rank),failed_seed_count=len(failures),registered_seeds='201..250',completed_training_time_s=float(rank.training_time_s.sum()) if len(rank) else 0,wall_budget_s=43200,elapsed_wall_s=state['elapsed_s'],retained_candidates=';'.join(map(str,selected.seed.tolist())) if both else '',source_sha256=sha(P/'configs/frozen_seed_screening.json'))])
 cs('outputs/SEARCH_BUDGET_PROVENANCE.csv',provenance)
 allsix=len(combined)==6 and bool(combined.qualified.all())
 summary=dict(status=state['status'],candidate_pool=SEEDS,attempted=attempted,completed=len(rank),A_pass=a_count,qualified_new_count=both,qualified_new_seeds=selected.seed.astype(int).tolist() if both else [],best_new_seed=int(rank.iloc[0].seed) if len(rank) else None,best_new_RMSE_mV=best,failed_or_timeout=failures,not_started=notstarted,combined_candidates=len(combined),six_candidates_all_strictly_qualified=allsix,elapsed_s=state['elapsed_s'],completed_training_time_s=float(rank.training_time_s.sum()) if len(rank) else 0,all_completed_RMSE_mean_mV=float(rank.RMSE_mV.mean()) if len(rank) else None,all_completed_RMSE_sample_std_mV=float(rank.RMSE_mV.std(ddof=1)) if len(rank)>1 else None,stop_reason=state.get('stop_reason','CANDIDATE_POOL_EXHAUSTED'),Test=False,Assessment=False,Validation_selected=True,thresholds=c['thresholds'],protocol_sha256=sha(P/'configs/frozen_seed_screening.json'))
 summary['best_qualified_new_seed']=int(selected.loc[selected.RMSE_mV.idxmin(),'seed']) if both else None
 summary['best_qualified_new_RMSE_mV']=float(selected.RMSE_mV.min()) if both else None
 js('outputs/FINAL_SUMMARY.json',summary)
 small=['seed','RMSE_mV','MAE_mV','R2','endpoint_RMSE_mV','max_abs_error_mV','selection_status']
 budget=f"原注册池201～250共50个；实际启动{attempted}个，完整完成{len(rank)}个，失败/超时{failures}，未启动{len(notstarted)}个。A级{a_count}个，A级+B级{both}个。停止原因：{summary['stop_reason']}。原12h墙钟已用{state['elapsed_s']/3600:.3f}h；完整拟合累计{summary['completed_training_time_s']/3600:.3f}h；若有失败训练，其耗时另见日志，未混入完整拟合均值。每次启动前保留3420秒覆盖继承3300秒watchdog及120秒收尾。"
 report('01_第二轮训练完成情况.md',f"# {state['status']}\n\n{budget}\n\n仅Seed改变。48×14历史、24步ΔV、0.5h、固定Top10、因果EWMAα0.3、Train-only scaler、原MSE/AdamW、batch32、450epoch/patience40/clip1完全继承。读取一次前3007行再过滤得到3005条；1380Train、205Validation窗口/4920目标与第一轮全部张量逐位一致。原第一轮全部文件与本轮开始时1087份历史checkpoint保护检查见audits。没有Test/Assessment。\n\n复用稳定v3日志修复，并将有界原子替换重试用于新阶段checkpoint，不改变数值计算。原种子不重训，失败不隐瞒、不进入正式合格名单。临时防睡眠退出恢复，不修改电源计划/代理。\n\n见FINAL_SUMMARY、ALL_50_CANDIDATE_STATES、INDEPENDENT_AUDIT及最终本地GIT_RECEIPT。所有选择基于Development Validation，不代表独立Test。")
 report('02_全部候选种子排名.md','# 全部完整新种子排名\n\n'+table(rank[small] if len(rank) else rank)+'\n\n排名仅用原pooled ΔV RMSE、同分按Seed升序。每个模型205起点×24目标，重叠预测按原4920个实例评价，不先平均预测。另保留target-time-equal指标。完整50个状态另见CSV，失败/未启动不填0、不参与平均。')
 report('03_新增优秀种子结果.md',f"# 运行前冻结的双门槛筛选\n\nA级RMSE≤6.30mV；B级同时要求12h终点RMSE≤10.0mV、最大单点绝对误差≤50.0mV；全部端点包含等号。规则未调整。新增合格{both}个。\n\n"+table(selected[small] if both else selected)+'\n\n门槛在阅读第一轮结果后提出，在本轮训练前固定；这属于进一步探索性Validation筛选，不是独立确认性实验。只满足A级的实例保留为LOW_RMSE_WITH_TAIL_RISK。B_pass列单独表示稳定性两条件，qualified列表示A与B全部通过。逐步误差及全部极端目标未删除。若不足3个，则按实际数量交付。')
 report('04_六个优秀候选种子汇总.md',f"# 历史与新增候选\n\n实际候选{len(combined)}个；形成6个且全部达到严格条件：{allsix}。历史42、103、193均使用既有权重只读GPU重放、逐点真值/标签/尺度及同一4920目标核验，不重新训练，不改旧结果。\n\n"+table(combined[small])+'\n\n相关F2候选池的完整搜索预算：\n\n'+table(provenance)+'\n\n此表追踪直接产生这些候选的Phase H及两轮种子池，不宣称覆盖项目早期Optuna、其他阶段或其他模型的全部研究成本。第一轮两次工程失败均保留，失败训练耗时不伪装成完整拟合。\n\nF2_SIX_CANDIDATE_SEEDS.csv文件名固定，实际行数如实体现。六个实例属于筛选后候选，不能用其均值冒充F2一般随机初始化性能。历史与本轮配置SHA不同用于区分执行身份，模型、数据、超参数和评价协议相同。历史权重保留原目录；新canonical预测仅增加时间/电压身份列。')
 report('05_误差与稳定性分析.md','# 全池与筛选后稳定性\n\n'+table(pd.DataFrame(stats))+'\n\n候选逐步诊断：\n\n'+table(pd.DataFrame(diag))+'\n\n全部新种子尾部诊断：\n\n'+table(pd.DataFrame(tail))+'\n\n第24步/第1步比值仅为描述，所有24步完整CSV和图同时保留。不要求误差单调增加，不强制所有seed收敛；最佳epoch450且尚在改善的模型如实记录，无追加epoch。R²按原ΔV定义，绝对电压误差与ΔV误差因同一起点相加而相同。阈值是实例筛选标准，不是失效阈值。不做将共同数据上的种子当独立电池的显著性检验。')
 report('06_下一步论文实验建议.md',f"# 仅整理论文，不自动训练\n\n可以整理可追溯的动态电压预测候选图、逐点误差、24步误差及选择过程；本轮新增{both}个严格合格实例，综合{len(combined)}个候选。论文必须明确经过Validation筛选，报告全部有效候选与计算预算，保留尾部失败。\n\n门控/双路径结构优势仍需同信息、同训练预算、同种子池与同选择策略的F0/F1/F2配对消融。不能只拿F2筛出的优质种子与未经筛选的其他模型平均值作主要比较，也不能只为对照挑选本轮F2优质seed。独立泛化结论需要规则锁定后获准的未参与选择评价；本轮不读取Test/Assessment，不扩充模型/超参数或SOH/RUL研究。六个候选的平均仅作为选择后描述。\n\n复现入口scripts/audit_round2.py、generate_figures.py；图表源CSV与SHA在outputs/figure_data及FIGURE_PROVENANCE.json。Git来源{BASE}；实际提交/备份状态以GIT_RECEIPT.local.json为准。")
 print(json.dumps(summary,ensure_ascii=False),flush=True)

if __name__=='__main__':main()
