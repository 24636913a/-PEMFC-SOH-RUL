"""Full-pool statistics and candid Chinese reports, never selects new seeds."""
from screen_common import *
def md(name,text):
 f=P/name;f.parent.mkdir(parents=True,exist_ok=True);f.write_bytes((text.strip()+'\n').encode('utf-8'))
def table(d,columns):
 q=d[columns].copy()
 for c in q.columns:
  if pd.api.types.is_float_dtype(q[c]):q[c]=q[c].map(lambda x:f'{x:.6f}')
 return '| '+' | '.join(q.columns)+' |\n| '+' | '.join(['---']*len(q.columns))+' |\n'+'\n'.join('| '+' | '.join(map(str,r))+' |' for r in q.itertuples(index=False,name=None))
def main():
 c=cfg();r=ranking();state=json.loads((P/'outputs/execution_ledger/PROGRESS_ENGINEERING_REPAIR.json').read_text());n=len(r);historical=json.loads((H/'outputs/H_FINAL_SUMMARY.json').read_text())
 statuses=pd.DataFrame(list(state['seeds'].values()));cs('outputs/ALL_20_SEED_STATES.csv',statuses)
 identities=[]
 for seed in r.seed if len(r) else []:
  initial=json.loads((P/f'outputs/training_logs/seed_{seed}/initial.json').read_text())
  identities.append(dict(seed=int(seed),base_commit=c['base_commit'],model='Frozen Phase H RegimeAwareDualPathTFT',parameter_count=332436,
   scientific_protocol_sha256=initial['protocol_sha256'],logger_amendment_sha256=sha(P/'configs/LOGGER_REPAIR_AMENDMENT.json'),actual_worker='scripts/train_seed_v2.py',
   actual_worker_sha256=sha(P/'scripts/train_seed_v2.py'),initialization_sha256=initial['initialization_sha256'],environment=json.dumps(initial['environment'],sort_keys=True)))
 cs('outputs/audits/RUN_IDENTITY_INDEX.csv',pd.DataFrame(identities))
 if not n:
  md('reports/01_通宵训练执行总结.md',f"# {state['status']}\n\n有效种子0；保留全部故障和未启动记录，禁止虚构排名/图表。\n\n{state.get('stop_reason','')}\n")
  js('outputs/FINAL_SUMMARY.json',dict(status=state['status'],completed=0));return
 cols=['RMSE_mV','MAE_mV','R2','endpoint_RMSE_mV','target_time_equal_RMSE_mV','P95_mV','P99_mV','max_abs_error_mV','training_time_s']
 convergence=r[['seed','best_epoch','stop_epoch','patience_early_stopped']].copy();convergence['reached450_epoch_budget']=convergence.stop_epoch.eq(450);convergence['best_at_final_epoch']=convergence.best_epoch.eq(convergence.stop_epoch)
 cs('outputs/CONVERGENCE_AND_BUDGET_DIAGNOSTICS.csv',convergence)
 tails=[];cache={}
 for row in r.itertuples():
  frame=rd(P/row.predictions);cache[int(row.seed)]=frame;err=frame.y_pred_delta_V-frame.y_true_delta_V;index=err.abs().idxmax();point=frame.loc[index];total=float(np.sum(err**2));same=frame.target_row.eq(point.target_row)
  tails.append(dict(seed=int(row.seed),origin_row=int(point.origin_row),target_row=int(point.target_row),horizon_step=int(point.horizon_step),target_age_h=float(point.target_age_h),
   max_abs_error_mV=float(1000*abs(err.loc[index])),single_worst_point_SSE_percent=float(100*err.loc[index]**2/total),all_origins_at_same_target_SSE_percent=float(100*np.sum(err[same]**2)/total),origins_covering_target=int(same.sum())))
 tails=pd.DataFrame(tails);cs('outputs/TAIL_CONCENTRATION_BY_SEED.csv',tails)
 global_tail=tails.loc[tails.max_abs_error_mV.idxmax()];key=['origin_row','target_row','horizon_step'];samekey=[]
 for seed,frame in cache.items():
  point=frame.loc[frame.origin_row.eq(global_tail.origin_row)&frame.target_row.eq(global_tail.target_row)&frame.horizon_step.eq(global_tail.horizon_step)];assert len(point)==1
  samekey.append(point.assign(signed_error_mV=1000*(point.y_pred_delta_V-point.y_true_delta_V)))
 cs('outputs/ALL_SEEDS_AT_GLOBAL_WORST_KEY.csv',pd.concat(samekey))
 stats=pd.DataFrame([dict(metric=k,n=n,mean=float(r[k].mean()),sample_std=float(r[k].std(ddof=1)) if n>1 else None,median=float(r[k].median()),minimum=float(r[k].min()),maximum=float(r[k].max())) for k in cols]);cs('outputs/F2_FULL_POOL_STATISTICS.csv',stats)
 top=r.head(5);best=r.iloc[0];horizon=rd(P/'outputs/F2_HORIZON_METRICS.csv');z=horizon.loc[horizon.seed.eq(int(best.seed))].sort_values('horizon_step')
 comparison=[]
 for old in historical['multi_seed_statistics']:comparison.append(dict(result_group=f"Historical {old['model']} all3seeds",n=3,RMSE_mV=old['RMSE_mean_mV'],sample_std_mV=old['RMSE_sample_std_mV'],selection='Frozen42/52/62, no new screening'))
 comparison.extend([dict(result_group='New F2 full valid pool',n=n,RMSE_mV=float(r.RMSE_mV.mean()),sample_std_mV=float(r.RMSE_mV.std(ddof=1)) if n>1 else None,selection='All completed predeclared candidates'),dict(result_group='New F2 selected best',n=1,RMSE_mV=float(best.RMSE_mV),sample_std_mV=None,selection='Selected on same Validation'),dict(result_group='New F2 selected Top5',n=len(top),RMSE_mV=float(top.RMSE_mV.mean()),sample_std_mV=float(top.RMSE_mV.std(ddof=1)) if len(top)>1 else None,selection='Selected on same Validation; optimistic')]);comparison=pd.DataFrame(comparison);cs('outputs/HISTORICAL_AND_NEW_COMPARISON.csv',comparison)
 number=int(best.seed);first=float(z.RMSE_mV.iloc[0]);last=float(z.RMSE_mV.iloc[-1]);std=float(r.RMSE_mV.std(ddof=1)) if n>1 else None
 summary=dict(status=state['status'],completed=n,registered=20,failed=statuses.loc[statuses.status.eq('FAILED'),'seed'].astype(int).tolist(),timeout=statuses.loc[statuses.status.eq('TIMEOUT'),'seed'].astype(int).tolist(),not_started=statuses.loc[statuses.status.eq('NOT_STARTED'),'seed'].astype(int).tolist(),
  top5=top.seed.astype(int).tolist(),best_seed=number,best_RMSE_mV=float(best.RMSE_mV),best_MAE_mV=float(best.MAE_mV),best_R2=float(best.R2),full_pool_RMSE_mean_mV=float(r.RMSE_mV.mean()),full_pool_RMSE_sample_std_mV=std,
  best_step1_RMSE_mV=first,best_step24_RMSE_mV=last,best_step24_to_step1_RMSE_ratio=last/first,top5_RMSE_mean_mV=float(top.RMSE_mV.mean()),training_time_sum_s=float(r.training_time_s.sum()),
  scientific_protocol_changed=False,logging_repair_disclosed=True,Validation_selected=True,independent_Test=False,new_Assessment=False,no_F0_F1_training=True,Test_values_read=False)
 js('outputs/FINAL_SUMMARY.json',summary)
 md('reports/01_通宵训练执行总结.md',f'''# {state['status']}

20个候选按原冻结顺序执行，有效完成{n}个；状态详见outputs/ALL_20_SEED_STATES.csv。失败{summary['failed']}，超时{summary['timeout']}，未启动{summary['not_started']}。已完成正式GPU拟合累计{summary['training_time_sum_s']/3600:.3f}小时。

本轮唯一科学变量是Seed。继承Phase H原生F2（332436参数）、48×14历史、24步ΔV、Top10固定mask、EWMAα0.3、Train-only scaler、AdamW及450epoch/patience40。原始N1只按nrows3007读取一次，再以<1819.997738筛出3005条；正式1380Train和205Validation窗口/4920目标与H精确一致。没有构造Assessment或Test窗口。

种子101在首批日志序列化时发生duplicate epoch keyword错误，尚未执行前向/反向/optimizer step。完整原文件、原冻结配置、失败与原控制器ledger均保留；101不重跑。LOGGER_REPAIR_AMENDMENT仅修复dict日志合并并继续原列表中未启动种子，不改变训练数值逻辑、截止时间或种子池。八项测试包含新旧worker仅这些已登记差异的逐字符核验。每个完成种子从独立初始化训练，无热启动。

8小时上限按第一次控制器start_UTC计时；剩余不足1小时不启动下一Seed，每Seed3300秒watchdog；遇关键有效性问题停止，不放宽条件。临时SetThreadExecutionState只抑制运行期间系统睡眠，finally恢复；不永久修改电源计划或代理配置。

全部结果是Development Validation的模型/epoch筛选成绩，不是独立Test。CUDA沿用H普通运行协议，不声称跨训练严格逐位确定性；保存权重的同GPU推理由独立审计精确重放。''')
 md('reports/02_F2随机种子完整排名.md','# 全部有效种子的Validation排名\n\n'+table(r,['rank','seed','RMSE_mV','MAE_mV','R2','endpoint_RMSE_mV','best_epoch','stop_epoch','training_time_s'])+'\n\n按RMSE升序、并列按seed升序；全部合法4920个重叠窗口目标均保留。R²针对冻结ΔV标签；不是绝对电压R²。逐目标物理时间见预测缓存，运行周期不参与本任务。')
 md('reports/03_F2优质种子分析.md',f'''# Validation筛选后的候选实例

Top{len(top)}：{summary['top5']}。最佳Seed{number}：RMSE {best.RMSE_mV:.6f}mV，MAE {best.MAE_mV:.6f}mV，R² {best.R2:.9f}。最佳epoch {int(best.best_epoch)}，训练停止epoch {int(best.stop_epoch)}。

Top{len(top)}平均RMSE {summary['top5_RMSE_mean_mV']:.6f}mV，属于同一Validation筛选后的乐观描述，不能冒充随机初始化的一般性能或独立泛化估计。

最佳种子第1步RMSE {first:.6f}mV，第24步/12h {last:.6f}mV，比值{last/first:.3f}。完整24步误差见F2_HORIZON_METRICS.csv，是否单调随步长增长应直接阅读曲线，不只用端点推断。P95/P99/全局最大误差均保留，典型图使用预先固定的205个起点中位索引；另展示最佳种子最差窗口，不拼接重叠预测窗口。''')
 md('reports/04_多种子稳定性分析.md','# 全候选集合统计\n\n'+table(stats,['metric','n','mean','sample_std','median','minimum','maximum'])+f'\n\n达到450轮预算上限{int(convergence.reached450_epoch_budget.sum())}个，最终一轮仍为最优{int(convergence.best_at_final_epoch.sum())}个；逐seed记录见CONVERGENCE_AND_BUDGET_DIAGNOSTICS.csv。不能声称全部模型已经收敛，也不因此增加epoch预算。'+ '\n\n'+table(tails,['seed','origin_row','target_row','horizon_step','max_abs_error_mV','single_worst_point_SSE_percent','all_origins_at_same_target_SSE_percent'])+'\n\n上表是事后尾部集中度诊断，SSE占比用于解释RMSE的敏感性，不删除任何目标。全池最大误差键上所有seed的真实配对输出见ALL_SEEDS_AT_GLOBAL_WORST_KEY.csv。少数目标上的巨大误差可能主导种子排名；仅凭这些差异不能归因于门控机制，也不据此调整后续种子、超参数或训练预算。'+'\n\n样本标准差ddof=1；不足两个种子时标准差为NA。失败/超时没有用0填补。不同种子共享同一数据，不能视为独立电池或独立寿命轨迹，不进行伪重复统计显著性检验。种子101是已披露的零步工程失败；剩余未启动列表按真实预算或有效性停止原因保留。')
 md('reports/05_与历史F0_F1_F2比较.md','# 历史与本轮的不同统计身份\n\n'+table(comparison,['result_group','n','RMSE_mV','sample_std_mV','selection'])+'\n\n历史Phase H已经提供同42/52/62的F0/F1/F2配对证据。本次新增候选预算只给F2，不能用筛选后的最佳/Top5对比历史F0/F1三种子均值，宣称完成了新增公平结构优越性证明。历史缓存不重训、不重写，不访问已复用Assessment来挑选种子。')
 md('reports/06_论文可用图表索引.md',f'''# 图表与数据索引

六组PNG（300dpi）及PDF均由scripts/generate_paper_figures.py从真实CSV生成；逐图CSV、单起点选择和科学身份见outputs/FIGURE_SOURCE_INDEX.csv、outputs/plot_data/及outputs/audits/FIGURE_SELECTION.json。没有人工构造曲线或连续窗口拼接。

F1为全部{n}有效种子排名；F2为全池分布；F3为最佳Seed的预先固定单起点24步电压；F4为同起点Top5；F5为全部24步误差和全池seed范围（不是置信区间）；F6同时保留最佳Seed最差RMSE窗口及全池最大绝对误差所在窗口，两组均展示真实电压和有符号误差。后一个负面诊断面板在观察到107尾部失稳后新增，只增加不利结果披露，不改变训练、排名或原图选择，见FIGURE_TAIL_DISCLOSURE_ADDENDUM.json。全部标明Development Validation及筛选身份。

正式checkpoint位于outputs/checkpoints，逐点预测位于outputs/predictions，完整逐batch日志位于outputs/training_logs/seed_*/batch_events.jsonl（仅本地、未备份，Git忽略）。本地绝对路径、大小、SHA256及best epoch见LOCAL_ARTIFACT_INVENTORY.csv；备份状态按清单生成时如实记录。checkpoint和预测计划随本阶段Git提交，成功推送后以Git提交与本地GIT_RECEIPT.local.json核验。''')
 md('reports/07_下一步正式对比建议.md','''# 下一步公平比较：今晚不执行

先保留历史H三种子配对结果作为既有结构证据。本轮20候选筛选仅增加F2模型实例候选，不能替代公平消融。

如果论文要采用筛选后的最佳实例作为主要结果，下一轮先登记相同20seed池、相同失败处理、同450/patience40和同样的Validation选择预算，再对F0/F1/F2进行匹配比较。可以复用本轮合格F2，但须核对全部环境、张量和代码身份；不能只为F0/F1挑选本轮F2 Top5。主表报告各方法所有有效种子的均值/样本SD，以及共同有效种子的配对差异；最佳和Top5另列为选择结果，并披露失败及计算预算。

还需在锁定候选模型和全部规则之后获得获准的独立评价，才能给出未参与选择的泛化估计。本轮不访问Test，也不能将已复用Assessment更名为独立验证。不新增SOH/RUL/PINN、主干、门控或超参数搜索。完成本轮后停止。''')
 print(json.dumps(summary,ensure_ascii=False),flush=True)
if __name__=='__main__':main()
