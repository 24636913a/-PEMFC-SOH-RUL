"""Frozen criteria, all-seed reporting and figures from REAL P3 results only."""
from __future__ import annotations
from p3_common import *
from audit_p3_predictions import audit_completed

def decide(results):
    indexed={(r['model'],r['seed']):r for r in results}
    if len(indexed)!=9 or set(indexed)!={(g,s) for g in ('G0','G1','G2') for s in (42,52,62)}:
        raise RuntimeError('NINE_RUN_RESULT_IDENTITY_INCOMPLETE')
    means={g:{metric:float(np.mean([indexed[g,s][metric] for s in (42,52,62)]))
                 for metric in ('RMSE_mV','MAE_mV','P99_abs_error_mV')}
           for g in ('G0','G1','G2')}
    comparisons={}
    for control in ('G0','G1'):
        gain=means[control]['RMSE_mV']-means['G2']['RMSE_mV'];threshold=max(.5,.05*means[control]['RMSE_mV'])
        comparisons[control]={'mean_RMSE_gain_mV':gain,'required_gain_mV':threshold,'mean_gain_PASS':gain>=threshold,
            'paired_RMSE_wins':sum(indexed['G2',s]['RMSE_mV']<indexed[control,s]['RMSE_mV'] for s in (42,52,62)),
            'endpoint_each_seed_PASS':all(indexed['G2',s]['endpoint_12h_RMSE_mV']<=1.05*indexed[control,s]['endpoint_12h_RMSE_mV'] for s in (42,52,62)),
            'target_time_equal_each_seed_PASS':all(indexed['G2',s]['target_time_equal_RMSE_mV']<=1.05*indexed[control,s]['target_time_equal_RMSE_mV'] for s in (42,52,62)),
            'mean_MAE_PASS':means['G2']['MAE_mV']<=1.05*means[control]['MAE_mV'],
            'mean_P99_PASS':means['G2']['P99_abs_error_mV']<=1.05*means[control]['P99_abs_error_mV']}
    all_pass=all(c['mean_gain_PASS'] and c['paired_RMSE_wins']==3 and c['endpoint_each_seed_PASS'] and
                 c['target_time_equal_each_seed_PASS'] and c['mean_MAE_PASS'] and c['mean_P99_PASS'] for c in comparisons.values())
    positive=all(c['mean_RMSE_gain_mV']>0 for c in comparisons.values())
    mean_threshold=all(c['mean_gain_PASS'] for c in comparisons.values())
    status=('GATE_INDEPENDENT_VALUE_ENGINEERING_CRITERIA_MET' if all_pass else
            'AVERAGE_GAIN_BUT_ROBUSTNESS_CRITERIA_FAILED' if mean_threshold else
            'SMALL_GAIN_BELOW_ENGINEERING_THRESHOLD' if positive else 'NO_CONSISTENT_MEAN_GAIN_OVER_BOTH_CONTROLS')
    return {'status':status,'all_predefined_criteria_PASS':all_pass,'comparisons':comparisons,'group_means':means,
            'historical_condition_independent_value_supported':all_pass,'statistical_significance_claim':False,
            'electrochemical_causality_claim':False,'historical_F2_automatically_replaced':False,
            'additional_model_or_training_authorized':False,'physics_F2':'PAUSED','SOH_RUL_completed':False,
            'Test_accessed':False,'confirmation_eligibility':'NOT_ESTABLISHED_REUSED_DEVELOPMENT_VALIDATION',
            'final_model_selection':'RESEARCH_OWNER_DECISION_REQUIRED'}

def markdown_table(rows,columns):
    def render(v):return f'{v:.6f}' if isinstance(v,(float,np.floating)) else str(v)
    return '| '+' | '.join(columns)+' |\n| '+' | '.join(['---']*len(columns))+' |\n'+''.join('| '+' | '.join(render(row[c]) for c in columns)+' |\n' for row in rows)


METRICS=['RMSE_mV','MAE_mV','endpoint_12h_RMSE_mV','target_time_equal_RMSE_mV','P95_abs_error_mV',
         'P99_abs_error_mV','max_abs_error_mV','top1_target_SSE_share','top5_target_SSE_share','top10_target_SSE_share']
def plot_figures(frame,horizons,comparison):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':['Arial','DejaVu Sans'],'font.size':13,'axes.spines.right':False,
        'axes.spines.top':False,'axes.linewidth':1.5,'legend.frameon':False,'svg.fonttype':'none'})
    colors={'G0':'#767676','G1':'#B64342','G2':'#0F4D92'}
    figdir=PHASE/'figures';figdir.mkdir(parents=True,exist_ok=True)
    def finish(fig,name):
        fig.tight_layout(pad=2)
        fig.savefig(figdir/(name+'.png'),dpi=300,facecolor='white')
        fig.savefig(figdir/(name+'.pdf'),facecolor='white');plt.close(fig)
    fig,axes=plt.subplots(3,1,figsize=(13,13))
    for ax,seed in zip(axes,(42,52,62)):
        for run_id,g,s in RUNS:
            if s!=seed:continue
            history=pd.DataFrame(read_json(OUT/'training_logs'/run_id/'epochs.json'))
            ax.plot(history.epoch,history.validation_RMSE_mV,color=colors[g],lw=1.5,label=g)
            row=frame[frame.model.eq(g)&frame.seed.eq(seed)&frame.checkpoint_role.eq('best')].iloc[0]
            stop=row.first_hypothetical_patience40_epoch
            if pd.notna(stop):ax.axvline(stop,color=colors[g],ls=':',alpha=.65,label=f'{g} first hypothetical p40: {int(stop)}')
            ax.scatter(row.best_epoch,row.RMSE_mV,marker='*',color=colors[g],s=110,zorder=4)
        ax.set_title(f'P3 independent fresh trajectories — Seed {seed}')
        ax.set_xlim(1,450);ax.set_ylim(0,60);ax.set_ylabel('Validation RMSE (mV)');ax.legend(ncol=2,fontsize=10)
    axes[-1].set_xlabel('Epoch (all 450 evaluated; display clipped at 60 mV, full values in logs)')
    finish(fig,'P3_TRAINING_CURVES')
    fig,axes=plt.subplots(1,3,figsize=(17,5))
    for ax,seed in zip(axes,(42,52,62)):
        for g in ('G0','G1','G2'):
            for role,ls in (('best','-'),('last450','--')):
                sub=horizons[horizons.model.eq(g)&horizons.seed.eq(seed)&horizons.checkpoint_role.eq(role)].sort_values('horizon_step')
                ax.plot(sub.forecast_hours,sub.RMSE_mV,color=colors[g],ls=ls,lw=2,label=g+' '+role)
        ax.set_title(f'Seed {seed}');ax.set_xlabel('Forecast horizon (h)');ax.set_ylim(bottom=0)
    axes[0].set_ylabel('Query-weighted RMSE (mV)');axes[-1].legend(fontsize=9)
    finish(fig,'P3_HORIZON_COMPARISON')
    fig,axes=plt.subplots(1,3,figsize=(16,5))
    for ax,metric,title in zip(axes,('RMSE_mV','endpoint_12h_RMSE_mV','target_time_equal_RMSE_mV'),('Formal queries','12 h endpoint','Target-time-equal')):
        for i,g in enumerate(('G0','G1','G2')):
            for j,role in enumerate(('best','last450')):
                ys=frame[frame.model.eq(g)&frame.checkpoint_role.eq(role)].sort_values('seed')[metric].to_numpy()
                x=i+(j-.5)*.32
                ax.bar(x,ys.mean(),width=.30,yerr=ys.std(ddof=1),capsize=3,color=colors[g],edgecolor='black',hatch='' if j==0 else '//',alpha=1 if j==0 else .55)
                ax.scatter(x+np.array([-.08,0,.08]),ys,c='black',s=15,zorder=3)
        ax.set_xticks([0,1,2],['G0','G1','G2']);ax.set_ylabel('RMSE (mV)');ax.set_title(title);ax.set_ylim(bottom=0)
    fig.suptitle('Best-of-450 (solid) / Last-450 (hatched); mean ± sample SD, all seeds')
    finish(fig,'P3_BEST_VS_LAST450')
    fig,axes=plt.subplots(1,3,figsize=(17,5))
    for ax,seed in zip(axes,(42,52,62)):
        for g in ('G0','G1','G2'):
            row=comparison[comparison.model.eq(g)&comparison.seed.eq(seed)].iloc[0]
            ax.plot([0,1],[row.P2_RMSE_mV,row.P3_best_RMSE_mV],color=colors[g],marker='o',label=g,lw=2)
            ax.annotate(f'P2 stop {int(row.P2_actual_stop_epoch)}',(0,row.P2_RMSE_mV),xytext=(4,5),textcoords='offset points',fontsize=9,color=colors[g])
        ax.set_xticks([0,1],['P2 actual early stop','P3 best of 450']);ax.set_title(f'Seed {seed}');ax.set_ylim(bottom=0)
    axes[0].set_ylabel('Formal query RMSE (mV)');axes[-1].legend()
    fig.suptitle('Separate runs, not a spliced training trajectory')
    finish(fig,'P3_P2_ACTUAL_STOP_COMPARISON')
    fig,axes=plt.subplots(1,2,figsize=(13,5))
    for ax,role in zip(axes,('best','last450')):
        for g in ('G0','G1','G2'):
            sub=frame[frame.model.eq(g)&frame.checkpoint_role.eq(role)].sort_values('seed')
            ax.plot(sub.seed.astype(str),sub.RMSE_mV,color=colors[g],marker='o',label=g,lw=2)
        ax.set_title(role);ax.set_xlabel('Seed');ax.set_ylim(bottom=0)
    axes[0].set_ylabel('Formal query RMSE (mV)');axes[-1].legend()
    finish(fig,'P3_PAIRED_SEED_RMSE')

def analyze(cfg,protocol_sha):
    audit=audit_completed(cfg,protocol_sha,require_all=True)
    aggregate=[read_json(OUT/'training_logs'/r/'result.json') for r,_,_ in RUNS]
    results=[x for r in aggregate for x in r['checkpoint_results']]
    frame=pd.DataFrame([{k:v for k,v in r.items() if not isinstance(v,(dict,list))} for r in results])
    frame.to_csv(OUT/'P3_NINE_RUN_METRICS.csv',index=False,float_format='%.17g')
    summary=[]
    for role in ('best','last450'):
        for g in ('G0','G1','G2'):
            for metric in METRICS:
                vals=frame[frame.model.eq(g)&frame.checkpoint_role.eq(role)][metric].to_numpy()
                summary.append({'checkpoint_role':role,'model':g,'metric':metric,'n_seeds':3,'mean':float(vals.mean()),
                                'sample_SD':float(vals.std(ddof=1)),'min':float(vals.min()),'max':float(vals.max())})
    pd.DataFrame(summary).to_csv(OUT/'P3_THREE_SEED_SUMMARY.csv',index=False,float_format='%.17g')
    horizons=pd.concat([pd.read_csv(OUT/'metrics'/f'{r}_{role}_horizons.csv',float_precision='round_trip').assign(checkpoint_role=role)
                        for r,_,_ in RUNS for role in ('best','last450')],ignore_index=True)
    horizons.to_csv(OUT/'P3_ALL_HORIZON_METRICS.csv',index=False,float_format='%.17g')
    best_last=[];comparison=[];pairs=[];manifest=[]
    by_id={(r['model'],r['seed'],r['checkpoint_role']):r for r in results}
    for run_id,g,s in RUNS:
        best=by_id[g,s,'best'];last=by_id[g,s,'last450']
        old=read_json(P2/'outputs/training_logs'/run_id.replace('P3-','P2-')/'result.json')
        best_last.append({'model':g,'seed':s,'best_epoch':best['best_epoch'],'first_hypothetical_patience40_epoch':best['first_hypothetical_patience40_epoch'],
                          **{f'{role}_{key}':by_id[g,s,role][key] for role in ('best','last450') for key in METRICS},
                          **{'last_minus_best_'+key:last[key]-best[key] for key in METRICS}})
        comparison.append({'model':g,'seed':s,'P2_run_id':old['run_id'],'P3_run_id':run_id,'P2_actual_stop_epoch':old['stop_epoch'],
                'P2_best_epoch':old['best_epoch'],'P3_best_epoch':best['best_epoch'],'P3_first_hypothetical_stop_epoch':best['first_hypothetical_patience40_epoch'],
                'P3_new_best_after_hypothetical_stop':best['new_best_after_hypothetical_stop'],'initialization_matches_P2':best['initial_model_sha256']==old['initial_model_sha256'],
                **{'P2_'+key:old[key] for key in METRICS},**{'P3_best_'+key:best[key] for key in METRICS},
                **{'P3_best_minus_P2_'+key:best[key]-old[key] for key in METRICS}})
        for r in (best,last):
            for path,kind in ((PHASE/r['checkpoint'],'checkpoint'),(PHASE/r['checkpoint_backup']['path'],'verified_backup'),(PHASE/r['validation_prediction'],'prediction')):
                manifest.append({'run_id':run_id,'role':r['checkpoint_role'],'kind':kind,'path':str(path.relative_to(PHASE)).replace('\\\\','/'),
                                 'sha256':file_hash(path),'bytes':path.stat().st_size})
        for name in ('batch_events.jsonl','epochs.json','epochs.csv','console.log','initial.json','environment.json','data_access_audit.json'):
            path=OUT/'training_logs'/run_id/name
            manifest.append({'run_id':run_id,'kind':'log_or_audit','path':str(path.relative_to(PHASE)).replace('\\\\','/'),'sha256':file_hash(path),'bytes':path.stat().st_size})
    comp=pd.DataFrame(comparison)
    comp.to_csv(OUT/'P3_P2_EARLY_STOP_VS_FULL450_COMPARISON.csv',index=False,float_format='%.17g')
    pd.DataFrame(best_last).to_csv(OUT/'P3_BEST_VS_LAST450_COMPARISON.csv',index=False,float_format='%.17g')
    for role in ('best','last450'):
        for s in (42,52,62):
            for g in ('G0','G1'):
                pairs.append({'checkpoint_role':role,'seed':s,'comparison':'G2-'+g,
                              **{key:by_id['G2',s,role][key]-by_id[g,s,role][key] for key in METRICS}})
    pd.DataFrame(pairs).to_csv(OUT/'P3_PAIRED_SEED_DIFFERENCES.csv',index=False,float_format='%.17g')
    reference={role:decide([by_id[g,s,role] for g in ('G0','G1','G2') for s in (42,52,62)]) for role in ('best','last450')}
    g2comp=comp[comp.model.eq('G2')]
    first_stats=[{'seed':r['seed'],'hypothetical_stop_epoch':r['first_hypothetical_patience40_epoch'],
                 'best_at_hypothetical_stop':r['best_at_first_hypothetical_stop'],'full450_best_epoch':r['best_epoch'],
                 'new_best_after_hypothetical_stop':r['new_best_after_hypothetical_stop']} for r in aggregate if r['model']=='G2']
    decision={'status':'P3_FULL450_SENSITIVITY_COMPLETED','experiment_nature':'POST_P2_EXPLORATORY_EPOCH_SENSITIVITY',
              'base_commit':BASE,'protocol_sha256':protocol_sha,'code_identity_sha256':cfg['code_identity']['sha256'],
              'nine_runs_completed':9,'epochs_per_run':450,'total_training_epochs':4050,'prediction_cache_roles':18,
              'initialization_matches_P2':True,'sampler_all_450_epochs_equal_within_seed':True,
              'P2_criteria_reference_only':reference,'G2_after_hypothetical_stop':first_stats,
              'G2_P3_best_better_than_P2_seeds':g2comp.loc[g2comp.P3_best_minus_P2_RMSE_mV.lt(0),'seed'].tolist(),
              'G2_P3_best_vs_P2_RMSE_difference_mean_mV':float(g2comp.P3_best_minus_P2_RMSE_mV.mean()),
              'P2_formal_decision_changed':False,'independent_confirmation':False,'CUDA_strict_bitwise_determinism_claimed':False,
              'Test_accessed':False,'Assessment_accessed':False,'additional_training_authorized':False,
              'checkpoint_backup':'18 HASH_VERIFIED_COPIES_SEPARATE_DIRECTORY_SAME_E_VOLUME_NOT_OFF_DEVICE'}
    atomic_json(OUT/'P3_FINAL_SENSITIVITY_DECISION.json',decision)
    atomic_json(OUT/'P3_LOCAL_ARTIFACT_AND_BACKUP_MANIFEST.json',{'artifacts':manifest,'backup_count':18,'all_backup_hashes_verified':True,
                   'same_E_volume':True,'off_device_backup':False,'sole_copy_deleted':False})
    plot_figures(frame,horizons,comp)
    cols=['checkpoint_role','model','seed','RMSE_mV','MAE_mV','endpoint_12h_RMSE_mV','target_time_equal_RMSE_mV',
          'P95_abs_error_mV','P99_abs_error_mV','max_abs_error_mV','best_epoch','selected_epoch']
    table=markdown_table(sorted(results,key=lambda r:(r['checkpoint_role'],r['model'],r['seed'])),cols)
    mean_table=markdown_table([r for r in summary if r['metric']=='RMSE_mV'],['checkpoint_role','model','mean','sample_SD','min','max'])
    q1=sum(r['new_best_after_hypothetical_stop'] for r in first_stats)
    q2=decision['G2_P3_best_better_than_P2_seeds']
    bestref=reference['best'];lastref=reference['last450']
    q5=markdown_table(g2comp.to_dict('records'),['seed','P3_best_minus_P2_endpoint_12h_RMSE_mV',
                'P3_best_minus_P2_target_time_equal_RMSE_mV','P3_best_minus_P2_P99_abs_error_mV','P3_best_minus_P2_max_abs_error_mV'])
    report=f'''# P3 — 完整450 Epoch敏感性实验

## 范围与身份

POST_P2_EXPLORATORY_EPOCH_SENSITIVITY。Base {BASE}，Protocol {protocol_sha}，code identity {cfg['code_identity']['sha256']}。
九次新鲜初始化训练均450 epoch，总4050，无重试。P2/F2/所有旧阶段只读，旧权重没有打开。
与P2对应初始权重、核心/门控及CPU/CUDA/Python/NumPy/loader RNG身份完全一致；本轮三组每seed实际450轮origin顺序完全一致。
CUDA_STRICT_BITWISE_DETERMINISM_NOT_CLAIMED：初始身份一致不代表与P2独立运行逐步bitwise一致。P2与P3没有拼接或从旧权重续训。

## 唯一训练改变

不因patience40停止；其余H/P1/P2配置和48→24 ΔV任务固定。Best-of-450为主，Last-450为强制敏感性。
best监测标准化Validation RMSE < best−1e−12。first hypothetical p40记录本条P3轨迹首次stale>=40时刻，并非P2实际stop。
AdamW、batch32、clip1、无scheduler/AMP、E→C eval/no_grad/batch32预测路径全部保留。

## 完整九训练、两种保存规则

{table}

## 三Seed等权均值与样本SD（ddof=1）

{mean_table}

4920预测查询等权RMSE为正式指标；205 origins、251 unique target rows，重叠查询不是独立样本。
target-time-equal是每target先平均平方误差再开根号，不是预测值平均集成。
所有2247/2248及极端查询保留。24个horizon和Top1/5/10 target SSE详表在outputs，不删除异常。

## 六个研究问题

1. G2在P3首次假设pat40停止之后产生新最优的seed数：{q1}/3。逐seed：{json.dumps(first_stats,ensure_ascii=False)}。
2. G2 Best-of-450优于P2实际best的seed：{q2}，平均P3−P2正式RMSE={decision['G2_P3_best_vs_P2_RMSE_difference_mean_mV']:.6f} mV。
3. 公平450预算下Best参考门槛状态：{bestref['status']}；Last参考状态：{lastref['status']}。各组均值与每seed完整公开，不挑保存规则。
4. 是否三个seed均改善：{len(q2)==3}（相对P2）；是否同时赢两对照三seed，见下面固定门槛表，不用个别seed代替稳定性。
5. 长终点、target-time-equal、极端误差是否同步，以下P3−P2负值才是改善：

{q5}

6. 本实验只能说明冻结预算及已复用Validation下的经验敏感性；更长搜索轨迹本身增加best选择机会，不能证明门控学到电化学机理。
优化盆地、延迟gate学习或CUDA数值差异等机制没有独立识别，均只是假设。不能把Best的选优收益说成Last也一致稳定。

## P2工程门槛：仅参照，不修改历史判定

{json.dumps(reference,ensure_ascii=False,indent=2)}

P2正式 NO_CONSISTENT_MEAN_GAIN_OVER_BOTH_CONTROLS 保留。不自动替换原F2，不升级为独立确认性证据。
O1历史正面结果：F2整体正式Validation RMSE优势可以报告，但为 F2_ADVANTAGE_METRIC_DEPENDENT，非全指标全seed一致；原J冻结结构与L确认性Test不合格限制继续有效。

## 数据与访问

仅N1 nrows3007后age<1819.997738得3005。Train/Validation范围[0,2103)/[2103,2404)，1380/205窗。
不做全N1 hash；沿用允许Development fingerprint、Train-only未滤波scaler、Top10/因果EWMA0.3。
没有Assessment、Test、raw、极化或其它新标签访问，Python审计/CSV入口防线及纯合成测试证据保留。
测试只验证工程防线，不生成科研模型结果。

## 产物与备份

18checkpoint原件和18逐文件SHA一致副本。副本在P3/backups/checkpoints，独立目录但同E卷，并非跨设备灾备。
没有删除唯一副本；权重和batch journal本地保留、Git忽略，清单记录path/bytes/SHA。
全部18预测缓存重读重算、checkpoint精确重载、450history与共同采样完整性PASS。
图表用固定颜色、清晰分栏、300dpi PNG及PDF；训练图展示上限60mV，未截断原始日志或排名数据。

## 停止

此处完成并停止。不调门槛，不增加融合比例、seed/backbone/Physics-F2，不进入SOH/RUL，不自动论文写作。
'''
    folder=PHASE/'reports';folder.mkdir(parents=True,exist_ok=True)
    (folder/'P3_FINAL_SENSITIVITY_REPORT.md').write_text(report,encoding='utf-8',newline='\n')
    short=f'''# 导师一页汇报：450轮训练敏感性

研究问题：P2提前停止是否掩盖工况动态融合G2的性能？冻结三模型×三seed，九次均从头450轮，仅取消pat40终止。

历史：原F2有整体Validation RMSE优势，但O1标注指标依赖；P2 G0/G1/G2均值为8.387/8.350/9.143mV，动态门控没有达到相对两对照的稳定独立收益。

P3完整结果（mV，mean±样本SD）：
{mean_table}

G2有{q1}/3个seed在本轮首次假设早停后获得新最优；比P2实际best改善的seed是{q2}，平均差{decision['G2_P3_best_vs_P2_RMSE_difference_mean_mV']:.3f}mV。
公平450轮下，Best参照门槛：{bestref['status']}；Last：{lastref['status']}。

可声称：已经执行并透明报告长训练敏感性，不能用单次优势替代三seed、终点及极端误差稳健性。
不能证明：独立Test优越、门控电化学机制、所有指标一致胜出、SOH/RUL完成。Best使用已复用Validation选优，属于事后探索。
P2负面结论原样保留。原F2不自动替换。Test/Assessment未访问；18权重及核验副本保留。完成后停止，交负责人决策。
'''
    (folder/'P3_SUPERVISOR_ONE_PAGE_SUMMARY.md').write_text(short,encoding='utf-8',newline='\n')
    print(json.dumps({'status':decision['status'],'best':bestref['group_means'],'last450':lastref['group_means'],'Test_accessed':False}),flush=True)

def technical_failure_report(ledger):
    folder=PHASE/'reports';folder.mkdir(parents=True,exist_ok=True)
    (folder/'P3_FINAL_SENSITIVITY_REPORT.md').write_text('# P3 技术硬停止\n\n'+json.dumps(ledger,ensure_ascii=False,indent=2)+'\n\n不重试、不改变Seed、不将技术失败写成负面科研结果；所有现场保留。',encoding='utf-8')
    atomic_json(OUT/'P3_FINAL_SENSITIVITY_DECISION.json',{'status':'P3_TECHNICAL_FAILURE','ledger':ledger,'scientific_decision':None,'Test_accessed':False})

