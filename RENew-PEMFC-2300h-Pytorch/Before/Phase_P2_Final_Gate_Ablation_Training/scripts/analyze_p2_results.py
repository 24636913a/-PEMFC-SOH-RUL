"""Frozen criteria, all-seed reporting and figures from REAL P2 results only."""
from __future__ import annotations
from p2_common import *
from audit_p2_predictions import audit_completed

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

def plot_figures(frame,horizons,gates):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':['Arial','DejaVu Sans'],'font.size':13,'axes.spines.right':False,
        'axes.spines.top':False,'axes.linewidth':1.5,'legend.frameon':False,'svg.fonttype':'none'})
    colors={'G0':'#767676','G1':'#B64342','G2':'#0F4D92'};markers={'G0':'o','G1':'s','G2':'^'}
    figdir=PHASE/'figures';figdir.mkdir(parents=True,exist_ok=True)
    def finish(fig,name):
        fig.tight_layout(pad=2);fig.savefig(figdir/(name+'.png'),dpi=300,facecolor='white')
        fig.savefig(figdir/(name+'.pdf'),facecolor='white');plt.close(fig)
    for metric,name,label in (('RMSE_mV','P2_RMSE_Comparison','Query-weighted RMSE (mV)'),
                ('endpoint_12h_RMSE_mV','P2_Endpoint_Comparison','12 h endpoint RMSE (mV)'),
                ('target_time_equal_RMSE_mV','P2_Target_Time_Equal_Comparison','Target-time-equal RMSE (mV)')):
        fig,ax=plt.subplots(figsize=(8,5))
        for i,g in enumerate(('G0','G1','G2')):
            ys=frame[frame.model.eq(g)].sort_values('seed')[metric].to_numpy()
            ax.bar(i,float(ys.mean()),yerr=float(ys.std(ddof=1)),color=colors[g],edgecolor='black',width=.62,capsize=5,hatch=('','/','..')[i])
            ax.scatter(np.full(3,i)+np.array([-.16,0,.16]),ys,c='black',s=28,zorder=3)
        ax.set_xticks([0,1,2],['G0 fixed 0.5','G1 global gate','G2 condition gate']);ax.set_ylabel(label)
        ax.set_ylim(bottom=0);ax.set_title('Development Validation: mean ± sample SD; all 3 seeds')
        finish(fig,name)
    fig,axes=plt.subplots(1,3,figsize=(17,5),sharex=True)
    for ax,seed in zip(axes,(42,52,62)):
        for g in ('G0','G1','G2'):
            section=horizons[horizons.model.eq(g)&horizons.seed.eq(seed)].sort_values('horizon_step')
            ax.plot(section.forecast_hours,section.RMSE_mV,label=g,color=colors[g],lw=2,marker=markers[g],markevery=3)
        ax.set_title(f'Seed {seed}');ax.set_xlabel('Forecast horizon (h)');ax.set_ylim(bottom=0)
    axes[0].set_ylabel('Query-weighted RMSE (mV)');axes[-1].legend()
    finish(fig,'P2_Horizon_Comparison')
    fig,axes=plt.subplots(1,3,figsize=(17,5),sharey=True)
    for ax,seed in zip(axes,(42,52,62)):
        for g in ('G0','G1','G2'):
            values=gates[gates.model.eq(g)&gates.seed.eq(seed)].gate_weight
            ax.hist(values,bins=np.linspace(0,1,31),histtype='step',lw=2,label=g,color=colors[g])
        ax.set_title(f'Seed {seed}');ax.set_xlabel('Best-checkpoint fusion weight g');ax.set_xlim(0,1)
    axes[0].set_ylabel('Validation origin count');axes[-1].legend()
    finish(fig,'P2_Gate_Weight_Distribution')

def analyze(cfg,protocol_sha):
    audit_completed(cfg,protocol_sha,require_all=True)
    results=[read_json(OUT/'training_logs'/r/'result.json') for r,_,_ in RUNS]
    decision=decide(results);decision.update(protocol_sha256=protocol_sha,code_identity_sha256=cfg['code_identity']['sha256'],base_commit=BASE)
    frame=pd.DataFrame(results);frame.to_csv(OUT/'metrics/P2_NINE_RUN_METRICS.csv',index=False,float_format='%.17g')
    summary=[]
    metric_names=['RMSE_mV','MAE_mV','endpoint_12h_RMSE_mV','target_time_equal_RMSE_mV','P95_abs_error_mV',
                  'P99_abs_error_mV','max_abs_error_mV','top1_target_SSE_share','top5_target_SSE_share','top10_target_SSE_share']
    for g in ('G0','G1','G2'):
        for metric in metric_names:
            vals=frame[frame.model.eq(g)][metric].to_numpy()
            summary.append({'model':g,'metric':metric,'n_seeds':3,'mean':float(vals.mean()),'sample_SD':float(vals.std(ddof=1)),
                            'min':float(vals.min()),'max':float(vals.max())})
    pd.DataFrame(summary).to_csv(OUT/'metrics/P2_THREE_SEED_SUMMARY.csv',index=False,float_format='%.17g')
    horizons=pd.concat([pd.read_csv(OUT/'metrics'/f'{r}_horizons.csv',float_precision='round_trip') for r,_,_ in RUNS],ignore_index=True)
    horizons.to_csv(OUT/'metrics/P2_ALL_HORIZON_METRICS.csv',index=False,float_format='%.17g')
    gates=pd.concat([pd.read_csv(OUT/'metrics'/f'{r}_gate_weights.csv',float_precision='round_trip') for r,_,_ in RUNS],ignore_index=True)
    gate_summaries=gates.groupby(['model','seed']).gate_weight.agg(['mean','std','min','median','max']).reset_index()
    gate_summaries.to_csv(OUT/'metrics/P2_GATE_WEIGHT_SUMMARY.csv',index=False,float_format='%.17g')
    paired=[]
    for seed in (42,52,62):
        for control in ('G0','G1'):
            g2=frame[frame.model.eq('G2')&frame.seed.eq(seed)].iloc[0];base=frame[frame.model.eq(control)&frame.seed.eq(seed)].iloc[0]
            paired.append({'seed':seed,'comparison':'G2-'+control,**{metric:float(g2[metric]-base[metric]) for metric in metric_names}})
    pd.DataFrame(paired).to_csv(OUT/'metrics/P2_PAIRED_SEED_DIFFERENCES.csv',index=False,float_format='%.17g')
    manifest=[]
    for result in results:
        run_id=result['run_id']
        for path in (PHASE/result['checkpoint'],PHASE/result['validation_prediction'],OUT/'training_logs'/run_id/'batch_events.jsonl',
                     OUT/'training_logs'/run_id/'epochs.json',OUT/'training_logs'/run_id/'epochs.csv',OUT/'training_logs'/run_id/'console.log'):
            manifest.append({'run_id':run_id,'path':str(path.relative_to(PHASE)).replace('\\','/'),'bytes':path.stat().st_size,
                             'sha256':file_hash(path),'backup_status':'LOCAL_ORIGINAL_PRESERVED_NO_SEPARATE_BACKUP' if path.suffix=='.pt' else 'REPOSITORY_OR_LOCAL_JOURNAL'})
    atomic_json(OUT/'audits/P2_LOCAL_ARTIFACT_MANIFEST.json',{'artifacts':manifest,'sole_copies_deleted':False})
    atomic_json(OUT/'P2_FINAL_RESEARCH_DECISION.json',decision)
    plot_figures(frame,horizons,gates)
    cols=['model','seed','RMSE_mV','MAE_mV','endpoint_12h_RMSE_mV','target_time_equal_RMSE_mV','P99_abs_error_mV','best_epoch','stop_epoch']
    table=markdown_table(sorted(results,key=lambda r:(r['model'],r['seed'])),cols)
    mean_table=markdown_table([r for r in summary if r['metric']=='RMSE_mV'],['model','mean','sample_SD','min','max'])
    evaluation='正式主指标为全部4920个合法预测查询等权RMSE（mV）；三Seed均值和样本SD，不混用pooled。目标时刻等权先平均平方误差再开根号，不平均预测值。目标2247/2248、极端误差全部保留。205 origins、251 unique target rows、重叠查询不能当独立统计样本。'
    identity='9个新P2 checkpoint只在P2/outputs/checkpoints，本地安全保留、Git忽略，无独立备份（已如实记录），从未加载/覆盖历史九权重。模型run_id、seed、protocol/code SHA、初始参数/RNG、输入fingerprints、最佳epoch、重载严格相同和缓存真值/键均审计。'
    limitation='原Development Validation已复用，本次不是独立确认性Test；Assessment与Test均未访问。预注册工程门槛不等于显著性或电化学因果机制。新九模型不自动替代冻结F2。Physics-F2继续暂停，不代表SOH/RUL完成，不继续gate改型、第四组、调参、重训或论文正文。'
    passed=decision['all_predefined_criteria_PASS']
    value=('历史工况条件依赖在本次受控Development实验中提供达到全部预设门槛的额外预测价值。' if passed else
           '本次未证明历史工况门控相对全局学习比例和固定融合具有达到全部预设门槛的稳定独立增益；不得以平均或个别Seed优势升级该主张。')
    documents={
        'P2_NINE_RUN_TRAINING_REPORT.md':f'# P2 九次正式训练\n\nBase `{BASE}`；协议 `{protocol_sha}`。全部9次真实训练完成，无重试。\n\n{table}\n\n## 三Seed汇总\n\n{mean_table}\n\n{evaluation}\n\n{identity}\n\n固定原H超参数、48→24、Raw14槽Top10零掩码/因果EWMA0.3、Train-only scaler；只读N1 nrows3007后filter为3005，Train/Validation1380/205。标准化RMSE < best−1e−12、pat40、max450、AdamW、clip1，无AMP/scheduler。\n\n逐epoch实际origin顺序和hash保留，并验证同Seed所有共同epoch一致。CPU/CUDA/Python/NumPy/loader初始状态复现P1。GPU串行按42:G0/G1/G2；52:G1/G2/G0；62:G2/G0/G1。\n\n{limitation}\n',
        'P2_MODEL_AND_CHECKPOINT_IDENTITY.md':f'# 模型及checkpoint身份\n\n{identity}\n\n完整本地文件清单、bytes、SHA、备份状态：outputs/audits/P2_LOCAL_ARTIFACT_MANIFEST.json。P1及H/F原始源码只读。初始核心共享，G1/G2门控57参数共享且g=0.5；G0332379，G1/G2332436。G1固定ones5、G2原历史五工况，核心decoder不变。\n\n产物完整性：PASS；全部预测重新读取重算一致。\n',
        'P2_GATE_ABLATION_RESULTS.md':f'# 完整门控消融结果\n\n{table}\n\n{mean_table}\n\n{evaluation}\n\n逐步24 horizon、P95/P99/max、Top1/5/10 target-SSE均完整公开在outputs/metrics。禁止异常排除。\n\n{limitation}\n',
        'P2_GATE_INDEPENDENT_VALUE_ASSESSMENT.md':f'# 门控独立价值审查\n\n主对比G2−G1，辅助G2−G0。\n\n{value}\n\n```json\n{json.dumps(decision["comparisons"],indent=2,ensure_ascii=False)}\n```\n\n不能把G1/G2相同57参数解释为相同函数有效自由度。G1全局可学习比例，G2历史条件依赖；本次是受控经验比较，不是某工况变量电化学因果识别。\n',
        'P2_FINAL_RESEARCH_DECISION.md':f'# 最终研究决定\n\n**{decision["status"]}**\n\n全部门槛通过：{passed}。\n\n{value}\n\n可保留：冻结双路径TFT结构、历史因果输入与工况门控设计作为已执行并完整评价的工程创新；{("工况条件依赖的受控增益主张（仅Development范围）。" if passed else "门控独立稳定有效的主张应削弱为未获本次完整门槛支持。" )}\n\n是否建议现有F2作为最终模型：保留其既有冻结身份，不自动以新G2替换；是否最终采用由负责人结合历史结果和本次九组完整结果决定。\n\n{limitation}\n\n技术完整性PASS，科学决定与技术失败分开。到此停止，等待研究负责人审核。\n'}
    reportdir=PHASE/'reports';reportdir.mkdir(parents=True,exist_ok=True)
    for name,text in documents.items():(reportdir/name).write_text(text,encoding='utf-8',newline='\n')
    print(json.dumps({'status':decision['status'],'all_criteria_PASS':passed,'group_means':decision['group_means'],'Test_accessed':False}),flush=True)

def technical_failure_report(ledger):
    folder=PHASE/'reports';folder.mkdir(parents=True,exist_ok=True)
    text='# P2 技术硬停止\n\n**P2_TECHNICAL_FAILURE**\n\n'+json.dumps(ledger,ensure_ascii=False,indent=2)+'\n\n不把技术失败写成无改善，不从头重跑，不替换Seed。既有checkpoint、journal和所有现场保留。Test/Assessment不访问。未完成九组，不生成假图或完整科研结论。\n'
    (folder/'P2_FINAL_RESEARCH_DECISION.md').write_text(text,encoding='utf-8',newline='\n')
    atomic_json(OUT/'P2_FINAL_RESEARCH_DECISION.json',{'status':'P2_TECHNICAL_FAILURE','ledger':ledger,'formal_scientific_conclusion':None,'Test_accessed':False})
