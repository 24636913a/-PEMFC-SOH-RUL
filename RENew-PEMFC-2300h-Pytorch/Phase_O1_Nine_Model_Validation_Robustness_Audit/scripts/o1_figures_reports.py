"""Figures/reports from verified in-memory Validation cache metrics only."""
import math
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

COLORS={'F0':'#B64342','F1':'#42949E','F2':'#0F4D92'}
LINES={'F0':'--','F1':':','F2':'-'}
SEEDS=(42,52,62)
MODELS=('F0','F1','F2')


def md_table(frame, fields=None):
    frame=frame[fields] if fields else frame
    def value(x):
        return f'{x:.8g}' if isinstance(x,(float,np.floating)) else str(x)
    return '|'+ '|'.join(frame.columns)+'|\n|'+'|'.join(['---']*len(frame.columns))+'|\n'+ '\n'.join('|'+ '|'.join(value(x) for x in row)+'|' for row in frame.itertuples(index=False,name=None))+'\n'


def save(phase, fig, name):
    folder=phase/'figures';folder.mkdir(exist_ok=True)
    fig.savefig(folder/(name+'.png'),dpi=300,bbox_inches='tight',pad_inches=.12)
    fig.savefig(folder/(name+'.pdf'),bbox_inches='tight',pad_inches=.12)
    plt.close(fig)


def grouped(ax, metric, frames):
    x=np.arange(3)
    for j,model in enumerate(MODELS):
        group=frames[frames.model==model].set_index('seed')
        ax.bar(x+(j-1)*.24,[group.loc[s,metric] for s in SEEDS],.23,
               color=COLORS[model],edgecolor='black',linewidth=.8,hatch=('','//','..')[j],label=model)
    ax.set_xticks(x,[f'Seed {s}' for s in SEEDS]);ax.set_ylim(bottom=0)


def figures(phase, metrics, horizons):
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':15,'axes.linewidth':2,
        'axes.spines.top':False,'axes.spines.right':False,'legend.frameon':False,
        'svg.fonttype':'none','pdf.fonttype':42})
    fig,axes=plt.subplots(1,3,figsize=(17,5),sharey=True,layout='constrained')
    ymax=horizons.RMSE_mV.max()*1.08
    for ax,seed in zip(axes,SEEDS):
        for model in MODELS:
            g=horizons[(horizons.seed==seed)&(horizons.model==model)].sort_values('horizon_step')
            ax.plot(g.horizon_step,g.RMSE_mV,color=COLORS[model],ls=LINES[model],lw=2.3,label=model)
        ax.set_title(f'Validation — Seed {seed}');ax.set_xlabel('Forecast step (0.5 h/step)')
        ax.set_xlim(1,24);ax.set_ylim(0,ymax);ax.set_xticks([1,6,12,18,24])
    axes[0].set_ylabel('Query RMSE (mV)');axes[2].legend()
    save(phase,fig,'Fig1_Validation_Horizon_RMSE_Three_Seeds')
    fig,ax=plt.subplots(figsize=(9,5),layout='constrained');grouped(ax,'endpoint_RMSE_mV',metrics)
    ax.set_ylabel('12 h endpoint RMSE (mV)');ax.set_title('Validation — all frozen Seeds');ax.legend()
    save(phase,fig,'Fig2_Validation_12h_Endpoint')
    fig,axes=plt.subplots(1,2,figsize=(15,5),sharey=True,layout='constrained')
    for ax,k,title in zip(axes,('RMSE_mV','target_time_equal_RMSE_mV'),('Formal query-weighted','Target-time-equal sensitivity')):
        grouped(ax,k,metrics);ax.set_title(title);ax.set_ylabel('Validation RMSE (mV)')
    for ax in axes:
        ax.set_ylim(0,metrics[['RMSE_mV','target_time_equal_RMSE_mV']].to_numpy().max()*1.12)
    handles,labels=axes[1].get_legend_handles_labels()
    fig.legend(handles,labels,loc='upper center',bbox_to_anchor=(.5,1.14),ncol=3)
    save(phase,fig,'Fig3_Formal_vs_Target_Time_Equal')
    fig,axes=plt.subplots(1,3,figsize=(19,5),layout='constrained')
    for ax,k,title in zip(axes,('P95_mV','P99_mV','max_abs_error_mV'),('P95 absolute error','P99 absolute error','Maximum absolute error')):
        grouped(ax,k,metrics);ax.set_title(title);ax.set_ylabel('Validation error (mV)')
    axes[-1].legend();save(phase,fig,'Fig4_Validation_Tail_and_Max_Error')
    fig,axes=plt.subplots(1,2,figsize=(15,5),sharey=True,layout='constrained')
    for ax,k,title in zip(axes,('top1_target_SSE_share','top5_target_SSE_share'),('Top1 target contribution','Top5 targets contribution')):
        grouped(ax,k,metrics);ax.set_title(title);ax.set_ylabel('Share of formal query SSE');ax.set_ylim(0,1)
    axes[-1].legend();save(phase,fig,'Fig5_Validation_Target_SSE_Concentration')
    # Equal-seed mean lines; SD is descriptive seed spread, not a CI.
    fig,ax=plt.subplots(figsize=(10,5),layout='constrained')
    for model in MODELS:
        g=horizons[horizons.model==model].groupby('horizon_step').RMSE_mV
        mean=g.mean();sd=g.std(ddof=1)
        ax.plot(mean.index,mean,color=COLORS[model],ls=LINES[model],lw=2.3,label=model)
        ax.fill_between(mean.index,np.maximum(0,mean-sd),mean+sd,color=COLORS[model],alpha=.15)
    ax.set_title('Validation — equal-Seed mean ± sample SD (not CI)')
    ax.set_xlabel('Forecast step (0.5 h/step)');ax.set_ylabel('Query RMSE (mV)');ax.set_xlim(1,24);ax.set_ylim(bottom=0);ax.legend()
    save(phase,fig,'Fig6_Validation_Horizon_Equal_Seed_Mean_SD')


def write_report(phase,name,text):
    folder=phase/'reports';folder.mkdir(exist_ok=True)
    (folder/name).write_text(text.strip()+'\n',encoding='utf-8')


def deliver(phase,metrics,horizons,targets,pairs,summary,extreme,status):
    figures(phase,metrics,horizons)
    base='7553fcf3ad5f2b10bcf1f590ce90d19d46468bb4'
    write_report(phase,'O1_VALIDATION_IDENTITY_REPORT.md',f'''
# O1 — Validation身份核验

基准 `{base}`；Stage0的协议SHA与允许来源/hash记录见protocol目录。九文件来自H正式Validation，role=MULTISEED_PRIMARY；Repeat-A/B未读取。

身份 `VALIDATION_CACHE_IDENTITY_PASS`。九文件实际SHA逐一匹配H保存的prediction_file_sha256；H protocol SHA匹配冻结文件。每run205origins、4920queries、24horizon各205。整数key无重复、每origin完整24步，target=origin+h且在[2103,2404)，预测/真值finite。九模型按显式键连接检查共同key集合与真值精确相同，不删除、补齐、自动去重或修改原文件。origin不连续沿用缓存宇宙，不重新构造raw segment。

模型/seed/epoch/参数量/权重路径通过H/L/J metadata对应；权重SHA仅对历史manifest，不读取权重bytes。H static audit窗口规则作为segment合法性的历史依据，O1未接触raw/N1重新验证每个segment。所有模型完整缓存通过之后才算指标，不按误差筛样本。

复算使用round_trip float64及冻结fsum/linear quantile。正式RMSE/MAE与H逐run、F2七项扩展与J learned一致，绝对容差1e-10mV，实际差异完整保留于 `outputs/O1_HISTORICAL_METRIC_DIFFERENCES.csv`。J已保存的4920 keys/truth/pred exact审计通过，但O1不重新读取J逐点预测；此为历史身份+本轮指标一致性，不冒充新的J逐点重推理。

唯一目标时刻数各run={int(metrics.n_unique_target_times.iloc[0])}，只是去重索引数量，不代表统计独立。所有缓存查询保留；没有新预测、模型加载、训练、Assessment逐点或Test访问。完整各文件证据见 `outputs/O1_IDENTITY_AND_RECOMPUTATION_AUDIT.json`。
''')
    selected=['model','seed','RMSE_mV','MAE_mV','P95_mV','P99_mV','max_abs_error_mV','endpoint_RMSE_mV','target_time_equal_RMSE_mV','top1_target_SSE_share','top5_target_SSE_share']
    report='# O1 — 九模型稳健性比较\n\n'+md_table(metrics,selected)
    report+='\n## 等Seed汇总（不是pooled、不是三fold）\n\n'+md_table(summary)
    report+='\n## 配对方向：新模型减基线\n\n'+md_table(pairs,['comparison','seed','RMSE_mV_difference','endpoint_RMSE_mV_difference','target_time_equal_RMSE_mV_difference','P95_mV_difference','P99_mV_difference','max_abs_error_mV_difference'])
    report+=f'''\n正式query-weighted RMSE保持主指标，target等权为敏感性。状态 `{status}`；不使用另一指标挑模型，不pool三Seed、不挑最好Seed。全部216行horizon和216行paired-horizon保留；配对query绝对误差/SSE差见独立CSV，不做重叠查询显著性检验。RMSE差不可当作加和贡献；目标SSE表提供严格平方误差贡献。

各指标劣化包括formal改善但其它seed/尾部/endpoint或target等权反转，不能删异常后重报。少量目标SSE集中不自动代表特定工况故障，也不表示这些目标彼此独立。当前只完成Validation描述性稳健性补证，不能修复Assessment复用和独立Test资格。
'''
    write_report(phase,'O1_NINE_MODEL_ROBUSTNESS_COMPARISON.md',report)
    s62=metrics[metrics.seed==62]
    analysis='# O1 — Seed62三模型极端残差诊断\n\n'+md_table(s62,selected)
    analysis+='\n## 每模型Top10绝对残差查询（仅诊断，不剔除）\n\n'+md_table(extreme)
    for model in MODELS:
        m=s62[s62.model==model].iloc[0]
        t=targets[(targets.model==model)&(targets.seed==62)].sort_values(['SSE_mV2','target_row'],ascending=[False,True]).head(10)
        analysis+=f'\n## {model} — Top10目标SSE\n\n'+md_table(t,['target_row','multiplicity','SSE_mV2','mean_SE_mV2','formal_SSE_share','target_equal_MSE_share','horizon_steps'])
        analysis+=f'\n最大贡献target_row={int(m.max_SSE_target_row)}，multiplicity={int(m.max_SSE_target_multiplicity)}；正式SSE占比={m.top1_target_SSE_share:.8g}，Top5占比={m.top5_target_SSE_share:.8g}。全部queries未删除。\n'
    f2t=targets[(targets.model=='F2')&(targets.seed==62)].set_index('target_row')
    combined_formal=float(f2t.loc[[2247,2248],'formal_SSE_share'].sum())
    combined_equal=float(f2t.loc[[2247,2248],'target_equal_MSE_share'].sum())
    analysis+=f'''
## 本次精确定位结论

Seed62的F0/F1/F2最大SSE目标都为2247；2247仅有两个queries（origin2223/h24、origin2224/h23），2248仅一个（origin2224/h24）。三模型均在这些相同目标出现极端残差，而不是F2独有事件。F2最大绝对误差307.984904mV来自origin2224→target2247/h23；端点另含origin2223→2247/h24及2224→2248/h24。

F2这两目标合计占正式query SSE的{combined_formal*100:.6f}%，占target等权MSE的{combined_equal*100:.6f}%。端点子集只有205queries，低重数极端点权重较大；因此正式8.436399mV、endpoint29.817395mV、target等权27.053894mV可以同时成立。

F2 Seed62正式RMSE优于F0/F1，但endpoint略差于两者；target等权也劣于F0/F1。F2的P99更低不代表这些稀少极端query消失。F0的max为309.062721mV，F1为306.414695mV，F2为307.984904mV，不能称F2所有尾部指标最优。该定位不删除或重评分任何目标。
'''
    analysis+='''
## 数学来源与解释边界

Formal MSE = Σtarget SSE_target/4920；target等权MSE = Σtarget(SSE_target/m_target)/N_unique；endpoint只使用h24的205queries。单个目标的meanSE在低multiplicity时可被目标等权放大，相对正式query权重形成不同排序。表中formal_SSE_share与target_equal_MSE_share分别对应两个总MSE中的贡献，不能把它们混称同一个比例。

Top10查询给出origin/target/horizon定位；目标表列出全部指向该目标的步长。每模型都用相同规则检查，不只审F2。是否同一target集中可从三个max_SSE_target_row和Top10交集直接核对，不按误差大小重新选择评估域。缓存没有原始工况/物理观测，禁止解释为催化剂、膜、供气或特定电化学机制。本轮未追加工况读取。
'''
    write_report(phase,'O1_SEED62_EXTREME_ERROR_ANALYSIS.md',analysis)
    write_report(phase,'O1_PHASE_O_TERMINOLOGY_ERRATA.md','''
# Phase O文字勘误（新增附录，不改原文件）

基准O提交7553fcf3，`reports/O_07_F2_EXPERIMENT_EVIDENCE_AUDIT.md`曾写“447独立target rows”。正确为“447个唯一目标时刻”。447沿用已保存Assessment摘要，本轮未读取Assessment目标值或重新计算该数。

去重只说明不同target索引数量，不等于统计独立。此勘误不改变旧数值、不改写原O报告或提交历史、不新增独立样本数；O1也不以其Validation唯一目标数作独立性假设。
''')
    write_report(phase,'O1_PAPER_FIGURE_AND_TABLE_INDEX.md','''
# O1 — 论文候选图表

六组原创图，每组PNG300dpi+PDF矢量，均来自身份全部通过的九个历史Validation缓存：

1. Fig1_Validation_Horizon_RMSE_Three_Seeds：3个Seed panel、全部24horizon、统一y尺度。
2. Fig2_Validation_12h_Endpoint：三个Seed三模型h24 endpoint，起点数量205/run。
3. Fig3_Formal_vs_Target_Time_Equal：正式query与target等权敏感性；不改变主指标。
4. Fig4_Validation_Tail_and_Max_Error：P95/P99/max完整对照，max独立轴避免隐藏极端值。
5. Fig5_Validation_Target_SSE_Concentration：Top1/Top5目标贡献，比例0–1，不等同RMSE指标。
6. Fig6_Validation_Horizon_Equal_Seed_Mean_SD：等Seed均值和sampleSD，不是CI或ensemble。

数据总表 `O1_NINE_MODEL_VALIDATION_METRICS.csv`；逐horizon `O1_HORIZON_METRICS.csv`；等Seed及horizon汇总两个CSV；目标SSE、配对模型/步长/query差与Seed62极端queries完整保存。CSV列名明确mV、mV²、share和counts，全部真实数值未格式化成字符串。spreadsheets技能用于单位/来源/去重纪律；用户要求机器可复现CSV和Python计算，未额外创建Excel工作簿。scientific-figure-making技能用于语义色、印刷线型/纹理、统一尺度和矢量输出。

图题均Validation；没有Independent Test、Cross-Stack Generalization或RUL Accuracy主张。可用作条件性F2论文实证图；图表完整不改变独立Test资格。正式主指标和全部Seed保留，不能只引用最有利图。
''')
    write_report(phase,'O1_FINAL_RESEARCH_DECISION.md',f'''
# O1 — 最终研究判断

身份：`VALIDATION_CACHE_IDENTITY_PASS`。

稳健性：`{status}`。

证据：`VALIDATION_EVIDENCE_COMPLETED`，指本次九模型Validation缓存指标补证范围，不是整篇论文证据全面合格。

九run各205origins/4920queries，共同键/真值精确相同、来源哈希一致。Formal RMSE/MAE复现H，F2 learned扩展复现J。15项合成生产防线测试通过；它们与真实缓存身份通过分别记录。

F2可以报告整体正式Validation RMSE优势，但必须同时呈现其它指标和各Seed的变动；若状态为metric-dependent，不可说全部误差/全部权重/全部Seed一致优胜。gate独立收益仍未证明，Assessment复用和L确认性Test资格限制不变。唯一targets与重叠queries都不自动独立；没有p值、显著性或随机重采样论证。

现已完成O列出的B类Validation只读稳健性缺口；仍未解决C类独立训练gate/容量反事实、D类独立确认数据、物理适用性或健康寿命标签问题。未训练、未推理、未重选Seed、未读Assessment逐点/Test/raw/N1、未修改F2和历史报告。

可继续条件性论文初稿，以透明的RealHyFC Development动态电压实证定位。无需为稿件美观删异常/换指标；不得据补证自动启动新实验。提交O1并推送main后停止。
''')
