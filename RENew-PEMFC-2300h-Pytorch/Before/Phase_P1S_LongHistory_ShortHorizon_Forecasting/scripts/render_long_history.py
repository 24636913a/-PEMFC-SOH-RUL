"""Actual saved results only: figures, Chinese reports and comparison tables."""
from pathlib import Path
import os,json,hashlib
P=Path(__file__).resolve().parents[1]
os.environ['MPLCONFIGDIR']=str(P/'runtime/matplotlib')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
COLORS={'M0':'#767676','M96':'#B64342','M192':'#0F4D92'}
plt.rcParams.update({'font.family':['Arial','DejaVu Sans'],'font.size':12,'axes.linewidth':1.4,
 'axes.spines.top':False,'axes.spines.right':False,'legend.frameon':False,'pdf.fonttype':42,'savefig.facecolor':'white'})

def read(name):return pd.read_csv(P/name,float_precision='round_trip')
def md(frame,columns):
    lines=['|'+'|'.join(columns)+'|','|'+'|'.join(['---']*len(columns))+'|']
    for _,row in frame.iterrows():
        values=[]
        for key in columns:
            v=row[key];values.append('不可评价' if pd.isna(v) else (f'{v:.6f}' if isinstance(v,(float,np.floating)) else str(v)))
        lines.append('|'+'|'.join(values)+'|')
    return '\n'.join(lines)
def report(name,text):
    path=P/name
    with path.open('x',encoding='utf-8',newline='\n') as out:out.write(text.strip()+'\n')
def save(fig,name):
    fig.tight_layout(pad=1.5)
    for ext in ['png','pdf']:fig.savefig(P/f'figures/{name}.{ext}',dpi=300,bbox_inches='tight')
    plt.close(fig)

def main():
    assert json.loads((P/'outputs/INDEPENDENT_VERIFICATION.json').read_text(encoding='utf-8'))['passed']
    if (P/'00_研究结论_通俗中文版.md').exists():raise RuntimeError('DELIVERABLES_ALREADY_EXIST')
    (P/'figures').mkdir(exist_ok=True)
    pred=read('predictions.csv');fold=read('metrics_fold.csv');summary=read('metrics_summary.csv');counts=read('sample_counts.csv')
    p1s=read('inputs/CAUSAL_P1S_ACTUAL_MEASUREMENT_IDENTITY.csv');models=json.loads((P/'outputs/FITTED_RIDGE_MODELS.json').read_text(encoding='utf-8'))
    decision=json.loads((P/'outputs/RESEARCH_DECISION.json').read_text(encoding='utf-8'));config=json.loads((P/'configs/FROZEN_LONG_HISTORY_PROTOCOL.json').read_text(encoding='utf-8'))
    comparisons=[]
    for h in ['H1','H3']:
        pivot=fold.loc[(fold.horizon==h)&fold.n_validation.gt(0)].pivot(index='fold',columns='model',values='MAE_mV_equivalent')
        for candidate,base in [('M96','M0'),('M192','M0'),('M192','M96')]:
            for f in ['fold_1','fold_2','fold_3']:
                ok=f in pivot.index
                comparisons.append({'horizon':h,'scope':f,'candidate':candidate,'baseline':base,
                    'difference_MAE_mV':float(pivot.loc[f,candidate]-pivot.loc[f,base]) if ok else None,
                    'relative_improvement_percent':float((pivot.loc[f,base]-pivot.loc[f,candidate])/pivot.loc[f,base]*100) if ok else None,
                    'status':'COMPARABLE_SAME_POINTS' if ok else 'NOT_EVALUABLE'})
            means=pivot.mean();comparisons.append({'horizon':h,'scope':'AVAILABLE_2_FOLDS_DIAGNOSTIC_ONLY','candidate':candidate,'baseline':base,
                'difference_MAE_mV':float(means[candidate]-means[base]),'relative_improvement_percent':float((means[base]-means[candidate])/means[base]*100),'status':'NOT_FORMAL_3_FOLD_RESULT'})
    comparison=pd.DataFrame(comparisons);comparison.to_csv(P/'outputs/PAIRED_COMPARISONS.csv',index=False,float_format='%.17g',lineterminator='\n')
    # Figure 1 uses a fixed first legal H1 Fold3 origin, chosen without errors.
    origin=pred.loc[(pred.horizon=='H1')&(pred.fold=='fold_3')&(pred.model=='M0')].iloc[0];t=float(origin.origin_measurement_end_h)
    fig,axs=plt.subplots(1,2,figsize=(13,4.8),sharey=True)
    for ax,L,c in zip(axs,[96,192],[COLORS['M96'],COLORS['M192']]):
        history=p1s.loc[(p1s.measurement_end_h<=t+1e-9)&(p1s.measurement_end_h>=t-L-1e-9)]
        ax.axvspan(t-L,t,color=c,alpha=.10);ax.scatter(history.measurement_end_h,history.R*100,c=c,s=35)
        ax.axvline(t,c='black',ls='--',lw=1);ax.set_xlim(t-200,t+5)
        ax.set(title=f"L{L}: n={len(history)}, observed span {np.ptp(history.measurement_end_h):.2f} h",xlabel='Actual past P1S measurement end (OpHrs2 h)')
        ax.text(.03,.06,f'Origin: {origin.origin_cycle_interval_id}\nt = {t:.2f} h',transform=ax.transAxes,fontsize=10)
    axs[0].set_ylabel('Raw P1S retention (%)');fig.suptitle('Same forecast origin, two past-only history windows | no gap filling')
    save(fig,'Fig1_96h_vs_192h_Observed_History')
    for h in ['H1','H3']:
        fig,axs=plt.subplots(1,3,figsize=(16,4.8))
        for ax,f in zip(axs,['fold_1','fold_2','fold_3']):
            g=pred.loc[(pred.horizon==h)&(pred.fold==f)]
            if g.empty:
                ax.set_axis_off();ax.text(.5,.55,f'{f}\nNO COMMON VALIDATION ORIGINS',ha='center',va='center',transform=ax.transAxes,fontsize=13)
                ax.text(.5,.28,'192h count/span rule not satisfied\nNo score or zero-valued prediction invented',ha='center',transform=ax.transAxes,fontsize=10)
                continue
            actual=g.loc[g.model=='M0'];ax.scatter(actual.target_measurement_end_h,actual.true_R*100,c='black',s=24,label='Observed target P1S',zorder=5)
            for model in ['M0','M96','M192']:
                rows=g.loc[g.model==model];ax.plot(rows.target_measurement_end_h,rows.predicted_R*100,c=COLORS[model],lw=1.6,label=model)
            ax.set(title=f'{h} {f}: n={len(actual)} matched targets',xlabel='Actual target measurement end (h)\nEvaluation coordinate only')
        axs[0].set_ylabel('Raw P1S retention (%)');axs[0].legend(ncol=2,fontsize=10)
        fig.suptitle(f'{h}: direct physical-cycle forecasts | proxy, not polarization SOH',fontsize=14)
        save(fig,f'Fig{2 if h=="H1" else 3}_{h}_Prediction_Curves')
    fig,axs=plt.subplots(1,2,figsize=(13,5));width=.22;x=np.arange(3)
    for ax,h in zip(axs,['H1','H3']):
        for i,model in enumerate(['M0','M96','M192']):
            rows=fold.loc[(fold.horizon==h)&(fold.model==model)].set_index('fold').loc[['fold_1','fold_2','fold_3']]
            ax.bar(x+(i-1)*width,rows.MAE_mV_equivalent,width,color=COLORS[model],edgecolor='black',linewidth=.7,hatch=['','//','xx'][i],label=model)
        ax.set_xticks(x,['Fold 1','Fold 2\nnot evaluable','Fold 3']);ax.text(1,.04,'No data',ha='center',transform=ax.get_xaxis_transform(),fontsize=10)
        ax.set(title=f'{h}: same-sample fold MAE',ylabel='P1S error (mV-equivalent)');ax.legend(ncol=3)
    fig.suptitle('No complete three-fold mean: Fold 2 cannot meet frozen 192h coverage',fontsize=13)
    save(fig,'Fig4_Matched_Fold_MAE')
    fig,axs=plt.subplots(1,2,figsize=(12,4.8))
    for ax,h in zip(axs,['H1','H3']):
        for model in ['M0','M96','M192']:
            e=np.sort(pred.loc[(pred.horizon==h)&(pred.model==model)].absolute_error_mV_equivalent.to_numpy())
            ax.step(e,np.arange(1,len(e)+1)/len(e),where='post',c=COLORS[model],label=model,lw=1.8)
        ax.set(title=f'{h}: pooled observed errors (descriptive)',xlabel='Absolute P1S error (mV-equivalent)',ylabel='Empirical fraction');ax.legend()
    fig.suptitle('Available Folds 1 and 3 only; no confidence interval or independence claim',fontsize=13)
    save(fig,'Fig5_Actual_Error_Distributions')
    fig,axs=plt.subplots(1,2,figsize=(12,4.8));width=.22
    for ax,h in zip(axs,['H1','H3']):
        g=counts.loc[(counts.horizon==h)&(counts.role=='validation')].set_index('fold').loc[['fold_1','fold_2','fold_3']]
        for i,(col,label,color) in enumerate([('original_n','Original frozen scope','#CFCECE'),('L96_history_eligible_n','96h only',COLORS['M96']),('common_history_n','Matched 96h + 192h',COLORS['M192'])]):
            ax.bar(x+(i-1)*width,g[col],width,label=label,color=color,edgecolor='black',linewidth=.6)
        ax.set_xticks(x,['Fold 1','Fold 2','Fold 3']);ax.set(title=f'{h}: validation availability',ylabel='Number of origins');ax.legend(fontsize=8)
    save(fig,'Fig6_History_Availability_and_Lost_Origins')
    # Actual lead reporting is allowed for evaluation, never a feature.
    leads=read('outputs/ACTUAL_LEAD_DESCRIPTIVE_ONLY.csv');leadstats=[]
    for h,g in leads.groupby('horizon'):
        v=g.actual_target_end_minus_origin_end_h.to_numpy()
        leadstats.append({'horizon':h,'n':len(v),'mean_h':v.mean(),'median_h':np.median(v),'min_h':v.min(),'max_h':v.max(),'P95_h':np.quantile(v,.95)})
    pd.DataFrame(leadstats).to_csv(P/'outputs/ACTUAL_LEAD_SUMMARY.csv',index=False,float_format='%.17g',lineterminator='\n')
    st=md(summary,['horizon','model','available_folds','formal_three_fold_equal_MAE_mV','descriptive_available_fold_equal_MAE_mV','fold3_MAE_mV'])
    best={h:summary.loc[summary.horizon==h].sort_values('descriptive_available_fold_equal_MAE_mV').iloc[0] for h in ['H1','H3']}
    report('00_研究结论_通俗中文版.md',f'''
# 长历史、短未来：实际实验已完成

预测对象是**原始P1S运行电压保持率健康代理**，不是极化参考SOH。执行H1/H3独立CPU Ridge拟合，96/192h同一五特征模板，同训练与验证样本、固定四alpha。没有GPU、TFT/TimesNet训练、长时间SOH外推、软物理重启或Test读取。

**预设规则下第2fold验证样本为空，不能认定完整三fold的长历史收益。** 不放宽10观测/70%跨度，不移动时间边界、不用更早数据填补。全部三个fold的模型均已拟合，但第2fold没有合法同样本验证成绩，MAE留空而非0。以下平均只对有数据的第1、第3fold等权，属于诊断。

{st}

在可用两fold诊断中：H1最低误差{best['H1'].model}，{best['H1'].descriptive_available_fold_equal_MAE_mV:.6f} mV-equivalent；H3最低{best['H3'].model}，{best['H3'].descriptive_available_fold_equal_MAE_mV:.6f}。192h相对96h的H1变化为{decision['H1']['diagnostic_M192_vs_M96_relative_improvement_percent']:.2f}%改善（负数更差），H3为{decision['H3']['diagnostic_M192_vs_M96_relative_improvement_percent']:.2f}%。不能把这两fold诊断写成预设三fold门槛成功。

完整逐点结果、fold3、P95/max及alpha见CSV和03/04报告。先读[是否有收益](05_长历史是否有收益.md)。M2/SOH接口已提供，现有参考发布证据不足，未制造SOH数值或真实SOH MAE。此轮到此停止，不自动增加窗口或复杂模型。
''')
    report('01_实验定义与数据来源.md',f'''
# 实验定义和冻结来源

R=V_P1S/{config['Vref_V']:.11f}，参考在171.82091h后可得，前10complete只暖启动，131因果事件。H1=origin raw cycle+1；H3=+3且全部中间raw cycle完整。来源H1/H3冻结样本292/248行、原CTAE/PAA测量、H1已保存实际middle_50时间重建、PHI周期身份及Development raw-cycle目录。没有重解析Raw、mixed N1读取或晚期目标。

原始三个fold角色不改：标签测量start≥角色下界、end<角色上界。来源SHA和执行代码SHA在configs/FROZEN_LONG_HISTORY_PROTOCOL.json，原始authority在inputs；保留同一origin在多个expanding folds的身份，不冒充独立电堆。

输入窗口终点用实际P1S测量结束小时，观测必须已结束且参考可得；origin本人包含。原CTAE保守结束记录也留存，不用名义周期9.5h取代真实间隔。年龄只用于构造历史窗口和斜率、不得直接作为模型特征；target小时仅定义标签/隔离/绘图评价。

两种长度共同输入资格决定同一个Train/Validation集合，依据数量/跨度，不看目标误差。额外执行标签可获得性purge：最终Train标签必须在首个共同Validation origin前结束；inner fit标签必须在inner选择首origin前结束。两模型同时应用，原样本身份保留、剔除原因另存。原H3仅目标区间划分的边界部分会产生模型参数晚于早期origin的风险，本轮关闭该风险，不改变旧历史结果。

![样本覆盖](figures/Fig6_History_Availability_and_Lost_Origins.png)

完整数量变化见sample_counts.csv；原历史Persistence全验证scope逐点身份/单位核验通过，新的排名仅用共同scope，不与历史完整scope的均值直接相减。
''')
    report('02_96h与192h历史特征.md','''
# 共同五特征与过去窗口

特征顺序固定：origin最新R、窗口mean R、population std R(ddof=0)、R对实际测量小时OLS斜率、origin R−窗口median R。除斜率单位h^-1外，均无量纲保持率。标准化仅inner-fit选alpha或final outer-Train refit，不用Validation。目标DeltaR保持原无量纲，不另标准化。

L96至少5观测和67.2h跨度；L192至少10观测和134.4h跨度。最早到最新真实观测的跨度计算，不能以cycle数量×9.5h替代。窗口边界[t−L,t]，保留全部合法观测，不按support删点，不插值/填gap，不加第三长度。

Ridge固定alpha=0.0001/0.01/1/100；后15%按ceil切出inner选择段，早85%在标签时间purge后fit scaler及模型。以inner MAE最小选alpha，同分选小alpha；随后完整合法outer-Train重新拟合。最终参数、尺度与inner评分均保存，五列之外没有future lead/target/工况/B2/平滑特征。

![两个窗口](figures/Fig1_96h_vs_192h_Observed_History.png)

图中只有真实历史P1S测量点与窗口范围，未虚构连续SOH轨迹。模型M0/M96/M192未来R分别是origin R、origin R+96h Ridge DeltaR、origin R+192h Ridge DeltaR；H1/H3独立拟合，不递推H1生成H3。
''')
    for h,index in [('H1','03'),('H3','04')]:
        f=fold.loc[fold.horizon==h];s=summary.loc[summary.horizon==h]
        report(f'{index}_{h}预测结果.md',f'''
# {h}实际预测结果

{md(f,['fold','model','status','n_train','n_validation','MAE_R','RMSE_R','MAE_mV_equivalent','RMSE_mV_equivalent','P95_abs_error_mV_equivalent','max_abs_error_mV_equivalent'])}

{md(s,['model','available_folds','formal_three_fold_equal_MAE_mV','descriptive_available_fold_equal_MAE_mV','pooled_n_validation','pooled_P95_abs_error_mV','pooled_max_abs_error_mV','fold3_MAE_mV'])}

三fold正式平均无法计算；有数据两fold平均及pooled仅诊断，分母明确。空fold每方法没有任何预测行，未补造，不记零误差。每fold预测CSV位于outputs/predictions，合并predictions.csv。n_train是同样本与参数时间purge后的数量；全部删除记录保留。

![{h}预测曲线](figures/Fig{2 if h=='H1' else 3}_{h}_Prediction_Curves.png)

图横轴未来测量时刻仅评价显示，绝不作为输入。黑点是实际离散P1S观测，不是极化SOH；H1/H3不称固定10/30h。实际lead统计见ACTUAL_LEAD_SUMMARY.csv，含时间单位/分布，仅描述。
''')
    report('05_长历史是否有收益.md',f'''
# 长历史收益判断

正式H1：**{decision['H1']['formal_status']}**；辅助H3：**{decision['H3']['formal_status']}**。缺第2fold，未达到预设完整三fold评价资格，不宣称CLEAR_GAIN或统计显著性。

{md(comparison,['horizon','scope','candidate','baseline','difference_MAE_mV','relative_improvement_percent','status'])}

负difference表示候选较好，正relative_improvement表示改善。正式门槛不改：H1三fold等权MAE低于M96至少5%、至少2fold胜M96且低于Persistence；缺fold不能用两fold均值代替。H3独立报告，不顶替H1主结论。

![每fold比较](figures/Fig4_Matched_Fold_MAE.png)

![误差分布](figures/Fig5_Actual_Error_Distributions.png)

若可用fold中观察到局部收益，只能称有限同堆诊断信号；样本损失、共同scope、fold3和极端误差均需一并解释。没有按结果改192/96h、特征、alpha网格或数据划分，没有挑最漂亮的fold。
''')
    report('06_M2与SOH衔接.md','''
# P1S预测到参考SOH的接口

本实验预测raw P1S健康代理。无周期级真实极化SOH目标，所以不报告真实SOH预测MAE，也不把retention直接改名SOH。

提供causal_soh_bridge接口：输入模型预测的未来R及至少3个当时已可知、原冻结映射的参考锚点；实际result_release_datetime必须有证据且≤origin_datetime，才按原AHC非负affine数学形成a*Rhat+b。未知发布时间或不足锚点返回不可用；不使用全六锚点后验B2，不用未来真实R。

目前已有资料没有精确参考发布时间，因此本轮实际auxiliary status全为NO_CERTIFIED_ANCHOR_RELEASE_TIME，SOH数字、真实目标、MAE均留空。接口合成测试验证数学和可获得性检查，不能冒充真实SOH成功。它不参加M0/M96/M192精度排名。

以后“短期代理→参考SOH→97%条件阈值”必须逐项验证时间、映射和参考测量。此轮不执行97%外推，不重跑250～350h实验，不定义真实EOL。
''')
    report('07_下一阶段建议.md',f'''
# 下一步与执行记录

先审阅本轮可用fold数值和样本可获得性，保留Persistence基线。第2fold的192h覆盖不足是实际数据限制，不能用调整阈值、增加窗口/模型或GPU容量解决。本轮不进入自动复杂网络训练；需要完整时间合格覆盖或另行预注册新数据问题才可推进。

实际执行：单元测试→freeze→一次--run→独立numpy重算→基于保存结果生成6张PNG/PDF和8报告。所有最终模型12个、固定inner-grid加final refit共60次Ridge拟合；这属于规定CPU模型选择，不是神经网络。单元测试的拟合仅检验隔离和scaler，不作为正式成绩。正式模型不重跑、失败现场不删除。

主输入/输出身份见protocol及HISTORY_WINDOW_IDENTITIES.json；alpha/训练样本/scale/coefficients均保存FITTED_RIDGE_MODELS.json。第一段历史H1 Ridge含实际future lead，只作旧历史，不用于本轮输入或严格causal排名。旧PAA/CTAE/H1/H3/TFT权重与报告未修改。

执行入口：python -B scripts/run_long_history.py --freeze；--run；python -B scripts/verify_long_history_results.py；python -B scripts/render_long_history.py。已有冻结协议/ledger会拒绝重复运行。结果缺fold仍标为完成实际执行，科学门槛另判NOT_EVALUABLE，不伪造三fold平均。

源SHA/代码SHA固定；最终哈希及图表QA见FINAL_INTEGRITY_AUDIT.json/DELIVERABLE_MANIFEST.json。Git提交/网络回执本地GIT_RECEIPT.local.json（忽略）；最终交接给实际SHA。保护历史结果、仅本阶段提交，不force push。到此停止，无新后台训练或自动下一轮研究。
''')
    report('README.md','''
# Long-History Short-Horizon P1S forecasting

Two fixed history windows and three matched methods; H1 primary, H3 auxiliary.
Actual CPU fitting completed, Fold2 validation unavailable under frozen 192h rules.
No complete three-fold gain claim; read 00_研究结论_通俗中文版.md.
No SOH ground truth, new neural/GPU training or late/Test access.
''')
    print('Rendered 6 real-result PNG/PDF figures and 8 Chinese reports; no new fitting.')

if __name__=='__main__':main()
