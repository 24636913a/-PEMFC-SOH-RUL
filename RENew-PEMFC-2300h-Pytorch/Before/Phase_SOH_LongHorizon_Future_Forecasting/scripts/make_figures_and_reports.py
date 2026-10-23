"""Render saved real results and write Chinese reports; never refit a model."""
import hashlib,json,os
from pathlib import Path
P=Path(__file__).resolve().parents[1]
os.environ['MPLCONFIGDIR']=str(P/'runtime/matplotlib')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

LABELS={'M0':'SOH persistence','M1':'SOH OLS trend','M2':'Calibrated P1S persistence','M3':'P1S 96h trend forecast'}
COLORS={'M0':'#767676','M1':'#B64342','M2':'#8BCF8B','M3':'#0F4D92'}
STYLES={'M0':':','M1':'--','M2':'-.','M3':'-'}
plt.rcParams.update({'font.family':['Arial','DejaVu Sans'],'font.size':12,'axes.linewidth':1.5,
    'axes.spines.top':False,'axes.spines.right':False,'legend.frameon':False,'pdf.fonttype':42,'savefig.facecolor':'white'})

def save(fig,name):
    fig.tight_layout(pad=1.6)
    for ext in ('png','pdf'):fig.savefig(P/f'figures/{name}.{ext}',dpi=300,bbox_inches='tight')
    plt.close(fig)

def md(df,cols):
    rows=['|'+'|'.join(cols)+'|','|'+'|'.join(['---']*len(cols))+'|']
    for _,row in df.iterrows():rows.append('|'+'|'.join(f'{row[c]:.6f}' if isinstance(row[c],(float,np.floating)) else str(row[c]) for c in cols)+'|')
    return '\n'.join(rows)

def write(name,text):
    path=P/name
    if path.exists():raise RuntimeError('REPORT_EXISTS_NO_OVERWRITE:'+name)
    with path.open('x',encoding='utf-8',newline='\n') as f:f.write(text.strip()+'\n')

def main():
    if not json.loads((P/'outputs/INDEPENDENT_VERIFICATION.json').read_text(encoding='utf-8'))['passed']:raise RuntimeError('VERIFY_FIRST')
    (P/'figures').mkdir(exist_ok=True)
    if list((P/'figures').glob('*.png')):raise RuntimeError('FIGURES_EXIST_NO_OVERWRITE')
    predictions=pd.read_csv(P/'predictions.csv',float_precision='round_trip');metrics=pd.read_csv(P/'metrics.csv',float_precision='round_trip')
    refs=pd.read_csv(P/'inputs/DEVELOPMENT_SOH_REFERENCES.csv',float_precision='round_trip')
    curves=pd.read_csv(P/'outputs/FORECAST_CURVE_PLOT_DATA.csv',float_precision='round_trip')
    window=pd.read_csv(P/'outputs/USED_P1S_TREND_WINDOWS.csv',float_precision='round_trip')
    origin=json.loads((P/'outputs/ORIGIN_INPUTS_AND_PARAMETERS.json').read_text(encoding='utf-8'))
    threshold=pd.read_csv(P/'outputs/THRESHOLD97_SCENARIOS.csv',float_precision='round_trip')
    config=json.loads((P/'configs/FROZEN_SOH_PROTOCOL.json').read_text(encoding='utf-8'))
    methods=list(LABELS);tasks=[r['task_id'] for r in origin]
    # Figure 1: measured anchors remain discrete; only prediction lines are connected.
    fig,axs=plt.subplots(1,3,figsize=(16,4.9),sharey=True)
    for ax,r in zip(axs,origin):
        o,t=r['origin_nominal_h'],r['target_nominal_h'];past=refs.loc[refs.nominal_h<=o]
        ax.scatter(past.nominal_h,past.Y*100,c='black',s=34,label='Known sparse references',zorder=5)
        target=refs.loc[refs.nominal_h==t].iloc[0];ax.scatter([t],[target.Y*100],c='black',marker='*',s=115,label='Held future reference',zorder=6)
        ax.axvspan(o,t,color='#CFCECE',alpha=.19);ax.axvline(o,color='black',lw=.7)
        for m in methods:
            line=curves.loc[(curves.task_id==r['task_id'])&(curves.method==m)]
            ax.plot(line.nominal_scenario_h,line.predicted_SOH*100,color=COLORS[m],ls=STYLES[m],lw=2,label=m)
            y=predictions.loc[(predictions.task_id==r['task_id'])&(predictions.method==m)].iloc[0].prediction
            ax.scatter([t],[y*100],color=COLORS[m],marker='o',s=30)
        ax.set(title=f"{r['task_id']}: {o} -> {t} nominal h ({t-o} h)",xlabel='Nominal checkpoint age (h)')
    axs[0].set_ylabel('Reference-condition SOH (%)');handles,labels=axs[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='lower center',ncol=6,bbox_to_anchor=(.5,-.025),fontsize=10)
    fig.suptitle('Sparse measured SOH and future forecasts | nominal-order replay, timing assumptions',fontsize=14)
    save(fig,'Fig1_Sparse_SOH_and_Future_Predictions')
    # Figure 2: absolute and signed errors, all three mandatory targets.
    fig,axs=plt.subplots(1,2,figsize=(14,5))
    x=np.arange(3);width=.18;hatches=['','//','..','xx']
    for j,m in enumerate(methods):
        rows=predictions.loc[predictions.method==m].set_index('task_id').loc[tasks]
        for ax,col in zip(axs,['absolute_error_pp','signed_error_pp']):
            ax.bar(x+(j-1.5)*width,rows[col],width,color=COLORS[m],edgecolor='black',linewidth=.7,hatch=hatches[j],label=m)
    for ax in axs:
        ax.set_xticks(x,[f"{r['origin_nominal_h']}->{r['target_nominal_h']}\n{r['nominal_horizon_h']} nominal h" for r in origin]);ax.axhline(0,c='black',lw=.7);ax.set_ylabel('SOH percentage points (pp)')
    axs[0].set_title('Absolute forecast error');axs[1].set_title('Signed forecast error (prediction - reference)')
    axs[0].legend(ncol=4);fig.suptitle('All three targets | nominal-order replay, timing assumptions',fontsize=14)
    save(fig,'Fig2_All_Target_Errors')
    fig,ax=plt.subplots(figsize=(8,5));values=metrics.set_index('method').loc[methods,'MAE_pp']
    bars=ax.bar(methods,values,color=[COLORS[m] for m in methods],edgecolor='black',linewidth=.9)
    for bar,m in zip(bars,methods):bar.set_hatch(hatches[methods.index(m)])
    for bar,value in zip(bars,values):ax.text(bar.get_x()+bar.get_width()/2,value+max(values)*.025,f'{value:.4f}',ha='center')
    ax.set_ylim(0,max(values)*1.2);ax.set_ylabel('Mean absolute error (SOH pp)');ax.set_title('Three fixed targets: equal-weight MAE\nNominal-order replay; no strict online qualification')
    ax.text(.02,-.16,'M0: SOH persistence | M1: SOH trend\nM2: P1S persistence | M3: P1S 96h trend',transform=ax.transAxes,fontsize=10)
    save(fig,'Fig3_Four_Method_MAE')
    # Figure 4 explicitly distinguishes actual observed P1S ages and assumed target coordinate.
    fig,axs=plt.subplots(1,3,figsize=(16,5.1))
    for ax,r in zip(axs,origin):
        w=window.loc[window.task_id==r['task_id']];latest=r['latest_R_actual_end_h'];target=r['target_nominal_h'];s=r['slope_R_per_h']
        ax.scatter(w.actual_measurement_end_h,w.R*100,c='black',s=30,label='Actual past P1S')
        grid=np.linspace(w.actual_measurement_end_h.min(),latest,50)
        ax.plot(grid,(r['R_OLS_intercept']+s*grid)*100,color=COLORS['M3'],lw=1.4,label='96h historical OLS fit')
        grid=np.linspace(latest,target,70);ax.plot(grid,(r['R_latest']+s*(grid-latest))*100,color=COLORS['M3'],lw=2,ls='--',label='Anchored extrapolation')
        ax.axvline(latest,c='black',lw=.9,ls=':',label='Latest legal measurement')
        ax.axvline(r['PAA_pretest_recorded_ophrs_h'],c='#767676',lw=1,ls='-.',label='Recorded pre-test cutoff')
        ax.axvspan(latest,target,color='#CFCECE',alpha=.2);ax.scatter([target],[r['R_pred_target']*100],c=COLORS['M3'],marker='X',s=50)
        ax.set(title=f"{r['task_id']}: n={r['trend_n_cycles']}; target {target} nominal h",xlabel='Actual past OpHrs2 / assumed target h',ylabel='Raw P1S retention (%)')
    handles,labels=axs[0].get_legend_handles_labels();fig.legend(handles,labels,loc='lower center',ncol=3,bbox_to_anchor=(.5,-.04),fontsize=9)
    fig.suptitle('M3 uses past measurements only | actual target operating age and release time unknown',fontsize=14)
    save(fig,'Fig4_P1S_Past_Trend_and_Extrapolation')
    # Real numeric tables and conclusions, not a new proposal.
    best=metrics.loc[metrics.MAE_pp.idxmin()];m0=float(metrics.loc[metrics.method=='M0'].iloc[0].MAE_pp)
    m2=float(metrics.loc[metrics.method=='M2'].iloc[0].MAE_pp);m3=float(metrics.loc[metrics.method=='M3'].iloc[0].MAE_pp)
    gain=(m0-m2)/m0*100;gain3=(m0-m3)/m0*100
    mt=md(metrics,['method','n_computed','MAE_pp','RMSE_pp','wins_vs_M0','improvement_vs_M0_percent','research_decision'])
    pt=md(predictions,['task_id','method','nominal_horizon_h','target_SOH','prediction','signed_error_pp','absolute_error_pp'])
    timing_table=pd.DataFrame([{k:r[k] for k in ['task_id','origin_nominal_h','PAA_pretest_recorded_ophrs_h','latest_P1S_cycle_id','latest_R_actual_end_h','latest_P1S_conservative_available_h','trend_n_cycles','extrapolation_h_under_coordinate_assumption']} for r in origin])
    tt=md(timing_table,list(timing_table))
    pair=[]
    for candidate,baseline in [('M2','M0'),('M3','M2'),('M3','M0'),('M3','M1')]:
        c=predictions.loc[predictions.method==candidate].set_index('task_id');b=predictions.loc[predictions.method==baseline].set_index('task_id')
        pair.append({'candidate':candidate,'baseline':baseline,'MAE_difference_pp':float(c.absolute_error_pp.mean()-b.absolute_error_pp.mean()),
            'improvement_percent':float((b.absolute_error_pp.mean()-c.absolute_error_pp.mean())/b.absolute_error_pp.mean()*100),
            'wins':int((c.absolute_error_pp<b.absolute_error_pp).sum()),'common_targets':3})
    comparisons=pd.DataFrame(pair);comparisons.to_csv(P/'outputs/PAIRED_METHOD_COMPARISON.csv',index=False,lineterminator='\n',float_format='%.17g')
    write('00_最终实验总结.md',f'''
# 未来SOH预测实验已完成（第一阶段）

已执行真实CPU预测：四种方法×三个固定目标，共12个未来预测，250/300/350h目标全部保留；14项必要测试和独立最小二乘重算通过。无GPU训练、未调窗口/阶数/映射，未打开晚期目标。

**这些是按标称检查点顺序的历史回放结果，不能称为已严格验证的实时在线250～350 operating hours预测。** 极化结果实际发布时间和精确OpHrs2缺失；例如标称850h前已有854.26343h记录。因此使用实际P1S时间筛选历史输入，但M3将目标标称小时替代未知实际目标小时，这项假设单独披露。

本回放误差最低为{best.method}，MAE={best.MAE_pp:.6f} SOH百分点。四组完整结果：

{mt}

M2相比M0的MAE变化为{gain:.2f}%改善（负数代表变差）；M3相比M0为{gain3:.2f}%改善。数字上的5%/2-of-3门槛与时间资格分开，未宣称正式CLEAR_GAIN。当前重点是看P1S信息更新和长期斜率外推的实际差别，不能把目标附近实际P1S代入校准后冒充预测。

参见[逐点预测](03_正式预测结果.md)、[误差比较](04_方法比较与结论.md)及[图3](figures/Fig3_Four_Method_MAE.png)。后续先解决时间身份与校准稳定性，不自动投入神经网络训练。
''')
    write('01_数据与预测任务定义.md',f'''
# 数据与时间身份

冻结SOH_V=V_pol(j*=1.0,t)/V_pol(j*=1.0,BoL)，nominal10-cell per-cell，BoL参考电压{config['BoL_reference_voltage_V']:.16f} V，标称坐标0h及SOH=1。BoL只是标准化参考原点，不伪造精确实验起始发布时间。1.6 A/cm²保留为既有辅助口径，本轮未重算/选择。

已读取CHM六个Development参考、18条冻结锚点映射、AHC前缀缓存、CTAE131合法事件、PAA middle_50测量、PAV时间表、H1已保存测量重建和PHI周期身份列。N1只解析SOH表第一条BoL记录，不读取后续晚期目标、不做混合表全文件SHA；不读raw860万行、mixedN1特征表或2000/2300目标。

{tt}

SOH-F1/F2/F3是本阶段任务ID，不是TFT模型F1/F2。起点850/1150/1500对应标称SOH检查点；目标1150/1500/1750，标称距离300/350/250。PAV pre-test cutoff分别为实测老化坐标的前测量记录，不是极化完成时刻/结果发布时间。P1S输入严格早于对应gap cutoff DateTime，该cutoff早于原点极化first_datetime；每个起点假设其本轮及既往结果在测试结束后可知，精确发布仍UNKNOWN。

M3历史时间来自H1的真实middle_50末测量OpHrs2；CTAE/PAA使用最后dwell结束作为更保守可获得记录，两者约差0.015h，均保存，不用文件名覆盖。96h从最新合法末测量往前计，至少4个不同完整物理周期；同堆周期不声称统计独立。参考前10周期不作为训练趋势观测，Vref在171.82091h之后可得。

**重要假设：**真实目标OpHrs2未知，M3公式中的target_time代入标称检查点数字。表中的外推跨度是该坐标代入假设的算术距离，不是经证实的实际未来运行时长。未知真实发布时间没有填造；严格在线三个任务均未获时间资格，回放数值仍完整报告。

所有具体周期ID、锚点参数、DateTime、原始和保守时间见outputs/ORIGIN_INPUTS_AND_PARAMETERS.json及USED_P1S_TREND_WINDOWS.csv。固定mapping每锚点三条全部保留，partial只用于原校准功能值，趋势仅complete；不按support删点。
''')
    write('02_四种预测方法.md','''
# 固定四种方法

- M0：目标SOH=起点最后已知参考SOH。
- M1：BoL和起点及以前全部参考点普通OLS：Yhat=a+b×目标标称小时。保留正斜率，不加单调、窗口或正则搜索。
- M2：同一原AHC数学规则拟合Y=aR+b、a≥0；R锚点是原三interval中位数。仅起点及以前映射校准，不使用全六点B2；对最新合法complete P1S估计当前proxy后保持到目标。
- M3：最近96实际operating hours、至少4个完整周期raw R拟合OLS斜率。Rhat_target=R_latest+slope_R×(nominal_target_h−actual_latest_middle50_end_h)，再使用与M2完全相同的a,b映射未来SOH。

原点本轮参考已知是在其极化结束、结果可获得假设之后，故校准可以包含原点参考；它没有包含未来目标参考。原AHC的“在本点前校准，再代入本点真实P1S”属于当前健康估计，本次是提前预测下一个参考，不能直接复制原缓存当新成绩。

M3锚定最新R，不使用OLS回归截距作为最新R替代；没有斜率负约束、全序列平滑、趋势剪裁或物理loss。M2/M3最新P1S、校准集合、目标时间口径完全相同，区别只在是否外推raw R。输入未来R、未来目标Y或实际未来lead均不允许。

代码先保存PREDICTIONS_BEFORE_EVALUATION.json（无真实未来Y），再由评价器附加三目标。测试修改未来Y、未来mapping或未来P1S后，同一原点预测不变。未来ref在后续任务成为过去时可以合法进入后续原点前缀。

误差单位：保持率误差乘100=SOH百分点（pp）；0.001保持率=0.1 pp。主MAE三目标等权，RMSE和signed/absolute逐点均保留；没有95%区间或显著性声明。
''')
    write('03_正式预测结果.md',f'''
# 三个固定未来目标的全部数值

本表“正式”指本轮冻结四法三目标真实执行的输出，不代表严格实时在线确认。所有行的scope=NOMINAL_CHECKPOINT_ORDER_REPLAY、strict_online_time_qualified=False。

{pt}

完整17位数值见predictions.csv。signed error=预测−参考，正数高估；绝对误差单位SOH百分点。未来目标实际SOH是保存的CHM参考值，没有用真实未来P1S进行预测。没有删点，没有添加额外起点、种子或重训。

M0/M1自检约0.2812/0.7714 pp与用户提供预期相符；这些检查没有硬编码进任何预测公式。独立numpy.linalg.lstsq与执行器centered covariance不同实现重算12值，最大差异见INDEPENDENT_VERIFICATION.json。

![稀疏SOH与预测](figures/Fig1_Sparse_SOH_and_Future_Predictions.png)

黑点/黑星是真实稀疏参考；连续彩线只是固定方法预测，不是实际连续SOH。图中小时是标称情景坐标，不能称精确运行时间。
''')
    write('04_方法比较与结论.md',f'''
# 四法比较及固定判断

{mt}

{md(comparisons,list(comparisons))}

MAE_difference_pp=候选−对照，负数更好。n=3是同一堆三个相关固定目标，不是三个独立寿命。这里没有缺失数值；若候选缺失，代码保留该行并拒绝完整三点判定，不能比较不同覆盖均值。

数字门槛固定为MAE至少比M0低5%且至少2/3目标更好；不能据结果修改。时间资格未知时，即使数字门槛满足也不得宣称正式CLEAR_GAIN；改善最多是LIMITED_SIGNAL，无法证明严格在线的评价为NOT_EVALUABLE。未改善者按NO_GAIN报告，而非追加参数寻找新优势。

![逐点误差](figures/Fig2_All_Target_Errors.png)

![MAE比较](figures/Fig3_Four_Method_MAE.png)

M3相对M2的对照隔离了长期外推贡献。不能用M2的当前SOH估计优势替代M3未来趋势预测优势，也不能以某单点取代全三点主MAE。
''')
    write('05_97百分比阈值衔接.md',f'''
# 97%阈值情景（仅探索）

主阈值0.97保持不变。仅使用本轮已经冻结的四法公式，不根据阈值调整预测模型，不读2000/2300目标。Development至1750尚未有SOH<0.97的观测，不能评价实际穿越精度或真实EOL/RUL。

{md(threshold,['task_id','method','horizon_nominal_h','scenario_origin_health','slope_health_per_scenario_h','status','remaining_h'])}

ALREADY_BELOW代表拟合/情景曲线在起点已低于，而不是实测原点故障。M1的OLS拟合起点值可以低于真实起点；M3的情景起点使用同一坐标代入公式，也不等同M2当前proxy。

斜率≥0或常数高于阈值没有有限向下crossing；只在各任务300/350/250h范围内报告有限root，范围外另列“范围内未跨越”，不得赋零寿命。等于阈值是边界，严格低于在边界之后；解析root是模型情景的边界而非观测首次事件。

这些是标称/实际坐标代入假设下的条件剩余时间，仅数学衔接。真实目标时间、恢复、工况变化、测量与映射误差未验证；不能以本表证明97%可准确预测。先证明健康轨迹预测优于Persistence，再考虑新独立资料的区间跨越评价。
''')
    write('06_研究限制与下一步.md',f'''
# 研究限制和下一步

本轮完成实际CPU数值和PNG/PDF图表，不停留于规划。最低回放MAE是{best.method}（{best.MAE_pp:.6f} pp）；是否值得继续看完整四法及三点对照，而不是升级网络名称。

主要限制：极化缺实际OpHrs2和结果发布时间；M3用标称目标坐标替代实际年龄；96h局部趋势被外推约250～350h，未来恢复/工况和校准稳定性未知；固定PAA映射部分median不在共同支持，保留全部；单堆稀疏三目标、共享BoL与年龄混杂，不能证明跨堆泛化。

先补源时间身份，明确标称原点的真正预测发布时刻及计划中的未来运行小时；利用已保存结果定位是当前proxy校准还是局部斜率外推造成误差。对未胜简单模型的M3不改变96h、目标或公式救回结果，不开展GPU大搜索。

只有更多独立、合法时间可获得的参考观测，以及同预算简单对照证据支持P1S未来外推时，才值得另立神经网络研究；目前优先简单模型和数据资格。现有TFT动态双路径模型保持冻结，本阶段没有把其12h电压输出改成SOH。

![P1S历史与外推](figures/Fig4_P1S_Past_Trend_and_Extrapolation.png)

图4黑点是使用的历史raw P1S；实线是过去96h拟合，虚线是锚定最新R的外推。灰区不是实际观测，目标横坐标采用标称代入，pre-test cutoff仅已有记录不是极化完成年龄。
''')
    write('07_执行与Git记录.md',f'''
# 执行和成果身份

起点main本地/实时远程均为{config['base_commit']}，工作区干净。所有新文件位于本独立阶段；历史AHC/PHI/CTAE/CMRF和模型权重未修改。没有GPU、raw重建、晚期目标打开或模型搜索。

执行：`python -B -m unittest discover -s tests -v` → `python -B scripts/run_soh_experiment.py --freeze` → `--run` → `python -B scripts/verify_saved_results.py` → `python -B scripts/make_figures_and_reports.py`。freeze绑定8个小型来源文件全SHA、BoL单记录逻辑指纹和全部脚本/测试；首次运行ledger已完成，禁止删除后重跑。

测试和控制台保存outputs，必要测试14项。第一次预冻结测试发现N1 CSV含UTF-8 BOM，修正reader为utf-8-sig后14项通过；保留TEST_CONSOLE.log失败现场及TEST_CONSOLE_ATTEMPT2.log通过记录，没有正式实验重跑。独立重算使用不同OLS实现，不只调用原方法。输入时间/周期ID及每个原点的参数保存JSON。模型数学/数据访问由固定源码和protocol哈希记录；报告/图表均使用已保存结果，没有重新拟合。

读取审计见DATA_ACCESS_AUDIT.json；未读取PHI全六点B2/SG/isotonic/PCHIP列作为输入，只消耗周期身份列。N1混合SOH表只解析第一BoL记录，不做全文件SHA。source SHA在configs/FROZEN_SOH_PROTOCOL.json，历史正式代码/报告作为只读依据，未调用旧runner。

输出CSV和PNG/PDF由真实预测生成，最终文件哈希清单见outputs/DELIVERABLE_MANIFEST.json。提交前检查历史tracked diff为空、图表视觉QA及Git暂存字节身份；提交/推送回执本地outputs/GIT_RECEIPT.local.json（忽略），最终回复提供实际SHA，避免为了回填自身SHA重复提交。禁止强推，远程更新需先审定。
''')
    write('README.md','''
# RealHyFC future SOH forecasting, first stage

Four fixed CPU methods, three fixed future targets. Read 00_最终实验总结.md first.
All results are nominal checkpoint-order replay with explicit time assumptions.
No strict online prediction, late-target evaluation, GPU or new neural training is claimed.
Completed protocol/run IDs must not be silently restarted.
''')
    hashes={str(p.relative_to(P)).replace('\\','/'):hashlib.sha256(p.read_bytes()).hexdigest() for p in P.rglob('*') if p.is_file() and not p.is_relative_to(P/'runtime') and p.name!='GIT_RECEIPT.local.json'}
    with (P/'outputs/DELIVERABLE_MANIFEST.json').open('x',encoding='utf-8',newline='\n') as f:json.dump(hashes,f,ensure_ascii=False,indent=2);f.write('\n')
    print('Generated eight reports and four figures in PNG/PDF; no new fitting.')

if __name__=='__main__':main()
