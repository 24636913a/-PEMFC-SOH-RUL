"""准备性资格/图表/中文报告：只读取本轮真实预测与既有冻结资格。"""
import hashlib,json,sys,zipfile,xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

TASK=Path(__file__).resolve().parents[1];GUIDE=TASK.parents[1];REPO=GUIDE.parent
RUN=TASK/'输出结果/正式计算';TABLE=TASK/'输出结果/表格';FIG=TASK/'输出结果/图片';PLOT=TASK/'输出结果/绘图数据'
LABEL={'A_Persistence':'目标域保持','A_TargetBounded':'目标域全历史有界率','B_FC1Direct':'FC1直接迁移','C_FC1PriorAdapted':'FC1先验＋早期适配','D_Target2B_Bounded':'同截止2B型有界率','D_Target2B_SlowReversible':'同截止2B型慢变＋可逆'}
COLORS=['#767676','#3775BA','#B64342','#9A4D8E','#42949E','#3C8D6C']
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def csv(df,p):df.to_csv(p,index=False,encoding='utf-8-sig',float_format='%.17g')
def md(df):
    def cell(x):
        if pd.isna(x):return 'NA'
        if isinstance(x,(float,np.floating)):return f'{x:.6f}'
        return str(x).replace('|','/').replace('\n',' ')
    lines=['| '+' | '.join(map(str,df.columns))+' |','| '+' | '.join(['---']*len(df.columns))+' |']
    return '\n'.join(lines+['| '+' | '.join(cell(x) for x in r)+' |' for r in df.itertuples(index=False,name=None)])
def write(name,text):(TASK/name).write_text(text.strip()+'\n',encoding='utf-8')
def main():
    protocol=json.loads((TASK/'配置/阶段2C冻结协议.json').read_text(encoding='utf-8'))
    pred=pd.read_csv(RUN/'表格/各方法逐起点预测.csv');scores=pd.read_csv(RUN/'表格/电压_SOH及损失误差.csv');main=scores[scores.scope=='COMMON_ALL'];gains=pd.read_csv(RUN/'表格/迁移收益与负迁移.csv');params=pd.read_csv(RUN/'表格/模型参数与速率.csv');source=json.loads((RUN/'模型记录/FC1源端参数.json').read_text(encoding='utf-8'))
    real=pd.read_csv(TASK/'数据准备/RealHyFC循环观测.csv');real=real[real.scheme=='B60'];diag=pd.read_csv(TASK/'数据准备/稀疏诊断观测.csv')
    oldpairs=pd.read_csv(GUIDE/'物理约束退化演化/阶段2B/输出结果/表格/诊断与运行历史配对及排除.csv')
    # 每条历史诊断单独一对；只按已冻结两参考支持及48日历小时，不搜索新阈值。
    pairs=[];elig=[];feature=[]
    for sc in protocol['scenarios']:
        hist=real[real.valid&(real.available_h<=sc['cutoff_h'])];ds=diag[(diag.scheme=='B60')&diag.primary&diag['标称小时'].isin(sc['diagnostic_nominal_h'])]
        eligible=[]
        for d in ds.to_dict('records'):
            end=pd.Timestamp(d['实际测试结束']);past=hist[hist.available_ns<=end.value].sort_values('available_ns')
            lag=None;last=None
            if len(past):last=past.iloc[-1];lag=(end.value-int(last.available_ns))/3.6e12
            ok=bool(d['双参考支持']) and last is not None and lag<=48
            reason='COMPLETION_ASSUMPTION_ONLY' if ok else 'NO_PRIOR_OPERATING_POINT' if last is None else 'LAG_GT_48H_OR_UNSUPPORTED'
            row=dict(scenario=sc['name'],diagnostic=d['文件名'],nominal_h=d['标称小时'],diagnostic_end=str(end),operating_upper_h=d['累计小时上界'],both_reference_supported=bool(d['双参考支持']),D_low=d['D_low_V'],D_ref=d['D_ref_V'],pair_eligible_completion_assumption=ok,operating_id=None if last is None else int(last.id),operating_h=None if last is None else last.available_h,calendar_lag_h=lag,reason=reason,release='UNKNOWN',strict_online=False)
            pairs.append(row)
            feature.append(dict(scenario=sc['name'],**{k:row[k] for k in ['diagnostic','nominal_h','D_low','D_ref','both_reference_supported','diagnostic_end','operating_upper_h']}))
            if ok:eligible.append(row)
        n=len(eligible);rank=0;cond=None;rank1=0;cond1=None
        if n:
            x=np.array([[e['D_ref'],e['D_low']] for e in eligible]);std=x.std(axis=0);z=(x-x.mean(axis=0))/np.where(std>0,std,1)
            a=np.column_stack([np.ones(n),z]);rank=int(np.linalg.matrix_rank(a));cond=float(np.linalg.cond(a)) if rank==3 else None
            a1=a[:,:2];rank1=int(np.linalg.matrix_rank(a1));cond1=float(np.linalg.cond(a1)) if rank1==2 else None
        elig.append(dict(scenario=sc['name'],operating_history=len(hist),diagnostics=len(ds),both_supported=int(ds['双参考支持'].sum()),completed_past_pairs=n,one_dim_parameters=2,min_1D_pairs=6,rank_1D=rank1,condition_1D=cond1,two_dim_parameters=3,min_2D_pairs=9,rank_2D=rank,condition_2D=cond,past_diagnostic_transitions=max(0,len(ds)-1),min_transitions=6,mapping_1D='NA_INSUFFICIENT_PAIRS',mapping_2D='NA_INSUFFICIENT_PAIRS',state_update='NOT_QUALIFIED',bounded_network16='NOT_QUALIFIED'))
    sourcepairs=oldpairs[(oldpairs.dataset=='FC1')&oldpairs.train_pair]
    csv(pd.DataFrame(pairs),TABLE/'各截止诊断与运行配对及拒绝.csv');csv(pd.DataFrame(elig),TABLE/'一维二维迁移与16单元网络资格.csv');csv(pd.DataFrame(feature),TABLE/'历史双参考诊断清单.csv');csv(sourcepairs,TABLE/'FC1源训练期名义诊断配对.csv')
    complexity=pd.DataFrame([
      ['A_Persistence','目标末次损失1个；速率固定0','0','1个观察锚点','无','不拟合未来状态'],
      ['A_TargetBounded','目标全历史斜率1个；经验cap','0','1个观察锚点','非负有界','截距仅OLS辅助、不作为额外预测自由度'],
      ['B_FC1Direct','源斜率1个及源cap；目标自身BoL','1个源速率','1个目标观察锚点','源模型固定','不使用目标早期漂移'],
      ['C_FC1PriorAdapted','源与目标各1个斜率；各自cap','1个源速率','1个目标观察锚点','源权重0.25固定','不拟合迁移权重'],
      ['D_Target2B_Bounded','目标q_prior、cap、accel共3个统计量','0','L锚点＋q滤波状态','k两个候选Train内部选择','无网络权重'],
      ['D_Target2B_SlowReversible','目标q_prior、cap、accel、noise、rev_scale共5个统计量','0','L,q,R三个因果滤波状态','tau三个候选Train内部选择','状态不等于自由拟合物理损伤参数']
    ],columns=['method','估计量','源输入','状态','固定或内选超参数','说明'])
    csv(complexity,TABLE/'参数规模与模型来源.csv')
    # 原方案文本与公式一并提取，用于可追溯核对；不改写原Word。
    doc=GUIDE/'step1/燃料电池三档研究方案.docx'
    with zipfile.ZipFile(doc) as z:root=ET.fromstring(z.read('word/document.xml'))
    texts=[]
    for p in root.iter('{http://schemas.openxmlformats.org/wordprocessingml/2006/main}p'):
        text=''.join(n.text or '' for n in p.iter() if n.tag.endswith('}t'))
        if text:texts.append(text)
    (TASK/'数据准备/原方案文字与公式核对.txt').write_text('来源SHA256: '+sha(doc)+'\n'+'\n'.join(texts),encoding='utf-8')
    # PNG/PDF均来自真实CSV，曲线不补缺测、不拼接滚动预测。
    plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Microsoft YaHei','Arial','DejaVu Sans'],'font.size':11,'axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':1.3,'legend.frameon':False,'axes.unicode_minus':False,'pdf.fonttype':42})
    figure_records=[]
    def savefig(fig,name,df,inputs):
        path=PLOT/(name+'.csv');csv(df,path);fig.tight_layout(pad=1.4)
        for ext in ['png','pdf']:fig.savefig(FIG/(name+'.'+ext),dpi=300,facecolor='white',bbox_inches='tight')
        plt.close(fig);figure_records.append(dict(figure=name,plot_csv=path.relative_to(TASK).as_posix(),csv_sha256=sha(path),script='程序/资格分析与报告.py',inputs=inputs,identity='探索性固定起点离线回放；真实未来时间仅为绘图横轴'))
    methods=protocol['methods'];scnames=[s['name'] for s in protocol['scenarios']]
    fig,axes=plt.subplots(1,3,figsize=(17,4.5),sharey=True)
    for ax,sc in zip(axes,scnames):
        sub=main[main.scenario==sc].set_index('method').loc[methods];bars=ax.bar(np.arange(6),sub.RMSE_mV,color=COLORS,edgecolor='black',linewidth=.6)
        for b,v in zip(bars,sub.RMSE_mV):ax.text(b.get_x()+b.get_width()/2,v+.4,f'{v:.2f}',ha='center',fontsize=9)
        ax.set_xticks(range(6),['保持','目标率','直接迁移','先验适配','2B有界','2B慢变'],rotation=30,ha='right');ax.set_title(sc+'；n=141');ax.set_ylabel('电压 RMSE / mV')
    fig.suptitle('相同141个后续目标上的固定起点迁移比较（探索性）',y=1.02)
    savefig(fig,'三场景共同目标RMSE',main,['正式计算/表格/电压_SOH及损失误差.csv'])
    for sc in scnames:
        sub=pred[pred.scenario==sc];fig,axes=plt.subplots(2,1,figsize=(11,8),sharex=True)
        true=sub[sub.method=='A_Persistence'].sort_values('target_id')
        for ax,ms in zip(axes,[methods[:4],['A_Persistence']+methods[4:]]):
            ax.scatter(true.target_actual_h,true.true_v,s=12,color='black',label='真实合格运行电压',zorder=4)
            for method in ms:
                g=sub[sub.method==method].sort_values('target_id');ax.plot(g.target_actual_h,g.pred_v,lw=1.7,color=COLORS[methods.index(method)],label=LABEL[method])
            ax.set_ylabel('平均单体电压 / V');ax.legend(ncol=2,fontsize=9);ax.set_title(sc+'；单一起点自主外推，141个实测目标')
        axes[-1].set_xlabel('真实累计运行小时 / h（仅事后坐标）')
        savefig(fig,sc+'_跨堆电压预测',sub,['正式计算/表格/各方法逐起点预测.csv'])
    fig,axes=plt.subplots(1,3,figsize=(17,4.5),sharey=True)
    for ax,sc in zip(axes,scnames):
        sub=pred[pred.scenario==sc];true=sub[sub.method=='A_Persistence'];ax.scatter(true.target_actual_h,true.true_soh*100,s=10,color='black',label='真实运行SOH')
        for method in ['A_Persistence','B_FC1Direct','C_FC1PriorAdapted','D_Target2B_SlowReversible']:
            g=sub[sub.method==method];ax.plot(g.target_actual_h,100*g.pred_soh,label=LABEL[method],color=COLORS[methods.index(method)],lw=1.4)
        ax.set_title(sc);ax.set_xlabel('累计运行小时 / h');ax.set_ylabel('运行SOH / %')
    handles,labels=axes[-1].get_legend_handles_labels();fig.legend(handles,labels,ncol=4,loc='lower center',bbox_to_anchor=(.5,-.06))
    savefig(fig,'运行SOH固定起点预测',pred,['正式计算/表格/各方法逐起点预测.csv'])
    fig,ax=plt.subplots(figsize=(10,4.5));pr=params[params.method.isin(['A_TargetBounded','B_FC1Direct','C_FC1PriorAdapted','D_Target2B_SlowReversible'])]
    for i,method in enumerate(pr.method.unique()):
        g=pr[pr.method==method];ax.plot(range(3),g.model_rate_per_h*1e6,marker='o',label=LABEL[method],color=COLORS[methods.index(method)])
    ax.set_xticks(range(3),scnames);ax.set_ylabel('相对损失速率 / $10^{-6}$ h$^{-1}$');ax.set_title('源端漂移明显高于目标域早期漂移；速率不等于不可逆机理');ax.legend(ncol=2)
    savefig(fig,'源目标速率与适配',pr,['正式计算/表格/模型参数与速率.csv'])
    timeplot=pred[(pred.method=='A_Persistence')].copy();fig,ax=plt.subplots(figsize=(10,4.5))
    for sc in scnames:
        g=timeplot[timeplot.scenario==sc];ax.plot(g.target_actual_h,g.nominal_target_h-g.target_actual_h,label=sc)
    ax.axhline(0,color='black',lw=.8);ax.set_xlabel('真实累计运行小时 / h（仅事后）');ax.set_ylabel('名义目标时间－实际时间 / h');ax.set_title('冻结每循环12h场景的时间误差：未将真实未来时长输入模型');ax.legend()
    savefig(fig,'名义与真实未来时间差异',timeplot,['正式计算/表格/各方法逐起点预测.csv'])
    dist=pd.read_csv(RUN/'表格/源目标归一化损失分布.csv');fig,ax=plt.subplots(figsize=(10,4.5));ax.errorbar(range(len(dist)),dist.loss_mean*100,yerr=dist.loss_std*100,fmt='o',capsize=4,color=COLORS[1]);ax.set_xticks(range(len(dist)),dist.group,rotation=20,ha='right');ax.set_ylabel('相对性能损失 / %');ax.set_title('固定数据域的均值与观测标准差（非独立重复测量置信区间）')
    savefig(fig,'源目标损失分布',dist,['正式计算/表格/源目标归一化损失分布.csv'])
    pd.DataFrame(figure_records).to_json(TABLE/'图片来源索引.json',orient='records',force_ascii=False,indent=2)
    # 报告只引用刚算出的数值，保留全部不利方法、跨度与场景。
    overview=main[['scenario','method','n','RMSE_mV','MAE_mV','RMSE_SOH_pp','MAE_SOH_pp']].copy()
    overview['method']=overview.method.map(LABEL)
    primarygain=gains[(gains.scope=='COMMON_ALL')&(gains.reference=='A_TargetBounded')]
    timeaudit=pred[pred.method=='A_Persistence'].groupby('scenario').agg(actual_min=('actual_lead_h','min'),actual_max=('actual_lead_h','max'),nominal_min=('nominal_lead_h','min'),nominal_max=('nominal_lead_h','max')).reset_index()
    srcpct=source['q_bounded']*100
    targetrates=params[params.method=='A_TargetBounded'][['scenario','model_rate_per_h']].copy();targetrates['source_to_target_rate_ratio']=source['q_bounded']/targetrates.model_rate_per_h
    csv(targetrates,TABLE/'源目标速率比.csv')
    ratio_text=f"{targetrates.source_to_target_rate_ratio.min():.2f}—{targetrates.source_to_target_rate_ratio.max():.2f}"
    clocks=pd.read_csv(TASK/'数据准备/严格截止日历包络.csv')
    write('跨堆迁移实验协议.md',f'''# 阶段2C跨堆迁移实验协议

预登记UTC：{protocol['utc']}。基准HEAD：`{protocol['head']}`。JSON SHA256：`{sha(TASK/'配置/阶段2C冻结协议.json')}`。配置和SHA在拟合源/目标速率、查看本轮误差之前保存，协议未因成绩更改。

本轮为**固定起点、标量运行性能、探索性离线迁移**。不是二维极化迁移，不是新独立Test。原2A/2B的滚动预测指标与本轮长程固定起点指标不能混为同条件排名。

## 数据与目标
只读2A三个冻结观测CSV；FC1源域只用原Train的669个合格小时观测。FC1参考约0.70 A/cm²，RealHyFC冻结B60运行负载约0.558161974 A/cm²，均使用自身已有基准。预测`d=1-SOH`，电压由`base_V*(1-d)`恢复。两负载、硬件、温压不同；归一化不能自动消除域差异。

## 三个时间起点
严格300h、严格600h、后移629.2038h。合法目标历史数23、47、48；末次观测分别296.81473、560.3537、629.2038h，原始循环23、49、51。截止前原始循环时钟和有效观测身份分开保存。600h没有原始运行记录精确时钟，日历只能包络在2025-02-24 01:10:00.005至2025-03-07 12:15:00.005之间；绝不插值制造精确截止日历。

从各自末次实测/滤波状态，先按已知过去小时传播到截止点；未来目标k名义时刻=`cutoff+12*(k-last_completed_cycle_id)`。截止时未完成循环相位未知，因此从截止重新登记名义计划，不声称已经获得真实未来运行计划。**所有真实未来累计小时、日历、电压只用于标签和事后审计；不参与预测传播。** 不在629h或中间循环读取真值修正300/600h模型。

## 冻结六方法（对应用户四组）
1. A保持：目标最后实测d保持。
2. A目标有界：截止前全历史OLS的d/h斜率，截到[0,Train局部斜率绝对值Q95]，最后实测d为锚。
3. B直接迁移：FC1原Train同样学到的非负有界全局d/h，迁入目标自身末次状态；不拟合目标速率。
4. C先验适配：`q=.25*q_FC1+.75*q_target`，权重0.25在运行前固定，不搜索。只多加入FC1信息，与第2项比较可分离其增量。
5. D同截止2B型有界率；6. D同截止2B型慢变＋可逆。

最后两项复用2B低阶数学结构，按本轮真实过去小时重新拟合；参数全部只来自截止前观测。末12个原始循环内合格观测形成因果状态。k=[0.25,1]，tau=[12,36,72]h；截止前合法点顺序前80%拟合、最后20%单起点验证，按RMSE、MAE和候选次序选定，随后全截止前历史重拟合。不是使用2B完整Train参数。

`x=[L,q,R]`、`d=L+R`，q非负、有Train经验上界和变化率界，R可正可负且均值回归；未来不更新。全部量纲、初始化与Q在冻结JSON及数学核心中完整给出。特别是原V/循环改为无量纲/h，过程噪声按参考12h步长参数化，不能直接照搬数值。

## 同样本评价与停止
共同评价为循环51之后、运行开始不早于629.2038h的全部冻结合格观测。141个唯一目标、每方法每场景1起点/141查询，18条模型轨迹共2538个记录；不是2538次独立寿命实验。仍记录149个候选循环中的8个缺失/不合格项。分别保存原分区标签结果，主指标为全部共同未来；旧Assessment为已查看的辅助分层。

固定辅助完整块1/4/12/25循环，目标从52开始，缺任一原始循环则NA，不补齐。25循环缺75，完整块拒绝。所有迁移方法对全部目标域参照报告收益，负迁移保留。不根据结果改权重、窗口、源域时段或融合公式，不继续训练16单元网络。
''')
    write('低阶迁移实验报告.md',f'''# 阶段2C低阶迁移实验报告

**这次没有获得FC1帮助RealHyFC预测的正面证据。** FC1学到的每小时相对损失速率约为RealHyFC早期的{ratio_text}倍，直接套用会过度预测电压下降。固定25%源信息的适配减轻了错误，但仍比不使用FC1的同形目标漂移模型差。该结论仅针对本次冻结低阶先导，不证明所有跨堆迁移均不可行。

## 实际成绩
全部为同一141个后续真实目标；电压误差mV，SOH误差为百分点。每个场景是单一起点不更新的长程预测。

{md(overview)}

300h和600h场景，保持法优于所有有漂移方法。629.2038h场景同截止慢变＋可逆模型RMSE={main[(main.scenario=='C_后移629h')&(main.method=='D_Target2B_SlowReversible')].RMSE_mV.iloc[0]:.6f}mV，低于保持法；**这是目标域状态估计的结果，并没有使用FC1先验，不能计为迁移成功**。保留全部三场景，不挑后移场景充当总体成功。

## 时间与样本
源Train669个观测，4—692h；只用原Train，不使用FC1后期观测重新找规律。目标适配23/47/48点。共同目标141个，原标签Train64、Validation35、Assessment40、跨边界2；在本轮所有这些目标均位于适配截止之后，不是本轮训练数据。原2A/2B分区标签和缺失资格完全保留。

{md(timeaudit)}

表中actual为实际未来运行小时跨度，仅供审计；nominal为预测真正使用的冻结时间场景。未来每循环12h不是测得的真实持续时间。600h位于原始记录缺口，实际截止日历未知；保守629.2038h对应2025-03-08T04:30:32.005000，属于循环51完成时刻，不是600h诊断精确完成小时。

完整辅助1/4/12循环均可计算；25循环因原始75缺失而NA。所有辅助结果及分区误差均在`输出结果/正式计算/表格/电压_SOH及损失误差.csv`，不以短跨度个别改善替代长期共同目标主结论。

## 速率与代价
FC1源速率为{source['q_bounded']:.12g}/h，即每小时{srcpct:.9f}个SOH百分点损失；转换到目标电压的斜率只乘目标自身base。禁止将源V/h直接当目标V/循环。

{md(targetrates)}

这只是观测漂移，不是已辨识的不可逆电化学损伤速率。目标域5项滤波统计量、状态/超参数数量与CPU耗时均保存，未训练神经网络。快是因为低阶统计和3维滤波，不代表真实数据足够支持复杂迁移。

## 复核与限制
独立程序不导入生产核心，用lstsq、独立滤波更新、传播和指标重算18条轨迹；详见独立复核JSON。原始数据、阶段0—2B共1050个文件SHA保持一致。单条目标轨迹、少量诊断、不同负载、未知真实发布时刻和已查看Assessment仍限制泛化主张。不输出RealHyFC虚构RUL，不改变FC1条件RUL。
''')
    write('迁移收益与负迁移分析.md',f'''# 迁移收益与负迁移分析

**已证实：本轮共同长程目标上，加入固定25%FC1源速率造成负迁移。** 同时，适配确实比直接迁移少犯错；这两件事不能混为“迁移有效”。

## 隔离源信息增量
保持目标锚点、名义未来时间与全历史有界目标速率估计完全相同，C相对A_TargetBounded只多出FC1速率权重。收益定义`100*(1-RMSE_transfer/RMSE_reference)`，负数表示恶化。

{md(primarygain[['scenario','transfer','reference','RMSE_gain_pct','MAE_gain_pct']])}

全部比较包括保持、目标有界、2B型有界与2B型慢变，保存`迁移收益与负迁移.csv`。原Assessment单独分层，也不能作为新的未触碰测试。个别短块上若有正收益，不用于重新挑起点或改0.25权重。

## 可能原因的证据级别
- 已证实：源q约为目标全历史早期q的{ratio_text}倍；同样时间场景下直接迁移预测下降过快，适配介于源与目标速率之间。
- 已证实：FC1约0.70与Real约0.558 A/cm²，基准、硬件、温压条件不同。归一化损失只统一数学量纲，不能保证演化规律一致。
- 较可信解释：将早期局部/全局漂移长期保持，会放大速率偏差；目标的后续损失演化不等于恒定源速率。
- 尚无法区分：负载差异、运行协议、可逆波动、真实时间排程误差各自占多少。仅有两个电堆及一次冻结先导，不能作物理因果归因。

后移目标域慢变模型较好，主要对应更小的估计外推q与不同初始L/R状态；**不证明L、R是两种物理损伤，也不证明未来恢复事件已可预测**。没有独立重复诊断验证分解。

## 停止决定
本轮未达到跨堆升级条件。不继续搜索权重、源训练时段、温压阈值或新网络。保留FC1→RealHyFC方向，但先补充观测映射与独立诊断证据。结果不支持立即用16单元网络解决负迁移。
''')
    write('二维极化迁移资格评估.md',f'''# 二维极化迁移资格评估

**二维仍不具备接入本轮迁移/16单元有界网络的条件。** 本轮只迁移了固定运行负载相对损失标量；运行电压约0.558 A/cm²绝不等同0.90 A/cm²极化电压。

保留`D_ref=V_BoL(.90)-V_t(.90)`、`D_low=V_BoL(.45)-V_t(.45)`。两者为综合数学表征，不拆成活化或欧姆损伤。

{md(pd.DataFrame(elig)[['scenario','operating_history','diagnostics','both_supported','completed_past_pairs','rank_1D','rank_2D','past_diagnostic_transitions','mapping_2D']])}

300h仅BoL，无可用诊断—运行历史配对；600h只允许BoL+已完成300h，得到1对；629.2038h可登记BoL+300h+600h，得到2对。两参考均有冻结B60支持，但支持两个负载不等于有足够独立健康时点。BoL早于第一份运行记录，无先前运行观测可配。

FC1源Train有6个名义诊断（含BoL），5个名义历史配对，**精确时间合法配对0个**。诊断小时仍NOMINAL_ONLY，不能把标称结束时刻当实际发布时刻。

沿用2B门槛：一维映射2参数至少6独立诊断对；二维3参数至少9对，设计满列秩且标准化条件数≤100；二维动态至少6个过去诊断转移，并有合法验证目标。本轮不降低门槛。目标各截止最多2对，一维和二维都NA，不拟合欠定映射。原2B用更多Train诊断的条件数/资格也不能借给严格早期截止。

诊断结果真实发布时间均未知；所谓完成后可用仅探索性离线假设。300h曲线运行上界327.81354h，600h曲线上界615.66968h，分别不能用于严格300/600h。629.2038h只是保守后移起点，不是600h诊断精确小时。

## 16单元原方案输入核对
共同参考负载损失值可以离线构造，历史稠密状态轨迹不能。诊断间转移、两状态变化率、观测映射及独立验证不够；运行状态不等于两个诊断分量。未知诊断/恢复事件的真实可用时刻不能用0或插值代替。网络损失及梯度未执行，记NA，不记0。

## 分层资格
- 标量相对损失数学迁移：已实际完成探索性先导，未获收益。
- 双参考特征离线描述：继续保留阶段1.2的CONDITIONAL_PASS；本轮不改历史结论。
- 严格因果二维跨堆迁移：NOT_QUALIFIED。
- 正式16单元有界网络训练：NO。低阶源先验没有可靠正收益、映射及独立时点不足；先补观测证据，不靠增加参数弥补。
- RealHyFC真实RUL评价：NO，无核实EOL；FC1此前4%首次观测越界及3.5%—5.5%敏感性/删失原记录保持原样。
''')
    write('使用说明.md','''# 阶段2C低阶迁移先导

本轮结论：固定25%源先验没有改善RealHyFC共同未来预测；直接迁移更差。后移起点的目标域慢变模型较好，但不是FC1贡献。全部为探索性离线回放。

## 可复现入口
Python3.11，依赖numpy、pandas、matplotlib。`程序/运行迁移实验.py --output <全新目录>`真实重算6个低阶模型×3场景；不得使用已有输出目录。只读三个复制的2A冻结观测，不重新处理原始860万CSV。

`程序/独立数值复核.py --run <上述目录>`独立复算，不导入数学核心。`程序/资格分析与报告.py`读取本轮正式计算CSV，重建资格表、300dpi PNG/PDF及中文报告。后者属于准备/绘图代码，不放入完整代码正式版本。

`程序/冻结与准备.py`为一次性预登记，已完成后拒绝覆盖；正常复现不要再运行。冻结JSON、来源SHA和1050个历史文件SHA均保留。

## 查找结果
`输出结果/正式计算/表格/各方法逐起点预测.csv`是2538行逐点主记录；`电压_SOH及损失误差.csv`含共同未来、旧分区和完整1/4/12/25块；25块缺75循环而NA。`迁移收益与负迁移.csv`保留全部正负收益。

`输出结果/图片`含三起点预测图、共同目标RMSE、SOH、源目标速率/分布与名义时间误差；每图数据及来源在`输出结果/绘图数据`和`输出结果/表格/图片来源索引.json`。

先读《低阶迁移实验报告》《迁移收益与负迁移分析》《二维极化迁移资格评估》。原2A/2B、原始数据、FC1条件RUL不变；本轮不训练网络、不产生RealHyFC真实RUL。
''')
    print('REPORTS_AND_FIGURES',len(figure_records),'target_pairs',[e['completed_past_pairs'] for e in elig])
if __name__=='__main__':main()
