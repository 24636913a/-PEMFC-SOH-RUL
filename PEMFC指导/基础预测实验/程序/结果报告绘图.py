"""从本次真实CSV生成报告、300dpi PNG/PDF与逐图数据索引。无预测选参。"""
from pathlib import Path
import os,json,sys
import numpy as np
import pandas as pd
WORK=Path(__file__).resolve().parents[1];GUIDE=WORK.parent;REPO=GUIDE.parent
OUT=WORK/'输出结果';TAB=OUT/'表格';FIG=OUT/'图片';PLOT=OUT/'绘图数据'
os.environ['MPLCONFIGDIR']=str(OUT/'绘图缓存')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from 运行基础预测 import sha,js,csv

plt.rcParams.update({'font.family':'Microsoft YaHei','font.size':10,'axes.spines.top':False,
    'axes.spines.right':False,'axes.unicode_minus':False,'pdf.fonttype':42,'ps.fonttype':42,
    'axes.prop_cycle':matplotlib.cycler(color=['#3775BA','#B64342','#3C8D6C','#9364A5']),
    'savefig.dpi':300,'figure.dpi':110})
INDEX=[]

def md(d):
    if d.empty:return 'NA：没有合格结果。\n'
    def fmt(x):
        if pd.isna(x):return 'NA'
        return f'{x:.6f}' if isinstance(x,(float,np.floating)) else str(x).replace('|','/')
    return '| '+' | '.join(d.columns)+' |\n| '+' | '.join(['---']*len(d.columns))+' |\n'+'\n'.join('| '+' | '.join(map(fmt,row))+' |' for row in d.itertuples(index=False,name=None))+'\n'

def save(fig,name,data,sources,note):
    fig.tight_layout();fig.savefig(FIG/(name+'.png'),dpi=300,bbox_inches='tight');fig.savefig(FIG/(name+'.pdf'),bbox_inches='tight');plt.close(fig)
    csv(PLOT/(name+'.csv'),data)
    INDEX.append(dict(图=name,绘图CSV=str((PLOT/(name+'.csv')).relative_to(WORK)),CSV_SHA256=sha(PLOT/(name+'.csv')),
                      来源=sources,说明=note,PNG_SHA256=sha(FIG/(name+'.png')),PDF_SHA256=sha(FIG/(name+'.pdf'))))

def main():
    for p in [FIG,PLOT]:p.mkdir(exist_ok=True)
    assert (OUT/'独立复核通过.json').exists()
    reused=[]
    for name in ['升降载描述性对照.csv','启停恢复配对.csv','实际候选支持平台条件.csv']:
        source=GUIDE/'健康特征提取/共同负载特征研究/输出结果/表格'/name
        target=TAB/('阶段1.2复用_'+name)
        target.write_bytes(source.read_bytes())
        reused.append(dict(来源=str(source.relative_to(GUIDE)),本轮副本=str(target.relative_to(WORK)),
                           SHA256=sha(source),数值重算=False,身份='既有描述性测量证据，不重新选阈值'))
    csv(TAB/'测量条件既有证据复用索引.csv',pd.DataFrame(reused))
    obs={'FC1':pd.read_csv(WORK/'数据准备/FC1小时观测.csv'),
         'RealHyFC':pd.read_csv(WORK/'数据准备/RealHyFC循环观测.csv')}
    preds={k:pd.read_csv(TAB/f'{k}基础预测逐起点结果.csv') for k in obs}
    mets={k:pd.read_csv(TAB/f'{k}连续预测指标.csv') for k in obs}
    counts=[];ends=[]
    for k,p in preds.items():
        for keys,d in p.groupby(['scheme','partition','horizon','method']):
            counts.append(dict(dataset=k,scheme=keys[0],partition=keys[1],horizon=keys[2],method=keys[3],
                origins=d.origin_id.nunique(),queries=len(d),unique_targets=d.target_id.nunique(),
                endpoint_lead_h_min=d[d.step==d.horizon].actual_lead_h.min(),
                endpoint_lead_h_max=d[d.step==d.horizon].actual_lead_h.max()))
            z=d[d.step==d.horizon];e=z.pred_v-z.true_v
            ends.append(dict(dataset=k,scheme=keys[0],partition=keys[1],horizon=keys[2],method=keys[3],n=len(z),
                             RMSE_mV=np.sqrt(np.mean(e**2))*1000,MAE_mV=np.mean(abs(e))*1000))
        d=obs[k];d=d[(d.scheme=='FC1_NOMINAL') if k=='FC1' else (d.scheme=='B60')].copy()
        d.loc[~d.valid,'v']=np.nan
        x=d.time if k=='FC1' else d.available_h
        fig,axes=plt.subplots(2,1,figsize=(9,6))
        axes[0].plot(x,d.v,lw=1,label='合格小时观测' if k=='FC1' else 'B60合格循环观测')
        axes[0].set(xlabel='实际运行小时 / h',ylabel='平均单片电压 / V',title=f'{k}：全轨迹离线观测；缺测保持断线')
        axes[0].legend(frameon=False)
        valid=p[(p.partition=='Assessment')&(p.horizon==(24 if k=='FC1' else 12))]
        origin=valid.origin_id.min();q=valid[valid.origin_id==origin].sort_values('step')
        for method,g in q.groupby('method'):
            axes[1].plot(g.target_operating_h,g.pred_v,'o-',ms=3,label=method)
        g=q[q.method=='Persistence'];axes[1].plot(g.target_operating_h,g.true_v,'k.--',label='真实观测')
        axes[1].set(xlabel='实际运行小时 / h（仅审计坐标）',ylabel='平均单片电压 / V',
                    title=f'后留最早合法固定起点 {int(origin)}；整块{int(q.horizon.iloc[0])}'+('小时' if k=='FC1' else '循环')+'预测')
        axes[1].legend(frameon=False,ncol=3)
        plot=pd.concat([d.assign(plot='observations'),q.assign(plot='single_fixed_origin')],ignore_index=True)
        save(fig,f'{k}实际电压与基础预测',plot,f"数据准备/{'FC1小时观测.csv' if k=='FC1' else 'RealHyFC循环观测.csv'}；输出结果/表格/{k}基础预测逐起点结果.csv",
             '全轨迹为真实观测；预测只取预先固定规则最早后留起点，不拼接重叠窗口。Real B60仅探索性。')
    counts=pd.DataFrame(counts);csv(TAB/'连续预测起点数量与实际跨度.csv',counts);csv(TAB/'固定终点辅助指标.csv',pd.DataFrame(ends))
    # Identical horizons on each stack, no cross-task unit ranking.
    fig,axes=plt.subplots(1,2,figsize=(10,4));data=[]
    for ax,k in zip(axes,obs):
        m=mets[k];m=m[(m.partition=='Assessment')&(m.scheme==('FC1_NOMINAL' if k=='FC1' else 'B60'))]
        hs=sorted(m.horizon.unique());xx=np.arange(len(hs))
        for n,method in enumerate(['Persistence','LocalLinear']):
            z=m[m.method==method].set_index('horizon').reindex(hs)
            vals=z.RMSE_mV.to_numpy();bars=ax.bar(xx+(n-.5)*.35,np.nan_to_num(vals),.35,label=method)
            for i,v in enumerate(vals):
                if np.isnan(v):ax.text(xx[i]+(n-.5)*.35,.03,'NA',ha='center',rotation=90)
            data.append(z.reset_index())
        ax.set(xticks=xx,xticklabels=hs,xlabel='未来小时' if k=='FC1' else '未来完整物理循环',
               ylabel='全块查询RMSE / mV',title=k+(' 后留探索性' if k=='FC1' else ' B60后留探索性'));ax.legend(frameon=False)
    save(fig,'不同预测跨度误差',pd.concat(data),'表格/FC1连续预测指标.csv；表格/RealHyFC连续预测指标.csv','不把缺失跨度画成0；柱高来自全部块内查询，不是仅终点误差。')
    diag=pd.read_csv(WORK/'数据准备/稀疏诊断观测.csv');fair=pd.read_csv(TAB/'一维二维健康特征公平对比.csv')
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    for ax,k,s in zip(axes,['FC1','RealHyFC'],['FC1_10','B60']):
        d=diag[(diag['电堆']==k)&(diag.scheme==s)&diag.primary].sort_values('标称小时')
        for field,label in [('D_ref_V','D(0.90)'),('D_low_V','D(0.45)')]:ax.plot(np.arange(len(d)),d[field]*1000,'o-',label=label)
        ax.set(xticks=np.arange(len(d)),xticklabels=d['标称小时'].astype(int),xlabel='真实诊断顺序（刻度为标称h）',ylabel='相对本堆BoL损失 / mV',title=k+' 离线诊断（非预测）')
        ax.tick_params(axis='x',rotation=45);ax.legend(frameon=False)
    save(fig,'一维二维诊断健康特征',diag,'数据准备/稀疏诊断观测.csv','FC1 NOMINAL_ONLY；Real实际时间边界另见时间资格表；B60 EXPLORATORY。')
    fig,axes=plt.subplots(1,2,figsize=(10,4))
    for ax,k in zip(axes,['FC1','RealHyFC']):
        d=fair[fair.dataset==k];targets=sorted(d.target_id.unique());xx=np.arange(len(targets))
        for n,method in enumerate(['Persistence','LocalLinear3','Ridge1D','Ridge2D']):
            z=d[d.method==method].sort_values('target_id')
            ax.plot(xx,z.pred_D*1000,'o-',label=method)
        z=d[d.method=='Persistence'].sort_values('target_id');ax.plot(xx,z.true_D*1000,'kx',ms=10,label='真实D_ref')
        ax.set(xticks=xx,xticklabels=z.target_nominal_h.astype(int),xlabel='目标诊断标称h（不是输入）',ylabel='D_ref / mV',title=f'{k}：同目标比较 n={len(targets)}')
        ax.legend(frameon=False,fontsize=8)
    save(fig,'一维二维公平预测对比',fair,'一维二维健康特征公平对比.csv','仅1和2个时间预测目标；不是独立寿命样本，不作显著性结论。')
    curves=pd.read_csv(TAB/'FC1_RUL外推曲线.csv');rr=pd.read_csv(TAB/'FC1_RUL阈值与删失结果.csv');fc=obs['FC1']
    fig,axes=plt.subplots(1,3,figsize=(12,4),sharey=True)
    for ax,origin in zip(axes,[400,500,600]):
        d=fc.copy();d.loc[~d.valid,'v']=np.nan;ax.plot(d.time,d.v,'k-',lw=.8,label='合格真实小时观测')
        for method,g in curves[curves.origin_h==origin].groupby('method'):ax.plot(g.time_h,g.pred_v,lw=1.2,label=method)
        threshold=rr[rr.main].threshold_V.iloc[0];ax.axhline(threshold,color='#9364A5',ls='--',label='4%条件阈值')
        ax.axvline(origin,color='gray',ls=':');ax.set(xlabel='实际小时 / h',ylabel='单片电压 / V',title=f'起点 {origin} h；非物理EOL验证')
        ax.legend(frameon=False,fontsize=7)
    save(fig,'FC1条件RUL与删失',pd.concat([curves.assign(kind='forecast'),fc.assign(kind='observations'),rr.assign(kind='threshold_record')],ignore_index=True),
         'FC1_RUL外推曲线.csv；FC1_RUL阈值与删失结果.csv；FC1小时观测.csv','600h的零斜率外推保留删失，不画虚构交点。')
    ts=pd.read_csv(TAB/'周期漂移时序绘图数据.csv');long=pd.read_csv(TAB/'温压周期波动与工况漂移对比.csv')
    # Representative selected only by file/measurement identity and distance to fixed reference, not forecast scores.
    candidates=long[long['文件名'].str.contains('_BoL')&long['主扫描']].copy()
    candidates['distance']=abs(candidates.j-.9);key=candidates.sort_values(['distance','平台键']).iloc[0]['平台键']
    z=ts[ts['平台键']==key]
    fig,axes=plt.subplots(2,1,figsize=(9,6),sharex=True)
    axes[0].plot(z['平台内秒'],z.temp,lw=.8,label='原始温度');axes[0].plot(z['平台内秒'],z.slow_temp_16s,lw=1.8,label='16点（约16s）慢变描述')
    axes[1].plot(z['平台内秒'],z.v*1000,lw=.8,label='原始电压');axes[1].plot(z['平台内秒'],z.slow_v_16s*1000,lw=1.8,label='16点（约16s）慢变描述')
    axes[0].set(ylabel='冷却液入口温度 / °C',title=key+'；仅回顾解释，不改变测量资格')
    axes[1].set(xlabel='平台内实际秒 / s',ylabel='平均单片电压 / mV')
    for ax in axes:ax.legend(frameon=False)
    save(fig,'温度周期波动与慢变漂移',z,'周期漂移时序绘图数据.csv；温压周期波动与工况漂移对比.csv','约4s模式不是已证实控制器原因；不按温度修正电压。')
    support=pd.read_csv(TAB/'筛选前后参考负载支持.csv')
    fig,axes=plt.subplots(1,2,figsize=(9,4))
    for ax,j in zip(axes,[.45,.9]):
        for method,color in [('A60','#3775BA'),('B60','#B64342')]:
            d=support[(support.j==j)&(support['方案']==method)].sort_values('文件名')
            vals=d['本曲线V'].where(d['本曲线支持'])
            ax.scatter(np.arange(len(d)),vals,label=method,c=color,s=28)
        ax.set(xlabel='文件名排序（仅位置索引）',ylabel='支持位置电压 / V',title=f'j={j:.2f} A/cm²；缺失不补齐');ax.legend(frameon=False)
    save(fig,'严格与探索性参考点保留',support,'筛选前后参考负载支持.csv','真实值变化是不同筛选支持，不代表温度修正或稳态认证。')
    # Read only three diagnostic CSVs, not full operating raw CSV; preserve source SHA.
    sys.path.insert(0,str(GUIDE/'数据处理/程序'));from 核验基础 import numeric,mean_cells,CELLS_REAL
    source=pd.read_csv(GUIDE/'健康特征提取/方法复核/输出结果/表格/原始极化来源核对.csv')
    reps=pd.read_csv(GUIDE/'健康特征提取/方法复核/输出结果/表格/合格代表点及条件.csv')
    names=['Polcurve_Nominal_RHFC_BoL.csv','Polcurve_Nominal_RHFC_600h.csv','Polcurve_Nominal_RHFC_2300h.csv']
    csv(TAB/'原始极化图源身份.csv',source[source['文件名'].isin(names)][['文件名','原始路径','SHA256','实际行数','扫描分段']])
    csv(TAB/'原始极化图筛选点来源.csv',reps[reps['文件名'].isin(names)&reps['方案'].isin(['A60','B60'])])
    fig,axes=plt.subplots(1,3,figsize=(12,4));pdata=[]
    for ax,name in zip(axes,names):
        info=source[source['文件名']==name].iloc[0];path=REPO/info['原始路径'];assert sha(path)==info.SHA256
        f=pd.read_csv(path,sep=';',encoding='utf-8-sig',usecols=['j.S.act [A/cm2]',*CELLS_REAL])
        raw=pd.DataFrame(dict(j=numeric(f['j.S.act [A/cm2]']),v=mean_cells(f,CELLS_REAL)[0],row=np.arange(len(f)),文件名=name,kind='RAW_ALL_DIRECTIONS'))
        ax.scatter(raw.j,raw.v,s=1,color='gray',alpha=.35,label='原始全部扫描（不拟合）');pdata.append(raw)
        for method,marker in [('A60','o'),('B60','x')]:
            z=reps[(reps['文件名']==name)&(reps['方案']==method)]
            ax.scatter(z.j,z.v,s=20,marker=marker,label=method+'降载代表点');pdata.append(z.assign(kind=method))
        ax.set(xlabel='实测电流密度 / (A/cm²)',ylabel='单片电压 / V',title=name.replace('Polcurve_Nominal_RHFC_','').replace('.csv',''));ax.legend(frameon=False,fontsize=7)
    save(fig,'原始极化与严格筛选对照',pd.concat(pdata,ignore_index=True),'表格/原始极化图源身份.csv；表格/原始极化图筛选点来源.csv','不同扫描方向不混合拟合；原始散点只展示，筛选点均为原规则降载代表点。')
    csv(TAB/'图表来源与SHA256.csv',pd.DataFrame(INDEX))
    reports(obs,preds,mets,counts,diag,fair,rr,long,support)


def reports(obs,preds,mets,counts,diag,fair,rr,long,support):
    cfg=json.loads((WORK/'配置/阶段2A冻结协议.json').read_text(encoding='utf-8'))
    fcmeta=json.loads((OUT/'FC1数据准备记录.json').read_text(encoding='utf-8'))
    realmeta=json.loads((OUT/'Real数据准备记录.json').read_text(encoding='utf-8'))
    def write(name,text):(WORK/name).write_text(text.rstrip()+'\n',encoding='utf-8')
    protocol='''# 阶段2A实验协议

本次为导师确认后的两套电堆分别建模。冻结配置、首次执行前SHA与Word原文位于配置目录。既有全寿命观测已用于方法研究，均为探索性回顾与时间后留，不是独立盲测。禁止以本轮误差反选测量门槛。

## 两类任务与信息流

连续运行：FC1额定0.70±0.02 A/cm²原始五单片完整均值，小时内按实际记录等权平均（不是伪造1Hz），最早10合格Train小时堆电压中位数为Ubase；存在重复时间30条，原样记录计数，未伪装独立测量。按名义诊断T的[T−1,T+3)保守排除，真实诊断钟不明，此规则有误排/漏排风险。Real由Train重复长平台计数固定状态、进入方向与恢复描述组，训练最早5合格循环中位数为基准；各基准积累完成前不预测。

各数据真实时间跨度60/20/20，不按合格样本数量重排。边界单元为CROSS_BOUNDARY。历史48h/12物理循环，最少24/6有效观测；无插值。每起点用当时已发生历史拟合带截距局部OLS（允许恢复），持续值用同一起点最新观测。它们是顺序滚动算法，后期已发生观测可作历史输入，不是利用未来标签重新选择模型。所有基准、运行平台仅Train确定。

未来24/72/168h或1/4/12/25循环整块必须真实有效且全部处于同一目标分区。所有块内查询等权主指标；固定终点另报。重叠查询不是独立电堆实验。Real使用循环索引做趋势坐标，未来实际运行小时只在审计和作图出现，不输入预测器。

稀疏极化：复用阶段1.2各自BoL的D_ref(0.90)和D_low(0.45)，按实际诊断顺序预测下一次主诊断，启停后单列，不能作同协议额外老化样本。持续值、最近3次诊断OLS，及固定alpha=1的ΔD_ref岭回归。1D/2D均使用相同完整过去转移；最少4/6转移，标准化后含截距设计矩阵满秩且条件数≤100。均值/总体标准差仅拟合时可用转移确定，截距不惩罚。四法同目标交集为主比较，另报全部可用起点，缺资格NA。

SOH：FC1=U/Ubase；Real运行=Vop/Vop0；Real极化=1−Dref/VBoL(.90)。三者单位均无量纲但测量定义不同，误差为百分点，不混合排名。B60保持探索性，无新温度认证，压力3kPa不放宽。

RUL：400/500/600h前历史按80/20选24/48/72h非负损失斜率OLS窗口，再用全部起点前历史同窗口重估；只按inner电压RMSE/MAE选择，阈值不参与选参。持续值并列。4%主阈值，3.5–5.5%敏感性，1000h上限。首次合格小时观测越界仅为条件阈值参考，不是物理EOL，保留前一观测到越界小时的间隔。任一侧不越界保留删失；起点已首次越界不计算新的寿命。

## 运行与复核

运行程序/准备基础数据.py（一次性，已完成禁止覆盖），程序/运行基础预测.py（已完成须--output新目录），程序/独立复核.py，再运行程序/结果报告绘图.py。正式代码不包含准备程序，验收后单独发布。

没有神经训练、迁移、物理损失或超参数搜索。300/600的检测完成后可用只是明确的离线假设，正式发布时间未知；FC1名义时钟不能升级严格在线。历史阶段0–1.3结论不变。
'''
    write('实验协议.md',protocol)
    partition_rows=[]
    for k,d in obs.items():
        bounds=fcmeta['bounds'] if k=='FC1' else realmeta['selected']['bounds']
        for scheme,g in d.groupby('scheme'):
            for part,q in g.groupby('partition'):
                partition_rows.append(dict(dataset=k,scheme=scheme,partition=part,records=len(q),qualified=int(q.valid.sum()),
                                           clock_boundaries_h=json.dumps(bounds)))
    partition_table=pd.DataFrame(partition_rows);csv(TAB/'连续观测时间划分与数量.csv',partition_table)
    lines=['# 基础预测实验报告','',
        '最简单的保持法已经给出可重复基线，但线性外推没有稳定优势。二维诊断特征尚未显示额外预测收益，不能立即升级到复杂物理或迁移模型。所有结果来自实际CPU运行，未训练神经网络。','',
        f"FC1原始143862条记录，合格小时{fcmeta['qualified_hours']}；Ubase={fcmeta['Ubase']:.9f} V（五片堆），单片基准{fcmeta['base_cell_V']:.9f} V。",
        f"Real缓存8604654行，只读复用而未重解析运行CSV；200个闭合候选周期中{realmeta['complete_cycles']}个符合新全寿命序列标准。最初156周期边界与旧缓存逐项完全一致。B60合格观测189个（含跨边界单元），A60为0；B60运行基准{realmeta['bases']['B60']:.9f} V。",
        f"Train中0.90±0.02运行平台行数为0。固定状态{realmeta['selected']['token']}，实测j≈{realmeta['selected']['j']:.9f} A/cm²，进入方向DOWN，恢复描述组LATER_OR_UNKNOWN。该状态不是0.90极化状态，也未假称旧P1S。",'',
        '### 实际时间边界与观测数','',md(partition_table),'CROSS_BOUNDARY单元不能作为预测目标，但在其结束后已发生的观测可成为后续历史。Train各项分布与评价结果是拟合期描述，不冒称样本外成绩。','',
        '## 连续运行结果','',
        '下表为后20%探索性评价，电压单位单片mV，SOH单位百分点。n为块内预测查询，起点数另表。', '']
    for k in obs:
        m=mets[k];m=m[(m.partition=='Assessment')&(m.scheme==('FC1_NOMINAL' if k=='FC1' else 'B60'))]
        lines += [f'### {k}', '', md(m[['horizon','method','n','RMSE_mV','MAE_mV','RMSE_SOH_pp','MAE_SOH_pp']])]
    lines+=['完整Train/Validation指标亦已保存；不以其与后留混合平均。FC1 168h、Real 25循环缺少完整合格目标块，A60没有基准，均为NA。Real验证期12循环趋势略好但后留反转，不能仅凭验证局部收益声称长期趋势可靠。','',
        '### 实际起点与真实跨度', '',md(counts[(counts.partition!='Train')&(counts.method=='Persistence')]),
        'Real按1/4/12/25个真实周期，不固定换算为小时；各完整块终点实际跨度列于上表。部分缺测切断目标块，不能改为只取终点凑样本。', '',
        '## FC1条件RUL', '',md(rr[rr.main][['origin_h','method','actual_first_cross_h','pred_first_cross_h','actual_RUL_h','pred_RUL_h','abs_error_h','status']]),
        '4%参考首次观测越界在809h，前一合格观测808h。小时均值的测量支持区间不代表精确瞬时故障钟。400/500起点线性外推可评价，600起点选择24h历史后斜率截断为0、保留右删失；持续值均预测删失。完整3.5–5.5%敏感性结果保存，不能改主阈值。Real不设RUL终点。','',
        '## 数据限制与原因', '',
        '诊断时间预测只有8条FC1和9条Real主曲线，不是多个独立电堆。二维至少6条过去转移后才可估计，使公平比较只剩1/2个目标。连续运行虽有较多重叠查询，仍只有各一条寿命轨迹。短期观测波动、可逆恢复、名义诊断时钟、工况筛选都限制外推；趋势法放大局部斜率噪声是可信解释，但本实验不能把具体误差归因为温度或某种损伤。', '',
        '## 温度与压力诊断', '',
        f"固定252个长平台：温度峰峰中位{long['原始温度峰峰_C'].median():.3f}°C，4s正弦投影幅值中位{long['四秒温度投影幅值_C'].median():.3f}°C，慢变温度首末差绝对值中位{long['慢变温度首末差_C'].abs().median():.3f}°C。慢变分解为16采样点居中完整窗口，按原数据约1秒节拍约16s；不是实时因果处理，不用于预测。固定4s投影不等于全部快波动功率；约4s周期可能漂移，不能据此认证传感器或温控机理。",
        md(support.groupby(['方案','j'])[['本曲线支持','共同支持']].sum().reset_index()),
        '支持改善来自明确不同筛选方案，不是温度修正。严格A仍缺共同支持；B60仅取消单项温度峰峰门槛，其他门槛仍保留，因此只作探索性。原始全扫描散点与降载代表点分开画，不把升降载混合。完整温压数值和所有252平台均已输出，不按预测成绩选温压条件。',
        '阶段1.2的升降载描述性对照（77条候选配对）、启停前后配对（14条）及实际支持平台条件（53条）以原始字节复用，见阶段1.2复用_*表及SHA索引。它们不是本轮新增独立实验；例如1150h的0.45位置升载减降载约−5.501mV，提示扫描条件可影响观测；1500h的0.45位置启停后减前约−2.212mV，并不支持所有位置必然恢复。存在缺点和温压差，不能据此将差异单独归因某一温压因素。', '',
        '## 300/600截止与下一步', '',
        '标称300h诊断运行小时区间312.15411–327.81354，严格300h只有BoL可在检测完成后可用假设下登记；标称600h区间566.16980–615.66968，无法确认600h前完成，严格600h可用BoL及已完成300h诊断，不能加入600h。若要用600h曲线，应单独登记实际日历完成且保守运行小时≥615.66968的后移起点，不能与严格600h等同；BoL前界未知但早于首运行记录。其他曲线逐条见300和600小时诊断可用性表。结果发布时间全部未知。时间上已完成不等于A60测量资格通过，B60可计算特征仍属探索性。', '',
        '下一步值得先补齐诊断重复性和时钟、检验一维低阶模型的假设；当前不足以宣称二维泛化或严格跨堆迁移有效，不建议直接大规模接入物理约束网络。若做下一阶段，只能是另行预注册的低复杂度探索性模型，不以本轮后留成绩继续搜索。', '',
        '## 验收与失败保留', '',
        f"独立复核{json.loads((OUT/'独立复核通过.json').read_text(encoding='utf-8'))['checks']}项通过：通用最小二乘复算、增广岭回归复算、同目标与历史截止、阈值及删失、缺块拒绝、原始阶段特征恒等与历史SHA。新增基准完成时点显式防护；原最低历史点数已隐含保证基准完成，独立比对确认首轮与最终全部合法预测和指标未改变，只有排除原因标签细化。首轮代码/输出保留在首次执行保留_基准可用性复核，不将防护完善误报成实际数据泄漏。没有替换权重或复用旧预测。",
        ('已发布首个正式基础预测代码版本，正式入口从冻结观测重新计算，17份预测、指标与参数文件字节完全一致；详见正式版本验收说明与正式入口独立重跑比较。' if (OUT/'正式版本发布记录.json').exists() else '正式代码须经过独立目录完整重跑比对后才发布。')]
    write('基础预测实验报告.md','\n'.join(lines))
    fm=pd.read_csv(TAB/'稀疏诊断公平交集指标.csv');allm=pd.read_csv(TAB/'稀疏诊断全部起点指标.csv')
    write('一维二维特征对比报告.md','''# 一维二维特征对比报告

结论：没有证据表明第二个参考负载改善了下一次诊断预测。此结论不否认二维曲线重构的信息价值，重构不是未来预测。

主比较同时满足四种方法可评价；1D/2D用完全相同的过去完整转移，不能给二维额外历史。FC1公平起点为823→991标称h（1个）；Real B60为1750→2000、2000→2300（2个），均按实际完成顺序，不能解释为独立盲测。严格A60无合格共同基准，NA。

## 同目标公平结果

'''+md(fm)+'''
FC1二维RMSE略高于一维，二者均明显差于持续值；Real二维也高于一维，且未超过持续值。三个目标来自两条寿命，不能将其当作三次独立堆实验进行显著性检验，也不能把两堆误差混合排名。

## 各方法全部可用起点（仅描述，不公平排名）

'''+md(allm)+'''
所有逐起点预测、NA和拒绝原因、模型系数、标准化均值/标准差、设计矩阵秩及条件数保存。启停后曲线独立登记，不混入主要老化序列。Real的B60是EXPLORATORY；FC1为NOMINAL_ONLY，结果发布时刻未知。D_ref/D_low首先是综合电压损失量，不能分别称为活化或欧姆损伤。

不增加特征、不重新调alpha、不按本轮误差改变负载点。下一步不应为追求二维正收益而加大模型；更重要的是增加真实独立诊断、复测与时间证据。
''')
    diff=pd.DataFrame([
        ['测量资格','原阶段严格稳态与时间证据不足','导师允许基础回放，A/B分别报告；不是改写旧BLOCKED'],
        ['温度筛选','末60s峰峰1°C为严格规则','固定4s/16s仅诊断；B60仍探索性，未认证新规则'],
        ['Real运行负载','优先0.90±0.02','Train实际不支持，按预登记计数选择0.558161974，不把旧P1S当0.90'],
        ['600h适配','原第三档提议600h、含BoL至少2次诊断','600h曲线跨截止，严格600不使用；需615.66968保守上界及实际完成后的单独起点'],
        ['时间可用性','结果发布未知','仅检测完成后可用离线假设；FC1仍名义时间'],
        ['模型阶段','原第三档包含物理/迁移模型','本次只持续值、局部趋势、固定岭回归，未执行神经或迁移'],
        ['跨度缺测','原方案建议24/72/168h和1/4/12/25循环','完整目标块资格不足的168h/25循环NA，不缩小跨度凑数'],
        ['RUL','原第5章4%阈值、400/500/600h','保留条件阈值与删失，不改称真实EOL'],
    ],columns=['项目','原方案或原资格','本次执行及边界'])
    csv(TAB/'导师意见与原方案差异.csv',diff)
    write('导师意见与原方案差异记录.md','# 导师意见与原方案差异记录\n\n已完整核阅Word第三档和第5章，配置目录保留逐段文本和原文件SHA。原Word及阶段0–1.3均未改写。\n\n'+md(diff))
    write('使用说明.md','''# 阶段2A基础预测实验

先读基础预测实验报告.md和一维二维特征对比报告.md。输出结果/表格含全部真实预测、NA、时间资格、模型参数索引；输出结果/图片含300dpi PNG及PDF；每图CSV在输出结果/绘图数据，索引为图表来源与SHA256.csv。

CPU环境Python3.11、NumPy、pandas、matplotlib、openpyxl。准备程序只在本任务中，已执行完成不可覆盖。预测入口：

`python -X utf8 -B 程序/运行基础预测.py --output 新的绝对输出目录`

默认读本阶段冻结观测。完成目录有保护标志，拒绝覆盖；重跑必须指定新目录。独立复核：`python -X utf8 -B 程序/独立复核.py --output 新的绝对输出目录`。报告绘图从本次最终CSV读取，不选择最优窗口；示例固定为最早后留合法起点。

原始大缓存只读引用不复制，SHA和来源登记于表格；这不是新造的原始数据。正式预测核心在完成独立运行检查后发布到完整代码，准备程序不会复制过去。Git按已有自动提交授权仅本地提交，不自动推送。
''')
    print('真实CSV报告与9组PNG/PDF已生成。')


if __name__=='__main__':main()
