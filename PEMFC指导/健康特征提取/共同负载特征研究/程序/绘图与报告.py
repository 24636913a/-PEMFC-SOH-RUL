"""只从真实CSV绘图/汇总；不重新提取、不挑选最优规则、不训练。"""
from pathlib import Path
import json,os,sys,re
import numpy as np
import pandas as pd

WORK=Path(__file__).resolve().parent.parent;GUIDE=WORK.parent.parent;OUT=WORK/'输出结果';TAB=OUT/'表格';FIG=OUT/'图片'
sys.path.insert(0,str(GUIDE/'数据处理/程序'))
from 核验基础 import table,save_json,sha
os.environ.setdefault('MPLCONFIGDIR',str(OUT/'绘图缓存'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
plt.rcParams.update({'font.family':['Microsoft YaHei','DejaVu Sans'],'axes.unicode_minus':False,
 'axes.spines.top':False,'axes.spines.right':False,'axes.labelsize':11,'axes.titlesize':12,
 'xtick.labelsize':9,'ytick.labelsize':9,'legend.fontsize':9,'pdf.fonttype':42})
COLORS=['#3775BA','#B64342','#3C8D6C','#8A6AB8','#D59338','#657786','#A34779','#387C87','#4D6D35']

def md_table(df,columns):
    out=['|'+'|'.join(columns)+'|','|'+'|'.join(['---']*len(columns))+'|']
    for _,r in df.iterrows():
        values=[]
        for c in columns:
            v=r[c]
            values.append('NA' if pd.isna(v) else f'{v:.6f}' if isinstance(v,(float,np.floating)) else str(v))
        out.append('|'+'|'.join(values)+'|')
    return '\n'.join(out)

def short(name):
    return name.replace('FC1_Pola_T','FC1 T').replace('.xlsx','').replace('Polcurve_Nominal_RHFC_','RH ').replace('_afterAirAirStart',' 启停后').replace('.csv','')

def main():
    assert json.loads((OUT/'独立复核结果.json').read_text(encoding='utf-8'))['状态']=='PASS'
    sup=pd.read_csv(TAB/'共同负载实际支持表.csv');time=pd.read_csv(TAB/'实际时间资格.csv')
    main_names=set(time.loc[~time.启停后,'文件名']);age_names=set(time.loc[~time.启停后&(time.标称小时>0),'文件名'])
    feature=pd.read_csv(TAB/'路线A中心参数.csv');dual=pd.read_csv(TAB/'路线B双参考特征.csv');points=pd.read_csv(TAB/'路线A逐点重构.csv')
    sens=pd.read_csv(TAB/'同域处理敏感性.csv');pert=pd.read_csv(TAB/'共享BoL条件扰动.csv');cv=pd.read_csv(TAB/'共享平台分组留出.csv')
    dcheck=pd.read_csv(TAB/'单参考与双参考留查询重构.csv');conditions=pd.read_csv(TAB/'RealHyFC代表平台温压与电压.csv')
    used=sup.loc[sup.方案.isin(['FC1_10','B60'])].copy()
    used=used.merge(time[['文件名','标称小时','启停后']],on='文件名').sort_values(['电堆','标称小时','启停后','j'])
    core=feature.loc[feature.方案.isin(['FC1_10','B60'])&(feature.支持域=='主曲线跨堆共同域')&feature.文件名.isin(main_names)]
    # 相同六查询域只有17条主曲线；启停后缺失的四点拟合不混入统计。
    q=np.sort(used.j.unique());names=used.文件名.drop_duplicates().tolist();index=[]
    def save(fig,name,source):
        fig.tight_layout()
        for ext in ['png','pdf']:fig.savefig(FIG/(name+'.'+ext),dpi=300,bbox_inches='tight')
        plt.close(fig)
        for ext in ['png','pdf']:index.append(dict(图片=name+'.'+ext,数据表=source,绘图程序='程序/绘图与报告.py',研究身份='离线探索性方法研究，非未来预测',SHA256=sha(FIG/(name+'.'+ext))))
    # 离散负载坐标覆盖图：不绘制未观测区间中的点。
    mat=np.array([used.loc[used.文件名==n].set_index('j').loc[q,'共同支持'].astype(int).to_numpy() for n in names])
    fig,ax=plt.subplots(figsize=(9,8));ax.imshow(mat,aspect='auto',cmap=ListedColormap(['#E9E9E9','#3775BA']),vmin=0,vmax=1)
    ax.set_yticks(range(len(names)),[short(n) for n in names]);ax.set_xticks(range(len(q)),[f'{j:.2f}' for j in q]);ax.set(xlabel='探索性候选电流密度 (A/cm²)，横坐标为离散类别',title='共同支持覆盖：FC1原分组 / Real B60数学诊断\n蓝=本次及自身BoL均支持；灰=拒绝/缺失；温度可比性未认证')
    for a in range(len(names)):
        for b in range(len(q)):ax.text(b,a,'✓' if mat[a,b] else '×',ha='center',va='center',color='white' if mat[a,b] else '#555555')
    save(fig,'各老化阶段可用负载覆盖图','共同负载实际支持表.csv;实际时间资格.csv')
    # 每个堆分别显示各时点，缺口处分线，绝不补出0.50至0.80观测。
    p=points.loc[points.方案.isin(['FC1_10','B60'])&(points.支持域=='主曲线跨堆共同域')&points.文件名.isin(main_names)]
    for stack in ['FC1','RealHyFC']:
        fig,axs=plt.subplots(1,2,figsize=(12,4.6))
        for k,n in enumerate(used.loc[(used.电堆==stack)&used.文件名.isin(main_names),'文件名'].drop_duplicates()):
            a=p.loc[p.文件名==n].sort_values('j');color=COLORS[k%len(COLORS)]
            for ids in np.split(np.arange(len(a)),np.flatnonzero(np.diff(a.j)>.2)+1):
                g=a.iloc[ids]
                axs[0].plot(g.j,g.D_V*1000,'o-',color=color,ms=4,label=short(n) if ids[0]==0 else None)
                axs[1].plot(g.j,g.二维重构_V*1000,'-',color=color,lw=1)
                axs[1].scatter(g.j,g.D_V*1000,color=color,s=15)
        for ax in axs:ax.set(xlabel='电流密度 (A/cm²)',ylabel='相对自身BoL电压损失 (mV)');ax.axhline(0,color='gray',lw=.7)
        axs[0].set_title('真实局部支持上的损失，保留符号');axs[1].set_title('二维重构仅标在有支持的位置')
        axs[0].legend(ncol=2,fontsize=8);fig.suptitle(stack+'：离线候选表示；插值点不是独立测量')
        save(fig,stack+'共同负载损失及重构','路线A逐点重构.csv')
    # 标称时间与实际小时区间明确区分。Real BoL单独画，无虚构0小时坐标。
    for stack in ['FC1','RealHyFC']:
        fig,axs=plt.subplots(1,2,figsize=(11.5,4.5));a=dual.loc[dual.方案.isin(['FC1_10','B60'])&(dual.电堆==stack)&dual.文件名.isin(main_names)].merge(time[['文件名','累计小时下界','累计小时上界']],on='文件名').sort_values('标称小时')
        for k,(field,label) in enumerate([('D_low_V','0.45 A/cm²'),('D_ref_V','0.90 A/cm²')]):
            for _,r in a.iterrows():
                if stack=='FC1':x=r.标称小时;err=0
                elif r.标称小时==0:continue
                else:x=(r.累计小时下界+r.累计小时上界)/2;err=(r.累计小时上界-r.累计小时下界)/2
                axs[0].errorbar(x,r[field]*1000,xerr=err,fmt='o',ms=5,color=COLORS[k],label=label if r.标称小时==(0 if stack=='FC1' else 300) else None)
            axs[0].set(ylabel='参考负载损失 (mV)')
        for _,r in a.iterrows():
            if stack=='RealHyFC' and r.标称小时==0:continue
            x=r.标称小时 if stack=='FC1' else (r.累计小时下界+r.累计小时上界)/2
            err=0 if stack=='FC1' else (r.累计小时上界-r.累计小时下界)/2
            axs[1].errorbar(x,r.S_V_per_A_cm2*1000,xerr=err,fmt='o',color=COLORS[2])
        for ax in axs:ax.set_xlabel('标称老化小时 (h)，NOMINAL_ONLY' if stack=='FC1' else '运行小时约束区间 (h)，横线不是精确时刻')
        axs[1].set_ylabel('双参考斜率 (mV/(A/cm²))');axs[0].legend();fig.suptitle(stack+'双参考表示：离线方法研究'+('；BoL零点另列CSV，早于首份运行记录' if stack=='RealHyFC' else '；没有实测诊断时钟'))
        save(fig,stack+'双参考健康特征变化','路线B双参考特征.csv;实际时间资格.csv')
        a=core.loc[(core.电堆==stack)&core.文件名.isin(age_names)].sort_values('标称小时')
        fig,ax=plt.subplots(figsize=(9,4.5));xx=np.arange(len(a));ax.bar(xx-.18,a.一维RMSE_V*1000,.36,label='同六点域最佳常数',color=COLORS[1],edgecolor='black',lw=.5);ax.bar(xx+.18,a.二维RMSE_V*1000,.36,label='同六点域中心线性',color=COLORS[0],edgecolor='black',lw=.5)
        ax.set_xticks(xx,[f'{v:g}' for v in a.标称小时]);ax.set(xlabel='标称测试标签 (h)，非精确预测时间',ylabel='同曲线重构RMSE (mV)',title=stack+'：同点同权重的一维/二维重构\n不是预测性能；参数更多本身会降低拟合残差');ax.legend();save(fig,stack+'一维与二维重构比较','路线A中心参数.csv')
    fig,axs=plt.subplots(1,2,figsize=(12,4.5))
    for k,stack in enumerate(['FC1','RealHyFC']):
        a=sens.loc[(sens.电堆==stack)&sens.文件名.isin(main_names)]
        for label,g in a.groupby('方案'):
            axs[k].scatter(g.h_ref相对主方案差_V*1000,g.s相对主方案差_V_per_A_cm2*.45*1000,label=label,s=28,alpha=.8)
        axs[k].axhline(0,color='gray',lw=.7);axs[k].axvline(0,color='gray',lw=.7);axs[k].legend();axs[k].set(title=stack+'：预定处理变化，同六点支持域',xlabel='h_ref变化 (mV)',ylabel='斜率变化×0.45 A/cm² (mV)')
    save(fig,'二维特征处理稳定性','同域处理敏感性.csv')
    fig,axs=plt.subplots(1,2,figsize=(11.5,4.8));native=cv.loc[cv.方案.isin(['FC1_10','B60'])&(cv.支持域=='主曲线跨堆共同域')&cv.文件名.isin(age_names)]
    for k,stack in enumerate(['FC1','RealHyFC']):
        g=native.loc[native.电堆==stack];good=g.loc[g.可评价]
        if len(good):
            axs[k].scatter(good.真实损失_V*1000,good.一维预测_V*1000,label='一维留组',color=COLORS[1]);axs[k].scatter(good.真实损失_V*1000,good.二维预测_V*1000,label='二维留组',color=COLORS[0]);axs[k].legend()
        else:axs[k].text(.5,.5,'无合法独立留平台重构\n移除共享支持组后需跨负载外推\n保持NA，不把留查询当留平台',transform=axs[k].transAxes,ha='center',va='center')
        axs[k].set(title=f'{stack}：合法{len(good)}/{len(g)}个留组查询',xlabel='真实损失 (mV)',ylabel='留组重构 (mV)')
    save(fig,'分原始负载组留出资格','共享平台分组留出.csv')
    condused=[]
    for r in used.loc[(used.电堆=='RealHyFC')&used.共同支持].to_dict('records'):
        for gid in str(r['支持组']).split(';'):
            if gid:condused.append((r['文件名'],int(gid)))
    keys=set(condused);c=conditions.loc[[(r.文件名,r.组号) in keys for r in conditions.itertuples()]]
    table(TAB/'实际候选支持平台条件.csv',c)
    fig,axs=plt.subplots(1,3,figsize=(14,4.6));axs[0].scatter(c.temp_range,c.v_p95_p5*1000,s=20,color=COLORS[0]);axs[0].axvline(1,color=COLORS[1],ls='--');axs[0].set(xlabel='冷却液入口温度峰峰 (°C)',ylabel='电压P95−P5 (mV)',title=f'候选支持所用平台 n={len(c)}\n虚线为旧保守阈值，非仪器精度')
    delta=pd.read_csv(TAB/'与BoL平台条件差.csv');axs[1].scatter(delta.temp_差,delta.pa_差,s=18,alpha=.6,color=COLORS[2]);axs[1].set(xlabel='相对BoL入口温差 (°C)',ylabel='相对BoL阳极入口压差 (kPa)',title='同负载附近平台条件差\n仅描述，不认证等价')
    sdir=pd.read_csv(TAB/'升降载描述性对照.csv');sg=sdir.loc[sdir.双方支持];axs[2].scatter(sg.j,sg.升减降_V*1000,s=18,alpha=.6,color=COLORS[3]);axs[2].axhline(0,color='gray',lw=.7);axs[2].set(xlabel='电流密度 (A/cm²)',ylabel='升载减降载电压 (mV)',title='同文件升降载描述性差异\n不是独立跨日期重复')
    save(fig,'测量条件及方向差异','实际候选支持平台条件.csv;与BoL平台条件差.csv;升降载描述性对照.csv')
    # 固定电流位置的真实时序例证，不选择相关系数最大的窗口。
    rawcache=pd.read_csv(GUIDE/'健康特征提取/方法复核/输出结果/表格/长平台时序绘图数据.csv')
    examples=[];fig,axs=plt.subplots(3,2,figsize=(13,8),sharex='col')
    for k,(name,lo,hi) in enumerate([('Polcurve_Nominal_RHFC_BoL.csv',.89,.91),('Polcurve_Nominal_RHFC_850h.csv',.71,.73)]):
        a=rawcache.loc[(rawcache.文件名==name)&rawcache.主扫描]
        med=a.groupby('平台键').j.median();keys=med.loc[med.between(lo,hi)].index
        assert len(keys)==1
        a=a.loc[a.平台键==keys[0]].copy();a['距平台结束秒']=a.平台内秒-a.平台内秒.max();a=a.loc[a.距平台结束秒>-90]
        examples.append(a)
        axs[0,k].plot(a.距平台结束秒,a.temp,color=COLORS[0],lw=1);axs[0,k].set(ylabel='冷却液入口 (°C)',title=short(name)+'，固定负载位置')
        axs[1,k].plot(a.距平台结束秒,a.v*1000,color=COLORS[2],lw=1);axs[1,k].set_ylabel('平均单片电压 (mV)')
        axs[2,k].plot(a.距平台结束秒,a.pa,color=COLORS[1],lw=1);axs[2,k].set(ylabel='阳极入口压力 (kPag)',xlabel='距原平台结束 (s)')
    table(TAB/'温度周期与压力拒绝原始例证.csv',pd.concat(examples,ignore_index=True))
    fig.suptitle('真实末90秒：持续温度记录周期与电压/压力分别展示；未识别控制器或传感器原因')
    save(fig,'温度周期与压力拒绝原始例证','温度周期与压力拒绝原始例证.csv')
    rec=pd.read_csv(TAB/'启停恢复配对.csv');fig,ax=plt.subplots(figsize=(9,4.5))
    for h,g in rec.loc[rec.双方支持].groupby('标称小时'):ax.plot(g.j,g.后减前_V*1000,'o',label=f'{h}h标签')
    ax.axhline(0,color='gray',lw=.7);ax.set(xlabel='电流密度 (A/cm²)',ylabel='启停后减前电压 (mV)',title='恢复变化依赖负载：低负载并非都回升\n缺失位置不补齐；不等同不可逆损伤修复');ax.legend();save(fig,'启停前后负载依赖变化','启停恢复配对.csv')
    # 保存每日期条件扰动分布；共享BoL不能将其误差按样本数缩小。
    unc=[]
    for (stack,name),g in pert.groupby(['电堆','文件名']):
        r=dict(电堆=stack,文件名=name,扰动次数=len(g),解释='平台内条件分块扰动范围，不是仪器/重复测量95%置信区间')
        for f in ['h_ref_V','s_V_per_A_cm2','D_low_V','D_ref_V','S_V_per_A_cm2']:
            if g[f].notna().any():r[f+'_p025']=g[f].quantile(.025);r[f+'_p975']=g[f].quantile(.975)
        unc.append(r)
    table(TAB/'条件扰动区间.csv',unc)
    bsens=[]
    for (stack,name),g in sens.groupby(['电堆','文件名']):
        base=g.loc[g.方案==('FC1_10' if stack=='FC1' else 'B60')].iloc[0]
        r=dict(电堆=stack,文件名=name,主双参考可用=bool(np.isfinite(base.S_V_per_A_cm2)),可用方案数=int(g.S_V_per_A_cm2.notna().sum()))
        for field in ['D_low_V','D_ref_V','S_V_per_A_cm2']:
            r[field+'_最大差']=float((g[field]-base[field]).abs().max()) if np.isfinite(base[field]) else np.nan
            r[field+'_最大相对差']=r[field+'_最大差']/abs(base[field]) if np.isfinite(base[field]) and abs(base[field])>1e-12 else np.nan
        bsens.append(r)
    table(TAB/'双参考处理稳定性.csv',bsens)
    bss=pd.DataFrame(bsens);bss=bss.loc[bss.文件名.isin(age_names)]
    covariance=[]
    for stack,g in pert.groupby('电堆'):
        n=g.文件名.unique()
        for i in range(len(n)):
            for j in range(i+1,len(n)):
                a=g.loc[g.文件名==n[i]].sort_values('扰动');b=g.loc[g.文件名==n[j]].sort_values('扰动')
                covariance.append(dict(电堆=stack,文件A=n[i],文件B=n[j],共享BoL_Dref方差_V2=float(np.var(a.仅BoL_Dref扰动_V,ddof=1)),
                    损失Dref协方差_V2=float(np.cov(a.D_ref_V,b.D_ref_V,ddof=1)[0,1]),解释='共享基准引入相关，不可当独立标签误差'))
    table(TAB/'共享BoL协方差影响.csv',covariance)
    assoc=[];perdate=[]
    realcore=core.loc[(core.电堆=='RealHyFC')&core.文件名.isin(age_names)]
    for _,r in realcore.iterrows():
        g=delta.loc[(delta.文件名==r.文件名)&delta.j.isin(c.loc[c.文件名==r.文件名,'代表j'])]
        d={'文件名':r.文件名,'h_ref_V':r.h_ref_V,'s_V_per_A_cm2':r.s_V_per_A_cm2,'平台对数':len(g)}
        for f in ['temp_差','tout_差','pa_差','pc_差','tgas_a_差','tgas_c_差']:d[f]=float(g[f].mean())
        perdate.append(d)
    dated=pd.DataFrame(perdate);table(TAB/'特征与实际支持平台条件差.csv',dated)
    for target in ['h_ref_V','s_V_per_A_cm2']:
        for field in ['temp_差','tout_差','pa_差','pc_差','tgas_a_差','tgas_c_差']:
            a=dated[[target,field]].dropna();r=float(a.corr().iloc[0,1]) if len(a)>2 and a.std().min()>0 else np.nan
            assoc.append(dict(特征=target,条件差=field,日期数=len(a),Pearson_r=r,解释='固定全部关联，不择优；老化/日期混杂，非因果，无显著性检验或校正'))
    table(TAB/'特征与温压描述关联.csv',assoc)
    # 逐堆、相同信息范围汇总；不将两路线不同输入点数做统一优劣排名。
    summary=[]
    for stack in ['FC1','RealHyFC']:
        a=core.loc[(core.电堆==stack)&core.文件名.isin(age_names)]
        se=sens.loc[(sens.电堆==stack)&sens.文件名.isin(main_names)]
        dr=dual.loc[dual.方案.isin(['FC1_10','B60'])&(dual.电堆==stack)&dual.文件名.isin(main_names)]
        cvg=native.loc[(native.电堆==stack)&native.可评价]
        d=dcheck.loc[dcheck.方案.isin(['FC1_10','B60'])&(dcheck.电堆==stack)&dcheck.文件名.isin(age_names)]
        r=dict(电堆=stack,非BoL主曲线数=len(a),共同候选点数=6,
          一维重构RMSE中位_mV=float(a.一维RMSE_V.median()*1000),二维重构RMSE中位_mV=float(a.二维RMSE_V.median()*1000),
          h_ref最大处理差_mV=float(se.h_ref相对主方案差_V.abs().max()*1000),
          s最大处理差_V_per_A_cm2=float(se.s相对主方案差_V_per_A_cm2.abs().max()),
          设计矩阵条件数=float(a.设计矩阵条件数.iloc[0]),参数相关=float(a.参数相关.iloc[0]),原参数相关=float(a.原参数相关.iloc[0]),
          双参考主曲线可计算含BoL=int(dr.双参考支持.sum()),留平台合法查询数=len(cvg),
          留平台一维RMSE_mV=float(np.sqrt(np.mean((cvg.真实损失_V-cvg.一维预测_V)**2))*1000) if len(cvg) else np.nan,
          留平台二维RMSE_mV=float(np.sqrt(np.mean((cvg.真实损失_V-cvg.二维预测_V)**2))*1000) if len(cvg) else np.nan,
          双参考留查询点数=len(d),单参考留查询RMSE_mV=float(np.sqrt(np.mean((d.真实损失_V-d.单参考重构_V)**2))*1000),
          双参考留查询RMSE_mV=float(np.sqrt(np.mean((d.真实损失_V-d.双参考重构_V)**2))*1000))
        summary.append(r)
    table(TAB/'同域表示诊断汇总.csv',summary);su=pd.DataFrame(summary).set_index('电堆')
    table(TAB/'图片与真实CSV索引.csv',index)
    decision={'数学表示':'CONDITIONAL_PASS','测量可比性':'BLOCKED','因果时间资格':'BLOCKED','允许正式退化与迁移训练':'NO',
       '解释':'数学计算和限定处理稳定不等于测量等价；缺传感器/温控与FC1同步条件、真实诊断/发布时间；主曲线负载有0.30中间空档，Real独立留平台不可评价',
       '阶段1结论':'BLOCKED保持不变','路线A':'探索性可计算，不恢复原正式资格','路线B':'主曲线可计算，2300h启停后主B60拒绝；更直观但非已认证二维状态'}
    save_json(OUT/'分层结论.json',decision)
    test=json.loads((OUT/'独立复核结果.json').read_text(encoding='utf-8'))
    rfc=su.loc['FC1'];rr=su.loc['RealHyFC']
    lines=['# 共同负载特征研究报告','',
      '**结论：找到可计算的共同负载表示，但尚未找到已经认证的跨电堆共同测量条件。值得继续第三档准备研究，当前不进入正式退化或迁移训练。**','',
      '本轮是一次探索性离线方法研究。阶段1仍为BLOCKED；没有模型训练、预测精度、SOH或RUL成绩。所有候选位置来自阶段1.1看过全寿命数据后的发现，不属于独立验证。',
      '', '## 1. 实际来源与共同负载',
      '重新读取并核对19份极化源：FC1 8条、RealHyFC主曲线9条和启停后2条。全部原始扫描及原始行号保留；仅最早完整降载段用于主分析。没有重读860万行运行数据。',
      '七个候选中，17条主曲线共同**数学支持**为0.45、0.50、0.80、0.85、0.90、0.95 A/cm²；FC1在1.00附近有原始记录，但原窄分组代表点不能两侧括定，保持NA。RealHyFC主曲线B60可计算全部七个候选；这不是温度可比性认证。',
      '六点跨度0.50，但0.50与0.80之间有0.30 A/cm²空档。查询点由局部平台插值得到，不是六次独立实测，也不代表空档连续有支持。不对空档填值或外推。',
      '850h约0.719474平台仍因阳极入口压力峰峰超过3 kPa拒绝；中段不能插值。2300h原曲线约0.359246平台拒绝，但0.45/0.90邻域仍可括定。2300h启停后约0.449167平台拒绝，B60和B90的0.45/0.50不得跨越它插值；B30可能有支持，但不挑B30替换主结果。',
      '本轮拒绝平台屏障比旧诊断的单纯括距检查更严格，是独立登记的方法研究变化，不修改旧结果。启停后曲线不能混入17条主曲线全交集；若全部19条都要求有支持，跨堆交集进一步缩至0.80、0.85、0.90、0.95。',
      '', '## 2. 路线A：中心负载表示',
      '`D=V_BoL−V_t=h_ref+s(j−0.9)`；h_ref单位V，s单位V/(A/cm²)，等价Ω·cm²，但不是膜内阻。每个查询等权，避免给未测大缺口赋积分权重。原`l1=h_ref−0.9s`、`l2=0.9s`。重参数化不增加数据。',
      md_table(pd.DataFrame(summary),['电堆','非BoL主曲线数','一维重构RMSE中位_mV','二维重构RMSE中位_mV','h_ref最大处理差_mV','s最大处理差_V_per_A_cm2']),
      f'表中FC1为7条非BoL，RealHyFC为8条；相同六点域。条件数均为{rfc.设计矩阵条件数:.6f}（j按A/cm²数值计算），中心参数相关{rfc.参数相关:.6f}，原坐标相关{rfc.原参数相关:.6f}。条件数依赖单位，相关来自设计协方差，不是独立重复测量的经验相关。',
      '相较旧的j=0截距，中心表示减少解释上的抵消，但不解决温压系统差和缺口。RealHyFC部分等价l1为负，与正l2抵消；允许其符号，不能分别叫活化/欧姆损伤。2300h启停后只有高负载4点，跨度0.15、条件数约17.90，其单独拟合不进入六点同域排名。',
      '', '## 3. 分负载留出与第二维的信息',
      f'按本次及BoL共享原始平台连通分组后，FC1非BoL共有{int(rfc.留平台合法查询数)}个合法留组查询，一维/二维RMSE为{rfc.留平台一维RMSE_mV:.6f}/{rfc.留平台二维RMSE_mV:.6f} mV；这只是部分中间负载的重构，不代表全部负载或未来预测。',
      'RealHyFC六个查询实际连成少数共享平台组，移除一个组后其他负载不能包围留出点；不做外推时，没有合法独立留平台误差，保留NA。因此不能只凭0.267 mV级训练重构误差宣布二维额外信息已得到独立验证。具体每个拒绝都在共享平台分组留出.csv。',
      '', '## 4. 路线B：双参考表示',
      '`D_low=D(0.45)`、`D_ref=D(0.90)`；`S=(D_ref−D_low)/0.45`。D_low/D_ref与D_ref/S完全可逆，只是相同两观测的两种表达，不是新增一个物理状态。',
      '主规则下FC1 8/8、RealHyFC主曲线9/9（均含BoL）可以计算；启停后1500h可计算，2300h主B60拒绝。每个堆保留自己的BoL，不能跨堆共用电压基准。',
      '只给路线A两个相同端点，其系数与路线B严格数学等价；数值差见路线B双参考特征.csv。路线A六点拟合比双参考多用观测，不能拿两者残差直接排名。',
      md_table(pd.DataFrame(summary),['电堆','双参考留查询点数','单参考留查询RMSE_mV','双参考留查询RMSE_mV']),
      '上表只使用非BoL主曲线的0.50/0.80/0.85查询：端点线性重构相比单一D_ref能表达负载依赖，但这些查询与端点共享插值平台。其误差不是独立留负载验证，不是时序预测。两负载健康信息有描述性价值，尚未证实可独立预测的二维状态。',
      '', '## 5. 稳定性与基准误差',
      'FC1使用原0.005/0.010/0.020分组宽度；无时钟，不假造30/60/90秒窗口。Real固定B30/B60/B90，不挑最优窗口；仅同域比较参数。平台末90秒另拆为三个互不重叠30秒段。全部有原始行和组号。',
      '100次分块扰动复用同一BoL扰动至所有日期。BoL自身损失严格为零只是定义，不是精度验证；共同基准导致日期间误差相关，见共享BoL协方差影响.csv。分块范围不包含传感器偏差、温压系统影响或跨日期重复性，不能当真实95%测量置信区间。',
      '末段均值和处理参数变化较小，支持有限的数值稳定性；没有独立重复极化测试，不能宣称跨日期重测可靠性通过。升降载按各自方向的BoL描述，不混合拟合；恢复前后存在负载依赖，不能强制单调损失。',
      f'双参考同样保留处理差异：非BoL主曲线中D_low、D_ref最大处理差分别为{bss.D_low_V_最大差.max()*1000:.6f}和{bss.D_ref_V_最大差.max()*1000:.6f} mV；S最大相对处理差{bss.S_V_per_A_cm2_最大相对差.max()*100:.3f}%。早期小斜率对相对误差更敏感，不能只报告中心h_ref稳定就称两特征完全可靠。逐曲线见双参考处理稳定性.csv。',
      '', '## 6. 哪个障碍最大',
      '原温度峰峰规则是阻断原提取的主要计数原因；温度规则不被实验公差认证，是目前测量解释的重要缺口。压力拒绝造成具体局部缺失，负载支持不足限制形状辨识和独立留组检验。现有观测不能定量判定“真实温度影响大于压力影响”；它们与老化/日期共同变化，没有因果分离依据。',
      '', '## 7. 分层验收及下一步',
      '|项目|结论|含义|','|---|---|---|','|数学表示|CONDITIONAL_PASS|主曲线可计算，预定处理下稳定；缺口与独立留负载验证不足|','|测量可比性|BLOCKED|FC1同步温压缺失，Real温度/压力允许范围及测量精度未认证|','|因果时间资格|BLOCKED|FC1标称时间；Real实际小时有区间且结果发布未知|','|允许正式退化与迁移训练|NO|不能用数学拟合替代测量与因果资格|',
      '建议继续第三档，但先以双参考损失作为更直观的候选测量表示，中心线性作为形状诊断。下一步优先补齐FC1诊断时钟/同步工况与Real温控、公差、压力位置、传感器及结果发布时间证据；确认至少两负载同条件重复性。只补资料与有限重测验证，不再加特征候选。',
      '具备这些条件后，再单独冻结小规模持续值/局部趋势等低阶退化基线的训练截止与验证协议。当前不启动正式神经网络或跨堆迁移；若只能补齐0.9参考，先保留一维表示而不是强行要求两个状态。',
      '', '## 8. 工程复核及历史保护',
      f'{test["源文件SHA数"]}个源SHA、{test["历史保护文件数"]}个历史文件通过；独立复算{test["独立代表点数"]}个代表点、{test["独立局部支持数"]}个局部值及{test["独立中心拟合数"]}次拟合。局部值最大差{test["局部插值最大差_V"]:.3e} V，中心系数最大差{test["中心系数最大差"]:.3e}（依字段为V或V/(A/cm²)）。八项边界/解析测试通过。严格A60与11条原曲线的逐组资格一致。',
      '工程故障完整保留：首次留出空训练集合索引类型错误；第二次全部CSV已完成，末尾测试夹具缺row_end字段失败。两者修复不改数学规则，第二次数值CSV经独立复算后续完，没有第三次提取或用缓存冒充训练。失败日志及当时代码在输出结果/失败记录与程序/历史副本。',
      '远端main两次连接失败，不能认证本地基准就是最新远端；本地ad57c2dc工作区起始干净，未修改网络设置。新程序仅在本任务程序目录；完整代码和阶段0/1/1.1原文件未动。本轮按此前明确授权仅作本地Git提交，不推送。',
      '', '复现/索引：使用说明.md、输出结果/表格/图片与真实CSV索引.csv、独立复核结果.json、配置/执行前证据.json。']
    report=re.sub(r'(\|[^\n]+\|)\n\n(?=\|)',r'\1\n','\n\n'.join(lines))
    (WORK/'共同负载特征研究报告.md').write_text(report+'\n',encoding='utf-8')
    spectrum=pd.read_csv(GUIDE/'健康特征提取/方法复核/输出结果/表格/全Nyquist温度频谱复核.csv')
    # 原队列的范围从历史真实CSV获取，不把4秒归因于已知硬件。
    period_col='最强全带周期_s'
    if '主扫描' in spectrum:spectrum=spectrum.loc[spectrum.主扫描]
    w=pd.read_csv(TAB/'平台末段不重叠窗口.csv');ww=w.groupby(['文件名','组号']).v_mean.agg(lambda v: v.max()-v.min())
    sd=sg.升减降_V*1000
    text=['# 测量条件可比性分析','',
      '**结论：有真实负载支持和平台内数值稳定性证据，但无法将不同日期、两个电堆的变化全部解释为同一种退化。测量可比性BLOCKED，非“所有测量无效”。**',
      '', '## 原论文与实际字段',
      'Fensterle等原论文doi:10.1016/j.ijhydene.2026.156060，本地PDF及先前逐页核对文本已阅读。表1名义冷却液入口70°C、气体入口75°C、出口阳极/阴极160/150 kPaG、RH50/35%、化学计量比1.5/1.7。表3给出j≥0.3每点5分钟、低于0.3每点2分钟、2 A/s斜坡；1 A/cm²预调理30分钟。',
      'T.Si.CL/T.So.CL为冷却液入口/出口；T.Si.A/T.Si.C为气体入口；p.Si.A/p.Si.C为入口压力，p.So.A/p.So.C为出口压力。原压力筛选用入口，不能拿名义出口160/150直接判断入口错误。阶段0登记的FC1运行压力为mbara绝压，不能和Real kPag表压直接混算；FC1极化记录本身没有同时温压，不用附近运行值冒充。',
      '原文未规定末60秒温度峰峰1°C，也未提供冷却液温度传感器型号、准确度、温控允许波动或各通道同步记录机制。不能声称作者认可取消限制。本轮B只是诊断，不是新物理稳态标准。3 kPa压力门槛保持原数值，未用P95范围替换峰峰使平台通过。',
      '', '## 平台与窗口实测',
      f'本轮候选真正引用的B60代表平台共有{len(c)}个（同一寿命轨迹内的技术平台），入口温度峰峰范围{c.temp_range.min():.6f}–{c.temp_range.max():.6f}°C；其电压P95−P5范围{c.v_p95_p5.min()*1000:.6f}–{c.v_p95_p5.max()*1000:.6f} mV。温度与电压稳定性不是同一个判据。',
      f'所有B90合格降载平台拆分三个不重叠30秒段后，段电压均值极差中位{ww.median()*1000:.6f} mV、最大{ww.max()*1000:.6f} mV。该比较不含被拒绝平台，也不能证明长期重测准确度。',
      f'复用阶段1.1全Nyquist记录周期数据：约{float(spectrum[period_col].median()):.6f}秒的主要温度记录周期仍只是观测事实。约0.099678°C数值间隔不是仪器精度；温度与电压相关性不是因果证据。控制器周期、传感器、采集同步等原因均未证实，没有做移位或温度修正。',
      '本轮还从已核验的原始时序缓存提取BoL约0.9及850h约0.72的固定平台末90秒，逐条显示温度、电压和入口压力，见温度周期与压力拒绝原始例证.png及同名CSV。例证只按文件和负载确定，没有挑选最大相关时段。',
      '', '## 同负载的跨日期差异',
      f'在原始负载邻近平台配对的全部描述记录中，相对BoL入口均温差范围{delta.temp_差.min():.6f}–{delta.temp_差.max():.6f}°C，出口温差范围{delta.tout_差.min():.6f}–{delta.tout_差.max():.6f}°C；入口压力差等完整列在与BoL平台条件差.csv。该表包括不止七候选所用的原生平台，不能和六点损失做不注明范围的相关排名。',
      f'所有共同有支持的升降载候选配对共{len(sg)}个查询，升减降电压中位{sd.median():.6f} mV，最大绝对差{sd.abs().max():.6f} mV。不同方向不可合并为同一等价测量，不能把同文件内滞后检查当跨日期独立重复。',
      '8条非BoL主曲线的h_ref/s与实际支持平台温压相对BoL差的全部固定Pearson关联见特征与温压描述关联.csv；未挑最好相关、未作显著性检验。年龄、日期、运行状态混杂，不能用这些关联拟合温度修正或判定老化机制。',
      '1500h恢复前后在0.45附近电压下降，而0.90附近上升；2300h启停后低负载压力失败不可配对，高负载可描述回升。变化不是统一常数，也不等于不可逆损伤修复。所有数值与测量日期见启停恢复配对.csv。',
      '', '## 共同BoL误差和二维参数',
      '每个电堆各用自己的BoL。100次相同BoL扰动传播到各日期，因此日期间损失误差相关；不能按9个点把基准系统误差当独立噪声平均掉。FC1分组敏感性、Real窗口敏感性仅覆盖部分处理不确定性。中心化降低原截距与斜率的代数相关，不创造缺失负载、温控知识或新的独立寿命轨迹。',
      '', '## 时间可用性',
      'FC1全部NOMINAL_ONLY。Real BoL早于首份运行记录，不能设成72.78869小时BoL；300h曲线运行小时边界312.15411–327.81354，不能用于严格300小时截止；600h边界566.16980–615.66968，不确认严格600小时已可得。图中只用实测运行小时区间横线，不把区间中点宣称精确测量时刻。所有诊断结果发布时间未知；测量结束不等于可训练标签发布。',
      '', '## 最低补齐条件',
      '1. FC1极化原始发布档案、诊断时钟、扫描/停留协议及同时温压或可信试验协议证据。',
      '2. Real冷却液测量链、同步、传感器准确度、允许温控范围；入口/出口压力传感器与允许波动依据。',
      '3. 至少0.45和0.90两个负载在可确认相同条件下的重复测量，保留压力失败拒绝；若没有条件，不用温度回归强行校正。',
      '4. 明确诊断结果可用时间和未来少样本训练截止；不从已看的2300h结果反选迁移参数。',
      '条件分层可作为下一步建议：同扫描方向、相同测试对象/恢复状态、按已确认温压范围分层；目前缺公差证据，不具体发明新的温压分层边界。',
      '', '准备代码、来源SHA、CSV与图像均在本任务内，不复制进完整代码。数学可计算CONDITIONAL_PASS；测量可比性BLOCKED；严格因果资格BLOCKED；正式训练NO。']
    (WORK/'测量条件可比性分析.md').write_text('\n\n'.join(text)+'\n',encoding='utf-8')
    readme='''# 阶段1.2共同负载特征研究

先阅读共同负载特征研究报告.md及测量条件可比性分析.md。所有结果是探索性离线方法研究，不是未来预测。阶段1原BLOCKED不变。

程序均在本目录的程序文件夹，未进入完整代码。无神经训练、SOH/RUL或迁移。

## 已完成结果复核（推荐）

在本目录用项目Python 3.11执行：

```powershell
& 'D:\\ruanjian\\python3.11.9\\python.exe' -X utf8 -B '程序\\独立复核.py'
```

它只读源SHA和历史保护清单，独立重聚合成员、闭式复算插值和中心参数，检查B特征及严格A60原资格。结果写入本任务的独立复核结果.json。不会重复全量运行数据审计。

## 全部分析复现

程序/共同负载研究.py为数值入口，只读19份极化源。为避免覆盖已完成结果，存在数值运行完成.json时拒绝重跑。复现应在另外的仓库副本中保留相同历史证据，先归档本任务输出目录再运行；不要删除本次成果。

执行顺序：共同负载研究.py → 独立复核.py → 绘图与报告.py → 交付验收.py。仅图表重绘会更新本任务派生文件；先留存旧输出。--核验后续完只用于本次已有CSV经过独立复核后的末尾标记修复，不重新提取。

## 文件索引

共同负载实际支持表.csv列出候选点左右负载、原始行、BoL支持与拒绝原因；逐扫描原始邻域支持.csv只证明原始覆盖。平台及分组筛选.csv与代表点原始成员.csv追溯全部原生负载。

原始行号统一为去掉表头后的零基记录索引；普通单表头CSV/Excel显示行号通常为该索引加2。相同数值负载也不能跨文件、扫描或被拒绝平台互换。

路线A中心参数.csv及路线A逐点重构.csv包含实际支持域；主曲线六点同域与单曲线域不能混排。启停后2300h缺低负载，其四点拟合单列。

路线B双参考特征.csv包含D_low/D_ref/S。单参考与双参考留查询重构.csv并非独立留平台，独立留组资格另见共享平台分组留出.csv。

同域处理敏感性.csv、平台末段不重叠窗口.csv、共享BoL条件扰动.csv及共享BoL协方差影响.csv保留处理稳定性与基准关联误差。实际时间资格.csv保留区间和未知发布时间。

图片与真实CSV索引.csv给出每张300dpi PNG和PDF对应的真实表格、生成代码与SHA。原始科学数值使用V、A/cm²；图中电压损失用mV，斜率用mV/(A/cm²)。

失败记录保留两次工程错误及修复前代码，不混入正式交付数值。新结果未改写阶段0/1/1.1。远端核对因连接失败未完成；仅本地Git提交，不push。
'''
    (WORK/'使用说明.md').write_text(readme,encoding='utf-8')
    print('图像、同域汇总与两份研究报告已生成；分层结论:',decision,flush=True)

if __name__=='__main__':main()
