"""只读真实计算CSV/状态记录生成图表与中文报告；不参与模型选择。"""
from pathlib import Path
import os,json
import numpy as np
import pandas as pd
WORK=Path(__file__).resolve().parents[1];GUIDE=WORK.parents[1];OUT=WORK/'输出结果';TAB=OUT/'表格';FIG=OUT/'图片';DATA=OUT/'绘图数据'
os.environ['MPLCONFIGDIR']=str(OUT/'绘图缓存')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from 运行低阶实验 import csv,sha,js
plt.rcParams.update({'font.family':'Microsoft YaHei','font.size':10,'axes.spines.top':False,'axes.spines.right':False,
    'axes.unicode_minus':False,'pdf.fonttype':42,'ps.fonttype':42,'savefig.dpi':300})
COL={'Persistence':'#777777','LocalLinear':'#B64342','BoundedRate':'#3775BA','SlowReversible':'#3C8D6C'}
LABEL={'Persistence':'持续值','LocalLinear':'局部线性','BoundedRate':'C 有界速率','SlowReversible':'D 慢变＋可逆'}
INDEX=[]

def table(d):
    if d.empty:return 'NA：无合法结果。\n'
    def fmt(x):
        if pd.isna(x):return 'NA'
        if isinstance(x,(float,np.floating)):return f'{x:.6g}'
        return str(x).replace('|','/')
    return '| '+' | '.join(d.columns)+' |\n| '+' | '.join(['---']*len(d.columns))+' |\n'+'\n'.join('| '+' | '.join(fmt(x) for x in r)+' |' for r in d.itertuples(index=False,name=None))+'\n'

def write(name,text):(WORK/name).write_text(text.rstrip()+'\n',encoding='utf-8')

def save(fig,name,data,sources,note):
    fig.tight_layout();fig.savefig(FIG/(name+'.png'),dpi=300,bbox_inches='tight');fig.savefig(FIG/(name+'.pdf'),bbox_inches='tight');plt.close(fig)
    csv(DATA/(name+'.csv'),data)
    origins={}
    for item in sources.split('；'):
        p=WORK/item
        if not p.is_file():p=OUT/item
        if not p.is_file():p=TAB/item
        assert p.is_file(),item
        origins[str(p.relative_to(WORK))]=sha(p)
    INDEX.append(dict(图=name,绘图数据=str((DATA/(name+'.csv')).relative_to(WORK)),CSV_SHA256=sha(DATA/(name+'.csv')),
        来源='；'.join(origins),来源SHA256=json.dumps(origins,ensure_ascii=False),说明=note,
        PNG_SHA256=sha(FIG/(name+'.png')),PDF_SHA256=sha(FIG/(name+'.pdf'))))

def main():
    assert (OUT/'独立复核通过.json').exists()
    for d in [FIG,DATA]:d.mkdir(exist_ok=True)
    met=pd.read_csv(TAB/'全部模型公平比较.csv');mainmet=met[met.method.isin(COL)].copy()
    states=pd.DataFrame(json.loads((OUT/'模型记录/逐起点状态与约束记录.json').read_text(encoding='utf-8')))
    chosen=json.loads((OUT/'模型记录/冻结选择结果.json').read_text(encoding='utf-8'))
    rate_records=[];recovery=[];obsall={};preds={}
    for dataset,h in [('FC1',72),('RealHyFC',12)]:
        obs=pd.read_csv(WORK/'数据准备'/('FC1小时观测.csv' if dataset=='FC1' else 'RealHyFC循环观测.csv'))
        scheme='FC1_NOMINAL' if dataset=='FC1' else 'B60';obs=obs[obs.scheme==scheme].sort_values('id');obsall[dataset]=obs
        p=pd.read_csv(TAB/f'{dataset}全部方法逐点预测.csv');preds[dataset]=p
        q=p[(p.partition=='Assessment')&(p.horizon==h)&p.method.isin(COL)];origin=q.origin_id.min();q=q[q.origin_id==origin].sort_values('step')
        fig,axes=plt.subplots(2,1,figsize=(9,6))
        valid=obs.copy();valid.loc[~valid.valid,'v']=np.nan
        axes[0].plot(valid.time if dataset=='FC1' else valid.available_h,valid.v,'k-',lw=.9,label='真实合格观测；缺测断线')
        axes[0].set(xlabel='实际带载运行小时 / h',ylabel='平均单片电压 / V',title=dataset+' 全轨迹观测；已看过的探索性研究');axes[0].legend(frameon=False)
        for method in COL:
            g=q[q.method==method];axes[1].plot(g.target_operating_h,g.pred_v,'o-',ms=2.5,lw=1.2,color=COL[method],label=LABEL[method])
        g=q[q.method=='Persistence'];axes[1].plot(g.target_operating_h,g.true_v,'k.--',ms=4,label='真实目标')
        axes[1].set(xlabel='实际带载运行小时 / h（仅审计坐标）',ylabel='平均单片电压 / V',
            title=f'后留最早完整{h}'+('小时' if dataset=='FC1' else '循环')+f'预测块；固定起点ID={int(origin)}')
        axes[1].legend(frameon=False,ncol=1,fontsize=8,loc='upper left',bbox_to_anchor=(1.01,1.))
        save(fig,dataset+'低阶模型预测曲线',pd.concat([obs.assign(kind='observation'),q.assign(kind='fixed_origin_forecast')],ignore_index=True),
            f'数据准备/{"FC1小时观测.csv" if dataset=="FC1" else "RealHyFC循环观测.csv"}；表格/{dataset}全部方法逐点预测.csv','图中预测只取最早合法起点，不拼接窗口。Real为B60、0.558A/cm²，非极化诊断SOH。')
        for _,r in states[states.dataset.eq(dataset)].iterrows():
            originobs=obs[obs.id==r.origin_id].iloc[0]
            rate_records.append(dict(dataset=dataset,origin_id=r.origin_id,model=r.model,constrained=r.constrained,q=r.q,
                operating_h=originobs.time if dataset=='FC1' else originobs.available_h,slow_D=r.L,reversible_D=r.R,
                rate_cap=chosen[dataset+'_'+r.model]['par']['rate_cap'],clipped_steps=r.clipped_steps,updates=r.updates))
        for (method,horizon,origin),g in p[p.partition.eq('Assessment')&p.method.isin(COL)].groupby(['method','horizon','origin_id']):
            o=obs[obs.id==origin].iloc[0];end=g[g.step==g.horizon].iloc[0]
            recovery.append(dict(dataset=dataset,method=method,horizon=horizon,origin_id=origin,
                actual_end_change_mV=(end.true_v-o.v)*1000,pred_end_change_mV=(end.pred_v-o.v)*1000,
                actual_rise=bool(end.true_v>o.v),predicted_rise=bool(end.pred_v>o.v),end_abs_error_mV=abs(end.pred_v-end.true_v)*1000))
    rates=pd.DataFrame(rate_records);csv(TAB/'退化速率与状态诊断.csv',rates);csv(TAB/'真实回升与预测方向逐起点.csv',pd.DataFrame(recovery))
    summary=pd.DataFrame(recovery).groupby(['dataset','method','horizon']).agg(
        n=('origin_id','size'),observed_end_rise=('actual_rise','sum'),predicted_end_rise=('predicted_rise','sum')).reset_index()
    csv(TAB/'回升预测方向汇总.csv',summary)
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    for ax,dataset in zip(axes,['FC1','RealHyFC']):
        m=mainmet[(mainmet.dataset==dataset)&(mainmet.partition=='Assessment')];hs=[24,72,168] if dataset=='FC1' else [1,4,12,25];x=np.arange(len(hs))
        for n,method in enumerate(COL):
            values=m[m.method==method].set_index('horizon').reindex(hs).RMSE_mV
            ax.bar(x+(n-1.5)*.2,values.fillna(0),.2,label=LABEL[method],color=COL[method],edgecolor='black',linewidth=.3)
            for i,v in enumerate(values):
                if pd.isna(v):ax.text(x[i]+(n-1.5)*.2,.02,'NA',rotation=90,ha='center',fontsize=7)
        ax.set(xticks=x,xticklabels=hs,xlabel='完整预测小时' if dataset=='FC1' else '完整物理循环',ylabel='全块RMSE / mV',title=dataset+' Assessment探索性再次验证')
        ax.legend(frameon=False,fontsize=8)
    save(fig,'各跨度四方法公平比较',mainmet,'表格/全部模型公平比较.csv','包含全部不利结果；168h/25循环仍NA，未补齐缺测。')
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    for ax,dataset in zip(axes,['FC1','RealHyFC']):
        d=rates[rates.dataset.eq(dataset)&rates.constrained]
        for model in ['BoundedRate','SlowReversible']:
            z=d[d.model==model].sort_values('operating_h');ax.plot(z.operating_h,z.q*1000,lw=1,label=LABEL[model],color=COL[model])
        ax.axhline(d.rate_cap.iloc[0]*1000,color='gray',ls='--',label='Train速率上界');ax.axhline(0,color='k',lw=.6)
        ax.set(xlabel='起点实际运行小时 / h',ylabel='损失速率 / '+('mV/h' if dataset=='FC1' else 'mV/循环'),title=dataset+' 因果起点速率（非物理损伤真值）');ax.legend(frameon=False,fontsize=8)
    save(fig,'长期速率变化与训练边界',rates,'模型记录/逐起点状态与约束记录.json；模型记录/FC1_Train拟合参数.json；模型记录/RealHyFC_Train拟合参数.json','每起点仅同2A历史窗估计；相邻状态可被新观测校正，不是全寿命双向分解。')
    differences=[]
    for dataset in ['FC1','RealHyFC']:
        for method in ['BoundedRate','SlowReversible']:
            a=met[(met.dataset==dataset)&(met.partition=='Assessment')&(met.method==method)]
            b=met[(met.dataset==dataset)&(met.partition=='Assessment')&(met.method==method+'_no_projection')]
            for _,r in a.iterrows():
                other=b[b.horizon==r.horizon].iloc[0]
                differences.append(dict(dataset=dataset,method=method,horizon=r.horizon,constrained_RMSE_mV=r.RMSE_mV,
                    no_projection_RMSE_mV=other.RMSE_mV,projection_gain_mV=other.RMSE_mV-r.RMSE_mV))
    dif=pd.DataFrame(differences);csv(TAB/'同结构去投影消融.csv',dif)
    fig,axes=plt.subplots(1,2,figsize=(10,4))
    for ax,dataset in zip(axes,['FC1','RealHyFC']):
        d=dif[dif.dataset==dataset];hs=sorted(d.horizon.unique());x=np.arange(len(hs))
        for n,method in enumerate(['BoundedRate','SlowReversible']):
            z=d[d.method==method].sort_values('horizon');ax.bar(x+(n-.5)*.3,z.projection_gain_mV,.3,label=LABEL[method],color=COL[method])
        ax.axhline(0,color='k',lw=.8);ax.set(xticks=x,xticklabels=hs,xlabel='预测小时' if dataset=='FC1' else '预测循环',
            ylabel='无投影减有投影RMSE / mV',title=dataset+'：正值才表示投影有利');ax.legend(frameon=False,fontsize=8)
    save(fig,'约束独立作用消融',dif,'表格/同结构去投影消融.csv','同训练参数、同选定k/tau、同容量，仅去速率投影；未给对照另调最优参数，不宣称普适因果收益。')
    rul=pd.read_csv(TAB/'FC1条件RUL误差与删失.csv');cur=pd.read_csv(TAB/'FC1条件RUL外推曲线.csv');fc=obsall['FC1'].copy();fc.loc[~fc.valid,'v']=np.nan
    fig,axes=plt.subplots(1,3,figsize=(12,4),sharey=True)
    for ax,origin in zip(axes,[400,500,600]):
        ax.plot(fc.time,fc.v,'k-',lw=.7,label='真实观测')
        for method in COL:
            d=cur[(cur.origin_h==origin)&(cur.method==method)];ax.plot(d.time_h,d.pred_v,color=COL[method],lw=1.2,label=LABEL[method])
        ax.axhline(rul[rul.main].threshold_V.iloc[0],ls='--',color='#9364A5',label='4%条件阈值');ax.axvline(origin,color='gray',ls=':')
        ax.set(xlabel='实际小时 / h',ylabel='平均单片电压 / V',title=f'{origin}h起点；未验证物理EOL');ax.legend(frameon=False,fontsize=7)
    save(fig,'FC1条件RUL四方法外推',pd.concat([cur.assign(kind='forecast'),fc.assign(kind='observation'),rul.assign(kind='threshold_record')],ignore_index=True),
        '表格/FC1条件RUL外推曲线.csv；FC1条件RUL误差与删失.csv；数据准备/FC1小时观测.csv','每个RUL起点独立80/20选择；没有借用完整Train或全局Validation。')
    fig,axes=plt.subplots(2,1,figsize=(9,6),sharex=True);z=fc[fc.time.between(780,850)]
    axes[0].plot(z.time,z.v,'k.-',ms=3,label='真实小时观测（缺测断线）');threshold=rul[rul.main].threshold_V.iloc[0]
    axes[0].axhline(threshold,color='#9364A5',ls='--',label='4%条件阈值');axes[0].axvline(809,color='gray',ls=':')
    axes[0].set(ylabel='平均单片电压 / V',title='809h首次观测越界与其后回升；不是不可逆失效');axes[0].legend(frameon=False)
    zz=cur[(cur.origin_h==600)&cur.time_h.between(780,850)]
    for method in COL:
        g=zz[zz.method==method];axes[1].plot(g.time_h,g.pred_v,lw=1.3,color=COL[method],label=LABEL[method])
    axes[1].plot(z.time,z.v,'k.-',lw=.7,ms=2,label='真实');axes[1].axhline(threshold,color='#9364A5',ls='--')
    axes[1].set(xlabel='实际小时 / h',ylabel='平均单片电压 / V',title='600h固定起点预测同一区间：无未来恢复事件输入');axes[1].legend(frameon=False,ncol=3,fontsize=8)
    save(fig,'电压下降及回升现象对照',pd.concat([z.assign(kind='observed'),zz.assign(kind='forecast_from_600h')],ignore_index=True),
        '表格/FC1_809h越界与后续回升.csv；FC1条件RUL外推曲线.csv','只说明已观察到恢复与模型允许回升；不声称可提前预测未知启停。822–827h存在缺测，恢复精确时刻未知。')
    extra=pd.read_csv(TAB/'RealHyFC_固定起点额外外推.csv');fig,ax=plt.subplots(figsize=(8,4))
    for method in ['BoundedRate','SlowReversible']:
        g=extra[extra.method==method];ax.plot(g.target_time,g.pred_v,'o-',ms=3,color=COL[method],label=LABEL[method])
    z=extra[extra.method=='BoundedRate'];ax.plot(z.target_time,z.true_v,'k.--',ms=3,label='真实；不合格位置NA')
    ax.set(xlabel='物理循环ID（不是小时）',ylabel='平均单片电压 / V',title='Real B60额外固定起点25循环展示；不进入主完整块成绩');ax.legend(frameon=False,fontsize=8)
    save(fig,'Real固定起点长程描述',extra,'表格/RealHyFC_固定起点额外外推.csv','图中有缺测；不能替代主25循环不可评价结论。')
    qualification=pd.read_csv(TAB/'一维二维状态演化资格.csv');fig,axes=plt.subplots(1,2,figsize=(10,4))
    scopes=[]
    for k in ['FC1','RealHyFC']:
        scopes.extend([
            dict(dataset=k,scope='直接运行电压损失标量C/D',status='EXECUTED_EXPLORATORY',reason='观测定义和基准已冻结；不依赖极化到运行映射'),
            dict(dataset=k,scope='极化一维/二维到运行联合状态',status='NOT_QUALIFIED',reason='时钟、配对数量和条件数见一维二维状态演化资格表')])
    csv(TAB/'标量与极化联合状态资格范围.csv',pd.DataFrame(scopes))
    for ax,dataset in zip(axes,['FC1','RealHyFC']):
        q=qualification[qualification.dataset==dataset].iloc[0]
        vals=[q.nominal_or_completion_train_pairs,q.exact_clock_train_pairs,6,9]
        ax.bar(np.arange(4),vals,color=['#777777','#3775BA','#B64342','#B64342'],edgecolor='black',linewidth=.3)
        ax.set(xticks=np.arange(4),xticklabels=['名义/完成配对','日历完成配对','1D预设最少','2D预设最少'],ylabel='不同诊断时点数',title=dataset+'：不能用密集采样扩充诊断数')
        ax.tick_params(axis='x',rotation=20)
    save(fig,'一维二维观测映射资格',qualification,'表格/一维二维状态演化资格.csv','门槛仅为预登记稳定性规则；FC1实际诊断完成未知，Real二维条件数也超限。')
    csv(TAB/'图表来源与SHA256.csv',pd.DataFrame(INDEX))
    reports(mainmet,met,rul,qualification,rates,dif,chosen)
    print('九组真实图表与四份主报告完成。')

def reports(mainmet,met,rul,qualification,rates,dif,chosen):
    validation=pd.concat([pd.read_csv(TAB/(k+'_Validation候选选择.csv')) for k in ['FC1','RealHyFC']],ignore_index=True)
    transfer=pd.read_csv(TAB/'后续300_600及后移起点协议.csv');pairs=pd.read_csv(TAB/'诊断与运行历史配对及排除.csv')
    identity=json.loads((OUT/'独立复核通过.json').read_text(encoding='utf-8'))
    recovery=pd.read_csv(TAB/'回升预测方向汇总.csv')
    params=pd.DataFrame([dict(dataset=k.split('_')[0],model=k.split('_',1)[1],selected=v['setting'],
        n_train=v['par']['n_train'],q_prior=v['par']['q_prior'],rate_cap=v['par']['rate_cap'],acceleration_cap=v['par']['acceleration_cap'],
        noise_V=v['par']['noise'],reversible_scale_V=v['par']['reversible_scale'],coordinate=v['par']['coordinate_unit']) for k,v in chosen.items()])
    csv(TAB/'已选配置与训练尺度.csv',params)
    improvements=[]
    for _,r in mainmet[mainmet.partition=='Assessment'].iterrows():
        p=mainmet[(mainmet.partition=='Assessment')&(mainmet.dataset==r.dataset)&(mainmet.horizon==r.horizon)&(mainmet.method=='Persistence')].iloc[0]
        improvements.append(dict(dataset=r.dataset,horizon=r.horizon,method=r.method,relative_RMSE_gain_pct=(p.RMSE_mV-r.RMSE_mV)/p.RMSE_mV*100))
    csv(TAB/'相对持续值改善比例.csv',pd.DataFrame(improvements))
    report=['# 物理约束模型实验报告','',
        '结论：低阶约束与可逆分量对FC1的24/72小时电压预测有实际收益；RealHyFC只在短跨度略有改善，12循环更差。条件RUL没有取得同步改善。因此不能把本轮升级为长期寿命或跨堆迁移已经成功。','',
        '## 真实执行与数学边界','',
        '本轮为确定性的CPU统计拟合与因果状态估计，未进行神经反向传播。所有速率、噪声、先验尺度由Train直接估计，Validation仅选预先固定的k或tau。保持2A的0.70/约0.558 A/cm²负载、基准、60/20/20分区、48h/12循环历史窗和完整目标块。Train用于拟合，不生成依赖完整Train参数的伪因果滚动成绩。Validation指标是选择期回放，不宣称其所选模型当时已在线冻结。','',
        'C：用Train先验速率与起点历史OLS速率按k混合，逐历史观测限制0≤q≤qmax及|Δq|≤amaxΔt，D(o+h)=D(o)+qh。它只代表无未来恢复场景，不能预测未知恢复。',
        'D：x=[L,q,R]，L+=qΔt，R*=exp(−Δt/tau)，D=L+R+epsilon。Q=diag((0.05σ)²Δt,a_max²Δt,s_R²(1−exp(−2Δt/tau)))，观测噪声σ²，因果滤波后投影q；协方差为近似记录，不声称严格截断高斯后验。预测时状态不再读未来观测，L的斜率非负但R可以均值回归，所以电压可以上升。',
        'qmax取Train局部速率绝对值Q95、下限1e−6；amax取Train局部速率变化率绝对值Q95的0.25倍、下限1e−8。σ由Train差分MAD估计、下限1e−5V；s_R由Train因果EWMA残差标准差估计。FC1单位V/h，Real单位V/物理循环（可换为名义12h时钟，但不使用未来真实带载时长）。完整公式见冻结JSON。',
        '这些是原方案启发的情景约束、训练经验尺度和统计均值回归，不是已测得电化学速率常数。两个分量不能解释为两种独立损伤；固定tau时状态结构可观测也不证明真实物理分解唯一。不同起点的新观测可修正状态，不能将其画成不可逆损伤真值。','',
        '## 恢复能力必须区分结构与实绩','',table(recovery[recovery.method=='SlowReversible']),
        '结构允许回升，合成验证也能生成回升；但本轮选定D在全部后留预测块内均没有相邻步电压回升，终点高于原始起点的预测数也为0。真实FC1 24h有34/179个终点上升，Real 1/4/12循环分别14/37、8/26、1/9个。符号比较未增加幅值阈值，仅作描述，不能将每个微小正差解释为物理恢复。当前不能宣布恢复预测成功，更不能按这些后留案例重新调tau或噪声。','',
        '## 已选参数','',table(params),'## 全部Validation候选（未删除不利配置）','',table(validation),
        '## Assessment探索性再次验证','',table(mainmet[mainmet.partition=='Assessment'][['dataset','horizon','method','n','origins','RMSE_mV','MAE_mV','RMSE_SOH_pp','MAE_SOH_pp']]),
        '各方法查询身份、真值和基准完全一致。重叠查询不是独立电堆重复。FC1 168h、Real 25循环和A60仍NA，未补齐数据。额外Real固定起点25循环仅描述，不能冒充完整块评价；FC1预登记的Train末端额外起点因尚未完成Validation选择而NA，合法长期图使用各自内层选择的400/500/600h RUL起点。','',
        '## 条件RUL与恢复','',table(rul[rul.main][['origin_h','method','actual_first_cross_h','pred_first_cross_h','abs_error_h','pred_right_censored','status']]),
        '首次合格小时越界保持809h；之后下一次已观测高于阈值在827h，前一低于阈值观测822h，823–826存在不合格/缺测，因此不能把真实恢复瞬间定为827h。809h不是已证实不可逆失效。C/D在400/500起点条件RUL误差均未超过原局部线性基线；600h虽产生越界预测，但误差较大。持续值删失不应填成零误差，不能在不同可评价子集上算误导性的统一平均。3.5–5.5%全部敏感性和删失保留。','',
        '## 状态与后续资格','',
        '已完成的C/D是直接运行电压损失标量模型，不需要极化到运行映射。下表拒绝的是极化诊断一维/二维联合状态接口，不能误读为已执行标量模型也未运行或被禁止。','',table(qualification),
        'FC1有5个Train名义配对，但真实诊断完成钟未知；Real有4个日历完成假设下的Train配对，少于预登记1D/2D的6/9个，二维标准化设计条件数约171.75>100。只算数学秩或密集采样不能补足独立诊断信息，故本轮不拟合不可靠映射，不进行二维状态校正。这个结论不删除二维研究方向。','',
        f"独立复核{identity['checks']}组通过，{identity['compared_origin_variants']}个起点变体逐点重算，最大差异{identity['max_prediction_difference_V']:.3g} V。身份字段严格一致，数值容差仅用于浮点复算，不掩盖样本错位。原阶段0–2A及所有冻结结果SHA保持不变。",
        '首次入口路径分隔符错误在拟合前修复并保留日志；没有按Assessment成绩改公式或候选。准备与绘图脚本不进入完整代码。',
        ('正式2B_v1已独立发布并重跑，23份结果/参数字节完全一致；阶段2A_v1原文件完整备份且均未覆盖，只有新增版本和版本索引。' if (OUT/'正式代码发布验收.json').exists() else '正式代码发布需先备份阶段2A_v1，再从独立入口重跑并比对。')]
    write('物理约束模型实验报告.md','\n'.join(report))
    write('模型与基础预测对比报告.md','# 模型与基础预测对比报告\n\n'+
        'FC1的C/D都减少了局部线性错误外推，D在24/72小时也优于保持法。Real只在1/4循环小幅优于保持法，12循环仍由保持法更好；不能用FC1的正面结论替代Real结果。\n\n'+
        table(pd.DataFrame(improvements))+ '\n## 同结构同信息去投影对照\n\n'+table(dif)+
        '\nFC1的72小时D去投影后误差明显增大，表明本实例的速率约束有实际数值作用。Real的C有无投影成绩一致，说明其后留收益来自先验混合而非激活的投影；Real D的1/4循环去投影略优，12循环投影略有帮助但仍输保持法。不能把全部D收益称为电化学物理贡献。对照沿用同一选定k/tau，未单独给去约束模型调参，故结论仅限这些冻结实例。\n\n'+
        '原局部线性不收缩局部斜率，短期恢复/波动会放大长期外推；Train先验收缩、有限速率和均值回归可减轻这种放大。但这仍是解释，不是温度、压力或不可逆退化的因果证明。D结构允许恢复，但本轮后留全部预测曲线仍持续下降；终点上升预测数为0，与实际终点上升不符，因此尚未取得实际恢复预测证据。RUL结果证明短中期RMSE改善不能直接转化为寿命预测改善。\n\n'+
        '所有SOH误差仅对应各自运行负载与基准，未生成诊断SOH新成绩。完整Validation/Assessment、尾部误差、逐起点恢复方向及NA均在表格内。')
    write('下一阶段迁移准备报告.md','# 下一阶段迁移准备报告\n\n'+
        '当前不满足原方案二维物理约束网络及FC1→RealHyFC正式迁移的最低条件。本轮只准备协议，不执行任何迁移训练。FC1的运行标量演化可继续作为低阶研究基础，但不能用其电压收益宣布极化状态可跨堆迁移。\n\n'+
        table(transfer)+
        '\n严格300h只允许BoL；严格600h允许BoL和已完成300h诊断，不含标称600h曲线。其运行小时区间566.16980–615.66968跨600截止。保守615.66968上界并满足检测日历结束后，现有B60观测中第一个可用循环末端为629.2038h（cycle51），因此另设该后移起点；它不是精确诊断完成运行小时。真实结果发布时间仍未知，只能用检测完成后可用的离线假设。\n\n'+
        '迁移四组保持FC1源、Real目标：目标域独立训练；直接迁移；分层适配；全量微调。全部使用目标自身BoL/负载观测标定，源标准化不能用目标后期重拟合。300/600与后移组单列；若比较不同截止，统一在最晚合法起点之后的相同目标上评分，较早截止模型只靠自身传播到该起点，禁止读中间真实目标状态。\n\n'+
        '最少需补：FC1诊断实际完成/可用时钟或明确导师认可的名义离线协议；足够不同Train诊断对以稳定估计运行—诊断映射；低负载与参考负载之间的条件差及扫描/温压证据；原方案事件历史字段可用性。Real二维设计条件数171.75还提示特征共线，增加神经参数无法自动修复。两套电堆各一条寿命，不能把窗口当成多电堆样本。\n\n'+
        '下一步优先补测量/时间与映射证据，并调查为何Real长跨度和FC1 RUL仍失效；不应直接大规模训练16单元网络。二维方向保留，不把样本不足改称二维无用。')
    write('16单元网络进入条件与预案.md','''# 16单元有界速率网络：条件性预案，尚未具备启动资格

原Word第3章的网络是单隐层16单元tanh，二维8输入/2输出、FC1标量5输入/1输出；局部漂移加b*tanh(g)，投影约束总参考损失率非负与分量修正范围。b的主尺度为Train状态变化率绝对值Q95×0.25，0及0.5倍只作预先固定敏感性。原优化设置Adam 1e−3、batch16、最多200轮、patience20；重复种子11/23/37/53/71，验证选择不得用Assessment。

实施前必须确定可观测的状态与事件历史、Train-only归一化、运行与极化映射和信息截止。当前没有合格运行—极化映射，FC1准确诊断钟与部分历史事件字段未确认，不能填零假称有效。二维7个历史特征/时间输入必须逐字段有证据；不擅自修改原输入来凑可训练网络。

若后续补齐资格，联合损失保留运行/极化/修正三项原定义，各项观测数、尺度下限和权重来源均需登记。不能将不存在的极化损失或梯度记0冒充物理作用。必须与同状态、同容量、同输入的无约束网络及低阶D对照，比较短中期、尾部和条件RUL，不能只报告最好种子。

本轮无神经损失、梯度或精度结果，均为NA而非0。当前结论为NETWORK_NOT_READY；标量低阶数值可运行不等于联合物理网络和迁移已获资格。
''')
    write('使用说明.md','''# 阶段2B使用说明

先读物理约束模型实验报告.md、模型与基础预测对比报告.md和下一阶段迁移准备报告.md。完整冻结公式见配置/阶段2B冻结协议.json，输入与历史SHA见同目录身份文件。

CPU入口：`python -X utf8 -B 程序/运行低阶实验.py`。已有完成目录禁止覆盖，复算使用`--output 新的绝对目录`。它从阶段2A冻结观测重新拟合C/D，不重跑阶段0–2A；A/B只读既有预测。

独立复核：`python -X utf8 -B 程序/独立数值复核.py --output 新输出目录`。准备性配对/迁移资格与报告绘图入口只在本任务程序目录。所有图来自真实CSV，提供PNG/PDF、绘图CSV与SHA索引；固定起点图不拼接重叠窗口。

正式完整代码更新前备份阶段2A_v1全部正式文件及SHA。正式包只发布数值核心、运行入口、配置和必要依赖，不复制本任务准备/绘图脚本。历史代码、结果与权重保留，未经本轮远端授权不推送。
''')


if __name__=='__main__':main()
