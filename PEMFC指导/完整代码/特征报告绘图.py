"""阶段1真实CSV绘图和中文报告；不修改任何提取参数。"""
from pathlib import Path
import os
import numpy as np
import pandas as pd
from 核验基础 import table, sha, save_json


def deliver(stage,out,cfg,feature,curves,sensitivity):
    os.environ.setdefault('MPLCONFIGDIR',str(out/'绘图配置缓存'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    chinese=next((x for x in ['Microsoft YaHei','SimHei','Noto Sans CJK SC'] if any(f.name==x for f in font_manager.fontManager.ttflist)), 'DejaVu Sans')
    plt.rcParams.update({'font.family':[chinese,'DejaVu Sans'],'axes.unicode_minus':False,
                         'font.size':11,'axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':1.2,
                         'pdf.fonttype':42,'savefig.dpi':300})
    main=feature.loc[feature.方案=='主规则'].copy();q=curves.loc[curves.方案=='主规则'].copy()
    index=[];not_generated=[]
    def save(fig,name,source):
        fig.savefig(out/'图片'/f'{name}.png',dpi=300,bbox_inches='tight')
        fig.savefig(out/'图片'/f'{name}.pdf',bbox_inches='tight')
        plt.close(fig)
        index.append(dict(图名=name,绘图CSV=source,PNG_SHA256=sha(out/'图片'/f'{name}.png'),
                          图像身份='离线描述性重构与条件扰动；非时间预测，非SOH'))
    def label(r):
        return ('BoL' if r.标称老化小时==0 else f'{int(r.标称老化小时)}h')+(' 启停后' if r.启停后 else '')
    def order(s):return s.sort_values(['标称老化小时','启停后'])
    for stack in ['FC1','RealHyFC']:
        fm=order(main.loc[main.电堆==stack]); colors=plt.cm.viridis(np.linspace(.05,.95,len(fm)))
        if not (fm.计算状态=='COMPUTED').any():
            audit=pd.read_csv(out/'表格/平台及分组筛选明细.csv')
            rows=[]
            for _,r in fm.iterrows():
                a=audit.loc[(audit.电堆==stack)&(audit.文件名==r.文件名)&(audit.方案=='主规则')&audit.实测j.between(.2,1.1)]
                rows.append(dict(文件名=r.文件名,标签=label(r),长平台数=int((a.平台秒>=180).sum()),合格平台数=int(a.保留.sum()),共同支持网格数=int(r.有效网格数)))
            support=pd.DataFrame(rows);table(out/f'表格/{stack}平台覆盖绘图数据.csv',support)
            fig,axes=plt.subplots(1,2,figsize=(13,5));xx=np.arange(len(support))
            axes[0].bar(xx-.18,support.长平台数,.36,label='≥180秒平台',color='#3775BA')
            axes[0].bar(xx+.18,support.合格平台数,.36,label='全部稳态检查通过',color='#B64342')
            axes[1].bar(xx,support.共同支持网格数,color='#767676');axes[1].axhline(cfg['拟合']['最少网格数'],ls='--',color='#B64342',label='最低10格')
            for ax in axes:
                ax.set_xticks(xx,support.标签,rotation=60,ha='right');ax.legend(fontsize=9)
            axes[0].set_ylabel('0.2–1.1 A/cm²附近平台数');axes[1].set_ylabel('与本堆BoL共同合法网格数')
            fig.suptitle(f'{stack}：冻结规则下不具二维提取资格（标签为文件名义身份，不是精确运行时钟）');fig.tight_layout()
            save(fig,f'{stack}平台覆盖与提取拒绝依据',f'{stack}平台覆盖绘图数据.csv')
            for suffix in ['相对初始极化电压损失','二维健康特征随老化变化','一维与二维拟合对比']:
                not_generated.append(dict(图名=stack+suffix,原因='REJECTED_SUPPORT：不生成无真实二维结果的空白图'))
            continue
        fig,ax=plt.subplots(figsize=(9,5))
        for color,(_,r) in zip(colors,fm.iterrows()):
            p=q.loc[(q.电堆==stack)&(q.文件名==r.文件名)]
            ax.plot(p.j,p.损失_V*1000,marker='o',ms=3,color=color,ls='--' if r.启停后 else '-',label=label(r))
        ax.axhline(0,color='gray',lw=.8);ax.set(xlabel='电流密度 j (A/cm²)',ylabel='相对本堆BoL电压损失 (mV)',
            title=f'{stack}：降载损失；仅绘制受支持网格，保留负值')
        ax.legend(ncol=3,fontsize=9,loc='best');fig.tight_layout()
        save(fig,f'{stack}相对初始极化电压损失','共同网格损失及一二维重构.csv')
        fig,axs=plt.subplots(1,2,figsize=(12,4.8),sharex=True)
        if stack=='FC1':
            xx=fm.标称老化小时.to_numpy();xlabel='标称老化小时 (h)，NOMINAL_ONLY'
        else:
            xx=pd.to_datetime(fm.实际测试开始,format='mixed');xlabel='实际极化测试日历日期；OpHrs2区间另列于CSV'
        for ax,field,lo,hi in zip(axs,['l_1','l_2'],['l1条件扰动下限_V','l2条件扰动下限_V'],['l1条件扰动上限_V','l2条件扰动上限_V']):
            normal=~fm.启停后.to_numpy();valid=np.isfinite(fm[field].to_numpy(float));m=valid&normal
            ax.plot(np.asarray(xx)[m],fm.loc[m,field]*1000,'o-',color='#0F4D92',label='主曲线')
            # 条件扰动范围可能非对称且非中心，使用vlines而非人为对称误差棒。
            ax.vlines(np.asarray(xx)[valid],fm.loc[valid,lo]*1000,fm.loc[valid,hi]*1000,color='#767676',lw=1)
            after=valid&~normal
            if after.any():ax.scatter(np.asarray(xx)[after],fm.loc[after,field]*1000,marker='s',color='#B64342',label='启停后单列')
            ax.axhline(0,color='gray',lw=.8);ax.set(ylabel=f'{field} (mV)',xlabel=xlabel)
            if stack=='RealHyFC':ax.tick_params(axis='x',rotation=25)
            ax.legend(fontsize=9)
        fig.suptitle(f'{stack}二维数学特征：竖线为条件分块扰动范围，不是独立测试置信区间',fontsize=12)
        fig.tight_layout();save(fig,f'{stack}二维健康特征随老化变化',f'{stack}二维健康特征.csv')
        fig,axes=plt.subplots(int(np.ceil(len(fm)/3)),3,figsize=(13,3.2*int(np.ceil(len(fm)/3))),sharex=True)
        for ax,(_,r) in zip(axes.flat,fm.iterrows()):
            p=q.loc[(q.电堆==stack)&(q.文件名==r.文件名)]
            ax.plot(p.j,p.损失_V*1000,'ko',ms=3,label='原始代表点局部插值损失')
            ax.plot(p.j,p.二维重构_V*1000,color='#0F4D92',label='二维重构')
            ax.plot(p.j,p.一维重构_V*1000,'--',color='#B64342',label='一维常数')
            ax.set(title=label(r),xlabel='j (A/cm²)',ylabel='损失 (mV)')
        for ax in list(axes.flat)[len(fm):]:ax.set_visible(False)
        handles,labels=axes.flat[0].get_legend_handles_labels();fig.legend(handles,labels,ncol=3,loc='lower center',fontsize=9)
        fig.suptitle(f'{stack}一维与二维拟合对比（同网格、同权重；非预测）');fig.tight_layout(rect=[0,.05,1,.96])
        save(fig,f'{stack}一维与二维拟合对比','共同网格损失及一二维重构.csv')
    fig,axes=plt.subplots(1,2,figsize=(13,5))
    for ax,stack in zip(axes,['FC1','RealHyFC']):
        s=sensitivity.loc[sensitivity.电堆==stack] if len(sensitivity) else pd.DataFrame()
        if not len(s):
            ax.set_axis_off();ax.text(.5,.5,f'{stack}\n共同拟合支持不足\n参数稳定性：NA\n见平台覆盖与拒绝依据图',ha='center',va='center',transform=ax.transAxes,fontsize=14)
            continue
        if len(s):
            for name,group in s.groupby('方案'):
                locs=[];vals=[]
                fm=order(main.loc[main.电堆==stack])
                for i,(_,r) in enumerate(fm.iterrows()):
                    z=group.loc[group.文件名==r.文件名]
                    if len(z):locs.append(i);vals.append(z.iloc[0].最大参数变化_V*1000)
                ax.plot(locs,vals,'o-',label=name,ms=4)
            ax.set_xticks(range(len(fm)),[label(r) for _,r in fm.iterrows()],rotation=60,ha='right')
        ax.axhline(cfg['稳定性']['最大参数绝对改变量_V']*1000,color='#B64342',ls='--',label='预设5 mV界限')
        ax.set(title=stack,ylabel='同支持网格最大参数改变量 (mV)');ax.legend(fontsize=9)
    fig.suptitle('二维健康特征稳定性：不挑选最优处理规则');fig.tight_layout()
    save(fig,'二维健康特征稳定性对比','同支持区间处理敏感性.csv')
    fig,axes=plt.subplots(1,2,figsize=(13,4.5))
    for ax,stack in zip(axes,['FC1','RealHyFC']):
        fm=order(main.loc[(main.电堆==stack)&(main.计算状态=='COMPUTED')]);x=np.arange(len(fm))
        if not len(fm):
            ax.set_axis_off();ax.text(.5,.5,f'{stack}\n没有合格的二维拟合\n一维/二维误差：NA',ha='center',va='center',transform=ax.transAxes,fontsize=14)
            continue
        ax.bar(x-.18,fm.负载留组一维RMSE_V*1000,.36,color='#E9A6A1',edgecolor='black',label='一维')
        ax.bar(x+.18,fm.负载留组二维RMSE_V*1000,.36,color='#3775BA',edgecolor='black',label='二维')
        ax.set_xticks(x,[label(r) for _,r in fm.iterrows()],rotation=60,ha='right');ax.set(title=stack,ylabel='留出负载组重构RMSE (mV)');ax.legend()
    fig.suptitle('按电流区间留组：不是未来时刻预测或独立老化验证');fig.tight_layout()
    save(fig,'一维二维负载留组重构比较','全部方案二维特征.csv')
    table(out/'表格/图片与来源索引.csv',index)
    table(out/'表格/未生成图表及原因.csv',not_generated)
    summary=[]
    for stack in ['FC1','RealHyFC']:
        s=main.loc[main.电堆==stack];a=s.loc[(s.标称老化小时>0)&~s.启停后]
        z=a.loc[a.计算状态=='COMPUTED']
        summary.append(dict(电堆=stack,主曲线含BoL=len(s.loc[~s.启停后]),启停后单列=int(s.启停后.sum()),
                    非BoL主曲线计算数=len(z),数值额外信息支持数=int((z.拟合质量=='CONDITIONAL_2D_INFORMATION').sum()),
                    完整共同网格数=int(s.完整共同网格受支持.sum()),
                    二维重构RMSE中位_mV=float(z.二维RMSE_V.median()*1000) if len(z) else np.nan,
                    一维重构RMSE中位_mV=float(z.一维RMSE_V.median()*1000) if len(z) else np.nan,
                    最大处理参数改变量_mV=float(z.处理规则最大参数变化_V.max()*1000) if len(z) else np.nan))
    table(out/'表格/阶段1验收汇总.csv',summary)
    count=sum(x['数值额外信息支持数'] for x in summary)
    # 总目标是两堆统一表示。单堆数值支持不能替另一堆取得资格。
    real_computed=main.loc[(main.电堆=='RealHyFC')&(main.计算状态=='COMPUTED')&(main.标称老化小时>0)&~main.启停后]
    status='CONDITIONAL_PASS' if count and len(real_computed) else 'BLOCKED'
    target=stage if out==stage/'输出结果' else out
    method='''# 二维极化健康特征提取方法

本轮配置在计算前冻结，具体SHA及时间见配置/计算前冻结记录.json。指导方案中档使用0.36–1.62 A/cm²，第三档明确0.3–1.0 A/cm²；本轮按第三档与本次指令采用后者，统一j_ref=0.9 A/cm²。

两堆分别用自身BoL，严格平均指定5/10个单片。取最早覆盖完整共同区间的降载扫描，不合并升载/后续不完整扫描。FC1无时间，只做窄电流组中位描述，不能认证稳态；RealHyFC同设定电流连续平台必须≥180秒，取末60秒均值，并检查电流、电压漂移、温压波动。全量保留平台拒绝理由和原始行成员。无合格代表点时允许拒绝。

名义网格0.30:0.05:1.00，仅相邻合格代表点线性插值，括距≤0.20 A/cm²，不外推、不跨扫描。没有参考点时标记缺失，不借用另一方向；拟合只在BoL和本曲线共同支持处进行，至少10格、跨度至少0.50 A/cm²。原文件覆盖完整不代表筛选后仍有完整网格。拒绝网格保持NA。禁止把局部插值点称为新的独立测量。

损失D=V_BoL−V_t，保留负值。拟合D≈l1+l2*j/0.9，同时拟合常数c。两个分量单位V，d_c=l1，d_r=l2/0.9（Ω·cm²）。权重为各网格代表的电流区间梯形积分权重，不依原始采样点数加权。没有使用高阶全局拟合。l1是j=0线性基函数截距，超出拟合区间，不能解释为真实开路损失；l2为电流相关斜率分量，不等同于膜内阻或特定机理。

除训练内重构外，将电流轴分5个连续组轮流留出，比较一维/二维重构误差。这是同曲线负载泛化诊断，不是时间外推，也不是独立电堆验证。额外信息要求同时满足：留组RMSE相对改善≥20%且绝对≥1mV；重构RMSE≤5mV；设计矩阵条件数≤30；参数相关绝对值≤0.98；l2条件扰动范围保持符号；全部预设处理变体均有足够共同支持，且最大参数变化≤5mV和相对≤25%。相对尺度max(|l1|,|l2|,10mV)。BoL按恒等式单列，不作为效果样本。

敏感性：FC1分组0.005/0.01/0.02 A/cm²，RealHyFC末30/60/90秒及严格电压波动筛选；所有结果均保存，不用结果挑规则。同支持区间复算后比较参数，避免支持区间变化冒充参数敏感性。

条件不确定性：每个代表点按原始顺序分5块，块均值或中位数重采样300次；同一堆BoL共享相同扰动，BoL自身损失严格为零。范围不包含测试间漂移、传感器校准、j测量误差或真实重复测试差异，不是独立置信区间；FC1更只能称扫描分组敏感性。设计矩阵相关性仅是条件数值诊断。不能把几十/几千测量点当作独立寿命样本。

保留实际DateTime、OpHrs2区间、标称小时及原时间资格，测量结束不等于结果发布。RealHyFC图用真实日历日期，不把标称300/600小时用作精确累计小时坐标。FC1图显式标记NOMINAL_ONLY。温压、硬件、扫描速度、缺失湿度和诊断发布时间的限制不能由数值拟合消除。
'''
    (target/'特征提取方法说明.md').write_text(method,encoding='utf-8')
    lines=[f'# 阶段1健康特征验证报告：{status}',
           '',f'实际重新读取19个极化源文件，含8条FC1、9条RealHyFC名义及2条启停后曲线。额外信息判据通过的非BoL主曲线共{count}条；这是数值与处理可靠性，不代表跨堆测量等价或在线可用。',
           '本轮跨堆统一二维特征的整体目标未通过：RealHyFC冻结规则下共同支持不足，不能用FC1单堆成功代替。原始覆盖完整不等于筛选后覆盖完整。这里的拒绝只针对已冻结提取协议，不证明RealHyFC永远无法构造健康表示；不事后调阈值取得正面结果。',
           '', '## 实际结果']
    for r in summary:lines.append('\n'+str(r))
    lines += ['', '## 每条主规则结果（V换为mV显示；启停后不混入主老化序列）', '',
              '|电堆/文件|l1 mV|l2 mV|一维RMSE mV|二维RMSE mV|留组一维/二维 mV|参数敏感性 mV|数值结论|',
              '|---|---:|---:|---:|---:|---:|---:|---|']
    for _,r in main.iterrows():
        if r.计算状态=='COMPUTED':
            lines.append(f'|{r.电堆}/{r.文件名}|{r.l_1*1000:.4f}|{r.l_2*1000:.4f}|{r.一维RMSE_V*1000:.4f}|{r.二维RMSE_V*1000:.4f}|{r.负载留组一维RMSE_V*1000:.4f}/{r.负载留组二维RMSE_V*1000:.4f}|{r.get("处理规则最大参数变化_V",np.nan)*1000:.4f}|{r.拟合质量}|')
        else:lines.append(f'|{r.电堆}/{r.文件名}|NA|NA|NA|NA|NA|NA|{r.计算状态}|')
    lines += ['', '## 可靠性和研究结论边界',
              '引入第二分量是否有价值，按逐曲线预注册的留负载重构与敏感性共同判断；不把训练残差更小当作成功。未达判据的曲线完整保留，不用其他方案替换主规则。参数相关性较大时，分量会相互抵消；即使重构稳定，也不支持将两者分别解释为活化和欧姆退化。',
              'FC1已计算的BoL特征严格为(l1,l2)=(0,0)，不把人为基准零点当作可靠性验证。RealHyFC基准自身支持不足时保持NA，不能用数学定义的零冒充提取成功。后期和启停后允许负损失，不能解释为不可逆损伤被修复。',
              'FC1只有Excel副本、没有极化采样时钟及同时温压，所有特征只能离线描述，稳态未认证。RealHyFC筛选检查了观测稳定性，但不同日期的温压、湿度缺失及硬件差异仍不支持宣称完全同工况；相对BoL条件差异在CSV逐项列出。没有跨堆共享基准。',
              'RealHyFC BoL早于第一份运行记录；300h曲线只能按312.15411–327.81354 h边界解释，不能用于严格前300 h；600h为566.1698–615.66968 h，不能假定前600 h可用。测量结束和结果发布区别保持，所有在线适配资格仍为UNCONFIRMED_RELEASE_TIME。本轮不确定正式少样本迁移训练边界。',
              '支持离线健康表示可靠性继续研究；不建议立即启动跨堆或物理约束退化网络。先解决FC1稳态/诊断时钟与工况证据、严格有效共同支持和未通过判据曲线。结果不包含未来预测精度，也不构成SOH/RUL验证。',
              '', '## 可追溯交付',
              '源SHA/行数：极化源身份复核.csv；平台拒绝与原始行：平台及分组筛选明细.csv、代表点原始行成员.csv；插值左右支持：局部插值左右支持.csv；拟合/留组/相关性：全部方案二维特征.csv；处理稳定性：同支持区间处理敏感性.csv；条件扰动：条件分块扰动参数.csv；绘图CSV：共同网格损失及一二维重构.csv、各堆二维健康特征.csv。图像PNG/PDF与SHA见图片与来源索引.csv。',
              '完整程序在程序目录；独立复算程序校验保存的损失、权重和系数，原始读取身份由本次源SHA与阶段0清单核对。正式代码更新前另执行完整备份和SHA核验，不改写阶段0结果。不自动commit/push。']
    (target/'健康特征验证报告.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    save_json(out/'阶段结论.json',dict(status=status,numerical_information_supported=count,summary=summary,
                                     online_qualified=False,physical_mechanisms_identified=False))
    return status
