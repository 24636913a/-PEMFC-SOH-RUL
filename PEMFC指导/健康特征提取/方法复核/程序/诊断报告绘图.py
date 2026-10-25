"""从真实诊断CSV制作图表与中文报告。所有对照保留DIAGNOSTIC_ONLY身份。"""
import os,json
import numpy as np
import pandas as pd
from 核验基础 import table,save_json,sha
from 平台诊断计算 import GUIDE


def deliver(work,out,cfg,features,points,sens,platforms,windows,audit):
    os.environ.setdefault('MPLCONFIGDIR',str(out/'绘图配置缓存'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':['Microsoft YaHei','DejaVu Sans'],'axes.unicode_minus':False,
        'axes.spines.top':False,'axes.spines.right':False,'font.size':10,'pdf.fonttype':42,'savefig.dpi':300})
    plots=[]
    def save(fig,name,source,pdf=True):
        fig.savefig(out/'图片'/f'{name}.png',dpi=300,bbox_inches='tight')
        if pdf:fig.savefig(out/'图片'/f'{name}.pdf',bbox_inches='tight')
        plt.close(fig);plots.append(dict(图名=name,来源CSV=source,PNG_SHA256=sha(out/'图片'/f'{name}.png'),身份='DIAGNOSTIC_ONLY'))
    short=lambda s:s.replace('Polcurve_Nominal_RHFC_','').replace('.csv','')
    temporal=pd.read_csv(out/'表格/长平台时序绘图数据.csv',low_memory=False)
    for i,(key,g) in enumerate(temporal.groupby('平台键',sort=False)):
        first=g.iloc[0];secs=g.平台内秒.to_numpy();duration=secs[-1]
        fig,axs=plt.subplots(4,1,figsize=(7.5,8),sharex=True)
        axs[0].plot(secs,g.current,lw=1,label='实测I');axs[0].plot(secs,g['set'],'--',lw=1,label='设定I');axs[0].set_ylabel('电流 (A)')
        twin=axs[0].twinx();twin.plot(secs,g.j,':',color='#42949E',label='实测j',lw=.8);twin.set_ylabel('j (A/cm²)');axs[0].legend(loc='lower left',fontsize=8)
        axs[1].plot(secs,g.v*1000,color='#0F4D92',lw=1);axs[1].set_ylabel('平均单片电压 (mV)')
        axs[2].plot(secs,g.temp,color='#B64342',label='冷却液入口');axs[2].plot(secs,g.tout,color='#42949E',label='冷却液出口');axs[2].set_ylabel('温度 (°C)');axs[2].legend(fontsize=8)
        axs[3].plot(secs,g.pa,label='阳极入口');axs[3].plot(secs,g.pc,label='阴极入口');axs[3].set_ylabel('入口压力 (kPaG)');axs[3].legend(fontsize=8)
        for ax in axs:
            for tail,color in [(90,'#767676'),(60,'#0F4D92'),(30,'#B64342')]:ax.axvline(duration-tail,ls='--',lw=.7,color=color)
        axs[-1].set_xlabel('本平台真实经过时间 (s)，虚线依次为末90/60/30秒起点')
        fig.suptitle(f'{short(first.文件名)} | 扫描{int(first.segment)} 平台{key.split("_p")[-1]}\n原始行{int(g.row.min())}–{int(g.row.max())} | {g.datetime.iloc[0]}\n主降载扫描={bool(first.主扫描)}；非因果诊断',fontsize=10)
        fig.tight_layout(rect=[0,0,1,.91]);save(fig,f'平台_{short(first.文件名)}_s{int(first.segment)}_p{key.split("_p")[-1]}','长平台时序绘图数据.csv',False)
        if (i+1)%30==0:print('平台图已生成',i+1,flush=True)
    cohort=pd.read_csv(out/'表格/历史69平台固定队列.csv')
    rows=[]
    for r in cohort.itertuples():
        m=windows.loc[(windows.文件名==r.文件名)&(windows.组号==r.组号)&windows.主扫描]
        for _,v in m.iterrows():rows.append(v.to_dict())
    cw=pd.DataFrame(rows);table(out/'表格/历史69队列分窗口.csv',cw)
    fig,axs=plt.subplots(1,2,figsize=(12,4.6))
    labels=['前60秒','末30秒','末60秒','末90秒']
    for ax,col,label in [(axs[0],'temp_range','温度峰峰差 (°C)'),(axs[1],'v_p95_p5','电压P95−P5 (mV)')]:
        data=[cw.loc[cw.窗口==w,col].to_numpy()*(1000 if col.startswith('v_') else 1) for w in labels]
        ax.boxplot(data,tick_labels=labels,showfliers=True);ax.set_ylabel(label)
    axs[0].axhline(1,ls='--',color='#B64342',label='原1°C保守筛选');axs[0].legend(fontsize=8)
    fig.suptitle('固定历史69个平台：早期与末段波动，保留全部异常值');fig.tight_layout();save(fig,'历史69平台的温度与电压波动','历史69队列分窗口.csv')
    fig,axs=plt.subplots(1,2,figsize=(12,4.5))
    w=cw.loc[cw.窗口=='末60秒']
    axs[0].scatter(w.temp_range,w.v_p95_p5*1000,c=w.j中位,cmap='viridis',s=28);axs[0].set(xlabel='末60秒温度峰峰差 (°C)',ylabel='末60秒电压P95−P5 (mV)')
    axs[1].hist(w.temp_v_r.dropna(),bins=np.linspace(-1,1,16),color='#3775BA',edgecolor='black');axs[1].set(xlabel='平台内T与V的Pearson相关',ylabel='平台数（技术分段，非独立老化试验）')
    fig.suptitle('温度与电压观测关联：不能解释为温度无影响或物理因果');fig.tight_layout();save(fig,'温度与电压观测关联','历史69队列分窗口.csv')
    names=features.loc[features.方案=='A60','文件名'].tolist();cases=[c['名称'] for c in cfg['诊断组']]
    counts=features.pivot(index='文件名',columns='方案',values='共同网格数').reindex(index=names,columns=cases)
    fig,ax=plt.subplots(figsize=(10,5.5));im=ax.imshow(counts.to_numpy(),vmin=0,vmax=15,cmap='viridis',aspect='auto')
    for i in range(len(names)):
        for j in range(len(cases)):ax.text(j,i,str(int(counts.iloc[i,j])),ha='center',va='center',color='white' if counts.iloc[i,j]<8 else 'black')
    ax.set_xticks(range(len(cases)),cases);ax.set_yticks(range(len(names)),[short(x) for x in names]);fig.colorbar(im,ax=ax,label='受支持网格数（15格候选；≥10仅为必要条件）')
    ax.set_title('单项温度限制与末窗口对共同支持的影响');fig.tight_layout();save(fig,'六组规则共同网格覆盖','诊断二维参数及支持资格.csv')
    enough=features.loc[features.足够数学支持]
    if len(enough):
        fig,axs=plt.subplots(1,2,figsize=(12,4.8))
        for case in ['B30','B60','B90']:
            f=enough.loc[(enough.方案==case)&~enough.启停后].sort_values('实际测试开始')
            if not len(f):continue
            x=pd.to_datetime(f.实际测试开始,format='mixed')
            for ax,field in zip(axs,['l_1','l_2']):ax.plot(x,f[field]*1000,'o-',ms=4,label=case)
        for ax,field in zip(axs,['l₁','l₂']):ax.set_ylabel(field+' (mV)');ax.tick_params(axis='x',rotation=30);ax.legend();ax.set_xlabel('实际测量日期（OpHrs2边界另见CSV）')
        fig.suptitle('探索性二维参数：各曲线原支持域可能不同；非正式健康轨迹');fig.tight_layout();save(fig,'诊断二维参数窗口对照','诊断二维参数及支持资格.csv')
        fig,axes=plt.subplots(3,3,figsize=(12,10),sharex=True)
        fs=features.loc[(features.方案=='B60')&~features.启停后].sort_values('实际测试开始')
        for ax,(_,r) in zip(axes.flat,fs.iterrows()):
            t=points.loc[(points.文件名==r.文件名)&(points.方案=='B60')]
            ax.plot(t.j,t.损失_V*1000,'ko',ms=3,label='本堆BoL相对损失')
            if r.足够数学支持:
                ax.plot(t.j,t.二维重构*1000,color='#0F4D92',label='二维');ax.plot(t.j,t.一维重构*1000,'--',color='#B64342',label='一维')
            ax.set(title=short(r.文件名),xlabel='j (A/cm²)',ylabel='损失 (mV)')
        handles,labels=axes.flat[0].get_legend_handles_labels();fig.legend(handles,labels,ncol=3,loc='lower center');fig.suptitle('B60诊断性损失重构：取消温度限制不代表认证恒温或物理等价');fig.tight_layout(rect=[0,.04,1,.96]);save(fig,'B60一维二维诊断重构','诊断网格损失及重构.csv')
    rule=pd.read_csv(out/'表格/逐规则独立排除贡献.csv');base=rule.loc[(rule.方案=='A60')&(rule.范围=='至少180秒且j在0.2至1.1')]
    fig,ax=plt.subplots(figsize=(12,4));xx=np.arange(len(base));ax.bar(xx-.2,base.边际拒绝平台数,.4,label='该项失败（可多标签）');ax.bar(xx+.2,base.仅该规则拒绝平台数,.4,label='仅此项失败');ax.set_xticks(xx,base.规则,rotation=30,ha='right');ax.set_ylabel('平台数');ax.legend();ax.set_title('原规则各项独立排除贡献：边际项不能相加');fig.tight_layout();save(fig,'原规则逐项拒绝贡献','逐规则独立排除贡献.csv')
    table(out/'表格/图片来源与SHA.csv',plots)
    summary=[]
    for case in cases:
        f=features.loc[features.方案==case];a=f.loc[~f.启停后];n=a.loc[a.标称小时>0]
        summary.append(dict(方案=case,主曲线数学支持数_含BoL=int(a.足够数学支持.sum()),老化主曲线数学支持数=int(n.足够数学支持.sum()),
          启停后数学支持数=int(f.loc[f.启停后,'足够数学支持'].sum()),有参考0_9主曲线数=int(a.参考0_9支持.sum()),
          完整0_3至1主曲线数=int(a.完整0_3至1覆盖.sum())))
    table(out/'表格/六组诊断验收汇总.csv',summary)
    sourcepdf=next((GUIDE.parent/'数据/《耐久性测试期间PEM燃料电池堆的空间解析电化学表征》的基础数据').glob('IJHE*.pdf'))
    lit=[dict(证据='原论文表1',页=3,结论='冷却液入口名义70°C；压力160/150 kPaG为出口，不是原入口字段',来源=str(sourcepdf),SHA256=sha(sourcepdf)),
         dict(证据='原论文2.2.1和表3',页=3,结论='≥0.3 A/cm²停留5min，低于0.3为2min，负载斜坡2A/s；未给末60秒1°C峰峰上限',来源=str(sourcepdf),SHA256=sha(sourcepdf)),
         dict(证据='原论文2.1',页=2,结论='Greenlight G200；未提供冷却液温度传感器型号、准确度与同步时序。CDDM校准讨论不能转作温度精度证据',来源=str(sourcepdf),SHA256=sha(sourcepdf))]
    table(out/'表格/实验说明与字段证据.csv',lit)
    # 控制波动、仪器误差和真实热变化不能仅凭相关性互相替代。
    therm=w[['temp_v_r','temp_v_diff_r']].median(numeric_only=True)
    lines=['# 平台失败原因分析报告','', '研究身份：DIAGNOSTIC_ONLY。原阶段1的BLOCKED及其CSV、报告、配置保持不变。',
       '', '## 主要结论',
       '首先区分规则拒绝和数据缺失。原1°C值是阶段1预登记的保守工程筛选，并无本地原论文规定的末60秒峰峰依据；因此原拒绝不能证明RealHyFC二维表示在数学上不可行。取消该单项仅用于归因，不认证温度无影响或新的正式稳态规则。',
       '', '六组一次性对照：',str(summary),'', '## 原规则执行与数据身份',
       '11份极化源SHA/行数与阶段1逐份核对，A60对原平台分组、保留标志、拒绝原因及代表电压逐条严格复现。没有访问任何旧预测实验Holdout/Test指标或重新读取860万行运行记录。',
       '', '## 温度是否仅是平台早期变化',
       f'固定历史69队列的末60秒T−V相关中位数{therm.temp_v_r:.6f}，一阶差分相关中位数{therm.temp_v_diff_r:.6f}。每个平台的前60秒及末30/60/90秒波动、温压和电压斜率均已保存，原始图逐平台生成。相关不能解释为温度因果响应；平稳电压也不能证明温度没有作用。',
       '', '## 原始说明和字段边界',
       '原论文（doi:10.1016/j.ijhydene.2026.156060）表1/3显示名义冷却液入口70°C、主要区间每点5min、低于0.3每点2min；没有规定1°C峰峰阈值。论文保持冷却液流量并以最高负载入口/出口10K温差选取流量，不等同于任意平台必须恒温。入口/出口温度不是电堆内部完整热状态。',
       '原始数值步距和重复比例只表征记录分辨特征，不能当成仪器测量误差。没有传感器准确度、记录同步元数据或控制器日志，故不能将周期性直接认定为控制器、传感器故障或电堆真实热状态变化。电流与密度的不一致只能提示通道同步问题，不能据此移位修正。',
       '', '## 独立排除数', '逐规则独立排除贡献.csv同时列多标签边际数、仅单项失败数、平台原始行和末窗口行、影响文件数；不同窗口重叠，均不能相加视作独立实验。升/降载分开，主扫描之外只作背景审计。',
       '', '## 后续建议', '保留二维路线的资格研究，先取得实验允许的温度范围、控制与传感器说明，预注册替代提取规则及独立验证；当前不直接进入退化预测、跨堆迁移或SOH/RUL。阶段0/1时间资格、300/600小时边界和发布时间未知继续保留。']
    (work/'平台失败原因分析报告.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    lines=['# 健康特征方法复核报告','', '本报告只讨论诊断性数学可行性，不替换阶段1。',
       '', '## 六组支持资格',str(summary),'', '共同负载范围详见全部主曲线共同支持.csv。只有BoL与本次曲线的交集满足至少10格、跨度0.50才计算；缺失保留NA，不外推。完整0.3–1.0与局部可用范围分开报告，不把不同支持域的原拟合参数直接当作可比健康状态。',
       '', '## 一维、二维及稳定性',
       '诊断二维参数及支持资格.csv保留一二维残差、留负载重构、条件数和参数相关；同支持网格参数敏感性.csv先匹配同样j网格，再比较B30/B60/B90及可用A组，不挑最优。条件分块扰动含共享BoL，范围不是重复测试置信区间。二维重构较低本身不作为健康特征有效证据。',
       '', '## 解释限制', '温压差异与参数的关联仍可能混有日期、负载、恢复和老化，不做因果分离，不进行温度校正，不以回归系数解释具体电化学机理。两个参数可相互抵消；共同网格的插值密度不能增加真实负载数或独立老化样本量。',
       '', '## 时间资格', 'BoL早于首份Loadcycle。300h文件保留312.15411–327.81354 h边界；600h为566.1698–615.66968 h，实际日期及已知OpHrs2逐条保留；结果发布时间未知。FC1原NOMINAL_ONLY结果只读对照，未重做FC1搜索。',
       '', '## 本轮判定', 'DIAGNOSTIC_ONLY_COMPLETED。可以提出有依据的新协议候选，但不能宣布阶段1转为PASS。本轮建议先完成测试条件处理方式的资格修订与验证，不立即训练退化网络。替代规则的依据、适用域和验证计划须单独冻结，禁止从六组中挑“最漂亮”的方案晋级。']
    (work/'健康特征方法复核报告.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    save_json(out/'诊断结论.json',dict(status='DIAGNOSTIC_ONLY_COMPLETED',original_stage1_status='BLOCKED_UNCHANGED',summary=summary,plots=len(plots),training=False))
    return summary
