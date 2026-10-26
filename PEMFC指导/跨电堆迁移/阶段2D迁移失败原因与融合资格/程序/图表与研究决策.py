"""从只读审计CSV生成图表与决策；不拟合预测器或重新评价未来时钟成绩。"""
import hashlib,json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
TASK=Path(__file__).resolve().parents[1];GUIDE=TASK.parents[1];REPO=GUIDE.parent;T=TASK/'输出结果/表格';P=TASK/'输出结果/绘图数据';F=TASK/'输出结果/图片';C=GUIDE/'跨电堆迁移/阶段2C低阶迁移先导'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def csv(d,p):d.to_csv(p,index=False,encoding='utf-8-sig',float_format='%.17g')
def md(d):
    def cell(x):
        if pd.isna(x):return 'NA'
        return f'{x:.6g}' if isinstance(x,(float,np.floating)) else str(x).replace('|','/').replace('\n',' ')
    return '\n'.join(['| '+' | '.join(d.columns)+' |','| '+' | '.join(['---']*len(d.columns))+' |']+['| '+' | '.join(cell(x) for x in row)+' |' for row in d.itertuples(index=False,name=None)])
def write(name,s):(TASK/name).write_text(s.strip()+'\n',encoding='utf-8')
def main():
    assert json.loads((TASK/'输出结果/独立检查通过.json').read_text(encoding='utf-8'))['status']=='PASS'
    pairs=pd.read_csv(T/'全部真实诊断身份与运行配对.csv');summ=pd.read_csv(T/'全寿命与各截止独立配对上限.csv');qual=pd.read_csv(T/'一维二维融合分层资格.csv');rates=pd.read_csv(T/'源目标速率差异核对.csv');st=pd.read_csv(T/'冻结状态与后移优势解释.csv');clock=pd.read_csv(T/'时间场景影响汇总.csv');rel=pd.read_csv(T/'特征相关性与第二维剩余变化.csv');inc=pd.read_csv(T/'源信息增量与误差平方分解.csv');changes=pd.read_csv(T/'相邻观测非单调变化.csv');directions=pd.read_csv(T/'运行与诊断相邻变化方向.csv');intervention=pd.read_csv(T/'启停诊断与共享运行锚点.csv');periods=pd.read_csv(T/'源目标阶段范围与描述性速率.csv')
    fc=pd.read_csv(GUIDE/'基础预测实验/数据准备/FC1小时观测.csv');real=pd.read_csv(GUIDE/'基础预测实验/数据准备/RealHyFC循环观测.csv');real=real[(real.scheme=='B60')&real.valid].copy();pred=pd.read_csv(C/'输出结果/正式计算/表格/各方法逐起点预测.csv')
    plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Microsoft YaHei','Arial','DejaVu Sans'],'font.size':11,'axes.linewidth':1.3,'axes.spines.top':False,'axes.spines.right':False,'axes.unicode_minus':False,'legend.frameon':False,'pdf.fonttype':42})
    palette=['#3775BA','#B64342','#3C8D6C','#9A4D8E','#42949E','#767676'];figures=[]
    def finish(fig,name,data,sources):
        csv(data,P/(name+'.csv'));fig.tight_layout(pad=1.6)
        fig.savefig(F/(name+'.png'),dpi=300,bbox_inches='tight',facecolor='white');fig.savefig(F/(name+'.pdf'),dpi=300,bbox_inches='tight',facecolor='white');plt.close(fig)
        normalized=[]
        for relpath in sources:
            source=TASK/relpath if (TASK/relpath).is_file() else GUIDE/relpath
            normalized.append(dict(path=source.relative_to(REPO).as_posix(),sha256=sha(source)))
        figures.append(dict(figure=name,csv=(P/(name+'.csv')).relative_to(TASK).as_posix(),csv_sha256=sha(P/(name+'.csv')),png=(F/(name+'.png')).relative_to(TASK).as_posix(),pdf=(F/(name+'.pdf')).relative_to(TASK).as_posix(),script='程序/图表与研究决策.py',script_sha256=sha(__file__),source_files=normalized,identity='探索性事后审计，无新模型训练，不是在线预测成绩'))
    data=[];fig,axes=plt.subplots(1,2,figsize=(13,4.5))
    for ax,(ds,d,x) in zip(axes,[('FC1',fc[fc.valid],'time'),('RealHyFC',real,'available_h')]):
        ax.scatter(d[x],100*(1-d.SOH),s=8,color=palette[0],label='真实合格运行损失')
        if ds=='FC1':
            source=d[d.partition=='Train'];ax.axvspan(source[x].min(),source[x].max(),color=palette[0],alpha=.13,label='源Train')
        else:
            for n,col in zip([300,600,629.2038],palette[1:4]):ax.axvline(n,color=col,lw=1.1,ls='--',label=f'适配截止{n:g}h')
        ax.set_title(f'{ds}；n={len(d)}；各一条运行轨迹');ax.set_xlabel('实际累计运行小时 / h');ax.set_ylabel('相对运行性能损失 / %');ax.legend(fontsize=9)
        for r in d.to_dict('records'):data.append(dict(dataset=ds,id=r['id'],actual_h=r[x],loss_percent=100*(1-r['SOH']),partition=r['partition']))
    finish(fig,'源目标完整运行损失与适配区间',pd.DataFrame(data),['基础预测实验/数据准备/FC1小时观测.csv','基础预测实验/数据准备/RealHyFC循环观测.csv'])
    fig,ax=plt.subplots(figsize=(10,4.5));x=np.arange(3);ax.bar(x-.17,rates.source_rate_per_h*1e6,.34,color=palette[1],label='FC1源Train');ax.bar(x+.17,rates.target_rate_per_h*1e6,.34,color=palette[0],label='RealHyFC截止前')
    for i,r in enumerate(rates.itertuples()):ax.text(i,r.source_rate_per_h*1e6+1,f'{r.source_target_ratio:.2f}倍',ha='center')
    ax.set_xticks(x,['300h','600h','629.2038h']);ax.set_ylabel('无量纲相对损失速率 / $10^{-6}$ h$^{-1}$');ax.set_title('真实过去小时统计速率；未重新优化模型');ax.legend()
    finish(fig,'源目标已冻结速率差异',rates,['输出结果/表格/源目标速率差异核对.csv'])
    plot=clock[clock.method=='B_FC1Direct'][['scenario','clock_difference_RMS_mV','frozen_RMSE_mV']].merge(inc[inc.method=='B_FC1Direct'][['scenario','rate_increment_RMS_mV']],on='scenario');fig,ax=plt.subplots(figsize=(11,4.5));x=np.arange(3)
    for i,(col,label) in enumerate([('frozen_RMSE_mV','原直接迁移RMSE'),('rate_increment_RMS_mV','源速率替换造成的预测差额RMS'),('clock_difference_RMS_mV','名义与实际时钟传播差额RMS')]):ax.bar(x+(i-1)*.25,plot[col],.25,label=label,color=palette[i],edgecolor='black',linewidth=.5)
    ax.set_xticks(x,['300h','600h','629.2038h']);ax.set_ylabel('电压量 / mV');ax.set_title('三种量分别报告；不是可相加的物理原因贡献');ax.legend(fontsize=9)
    finish(fig,'负迁移速率与时间场景量级',plot,['输出结果/表格/源信息增量与误差平方分解.csv','输出结果/表格/时间场景影响汇总.csv'])
    counts=summ[(summ.dataset=='RealHyFC')|(summ.scope=='全寿命_仅描述')].copy();counts['label']=counts.dataset+' '+counts.scope;fig,ax=plt.subplots(figsize=(11,5));x=np.arange(len(counts));ax.barh(x,counts.main_unique_operating_anchors,color=[palette[1] if d=='FC1' else palette[0] for d in counts.dataset],edgecolor='black',linewidth=.5)
    ax.axvline(6,color=palette[2],ls='--',label='一维至少6对');ax.axvline(9,color=palette[3],ls='--',label='二维至少9对');ax.set_yticks(x,counts.label);ax.set_xlim(0,10.5);ax.set_xlabel('主诊断序列中不同运行锚点的配对数');ax.set_title('FC1仅名义时钟；全部仍来自各一个电堆，不等于独立寿命重复');ax.legend(fontsize=9)
    for i,n in enumerate(counts.main_unique_operating_anchors):ax.text(n+.1,i,str(n),va='center')
    finish(fig,'诊断运行配对数据上限',counts,['输出结果/表格/全寿命与各截止独立配对上限.csv'])
    plot=pairs[pairs.primary&pairs.pair_dual].copy();fig,axes=plt.subplots(1,2,figsize=(12,4.8))
    for ax,ds in zip(axes,['FC1','RealHyFC']):
        g=plot[plot.dataset==ds];ax.scatter(g.D_ref*1000,g.D_low*1000,color=palette[0],s=38)
        for r in g.itertuples():
            offset=(-28,12) if ds=='RealHyFC' and r.nominal_h==2000 else (-28,-18) if ds=='RealHyFC' and r.nominal_h==2300 else (3,3)
            ax.annotate(f'{r.nominal_h:g}',(r.D_ref*1000,r.D_low*1000),xytext=offset,textcoords='offset points',fontsize=8,arrowprops=dict(arrowstyle='-',lw=.5) if offset!=(3,3) else None)
        cr=rel[rel.dataset==ds].iloc[0].corr_Dref_Dlow;ax.set_title(f'{ds}；n={len(g)}；r={cr:.4f}');ax.set_xlabel('D_ref(0.90 A/cm²) / mV');ax.set_ylabel('D_low(0.45 A/cm²) / mV')
    fig.suptitle('两参考损失的全寿命描述；标注为名义测试小时，不是独立验证',y=1.02)
    finish(fig,'双参考损失观测关系',plot,['输出结果/表格/全部真实诊断身份与运行配对.csv'])
    fig,axes=plt.subplots(1,2,figsize=(12,4.8))
    for ax,ds in zip(axes,['FC1','RealHyFC']):
        g=plot[plot.dataset==ds];ax.scatter(g.operating_loss_V*1000,g.D_ref*1000,label='极化D_ref',color=palette[0]);ax.scatter(g.operating_loss_V*1000,g.D_low*1000,label='极化D_low',marker='s',color=palette[2]);ax.set_title(f'{ds}；n={len(g)}；不同负载/测量协议');ax.set_xlabel('固定负载运行电压损失 / mV');ax.set_ylabel('极化参考电压损失 / mV');ax.legend()
    finish(fig,'运行与极化损失描述性对应',plot,['输出结果/表格/全部真实诊断身份与运行配对.csv'])
    fig,ax=plt.subplots(figsize=(9,4.8));x=np.arange(2)
    for j,(col,label) in enumerate([('after_minus_before_Dref_mV','D_ref变化'),('after_minus_before_Dlow_mV','D_low变化')]):
        for i,r in enumerate(intervention.to_dict('records')):
            value=r[col]
            if np.isfinite(value):ax.bar(i+(j-.5)*.28,value,.28,color=palette[j],label=label if i==0 else None,edgecolor='black',linewidth=.5)
            else:ax.text(i+(j-.5)*.28,0,'NA',ha='center',va='bottom',fontsize=10)
    ax.axhline(0,color='black',lw=.8);ax.set_xticks(x,['1500h启停配对','2300h启停配对']);ax.set_ylabel('启停后－前极化损失 / mV');ax.set_title('两负载响应可相反；同阶段共用运行锚点，不计新增独立老化样本');ax.legend()
    finish(fig,'启停前后负载响应差异',intervention,['输出结果/表格/启停诊断与共享运行锚点.csv'])
    (T/'图片来源索引.json').write_text(json.dumps(figures,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    # 可验证主张与最少缺失条件，按用户四级标签逐项决策。
    decisions=pd.DataFrame([
      ['单堆低阶运行预测','已验证','限2B FC1 24/72h及Real短跨度；长期/条件RUL并未同步改善','保留基线和已完成结果，不扩大成全部长期预测有效'],
      ['运行标量跨堆正收益','目前不具备资格','2C三个截止加入源速率均更差；只证明该固定先导负迁移','不将标量负迁移改写为二维迁移失败的终极证明'],
      ['一维极化-运行融合','有条件可探索','全寿命7名义/8完成假设配对可越过6对数学计数，但严格早期0/1/2，条件和发布未知','仅离线描述或另行预注册小规模验证；本轮不拟合映射'],
      ['二维极化-运行融合','数据自身无法支持，需要补充来源','主序列全寿命FC1 7、Real 8对，均小于9；BoL无运行对，启停后不凑数','补充独立配对、时钟和条件，不能调低门槛'],
      ['16单元有界网络','目前不具备资格','二维观测映射、历史状态输入、验证和源增量证据未成立','暂停网络训练；网络大小不是资格证明'],
      ['严格300/600h运行标量离线回放','有条件可探索','2C已执行截止安全，但未来是名义排程；未知真实发布，历史后留已查看','不能称全新盲测或严格在线极化融合'],
      ['严格300/600h二维适配','目前不具备资格','Real只有0/1对，源端缺真实诊断时钟','不能把300/600文件名当信息已可用的时刻'],
      ['跨堆独立泛化证据','数据自身无法支持，需要补充来源','当前仅1源+1目标，各一条寿命，无独立目标堆验证','先追索已有档案；如坚持跨堆主张再补独立电堆'],
      ['真实RUL','目前不具备资格','Real无核实EOL；FC1 809h仅首次阈值越界并有回升','只保留FC1条件阈值寿命、敏感性和删失']
    ],columns=['研究项','分层结论','实际证据','行动边界']);csv(decisions,T/'第三档分层研究决策.csv')
    gaps=qual[['dataset','scope','dimension','distinct_pairs','minimum_pairs']].copy();gaps['count_gap']=np.maximum(0,gaps.minimum_pairs-gaps.distinct_pairs);gaps['warning']='仅计数缺口，不含独立验证、条件、真实时钟；满足数目不等于网络充分';csv(gaps,T/'达到原门槛的配对数量缺口.csv')
    requirements=pd.DataFrame([
      ['独立电堆','跨堆研究至少源与目标身份分开；若要独立目标泛化，建议1源+1适配目标+1未参与选择的验证目标（至少3堆）','当前2堆可做这对电堆的探索；3堆是验证设计建议，不是新发现的物理定理'],
      ['独立诊断配对','每堆二维标定至少9个合格、不同健康阶段的运行-诊断配对，并另留真实未来验证目标；配对转移至少6','现有全寿命FC1 7名义、Real 8完成假设；仅补2/1对只能补数学计数，不能自动解决时间和验证'],
      ['早期适配','如果坚持严格300/600h，所需标定配对必须在截止前实际可用；后来的测试不能回填','Real严格300/600目前0/1对；也可另行登记更晚诊断完成后的研究问题'],
      ['共同负载与BoL','保留0.45、0.90诊断参考；记录各自BoL，明确运行负载与极化协议，最好有诊断邻近同负载运行观测','不能用一个电堆的BoL代另一个；不能造BoL运行零点'],
      ['时钟与发布','运行累计小时、日历、扫描首末、停机、处理与可用时间或可信上界；说明缺口和采样导出链','FC1真实诊断时钟/约30s导出链仍缺；Real600h物理时钟只有区间，发布未知'],
      ['测量与重测','温度位置/设定/波动、公差、压力绝压/表压和位置、流量湿度、校准、预调理/停留、扫描方向；同阶段独立重测','例如至少两次真正重复测量用于估计不确定性，是设计建议；不能把一条曲线密集点当重测'],
      ['真实寿命','若主张真实RUL，需与该SOH定义一致的EOL事件、首次观测和右删失记录','否则仅做SOH/退化轨迹与条件阈值寿命']
    ],columns=['需要补充','最低条件或设计建议','现有边界']);csv(requirements,T/'补充数据最低条件.csv')
    roles=[]
    cutoff=pd.read_csv(T/'各截止时点诊断可用及排除明细.csv')
    for r in cutoff.itertuples():
        p=pairs[pairs.diagnostic==r.diagnostic].iloc[0]
        if p.after_start:role='启停干预描述；不混入主老化标定'
        elif not r.available_completion_or_nominal_assumption:role='该截止不可用于训练或校正；仅后续真实诊断验证标签候选'
        elif p.nominal_h==0:role='自身极化BoL零基准；无同刻运行配对，不造运行零点'
        elif not p.both_supported or not p.pair_dual:role='保留缺失/拒绝；仅已有合格单负载或条件描述'
        elif p.dataset=='FC1':role='名义时间离线标定/验证候选；真实时钟未确认'
        else:role='完成后可用假设下的离线标定候选；组级资格另查，不自动允许状态校正'
        roles.append(dict(scope=r.scope,dataset=r.dataset,diagnostic=r.diagnostic,role=role,one_dim_group_gate='见一维二维融合分层资格.csv',two_dim_joint_training='NO',state_correction='NO_UNQUALIFIED_MAPPING',online='NO_RELEASE_TIME_UNKNOWN',validation='仅潜在观察标签，未开展新模型评分或盲测'))
    csv(pd.DataFrame(roles),T/'逐诊断训练_校正_验证用途.csv')
    scans=GUIDE/'健康特征提取/共同负载特征研究/输出结果/表格/全部扫描身份.csv';csv(pd.read_csv(scans),T/'全部升降载扫描身份只读引用.csv')
    source_reports=['step1/燃料电池三档研究方案.docx','健康特征提取/测量证据与建模资格/测量条件证据核验报告.md','健康特征提取/测量证据与建模资格/建模资格最终评估报告.md','基础预测实验/基础预测实验报告.md','基础预测实验/一维二维特征对比报告.md','物理约束退化演化/阶段2B/物理约束模型实验报告.md','跨电堆迁移/阶段2C低阶迁移先导/低阶迁移实验报告.md','跨电堆迁移/阶段2C低阶迁移先导/迁移收益与负迁移分析.md','跨电堆迁移/阶段2C低阶迁移先导/二维极化迁移资格评估.md']
    csv(pd.DataFrame([dict(path=(GUIDE/p).relative_to(REPO).as_posix(),sha256=sha(GUIDE/p),use='既有原方案与报告只读；本轮未联网重查文献') for p in source_reports]),TASK/'配置/原方案与报告来源SHA256.csv')
    realfull=summ[(summ.dataset=='RealHyFC')&(summ.scope=='全寿命_仅描述')].iloc[0];fcfull=summ[(summ.dataset=='FC1')&(summ.scope=='全寿命_仅描述')].iloc[0];realrel=rel[rel.dataset=='RealHyFC'].iloc[0];fcrel=rel[rel.dataset=='FC1'].iloc[0]
    delayed=st[(st.scenario=='C_后移629h')&(st.method=='D_Target2B_SlowReversible')].iloc[0];dt=clock[clock.method=='B_FC1Direct'];sourceinc=inc[inc.method=='B_FC1Direct'];primaryqual=qual[qual.scope=='全寿命_仅描述']
    phase_table=periods[periods.group.isin(['源原Train','A_严格300h','B_严格600h','C_后移629h'])][['dataset','group','n','first_h','last_h','observed_record_span_fraction','OLS_loss_per_h']]
    source_late=periods[(periods.dataset=='FC1')&(periods.group=='原Assessment')].iloc[0];target_future=periods[periods.group.str.startswith('固定区间_629.2038_')].iloc[0];target_late=periods[(periods.dataset=='RealHyFC')&(periods.group=='原Assessment')].iloc[0]
    rise_summary=changes[changes.adjacent_raw_id].groupby('dataset').agg(adjacent_pairs=('delta_v_mV','size'),rise_pairs=('delta_v_mV',lambda x:int((x>0).sum())),largest_rise_mV=('delta_v_mV','max')).reset_index();csv(rise_summary,T/'真实相邻回升符号汇总.csv')
    write('负迁移原因分析报告.md',f'''# 负迁移原因分析报告

**FC1没有帮助RealHyFC，已被核实的直接原因是源端漂移更快、模型又把它长期保持。时间排程会放大偏差，但不是全部解释；不能归因于某一种电化学损伤。** 本轮零次模型训练、零次权重搜索，只用冻结记录做描述和代数检查。2C原成绩和负迁移结论不改。

## 1. 公式、数据合同和单位
逐点核对2C全部2538条预测，按保存的起点状态`d=L+q*h+R*exp(-h/tau)`及各自base还原SOH和电压。起点/目标身份和未来名义时间均核验。2A/2B/2C三份数据副本SHA一致，原代码仅阅读和AST解析，未调用旧运行/拟合入口。详细差异见`冻结预测代数核对.csv`。

2B Real参数为V/物理循环；2C重新在真实**过去**小时上统计无量纲`d/h`，未来按固定名义12h/循环。不是把V/h直接当V/循环。FC1五片电压转平均单片、各堆运行基准和`1-SOH`均保留。源原Train669点，目标截止历史23/47/48点。源训练区间较长，目标是早期，不能视为相同老化阶段。

{md(rates[['scenario','source_n','target_n','source_rate_per_h','target_rate_per_h','source_target_ratio']])}

源Train与目标早期实际范围、全记录范围比例见`源目标阶段范围与描述性速率.csv`；比例只是已记录时段占比，Real没有核实EOL，不能叫物理寿命百分比。

{md(phase_table)}

## 2. 源信息如何精确放大误差
目标锚点和时间排程相同时，直接迁移相对目标有界率的预测差为`delta_V=-base*(q_source-q_target)*(已知过去传播时长+名义未来时长)`；25%先验模型的差正好是其四分之一。

{md(inc[['scenario','method','rate_increment_RMS_mV','cross_term_mV2','squared_increment_mV2','actual_MSE_increase_mV2']])}

MSE差分为`2mean(e_target*delta_V)+mean(delta_V²)`，只是代数恒等式。三个场景交叉项和平方项都为正，说明这次源信息使既有偏差继续同向增大。不能把这些数当温度/负载/电化学原因占比。

## 3. 名义时间影响
真实未来小时仅用于事后计算“同一冻结状态在两种时钟下的传播差额”。没有产生真实时钟替代预测CSV，也没有以它重排新模型成绩。

{md(dt[['scenario','time_difference_mean_h','time_difference_min_h','time_difference_max_h','clock_difference_RMS_mV','frozen_RMSE_mV']])}

直接迁移时钟差额RMS约{dt.clock_difference_RMS_mV.min():.3f}—{dt.clock_difference_RMS_mV.max():.3f}mV，原RMSE约{dt.frozen_RMSE_mV.min():.3f}—{dt.frozen_RMSE_mV.max():.3f}mV。时钟场景重要，但量级不能解释全部误差。两种差额与总误差不可直接相加成因果百分比。严格600h在运行数据缺口内，不能制造精确日历时刻；629.2038h是循环51末端，不是600h诊断精确完成小时。

## 4. 非单调与漂移变化
真实相邻原始ID的回升符号统计如下，未设恢复幅度阈值，细小电压正差不自动代表物理恢复。跨缺口变化单列，不当作连续采样。

{md(rise_summary)}

固定旧分区及300/600/629.2038边界的描述斜率并不恒定，说明恒定源漂移长期外推假设过强。没有按这些后期斜率重新估计预测器。FC1 809h首次阈值越界后有回升的原事实保留，不认定不可逆EOL。

更具体地，FC1旧Assessment描述斜率约`{source_late.OLS_loss_per_h:.9g}/h`，也明显低于源Train；Real629h后全部合法目标约`{target_future.OLS_loss_per_h:.9g}/h`，旧Assessment约`{target_late.OLS_loss_per_h:.9g}/h`，后者为负。它们是审计统计，不拿来重选源训练区间或构造改进预测。单点区间速率NA、跨边界仅两点的斜率不作为连续阶段趋势证据。

## 5. 为什么后移目标模型较好
629.2038h慢变模型冻结值：`L={delayed.L:.12g}`、`R={delayed.R:.12g}`、`q={delayed.q_per_h:.12g}/h`、`tau={delayed.tau_h:g}h`，初始总损失`{delayed.predicted_initial_loss:.12g}`，原末次实测`{delayed.last_measured_d:.12g}`。其长期q小于目标全历史率及FC1率，且初始分量不同。R为负，其均值回归会使损失上升；该实际预测轨迹中电压回升相邻对为{int(delayed.future_voltage_rise_pairs)}。所以成绩好不等于预测了未来恢复，更不等于辨识出两种损伤。

## 6. 证据分层
- **已证实**：速率比2.79—3.21、固定漂移累积、名义时间差、观测非单调、后移状态数值以及本轮负迁移。
- **有依据的限制/解释**：FC1与Real运行负载0.70/约0.558不同，温压、硬件、预调理和扫描协议不同；相对归一化不保证相同演化规律。
- **尚无法区分**：各工况因素各占多少、哪种损伤导致速率差、L/R是否可独立物理验证。不能用相关性替代控制实验。
''')
    write('共同健康表征可比性评估.md',f'''# 共同健康表征可比性评估

**数学共同定义成立，物理等价和联合状态训练资格尚未成立。** 两堆用自身BoL，继续冻结`D_ref=D(0.90)`与`D_low=D(0.45)`，无新负载、无跨扫描插值、无压力放宽。原阶段1的BLOCKED与1.2数学CONDITIONAL_PASS等结论保持不变。

## 真实支持和不同测量条件
19个真实诊断文件：FC1 8个、Real 11个；Real主序列9个，另有2个启停后测试。FC1 8个与Real主9个有双参考数学支持；Real2300h启停后低负载拒绝仍保留。严格A60仍无合格共同基准，不能拿B60值冒充严格结果。

逐点左右实测j、原始行号、平台支持组和扫描身份保存在`两参考点实测支持与原始行号.csv`。本轮与冻结D值逐项核对，不重新提取。相关原始温压/流量/平台波动只从已有支持表引用，不重新筛选。

每条诊断在四个截止范围内可作基准、标定候选、未来验证候选或仅启停描述，列在`逐诊断训练_校正_验证用途.csv`；全部升/降载原身份另存，不把其他扫描混入当前特征。

阶段1.3已核实：FC1为连续降载，名义冷却液出口55°C、气体入口1300mbar绝压等；Real为阶梯负载平台，名义冷却入口70°C、气体出口160/150kPaG。**温度测点、压力位置和绝压/表压不同，不能直接按数字比较。** 原作者名义设定有依据，但FC1逐次同步条件、Real传感器准确度/温控允许波动、独立重测和实际发布仍缺。上述为既有资料结论，本轮未联网重查，也未借用其他同名BoL或敏感性数据修正特征。

FC1操作负载0.70、Real约0.558，不等于诊断0.90。相对各自BoL的损失只统一数学单位，不证明同样的机理或绝对健康状态。

## 观测关系
主配对全寿命描述结果：

{md(rel[['dataset','n_main_pairs','corr_op_Dref','corr_op_Dlow','corr_Dref_Dlow','second_feature_residual_std_V','second_feature_unexplained_fraction']])}

运行与诊断同向的描述关联很高，但共同老化时间趋势、共享BoL误差和不同测量条件均可能贡献。没有拟合一个全寿命运行—诊断预测映射，没有计算其未来预测成绩。逐阶段方向和诊断前后运行差保存，不能把不同负载的损失相减并归因为老化。

Real第二特征对D_ref线性投影后的剩余标准差约{realrel.second_feature_residual_std_V*1000:.3f}mV，剩余方差比例约{realrel.second_feature_unexplained_fraction*100:.3f}%；FC1对应约{fcrel.second_feature_residual_std_V*1000:.3f}mV。这里只是冗余诊断，不是误差条或独立噪声估计；缺重测，不能宣布剩余量已经可测或完全无用。

## 启停后确有不同负载响应，但不能凑独立样本
1500h启停后D_ref改变{intervention.iloc[0].after_minus_before_Dref_mV:.3f}mV、D_low改变{intervention.iloc[0].after_minus_before_Dlow_mV:.3f}mV，方向相反。两参考包含不同响应的观测事实，应保留研究价值；不能因此解释为两种独立损伤。2300h启停后D_low缺支持，NA不填。

1500h前后测试都配运行锚点125，2300h前后都配锚点200；介入身份不同，不是新增同协议独立老化状态。诊断前后运行电压变化只是相邻合格观测差，并不等于极化测试或启停的因果效应。

## 单负载能否恢复两个状态
仅在**假设损失对j线性**时，两参考可写成`D(j)=w*D_ref+(1-w)*D_low`，`w=(j-.45)/.45`。FC1操作负载w≈0.5556、Real≈0.24036。固定负载的瞬时观测方向秩为1，两个未知状态加未知运行偏差不能靠这一条观测唯一拆开。两个电堆各有不同状态，不能把两个负载方向堆叠成对同一个健康状态的两条测量。

时间演化先验和诊断可能增加信息，但须独立辨识和验证，不能先假定正确再宣布可观测。原完整0.3–1.0网格不足也没有被双参考替代表述自动修复。
''')
    write('二维观测融合数据充分性报告.md',f'''# 二维观测融合数据充分性报告

**即使用完整寿命，主协议仍只有FC1 7个名义配对、Real 8个完成时间假设配对，小于冻结二维门槛9。** 这已是数据自身的结构性上限，不是换优化器或多训练能解决的缺口。

本报告“全寿命”指当前数据的完整记录范围，不代表已经覆盖核实的真实EOL。

## 各截止及全寿命上限
FC1早期计数仅按标称时钟描述，不用于改写2C源Train；Real为目标真实截止。

{md(summ[['dataset','scope','main_diagnostics','main_dual_pairs','main_unique_operating_anchors','actual_completed_pairs','main_diagnostic_transitions','paired_transitions']])}

BoL数学上为零，但两堆在同一时钟规则下都没有BoL以前的合格运行观测；不能补一个(0,0)运行配对。FC1有7名义对、真实完成钟可认证对为0。Real完整11文件中主9包含BoL，剩8个主诊断有配对。

若将启停后也混入，Real可列9次双参考配对测试、10次仅D_ref配对测试，但最多仍8个不同运行锚点，且干预/条件不同。不能通过这一口径达到9个主协议独立老化配对。不同锚点也仍来自一条电堆寿命，不能冒称8条独立轨迹。

## 原门槛逐项检查
一维2参数至少6对；二维3参数至少9对；标准化设计满秩、条件数≤100；历史可配对诊断转移至少6次，并有未参加标定/选参的合法验证目标。门槛是2B预登记稳定性规则，不是本轮发明，也不是满足就保证可靠的物理定理。

{md(primaryqual[['dataset','dimension','distinct_pairs','minimum_pairs','rank','standardized_condition','paired_transitions','mathematical_sample_gate','clock_gate']])}

全寿命二维设计矩阵数学上满秩，条件数FC1约{primaryqual[(primaryqual.dataset=='FC1')&(primaryqual.dimension==2)].standardized_condition.iloc[0]:.2f}、Real约{primaryqual[(primaryqual.dataset=='RealHyFC')&(primaryqual.dimension==2)].standardized_condition.iloc[0]:.2f}，均未超过100。**因此不能继续说全寿命都因秩/条件数被阻断；此处硬缺口是独立配对数和测量/时间/验证资格。** 原2B Train的4对Real条件数171.75仍是原子集事实，并不与全寿命31.34矛盾。

全寿命主配对链FC1有6次名义转移、Real有7次完成假设转移，数量上可越过6；但二维映射仍缺9对，且FC1真实时间未证。不能用一个条件达标替代其他条件。2C早期表的全诊断链0/1/2与本轮配对链0/0/1语义不同，均明确列出，不改变旧资格拒绝结果。

## 一维不是同样的结论
全寿命FC1名义7对、Real完成假设8对，**一维可以越过纯数学计数与秩门槛**，因此不能笼统说“一维全寿命也绝对不可计算”。但这只允许谨慎离线观测关系研究；严格早期0/1/2仍不足，真实发布时间/测量条件仍缺，且把6对留作拟合后只余1或2个主目标，验证很弱。本轮不拟合这个映射，不把全寿命值借给300/600h。

## 16单元网络
原Word3.3规定二维8→16→2，普通含偏置MLP有178个权重；FC1标量5→16→1有113个。这不等于本轮已建网，也不以权重数与诊断数简单相除断定必然过拟合。关键是历史双状态及变化率/标准差、事件可用时间和合法观测映射仍缺，连续运行只能直接观测标量组合；没有经过验证的共享二维演化教师或验证目标。

结论：**暂停正式二维融合和16单元网络训练。** 缺失损失/梯度记NA，不记0；不得用插值SOH、未来诊断或后期全寿命回归填这些输入。
''')
    write('第三档下一阶段决策报告.md',f'''# 第三档下一阶段决策报告

**第三档问题值得保留，但当前不能把“二维状态融合＋跨堆迁移”写成已实现主贡献。建议近期以路线B的单堆低阶观测演化、SOH与FC1条件寿命为可验证主线；路线A保留目标、暂停扩网；路线C用于补证据，而非立即宣布换数据。** 是否调整论文定位应带这份证据给导师决定，本轮不启动下一阶段。

## 七项资格与边界

{md(decisions)}

## 推荐顺序
1. **路线B，近期优先**：整理已有单堆证据。2B FC1 24/72h慢变模型有真实收益，Real 1/4循环仅小幅改善、12循环更差；168h/25循环完整块不足仍NA。SOH只在各自负载/基准下解释。FC1条件RUL未同步改善，809h只是首次观测阈值越界且后有回升，不能变成真实不可逆失效；Real无真实EOL，不做真实RUL成绩。
2. **路线C，若跨堆仍为论文必需，优先补来源/观测**：先追索现有原始测试时钟、发布、FC1导出链及未纳入的独立配对；若原数据硬上限无法改变，再寻找有诊断—运行对应的新电堆数据或安排重测。当前两堆足以做这对电堆的探索和负面研究，不足以支持一般跨堆泛化主张。
3. **路线A，有条件恢复**：只有观测映射、时间、独立样本及验证来源都补齐，再预注册二维状态与分层适配实验。2C只否定本轮固定速率先导的正收益，并没有逻辑上否定所有可能的原方案；但也没有证据允许现在跳到178权重网络。

## 最少应补什么

{md(requirements)}

FC1全寿命单论9对门槛还差2个合格独立配对，Real差1个；这只是计数差。严格300/600h目标历史实际只有0/1对，未来补的晚期曲线不能放回过去。必须另有验证观测，不能用凑到9的同一组数据同时证明未来性能。

最低数据设计建议是1个源堆、1个适配目标堆、再有1个未参与选择的验证目标堆，便于分离适配与独立泛化。**“至少3堆”是面向新独立泛化实验的设计建议，不是宣称两个电堆完全不能做任何迁移研究，也不是现有冻结门槛。** 重复诊断用于估计测量可靠性，不能替代不同健康阶段的独立配对。

## 停止决定
本轮完成审计即停止：不重训2C、不搜迁移权重、不改温压、不开16单元/PINN/TFT，不修改完整代码，不把描述性全寿命关系冒充在线预测。保留数学双参考研究及负迁移结果，减少现有数据无法支持的联合网络实验。
''')
    write('使用说明.md','''# 阶段2D：迁移失败原因与融合资格

本目录只含准备/审计代码，不进入完整代码。先看四份报告，结论是源速率和恒定外推造成可核验负迁移；全寿命主配对FC1 7名义、Real 8完成假设，二维仍不足9；一维全寿命数学计数可达标但早期/测量/发布资格仍不足。

原始2A/2B/2C结果全部只读，不重新运行历史模型。

程序顺序：`冻结审计范围.py`一次性登记并拒绝覆盖；`执行只读审计.py`真实计算审计表并拒绝覆盖完成结果；`独立检查.py`独立复核；`图表与研究决策.py`从真实CSV生成四份报告和7组300dpi PNG/PDF；再运行`独立检查.py`核验绘图来源及历史SHA。依赖numpy、pandas、matplotlib，图像检查用Pillow；无PyTorch训练依赖。

所有入口可用`python -X utf8 -B "完整脚本路径"`运行，从自身位置定位资料。既有分析已经完成，普通复核只运行独立检查；不要覆盖历史目录。全寿命相关性/斜率是描述，不输出新预测成绩。每图绘图CSV和来源见`输出结果/绘图数据`及`输出结果/表格/图片来源索引.json`。

优先看：独立配对上限图、负迁移速率/时钟量级图、双参考关系图。表格保留全部拒绝、启停后身份和未知时间，不把缺失置零。
''')
    print('REPORTS_READY',len(figures),'figures; original codes untouched')
if __name__=='__main__':main()
