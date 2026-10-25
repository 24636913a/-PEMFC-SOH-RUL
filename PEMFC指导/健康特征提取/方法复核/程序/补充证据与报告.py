"""根据已完成的真实CSV整理证据与最终报告，不重复六组比较、不改历史结果。"""
from pathlib import Path
import os,sys,shutil,json
import numpy as np
import pandas as pd

WORK=Path(__file__).resolve().parent.parent;GUIDE=WORK.parent.parent;OUT=WORK/'输出结果'
sys.path.insert(0,str(GUIDE/'数据处理/程序'))
from 核验基础 import table,sha,save_json


def run():
    features=pd.read_csv(OUT/'表格/诊断二维参数及支持资格.csv')
    rules=pd.read_csv(OUT/'表格/逐规则独立排除贡献.csv');co=pd.read_csv(OUT/'表格/历史69平台固定队列.csv')
    windows=pd.read_csv(OUT/'表格/长平台分窗口统计.csv').merge(co[['文件名','组号']],on=['文件名','组号'])
    windows=windows.loc[windows.主扫描]
    spectrum=pd.read_csv(OUT/'表格/全Nyquist温度频谱复核.csv').merge(co[['文件名','组号']],on=['文件名','组号'])
    spectrum=spectrum.loc[spectrum.主扫描]
    lag=pd.read_csv(OUT/'表格/69平台全部滞后汇总.csv');sens=pd.read_csv(OUT/'表格/同支持网格参数敏感性.csv')
    b=features.loc[(features.方案=='B60')&features.足够数学支持&~features.启停后&(features.标称小时>0)]
    tiny=features.loc[(features.方案=='B60')&~features.足够数学支持]
    late=windows.loc[windows.窗口=='末60秒'];early=windows.loc[windows.窗口=='前60秒']
    period=float(spectrum.最强全带周期_s.median());maxchange=float(sens.最大参数差_V.max()*1000)
    os.environ.setdefault('MPLCONFIGDIR',str(OUT/'绘图配置缓存'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':['Microsoft YaHei','DejaVu Sans'],'axes.unicode_minus':False,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
    fig,ax=plt.subplots(figsize=(9,4.5));ax.plot(lag.滞后秒,lag.中位,'o-',color='#0F4D92',label='69平台相关中位数')
    ax.fill_between(lag.滞后秒,lag.最小,lag.最大,alpha=.15,color='#3775BA',label='全部平台最小–最大')
    ax.axhline(0,color='gray',lw=.8);ax.set(xlabel='固定滞后 (s)：corr(T(t),V(t+lag))',ylabel='Pearson相关',title='全部预定滞后完整展示，不据最佳滞后修正通道');ax.legend();fig.tight_layout()
    for ext in ['png','pdf']:fig.savefig(OUT/f'图片/固定滞后温度电压关联.{ext}',dpi=300,bbox_inches='tight')
    plt.close(fig)
    fig,axs=plt.subplots(1,2,figsize=(11,4.5));axs[0].hist(spectrum.最强全带周期_s,bins=12,color='#3775BA',edgecolor='black');axs[0].set(xlabel='最强离散频率对应周期 (s)',ylabel='平台数',title='固定69队列：全Nyquist频谱')
    for i,w in enumerate(['前60秒','末30秒','末60秒','末90秒']):
        a=windows.loc[windows.窗口==w];axs[1].scatter(np.full(len(a),i),a.temp_range,s=10,alpha=.5)
    axs[1].set_xticks(range(4),['前60秒','末30秒','末60秒','末90秒']);axs[1].axhline(1,ls='--',color='#B64342');axs[1].set(ylabel='温度峰峰差 (°C)',title='记录周期性持续至平台末段，原因未识别')
    fig.tight_layout()
    for ext in ['png','pdf']:fig.savefig(OUT/f'图片/温度周期性与末段持续性.{ext}',dpi=300,bbox_inches='tight')
    plt.close(fig)
    archive=OUT/'报告初稿存档';archive.mkdir(exist_ok=True)
    for name in ['平台失败原因分析报告.md','健康特征方法复核报告.md']:
        if not (archive/name).exists():shutil.copy2(WORK/name,archive/name)
    source=rules.loc[(rules.方案=='A60')&(rules.范围=='至少180秒且j在0.2至1.1')]
    # 报告指标均从真实CSV汇总，不硬编码预期实验成绩。
    thermal=source.loc[source.规则=='CONDITION_UNSTABLE_temp'].iloc[0]
    pressure=source.loc[source.规则=='CONDITION_UNSTABLE_pa'].iloc[0]
    lines=['# RealHyFC平台失败原因分析报告','',
      '**结论：原1°C筛选与实际温度记录特性不相适应，是主要阻断因素；同时仍有压力、共同负载支持和测试条件证据不足。不能归结为“数据完全不支持二维特征”，也不能直接开始退化模型。**',
      '', '研究身份为DIAGNOSTIC_ONLY。阶段0、阶段1原报告、配置、结果保持只读；阶段1仍为BLOCKED。',
      '', '## 已证实的原规则影响',
      f'原A60在77个相关长平台中保留8个；温度条件边际拒绝{int(thermal.边际拒绝平台数)}个，其中{int(thermal.仅该规则拒绝平台数)}个仅这一项失败；阳极入口压力另有{int(pressure.边际拒绝平台数)}个平台失败，均与温度失败重合。取消单一温度限制后保留74个，未取消任何其他规则。多标签计数不能相加。',
      f'温度拒绝涉及{int(thermal.边际受影响原始平台行数):,}条平台原始记录、{int(thermal.边际受影响末窗口行数):,}条末窗口记录及{int(thermal.边际影响文件数)}个文件。这是同一条寿命轨迹中的技术分段，不是独立电堆实验。各规则在全部候选段与关注区间的排除数都在逐规则独立排除贡献.csv。',
      '', '## 1°C有无实验依据',
      '原论文2.2.1、表1及表3给出名义冷却液入口70°C、j≥0.3 A/cm²每点5分钟、低于0.3每点2分钟、2 A/s斜坡和1 A/cm²处30分钟预调理。没有给出末60秒温度峰峰≤1°C要求，也没有给出冷却液温度传感器型号或精度。1°C是原阶段1事先登记的保守规则，不能说成作者协议或已验证物理公差。3 kPa入口压力峰峰值同样没有由论文认证，暂不擅自取消。',
      '论文表1的阳极/阴极160/150 kPaG是出口压力，原筛选使用p.Si.A/p.Si.C入口压力，不能将两者差异当错误。T.Si.CL为冷却液入口，T.So.CL为出口；单个入口温度通道不代表完整电堆热状态。CDDM段电流校准与EIS测量误差描述不适用于温度传感器。',
      '', '## 温度并非只在平台开头变化',
      f'固定69队列：前60秒温度峰峰差中位数{early.temp_range.median():.6f}°C，末60秒仍为{late.temp_range.median():.6f}°C；末30秒中位数{windows.loc[windows.窗口=="末30秒","temp_range"].median():.6f}°C。因此不能靠延长已达5分钟的平台或只删开头解释全部拒绝。',
      f'全Nyquist频谱复核显示，69个平台最强温度记录周期为{spectrum.最强全带周期_s.min():.6f}–{spectrum.最强全带周期_s.max():.6f}秒，中位{period:.6f}秒，谱峰占比中位{spectrum.全带谱峰占比.median():.6f}。原主表中的4–300秒频段结果保留；完整频带复核在全Nyquist温度频谱复核.csv。约4秒重复记录是观测事实，不是已辨识的控制器或传感器原因。',
      '', '## 温度与电压有什么关系',
      f'69队列前60秒电压P95−P5中位数{early.v_p95_p5.median()*1000:.6f} mV，末60秒为{late.v_p95_p5.median()*1000:.6f} mV。末60秒T−V原值相关中位数{late.temp_v_r.median():.6f}，一阶差分相关中位数{late.temp_v_diff_r.median():.6f}，不同平台正负均可出现。全部±10秒滞后均展示，没有挑最佳滞后、同步校正或温度校正。',
      '电压末段较平稳、温度仍周期波动，两者可以同时发生；不能据此说温度对性能无影响。入口/出口温度、气体温度及温压差均已统计。电流改变后的前段与末段均值可提示局部响应，但缺乏热状态、控制量和测量链元数据，不能识别热响应时间常数或因果方向。',
      '', '## 仪器误差、控制波动与真实热变化的区分',
      '已证实：主要日历采样约1秒，温度记录呈约0.099678°C最小正数值间隔，并有持续周期性。较可信解释：原峰峰阈值对记录链或控制过程中的持续波动很敏感。无法确定：真实温度测量准确度、正常温控允许范围、内部热状态是否同幅变化、4秒现象来自控制器还是采集链。没有足够依据将其认定为传感器故障。电流/密度固定滞后敏感性仅登记，不自动移位通道。',
      '', '## 可复核数据与边界',
      '11份原始极化CSV重新读取并SHA核对；A60分组、拒绝原因和代表电压严格复现原结果。252个全部扫描内长平台保留真实行号、日期、方向和主扫描标记，分别生成原始图；六组规则仅用于最早完整降载段。没有重新扫描860万行运行数据，没有访问旧预测Holdout/Test指标。',
      '', '## 建议',
      '**需要修改测试条件处理方式并独立验证，暂不进入退化预测模型。**二维路线仍值得保留为候选；不能把取消单一温度筛选直接当作最终替代规则。需要原作者或试验台资料确认温度测量和温控含义，再用可确认同负载同条件的重复记录验证窗口均值及温压分层。不能在已看过的全寿命曲线上反复调阈值后宣称独立证据。',
      '', '原论文：Fensterle等，International Journal of Hydrogen Energy 250 (2026) 156060，doi:10.1016/j.ijhydene.2026.156060。本地PDF的第2/3页已人工核对；来源路径、SHA及逐页文本见文献核对目录和实验说明与字段证据.csv。']
    (WORK/'平台失败原因分析报告.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    pressure_detail=pd.read_csv(OUT/'表格/取消温度后独立压力障碍.csv')
    with (WORK/'平台失败原因分析报告.md').open('a',encoding='utf-8') as stream:
        stream.write('\n## 压力障碍的边界检查\n\n')
        for r in pressure_detail.itertuples():
            stream.write(f'- {r.文件名}，j≈{r.实测j:.6f}：末60秒阳极入口压力峰峰差{r.pa_range:.6f} kPa、P95−P5为{r.pa_p95_p5:.6f} kPa。\n')
        stream.write('\n850h的峰峰差约3.032 kPa，距原3 kPa规则很近。这提示原峰峰统计对极端值和记录分辨率敏感，但不足以证明压力变化无关紧要。本轮保留原拒绝，未改用P95统计使其通过。\n')
    summary=pd.read_csv(OUT/'表格/六组诊断验收汇总.csv')
    lines=['# RealHyFC健康特征方法复核报告','',
      '**二维表示在部分曲线上具有数学可行性，原失败不能单独归因于数据不足；跨全部主曲线的统一特征资格仍未成立。所有新结果为DIAGNOSTIC_ONLY，不改变阶段1的BLOCKED。**',
      '', '## 一次性六组诊断',
      'A保留全部原限制；B只取消温度峰峰限制。各自比较末30/60/90秒，配置在比较前冻结。没有新增温度阈值搜索。D替代稳态定义未执行：缺少传感器和允许温度波动的独立依据。',
      '', '|方案|具数学支持的主曲线，含BoL|其中老化主曲线|启停后曲线|0.9参考支持主曲线|完整0.3–1.0|',
      '|---|---:|---:|---:|---:|---:|']
    for _,r in summary.iterrows():lines.append('|'+ '|'.join(str(r[c]) for c in summary.columns)+'|')
    lines += ['', 'B组每个窗口均有10/11条曲线达到各自数学条件：8条主曲线（含BoL）和2条启停后曲线。不是10条独立寿命实验，也不代表这些特征已经具备正式健康含义。',
      '', '## 温度以外仍有实际障碍',
      'B60的850h曲线约0.719474 A/cm²平台仍因阳极入口压力波动被拒绝。相邻代表点之间间隔超过0.20，故0.55–0.75中间网格不得跨缺口插值，仅剩8格，未拟合。2300h原曲线约0.359246平台、2300h启停后约0.449167平台也因压力被拒绝。没有暗中放宽3 kPa或0.20括距。',
      '原共同0.3–1.0区间没有完整通过者。主要支持为0.40–1.00（13格），2300h为0.45–1.00（12格）。九条主曲线全部交集只有0.45、0.50、0.80、0.85、0.90、0.95、1.00共7格，仍少于冻结的10格；有不连续缺口。低于0.3平台原停留约2分钟不满足180秒，不能外推凑齐0.30/0.35。',
      '', '## 一维与二维的实际表现',
      f'B60中7条有资格的非BoL主曲线，一维重构RMSE中位数{b.一维RMSE_V.median()*1000:.6f} mV、二维{b.二维RMSE_V.median()*1000:.6f} mV；留出负载组重构分别为{b.负载留组一维RMSE_V.median()*1000:.6f}/{b.负载留组二维RMSE_V.median()*1000:.6f} mV。这些是同曲线损失重构，不是未来退化/SOH预测精度。',
      '保留每条曲线的一维/二维残差和缺失网格。残差并非纯随机：低电流端与中段可有系统性偏差，线性两分量不能完整解释极化损失。拟合点来自局部插值，不能把13格当13个独立真实负载或老化样本。',
      '', '## 参数是否稳定、能否解释机理',
      f'在相同受支持网格上，B30/B60/B90最大参数差为{maxchange:.6f} mV；窗口改变没有让已计算曲线的参数大幅翻转。但数值相关系数仍约−0.970至−0.976，两个分量高度抵消，l1又对应拟合区间外j=0截距，不能将其解释为真实开路或活化损失。',
      '条件分块扰动包含共享BoL，但不包含仪器误差、温压系统偏差或独立重复诊断差异。窗口稳定不等于物理稳态，更不等于跨堆可比。可在后续讨论参考负载处损失与斜率的等价参数化以改善解释，但该重参数化不增加观测支持，本轮未借此改变原方案。',
      '', '## 温压是否系统性影响参数',
      f'7条B60老化主曲线相对BoL的同网格冷却液入口温差均值约{b["temp相对BoL均差"].min():.6f}–{b["temp相对BoL均差"].max():.6f}°C。原始稳态均值仍有跨日期偏差；描述关联表保留l1/l2与温压差相关，没有做校正或显著性检验。年龄、恢复和工况共变，不能从7个时点分离因果温度影响。',
      '', '## BoL与在线时间资格',
      'B组下BoL可在0.40–1.00括定且0.9有支持，BoL对自身损失严格为零；A组BoL支持不足，保留NA。BoL早于首份运行记录、结果发布时间未知。300h实际边界312.15411–327.81354 h；600h为566.1698–615.66968 h，均不按文件名假定提前可用。FC1时钟仍NOMINAL_ONLY，未重新搜索FC1。',
      '', '## 有根据的候选方案及独立验证计划',
      '可保留的候选是：按原论文明确的完整降载平台协议提取窗口均值，同时显式报告温度周期、跨测试均温偏差和压力条件，不将单个入口峰峰值等价于内部稳态。B只是该候选的消融诊断，并非已认证替代规则。',
      '最小下一步为确认测量链、温控公差与压力字段；使用同一电堆、同一负载且可确认测试条件的重复极化记录检查均值重复性和温压敏感性，再预先固定负载支持与筛选规则。论文提及BoL完整EIS敏感性来自另一8片新堆，不能未经硬件身份核对就用于10片耐久堆的温度校正。现有升/降载只能作滞后对照，不能冒充独立老化重复。',
      '停止条件：仍无法得到跨主曲线可比的充分负载支持，或参数受条件处理影响超过信号量级，就继续修改表示/条件处理，必要时保留可验证的一维参考损失；不要训练网络掩盖测量资格。取得依据后另阶段冻结，不从本轮六组挑最优方案升级。',
      '', '## 最终科研建议', '**需要修改健康特征表示或测试条件处理方式；数据证据尚不足以进入退化预测模型。**第三档可继续做资格修订，不能宣称统一健康状态已经完成。没有依据直接否定二维路线，也没有依据直接启动物理约束或跨堆训练。',
      '', '工程验证：11源SHA及A60原始执行严格复现；独立闭式复算30个诊断拟合，最大系数差约2e-16 V。准备代码已归回任务，旧版本完整归档；完整代码当前无正式实验程序，未执行Git提交。']
    (WORK/'健康特征方法复核报告.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    extra=[]
    for n,src in [('固定滞后温度电压关联','69平台全部滞后汇总.csv'),('温度周期性与末段持续性','全Nyquist温度频谱复核.csv;历史69队列分窗口.csv')]:
        extra.append(dict(图名=n,来源CSV=src,PNG_SHA256=sha(OUT/f'图片/{n}.png'),身份='DIAGNOSTIC_ONLY'))
    table(OUT/'表格/补充图片来源与SHA.csv',extra)
    save_json(OUT/'最终研究判定.json',dict(status='DIAGNOSTIC_ONLY_COMPLETED',original_phase1='BLOCKED_UNCHANGED',
       recommendation='需要修改测试条件处理方式，暂不进入退化预测模型',A60_pass=8,B60_pass=74,
       B60_fit_count=int(features.loc[features.方案=='B60','足够数学支持'].sum()),all_main_common_grid=7,
       parameter_window_max_difference_mV=maxchange,temperature_period_median_seconds=period,training=False,git=False))
    print('报告与补充证据完成；原阶段1保持BLOCKED')


if __name__=='__main__':run()
