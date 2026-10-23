"""Reports derived from completed numeric artifacts, never fabricate training results."""
import sys
from common import *
def table(d):
 return '| '+' | '.join(d.columns)+' |\n| '+' | '.join(['---']*len(d.columns))+' |\n'+'\n'.join('| '+' | '.join(str(x) for x in row)+' |' for row in d.itertuples(index=False,name=None))
def stage_a():
 cfg=config();e=pd.read_csv(P/'outputs/SOH_EVENT_PREDICTIONS.csv');m=pd.read_csv(P/'outputs/SOH_EVENT_METRICS.csv');links=pd.read_csv(P/'outputs/P1S_TO_SOH_EXACT_FORECAST_LINKS.csv');t=pd.read_csv(P/'outputs/TFT_PAST_INFORMATION_AND_PARAMETER_AVAILABILITY.csv')
 md('reports/01_统一时间与健康参考协议.md','''# 三种目标与时间协议

F2：48步历史、24步未来、0.5h分辨率，预测动态ΔV，跨度12h。P1S：96h历史，H1/H3分别下第1/3完整物理周期，并不是固定12/36h。参考SOH：Vpol(j=1.0)/0.6989533305168152。P1S分母0.71383023635，两者不可互换。

全部数值只读已冻结Development派生缓存，没有读取原始数据、混合N1、Test、2000/2300h目标或加载TFT权重。来源SHA及原有独立Development读取规则核验仍成立。七个参考只当七个稀疏事件，不插值造标签。

新协议先冻结，再运行。事件选择为每次极化前48个日历小时内最近完整middle_50 P1S；事件时间及target身份用于回顾性队列选择，未来时间/lead/真实R均不进入预测函数。该策略不能证明实时预约已知。原始cycle身份来自回顾性目录，未证明在线识别完整周期。

极化内部DateTime证明先后，极化实际OpHrs2及结果发布时间未知。前一durability cutoff只表示邻近的已知坐标，不是极化真实年龄。BoL只有已审查的2024-12-09日期身份，本轮不伪造精确秒级起点。所有SOH成绩属于按测量完成时间、假定前次结果已发布的回顾性顺序评价，严格在线资格为0。

M0：最后参考保持不变。M1：使用既有、Train内选alpha且标签隔离的96h Ridge H1/H3预测，再以至少3个此前post-BoL锚点做低自由度仿射校准。BoL与P1S warm-up不是同一时刻，不作为校准对。''')
 md('reports/02_合法事件与预测起点.md',f'''# 实際事件与起点

六个后续参考有严格在极化前的P1S。300/600/850没有指向最近完整周期的现有合法Validation预测，且历史校准不足，排除M1。1150/1500/1750的H1和H3各3个事件，共6条精确缓存链接。所有origin、target raw cycle ID、Train标签截止和inner选参隔离均逐行核验。

{table(links[['nominal_h','horizon','origin_cycle_interval_id','target_cycle_interval_id','origin_calendar','calendar_lead_to_reference_h_audit_only','calibration_prior_count']])}

日历跨度是真实测量间隔，仅供评价审计，不作为未来已知输入；操作小时跨度只在P1S目标之间计算。SOH事件资格、全部排除、邻近P1S工况及可比性见输出CSV。P1S的j≈0.78923与参考j=1不同，校准为经验假设，不声称物理测量等价。原有全六锚点AHC不用于早期预测。''')
 md('reports/03_未来SOH基线与实际实验.md',f'''# CPU实际结果

{table(m)}

M0全6事件均值不能与M1三事件比较；正式呈现的是H1/H3各自共同的1150/1500/1750三个事件。三事件既少且来自单堆，还存在测量参考和校准假设，不能称独立确认或统计显著。

本轮新校准命名M2_NC，复用aR+b形式，输入对为各历史参考前“最近完整周期”的R。它不同于原AHC三条selected记录的中位数（含partial记录）；改变算子是为保证校准和预测输入一致。原AHC参数/成绩全部保留，只作历史参照。每个事件分别拟合过去3/4/5对，H1/H3参数相同，不调参；保存6份参数记录。96h模型、alpha、权重没有重拟合。

该实验不以真实future P1S作M1输入。target真实R只保存在原缓存用于误差审计。独立程序用矩阵最小二乘复算仿射，用缓存系数/标准化复算Ridge及所有误差。''')
 md('reports/04_TFT辅助健康信息增益.md',f'''# TFT额外信息资格

{table(t[['nominal_h','origin_calendar','frozen_F2_selection_end_calendar','model_selection_labels_available_before_origin','causal_cached_queries_available']])}

本轮检查过去动态电压、历史工况、当前P1S及既有12h预测，也检查模型参数形成时间，而不是重新做同时窗直接匹配。历史F2最佳检查点用完整Validation早停/选模，最晚标签结束运行小时{t.frozen_F2_selection_label_end_h.iloc[0]:g}。所有现有Validation origin均早于该选模标签截止；把它们当时可用的健康映射输入会遗漏参数时间泄漏。常规冻结模型Validation误差仍是有效历史指标，不因此改写H/O1。

1750前可以在选模完成之后运行新F2推理，但此前没有足够合格TFT→参考事件训练对，只有一个后选模参考事件。单做一次推理也不能证明增量，故未加载模型或制造映射。历史动态信息统计见PAST_DYNAMIC_AND_P1S_INFORMATION.csv；信息可用不等于目标条件等价。12h动态ΔV不外推数百小时。

结论TFT_HEALTH_ALIGNMENT_INSUFFICIENT；没有匹配目标与可用信息下的TFT辅助SOH误差，不声称增益。''')
 md('reports/05_第一阶段研究判断.md','''# 第一阶段结论

三个事件的P1S预测+过去校准回放确实改善M0，可保留为探索性健康衔接证据。缺结果发布时间与准确极化OpHrs2，不能声称已验证严格在线未来SOH。参考定义不变，不建立连续真实标签。

TFT仍是短尺度动态分支；独立健康增益不可评价。已完成本阶段CPU计算、缓存复算与真实图表，按要求独立提交，随即进入软物理CPU资格和寿命定义核验。''')
 md('README.md','''# RealHyFC一体化研究

仅在本目录写入；原历史只读。Python3.11，numpy/pandas/matplotlib；正式新拟合只包括低自由度过去仿射校准与描述性轨迹，不需要torch或GPU。

已完成输出存在时不得再次run。复核命令从研究主目录运行：

```powershell
$env:PYTHONUTF8='1'
D:\\ruanjian\\python3.11.9\\python.exe -B Phase_Integrated_SOH_Physics_RUL_Research\\scripts\\verify_results.py a
D:\\ruanjian\\python3.11.9\\python.exe -B -m unittest discover -s Phase_Integrated_SOH_Physics_RUL_Research/tests -v
```

一次性复现需将本目录完整复制到另一干净副本，删除副本生成物（保留代码），先运行run_stage_a.py freeze，再run；此说明不授权删除原始成果。原协议、源SHA、模型系数与全部事件结果均保存。最终的b/c、final复算命令随交付脚本保存。''')
 print('A reports written')
def final():
 stage_a();decision=json.loads((P/'outputs/FINAL_RESEARCH_DECISION.json').read_text());m=pd.read_csv(P/'outputs/SOH_EVENT_METRICS.csv');b=json.loads((P/'outputs/PHYSICS_CPU_DIAGNOSTICS.json').read_text());sc=pd.read_csv(P/'outputs/CONDITIONAL_CALENDAR_LIFETIME_SCENARIOS.csv');e=pd.read_csv(P/'outputs/EVIDENCE_MATRIX.csv')
 docs={
 '00_整体研究总结_通俗版.md':f'''# 真实结果与总体判断

{decision['status']}。已有动态电压和96h P1S证据可保留，本轮得到三个参考SOH事件的过去信息回放，但整合的物理SOH/RUL方法尚未验证。

同三个事件：M0 MAE0.281172 SOH百分点，H1预测+校准0.079730，H3预测+校准0.032385。严格在线事件0，因为参考结果发布时间未知、极化精确运行小时缺失、预测事件排程回顾性选择。不能把这些成绩写成已认证在线SOH。

TFT没有被证实帮助SOH；过去预测参数的选模时间不满足早期起点。本轮未重新训练TFT/96h Ridge。

软物理CPU检查发现状态分解不可辨识，慢变项与普通平滑的损失和梯度完全相同；EIS没有认证HFR/ASR。真正物理训练损失/参数梯度为NA，匹配训练消融未获资格，GPU0。

未找到与参考SOH一致的真实失效标签。完成事先冻结97/95/90%假设阈值的日历时间条件外推，包含零趋势恢复情景；没有RUL准确率，没有把2300h当EOL。''',
 '01_单数据集统一实验协议.md':'''# 冻结协议

见configs/FROZEN_PROTOCOL.json及sha256；A阶段实验代码单独冻结，B/C代码运行前保存独立身份，原冻结协议不覆盖。三个参考SOH目标定义、物理周期身份、结果发布时间未知边界、日历/运行小时、单位和未来信息处理详见reports/01。来源只读白名单及SHA逐次检查，原始数据/Test/GPU均未访问。

P1软先验λ固定0.1，只作量纲归一和梯度资格诊断；未按Validation调参，没有资格则不训练。条件轨迹预设all post-BoL、last3及零斜率恢复敏感性，365日日历外推，阈值97/95/90%均为场景假设。''',
 '02_历史成果复用与模型关系.md':'''# 复用身份与范围

十个指定历史阶段已逐一审查并记录SHA。H/O1的九个缓存重新复算：F2三Seed RMSE均值7.245400mV，F0 9.067563mV。O1仍为F2_ADVANTAGE_METRIC_DEPENDENT，尤其Seed62尾部误差和target时间等权指标不被RMSE均值掩盖。P2/P3未证明gate独立稳定收益。

96h H1/H3三折MAE改善10.641466%/10.420448%，188条原预测逐折重算；没有重拟合。AHC只参考形式，全六锚点参数不进入早期模型。既有长期SOH任务M0最佳，不能与本次较短lead回放的新三事件数值混作同一实验。SOH桥/EIS历史失败门槛仍保留。

动态预测单位mV，P1S单位mV-equivalent，SOH单位百分点，寿命情景日历h/日；不能互换。''',
 '03_未来SOH实验与合法时间对齐.md':f'''# 稀疏事件回放

{table(m)}

逐点真值、预测、误差、真实起点见SOH_EVENT_PREDICTIONS.csv；身份与模型标签截止见P1S_TO_SOH_EXACT_FORECAST_LINKS.csv。M2_NC以最近完整P1S为校准算子，过去3/4/5锚点拟合，与AHC中位数算子区别明确。仅三事件、不假定在线发布时间，无统计显著性或泛化论证。''',
 '04_TFT对SOH的独立信息价值.md':'''# 尚无独立增量证据

结论TFT_HEALTH_ALIGNMENT_INSUFFICIENT。本轮用“极化前过去信息”路径检查并新增模型参数时间审计。F2经过完整Validation选模，现有缓存起点均在选模标签截止之前；不能用这些缓存作为早期在线健康特征。1750后选模可推理，但没有足够过去事件建立辅助映射的训练和保留评价对。没有新TFT映射训练成绩，也不以12h输出代替几百小时SOH轨迹。''',
 '05_软物理约束及物理依据.md':f'''# 候选资格的实际证据

P1：H=D+Q+noise。Train n={b['Train_n']}，观测矩阵[I,I] rank={b['state_rank']}、nullity={b['state_nullity']}；D+q,Q−q完全等价，两套不同非增D均精确重建实测R。真实Train中{b['observed_recovery_steps']}次观测上升，参考1150→1500也回升，不能令观测SOH全局单调。输出恢复项不能任意抵消慢退化而仍宣称可辨识。

Lslow=mean[(D[i+1]−D[i])²]/s_train²，单位无量纲。输出/潜变量节点梯度已实际计算并有限差分验证；同信息、同参数量普通平滑L与梯度精确相等。这是退化启发式正则化，不是独立认证电化学机理。稀疏不等时点的该统计先验也不是可解释物理退化速率。本轮不以它训练制造物理胜利。

P2：沿用真实65份EIS/3620点及33电路拟合，0.157459/0.177282mΩ为模型依赖R0，不是认证HFR或老化电阻。同堆序列号/布线/寄生校正等不足，认证ASR0，不忽略活化和浓差损失，不能计算真实物理监督项。''',
 '06_物理约束实际实验与消融.md':f'''# CPU研究已执行，受控训练未获资格

两类候选都完成适用门槛核验，见PHYSICS_CPU_QUALIFICATION.csv。真实Train平滑诊断损失={b['actual_Train_statistical_smoothing_loss']:.9g}，节点梯度L2={b['actual_Train_statistical_smoothing_gradient_L2']:.9g}，梯度有限差分最大差={b['finite_difference_gradient_max_diff']:.3g}。这些是统计先验的真实数据诊断，不是训练中的电化学物理损失。合成线性参数化梯度亦通过。

同信息平滑控制损失差/梯度差都0，实际体现它不能独立归因于物理。真正认证Train物理loss、物理对模型参数梯度、损失/预测损失比均NA。没有实际A/B/C优化训练和收益数值；因为P1不可辨识、P2无认证参数，B5预条件未满足，按指令停止候选。没有更换λ、超参数扫描、GPU或随机Seed成绩。''',
 '07_健康轨迹与RUL资格判断.md':f'''# EOL与条件场景

EOL_NOT_ESTABLISHED。历史资料中确有条件特定500mV测量下限，但它不等于j=1参考SOH的终寿准则；约200周期后研究性完整表征支持EoT，而非已知failure。不能采用2300−t，也未认证右删失。

下表是以最后Development参考SOH为出发点，使用过去参考日历日期的固定OLS斜率、并设置零斜率恢复敏感性所得条件阈值时间。所有值为日历时间，不是运行小时RUL。全量post-BoL拟合、last3拟合和leave-one-out斜率包络是外推假设敏感性，不是置信区间。稀疏七参考不足以验证数年外推，超过365日的交点还标记out_of_plot_horizon；未达阈值不是无限真实寿命。

{table(sc)}

固定过去日历斜率的三事件回测MAE为0.279603 SOH百分点，M0为0.281172，平均差极小；1500恢复事件的斜率误差0.754331明显大于M0的0.123993。它未建立稳定长期趋势优势。

同一95%假设阈值，全post-BoL斜率为239.65日、last3为3629.03日，零斜率不穿越。巨大的条件差异说明寿命不可识别，而不是已得到可信数年寿命估计。绘图区仅365日，更远解析交点只做敏感性证据，不能作为推荐寿命。

未使用Test验证轨迹或阈值，真实RUL准确率NA。条件轨迹仅展示不可辨识寿命的假设依赖，不能视为长期模型已合格；也未将M1短周期预测直接外推到这些长期情景。''',
 '08_一体化论文方法框架.md':'''# 实际信息连接

F2：动态V历史→12h ΔV。Ridge96：96h P1S→未来物理周期R。新M2_NC：过去极化/P1S锚点→a,b；未来周期R预测→参考SOH回放。只有后一条衔接实际执行并评价。

SOH参考→假设健康斜率→阈值日历情景已计算，但无真实EOL验证。TFT→SOH及认证物理loss→健康模型仍被资格门槛阻断。F7流程图明确显示阻断处，没有端到端网络或融合增益主张。单RealHyFC研究可以形成透明的动态预测/健康代理研究稿，但标题中的软物理与真实RUL贡献尚不成立。''',
 '09_最终实验指标总表.md':f'''# 实际证据矩阵

{table(e)}

全部指标明细：RECOMPUTED_DYNAMIC_METRICS、RECOMPUTED_P1S_FOLD_METRICS、SOH_EVENT_METRICS、PHYSICS_CPU_DIAGNOSTICS、CONDITIONAL_CALENDAR_LIFETIME_SCENARIOS。不同单位、样本和信息条件不可在一个排名中混比。''',
 '10_投稿资格及缺口.md':'''# 不可省略的缺口

1. 为SOH补可追溯参考结果发布时间、极化实际操作小时、事件预约和在线完整周期身份；本轮仅探索性回放。
2. 保留更多未来参考事件做冻结后评价；三事件校准收益不足以建立稳定性/独立确认。保持单RealHyFC范围，无需本轮跨数据集。
3. TFT增量需要模型已冻结时的合法历史特征、足够参考事件及同目标对照；12h动态输出不直接代表长期SOH。
4. 物理状态需要独立恢复/退化测量，或认证同堆同对象EIS/HFR/ASR及适用工况，才可做实际有效梯度与匹配平滑控制训练。
5. 定义与参考一致的真实EOL准则，提供failure或可靠删失证据与长期轨迹验证；场景阈值不是真实RUL。

当前PARTIALLY_SUPPORTED_NEEDS_EVIDENCE。不能以完成文件数量代替以上科研条件。本轮停止，不开放Test、不换数据集/模型、不再搜索gate或窗口。'''}
 for name,s in docs.items():md('reports/'+name,s)
 md('paper_materials/CLAIM_BOUNDARIES.md','本轮论文图统一声明Development探索性分析。动态分支与P1S未来代理可报告现有误差；参考SOH仅三个事件回放；TFT增量、认证物理优化、真实RUL均未验证。图F5是资格诊断，F6是条件外推，不是训练/准确率。')
 md('README.md',(P/'README.md').read_text(encoding='utf-8')+'\n最终复核：verify_results.py final；一次性B/C计算run_stage_bc.py freeze后run；绘图render_results.py a/bc，报告write_reports.py a/final。A/B/C完成标记阻止重复计算；未来复核只用verify/tests。\n')
 print('Final reports written')
if __name__=='__main__':
 firewall();{'a':stage_a,'final':final}[sys.argv[1]]()
