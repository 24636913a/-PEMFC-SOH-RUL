"""Seven requested reports, evidence tables and one bounded future protocol."""
from common import *
def table(d):
 def f(x):
  if pd.isna(x):return 'NA'
  return f'{x:.6f}' if isinstance(x,(float,np.floating)) else str(x)
 return '| '+' | '.join(d.columns)+' |\n| '+' | '.join(['---']*len(d.columns))+' |\n'+'\n'.join('| '+' | '.join(f(x) for x in r)+' |' for r in d.itertuples(index=False,name=None))
def main():
 dec=json.loads((P/'outputs/FINAL_DECISION.json').read_text());unit=rd(P/'outputs/SAMPLE_UNIT_LEDGER.csv');inv=rd(P/'outputs/FIXED_STATE_INVENTORY.csv')
 profile=rd(P/'outputs/PRIMARY_STATE_MEASUREMENT_PROFILE.csv');split=rd(P/'outputs/T0_T1_SPLIT_COUNTS.csv');cy=rd(P/'outputs/FIXED_PRIMARY_CYCLE_OBSERVATIONS.csv')
 fold=rd(P/'outputs/FUTURE_CYCLE_TASK_TIME_GATES.csv');drift=rd(P/'outputs/LONG_TERM_AND_WITHIN_CYCLE_VARIATION.csv');seq=rd(P/'outputs/IRREGULAR_48OBS_TO_24OBS_NOT_NATIVE_TASK.csv')
 matched=rd(P/'outputs/FROZEN_PAA_CONDITION_MATCHING.csv');matched_mV=matched.late_minus_early_retention.median()*.71383023635*1000
 # Additive provenance: inspect existing seven-event metadata only, no raw/Test polarization.
 pol=ROOT/'Phase_Polarization_Physics_to_Health_Qualification/outputs/POLARIZATION_QUALIFICATION.csv';rel=pol.relative_to(ROOT).as_posix()
 cs('data_contracts/ADDITIONAL_METADATA_SOURCE_SHA256.csv',pd.DataFrame([dict(path=rel,sha256=sha(pol),git_blob=subprocess.check_output(['git','rev-parse','HEAD:'+ROOT.name+'/'+rel],cwd=REPO,text=True).strip(),role='Seven-event object/protocol metadata; no threshold changes')]))
 q=rd(pol);assert len(q)==7 and q.cells.eq(10).all() and q.area_cm2.eq(280).all();cs('outputs/SEVEN_POLARIZATION_OBJECT_AND_PROTOCOL_METADATA.csv',q)
 leaks=[
 ('Full-cycle grouping','PASS','N1 Train/Validation supervised target cycle overlap0; cycles crossing split boundaries excluded'),
 ('Future target vs history','PASS','Only measured windows strictly after origin; input ends at latest available N1bin; future conditions not inputs'),
 ('Reference availability','CONDITIONAL','0.71383023635 is first10cycle median, available171.82091h; earlier retention not prospective; rawV labels do not need denominator'),
 ('Test-driven state selection','PASS','All long positive state tokens selected from frozen metadata before new voltages; no Test numbers'),
 ('Model-selection timing','NOT_EXECUTED','No new model; future fits must finish before first origin. Two equal-time labels need stricter purge'),
 ('Train-only scaler','NOT_EXECUTED','No new scaler fitted; future protocol Train-only, including inner split'),
 ('Polarization object and protocol','PARTIAL','Same nominal stack,10active cells/280cm2; sweep/time/conditions differ, publication and exact durability clocks unknown'),
 ('Independent validation events','LIMITED','T1bin151labels but14cycle blocks; H1 with48input Val13/6/22, one life trajectory'),
 ('Within-cycle observations','DEPENDENT','3503times/146cycles, not3503independent health states'),
 ('Retrospective physical QC','AUDIT_ONLY','Uses observed target drift/temperature for measurement-quality audit; not an origin-known condition plan'),
 ('Cycle completeness identity','RETROSPECTIVE_CATALOG','Last observedP1window can precede cycle completion; future operational protocol must define completion confirmation before using complete-cycle identity')]
 cs('outputs/LEAKAGE_AND_IMPLEMENTABILITY_MATRIX.csv',pd.DataFrame(leaks,columns=['item','status','evidence']))
 md('01_现有TFT预测目标审查.md',f'''# 保留已验证的动态电压任务

本轮推荐方案C：保留现有动态双路径TFT为论文主模型，Ridge96作为周期级性能预测的辅助分析，暂不启动新健康TFT训练。结论依据真实标签覆盖与测量条件，不是要求必须得到正面健康结果。

冻结合同：48个半小时历史分箱、Raw14输入、Spearman Top10、历史电压EWMA α=0.3、LSTM+因果多尺度CNN及历史工况门控融合，输出24步原始动态平均单体电压ΔV。48步历史不是48个周期；未来24步不是24个周期。原三种子Development Validation RMSE均值7.2454mV只属于此动态目标，不能转移给固定状态电压任务。

实际重建N1历史规则：先读前3007条身份元数据，按bin_end_ophrs2_h<1819.997738保留3005条，再只解析这3005条已证明非Test的特征。与历史35共用字段105175格位级一致；没有整文件读取或整混合文件哈希。

既有TFT→参考SOH结论仍为TFT_TO_SOH_INSUFFICIENT_SUPPORT。上一阶段动态条件保持率为部分可行，但实测比值与电流秩相关约−0.738、与时间约−0.046，未证明标准SOH。现在审查固定状态原始电压，而非重复该映射。

实际复核Phase F/H、PAA、RCR、PHI、旧SOH桥接和上一阶段。PHI只读README及冻结配置身份，没有读取其含既有晚期证据的数值输出；不采用全锚点B2或单调平滑结果作新标签。没有读取Test、运行TFT推理或训练。
''')
 md('02_RealHyFC固定状态电压数据清单.md',f'''# 数据确实更密集，但独立样本不能虚增

本轮按原始文件各自的RCR认证行数读取14个Development前缀，共6708040条记录；最后一个混合源仅取其认证前268684条。先单独核验运行小时，再读受界限限制的数值列。没有把N1的3007行规则套到高频数据，也没读后续文件。14份原始前缀SHA见RAW_SOURCE_PREFIX_PROGRESS.csv。

主状态严格为operation221、设定电流约221.2A，实测电流密度中位数0.789229989 A/cm²。实际电流与设定值并非完全相同；逐片段j标准差中位数约0.000602、最大约0.000754 A/cm²。不能改称0.7A/cm²数据。

{table(unit)}

146个完整周期里145个有24次长片段、1个有23次，合计3503次非重叠观测。每片段原始持续约134.999–392.001s，中位248s；middle50有67–196个点，采样间隔中位1s，最大间隔2s。窗口电压采用10个有效单体均值的中位数，排除U6，与原协议相同。

原PAA同时生成middle50和last25，75%边界点归入last25，因此middle50实际为[25%,75%)。本轮先复现该边界，再逐项核对3796个历史P1/P2片段及141个周期。错误的首次包含边界版本保存在outputs/failures，不进入正式统计。

141个冻结周期值已精确复现到浮点精度，最大差2.22e−16V。另5个周期raw152–156属于合法Development，可独立重建至1810.70h，但仅为审查补充，未更新原PAA/Ridge标签、系数或成绩。146个周期末次实际测量覆盖{cy.last_measurement_h.min():.5f}–{cy.last_measurement_h.max():.5f}h。

稳定性诊断预先冻结：j标准差≤0.005A/cm²、|电压OLS漂移|≤2mV/min、3个温度通道各自跨度≤2°C、至少30点并有完整通道。基于已有极化诊断标准作测量稳定性检查，不等于新的物理等价认证。主状态2703次通过，覆盖146个周期，但没有一个周期的全部片段都通过该更严格诊断。QC子集周期中位数仅作诊断，不能替换原24次汇总标签。

其他状态分开列出，不把不同电流标签混合：

{table(inv.loc[inv.Development_runs.ge(100),['state_token','actual_j_A_cm2','Development_runs','complete_cycle_dwells','complete_cycles','QC_dwells','QC_cycles']])}

0.558162与1.120683A/cm²状态也存在重复观测；低电流0.047958及0.299081平台虽存在长dwell，但没有通过本轮完整稳定诊断，不能只凭电流恒定叫稳态健康标签。稀有异常token仍保留在清单，不另行搜索替代状态。

所有状态合起来仍为同一对象的一条耐久轨迹。不同时间的片段有真实预测意义，可以研究短期运行性能；相同周期内的多次片段、middle50/last25和高频采样不是额外独立寿命标签。
''')
 md('03_候选预测目标T0_T1_T2对比.md',f'''# 三类目标不能混为同一标签

| 候选 | 实际数据 | 时间与标签 | 对现有TFT的适配 |
| --- | --- | --- | --- |
| T0动态ΔV | 3005个Development分箱；1380/205合法Train/Validation起点 | 48×0.5h历史，未来24×0.5h动态目标 | 原冻结模型已有真实证据，保持 |
| T1周期固定电压 | 141冻结周期、146审查重建周期；3503个片段中2703通过诊断 | 非均匀真实测量；H1/2/3是物理周期 | 需新标签、起点和输出协议；原成绩不可继承 |
| T1半小时固定状态标签 | 只保留窗口及保守可用时刻完整落入同一分箱的真实片段，中位聚合；2026个分箱至少有原始片段 | 缺失保持缺失，不插值；QC有效覆盖另计 | 能保留48×14历史，但24步未来标签不完整，需新损失与评价 |
| T2同一0.78923状态参照 | 与T1相同观测，仅换成保持率解释 | 分母0.71383023635V身份冻结，非认证BoL | 不新增独立样本 |
| T2极化j=1.0 | 7个真实Development事件 | 不规则稀疏测量，另有精确时钟/发布时间缺口 | 不能训练原48→24序列神经网络；保留稀疏外部对照 |

同一完整周期只属于一个监督标签分区；跨越原N1 Train/Validation边界的整周期不进入这两区标签。实际结果：

{table(split)}

Train的21144、Validation的3077是重叠窗口重复使用的查询—标签对，独特QC分箱只有982和151，所属完整周期82和14。不是21144/3077个独立健康事件。完整24/24标签起点在两区均为0，未加稳定性QC时也为0，因此缺口不是通过放宽稳定阈值即可解决。

若把146个周期直接枚举为48观测→24观测，表面得到75个高度重叠窗口，但均跳过缺失/不完整物理周期，不再是原任务。历史测量跨度{seq.history_measurement_span_h.min():.2f}–{seq.history_measurement_span_h.max():.2f}h，未来跨度{seq.future_measurement_span_h.min():.2f}–{seq.future_measurement_span_h.max():.2f}h。要求72个连续raw物理周期时合法窗口为0。没有把它们冒称为半小时序列。

H1/H2/H3采用原合法cycle键再检查48步动态输入后的实际数量：

{table(fold[['H','fold','Train','Train_strict_before_first_Validation','Validation','purged_equal_or_later']])}

H1/2/3实际提前约9.52–9.89/19.04–19.41/28.57–28.93运行小时，日历跨度还受停机影响；不得写成未来1/2/3小时。
''')
 md('04_电压健康指标可行性.md',f'''# 固定状态保持率更直接，仍是健康代理

周期固定电压保持率可直接按100×V_cycle/0.71383023635计算。分母确实来自前10个合法周期middle50周期电压中位数，原值不重算替换，参考数据最后可用171.82091h；前10个周期本身属于参考构建，不能声称此前已知这一分母。它是冻结历史参照，不是自动认证的BoL。保留原始电压作训练标签可以避免归一化分母的未来信息问题。

极化参考仍为V_pol(j1.0)/0.6989533305168152，与0.78923状态分母不能混用。

{table(drift[['state_token','cycles','late_minus_early_median_mV','median_adjacent_change_mV','upturns','within_cycle_IQR_median_mV']])}

主状态新增146周期审查的晚—早中位电压差约−5.043133mV、相邻周期变化绝对值中位约0.148736mV，存在长期性能差异；136对连续raw周期中30次上升，未做单调化。原141周期PAA的七工况匹配方向仍下降，复用原匹配得到约{matched_mV:.6f}mV。这支持部分变化并非完全由所记录工况的简单变化解释，但不是因果分解或不可逆损伤证明。

固定状态比任意动态平均电压更易解释，因为电流平台一致、观测算子更明确；但固定电流不代表恒定环境。主状态片段阳极温度55.20–83.45°C、RH28.42–98.25%、压力164.09–164.89kPa(g)。周期中位数也出现55.70°C及95.71%RH的冷湿阶段，不能将该阶段的电压回升全部解释为健康修复。

同周期dwell电压IQR中位约1.392mV，对比长期约5.043mV差异不可忽略。更密集的片段会携带负载前史、热/水合状态和恢复信息；它们可预测运行性能，不能未经验证变成密集退化真值。

7次极化来自同一名义耐久对象及10个有效单体、面积280cm²，标称扫描协议相近，但各段停留/热状态/实际可比区间并非完全相同；原文件实际RH列缺失、存在露点信息，不能伪造RH。精确耐久运行时钟与结果发布时间仍未知。可作稀疏参考和过去信息校准背景，不能插值成数千个SOH标签或用全事件回顾B2制造因果监督。
''')
 md('05_双路径TFT目标适配性.md',f'''# 能构造张量，不等于已经具备健康训练资格

保留48个半小时动态历史时，原14变量和Top10/EWMA合同可以继续作为输入；温湿压只参与本轮审查，不悄悄加入输入。24步的未来固定状态电压则是新的标签定义。因完整24步标签起点为0，原全目标MSE不能直接复用。

稀疏监督在工程上可实现：只对实际测量位置计算mask损失。可是它必须明确预测“该固定状态出现时的电压”，而不是连续12h健康轨迹；未来是否测量以及是否通过温度/漂移诊断是事后资格，不能充当起点已知的输入/计划。需要新的缺失机制审查、评价目标和消融，不能把软件可运行当作健康主张已成立。

另一种周期事件级任务会改变输出维度及时间含义。H1保留48步动态历史后Train37/50/56、Validation13/6/22；H2/H3更少，第二折仅5/4个验证周期。并不存在通用的“低于某个样本数绝对不能训练”定理，但当前数据不支持在尚未解决标签与工况资格的情况下投入新GPU并承诺健康泛化收益。

数据泄漏及可实现性逐项见outputs/LEAKAGE_AND_IMPLEMENTABILITY_MATRIX.csv。N1方案Train标签最晚1303.94h，早于首Validation起点1310.5h，目标周期交集0。共享已知历史输入不等于共享未来监督标签。

新周期任务还发现两条严格时间问题：H2第三折训练目标raw117与首验证起点同一时刻；H3第三折raw116亦如此。如果要求真实模型拟合/选择完成严格早于起点，需要各剔除一条，Train50→49和44→43。它们是未来协议的额外严格门槛，未改写原冻结Ridge结果及其原判据。

没有拟合新scaler或模型，故真实模型训练/选择完成时间为NA；未来必须Train-only scaler、内部时间选择、保留实际计算延迟。原F2现有缓存也不能用于证明过去已经部署的健康预测模型。

现有Ridge96和Persistence可作相同目标的未来对照，但缩减到48步历史合格交集后Train集合改变，冻结Ridge参数不再对应共同Train，不能直接借其完整验证成绩作公平新比较。
''')
 md('06_推荐方案与最小训练协议.md',f'''# 推荐C：本轮不改TFT主目标，不启动GPU

推荐不是否定固定电压的科学价值。真实数据已支持固定状态性能差异与轻量预测研究，但尚不足以直接替换现有动态TFT目标：141冻结/146审查周期只有一条寿命轨迹，原24步完整固定标签为0，密集片段存在工况/周期依赖，健康任务需要重新定义而非换一个列名。

方案A条件未满足；方案B工程上可设想，但会引入新标签合同、稀疏/事件损失、重新训练、配对基线和消融，当前证据不足以证明新增工作量值得。因此本轮不把它升级为推荐的新主线，保留方案C。TFT仍是论文主模型，Ridge96为辅助，不默认用Ridge取代TFT。

当前能否完成公平新消融：尚未完成，不能拿原F0/F1/F2缓存作为固定目标成绩。原架构动态消融仍保留；新目标必须在同标签、同Train、同起点、同容量控制下重新训练。

如果以后负责人仍决定验证新健康TFT，只保留一份最小候选协议，不开展多路线搜索：

1. 首先独立冻结H1“下一完整物理周期的原定义固定状态汇总电压”，固定0.78923状态，不混入其他电流或极化标签。目标为真实电压，保持率仅作后处理；先明确完整周期完成确认与预测起点的可用时间。
2. 动态历史仍48×14，Top10/EWMA及骨干容量冻结；只将输出头登记为1个事件级电压标量。明确它是新任务，未来事件时间不输入模型，不称未来12h同任务。旧权重保持只读。
3. 使用原expanding folds的合法键及已计算48步输入交集；H1现有候选Train37/50/56、Validation13/6/22，进一步按新确认时间/实际拟合延迟严格清理，只能减少，不能为凑数移动边界。
4. 先证明各折参考状态与测量算子可比。Train-only scaler、Train内部时间选择；原超参数候选不重搜。Persistence和Ridge96在完全同一Train/Validation键上作为对照，Ridge若需重拟合必须另行授权，原成绩不替代新对照。
5. 若获独立训练授权，再做冻结F0/F1/F2的同任务配对消融，种子42/52/62；预先固定主MAE、RMSE/尾部、逐折结果和停止规则，不按结果改窗口/状态/阈值。
6. 任一折没有合法样本、参考/完整周期可用时间无法认证、实际训练无法在起点前完成，或公平基线/消融无法建立，即停止。若未获稳定的同样本收益，保留原动态主任务，不扩大模型。

最小需新增的代码是独立事件标签/时间身份构造器、仅新模型的输出头配置、Train-only样本scaler、配对基线和评价适配器；不改旧实现。本轮没有执行以上模型改动或训练。当前不建议下一轮直接启动GPU；先由本次数据清单判断能否补齐测量资格，不能用训练替代资格证据。
''')
 md('07_通俗中文总结.md',f'''# 实际推荐：保留原动态TFT目标（方案C）

1. 原冻结固定状态观测是141个周期点。本轮从合法Development原始数据精确复现它们，还找到5个已有完整周期，可审查重建为146个，末次测量到1810.70h；新5个仅审查，不改原模型成绩。
2. 确实有更密集的真实数据：146个周期中3503次非重叠固定状态片段、452220个middle50原始点；其中2703个片段通过电气/热稳定性诊断。但它们仍属于146个周期、一条耐久轨迹，不能充当3503条独立健康标签。另有0.558162、1.120683A/cm²固定平台，不能混合成同一种电压标签。
3. 不能直接套入当前TFT：原Train1380、Validation205个起点都没有完整24步固定状态目标。稀疏标签可以构造，但Validation只有151个不同QC分箱、14个周期；换成H1事件目标，原48步动态输入可用的三折验证也仅13/6/22。能做新任务的软件实验，不等于足以承诺健康泛化收益，当前不推荐GPU训练。
4. 直接固定状态电压除以0.71383023635更简单、比任意动态电压比例更可解释。但这不是认证BoL；温湿度和恢复仍影响电压，不能自动得到标准SOH。7个j1.0极化参考仍稀疏，不能插值伪造标签。
5. 保留TFT动态预测为主、Ridge96周期性能预测为辅助。固定电压数据有价值，暂不值得为它重做一整套主模型训练与消融。下一步若必须验证新目标，只考虑报告06中的单一H1最小协议，先补资格、再另行决定训练。

推荐路线：Development预处理→Top10/EWMA→冻结双路径TFT→动态电压预测（主）；固定状态观测→冻结Ridge96 H1/H2/H3→周期性能保持率（辅）；7次极化→按过去信息校准/稀疏SOH对照。动态TFT到标准SOH的连接尚未建立，不画成既有贡献。

图见figures/F4_recommended_route.png；原始CSV、代码与图片对应关系见data_contracts/FIGURE_SOURCE_INDEX.csv。所有新成绩均为NA，因为本轮没有训练新模型。
''')
 md('README.md','''# 健康导向电压预测目标资格审查

实际推荐C：保留动态双路径TFT主目标和冻结Ridge96辅助；不是宣称固定电压不可预测，而是当前不足以直接替换主任务并开展健康TFT训练。

先读07_通俗中文总结.md和06_推荐方案与最小训练协议.md。代码只在本阶段写入，无模型训练/推理、Optuna、Test数值访问或GPU。

一次性执行顺序：freeze.py→extract_development.py→audit_targets.py→supplement.py。完成标志禁止重复原始提取/审查；原始复现须在独立干净副本执行，不能删除现有结果。提取读各原始源认证的Development前缀，30分钟数据另用3007身份→3005数值规则。

现有结果复核（本Phase为工作目录）：

```powershell
$env:PYTHONUTF8='1'
D:\\ruanjian\\python3.11.9\\python.exe -B -m unittest discover -s tests -v
D:\\ruanjian\\python3.11.9\\python.exe -B scripts/verify_independent.py
```

Python3.11 + numpy/pandas/PyYAML/matplotlib。独立程序核对全部新清单/稳定门槛/窗口标签数量、3796个历史片段和141个冻结周期，并以csv+datetime+statistics单独抽查三种主要电流状态首/中/末共9个真实片段。不是声称对10680个片段全部重新读原始数值。prefix SHA追溯14份合法源，旧代码与checkpoint的SHA保持。

失败的原始包含75%边界版本及错误记录保存在outputs/failures，不进入正式分析。原始有效表为RAW_FIXED_STATE_DWELL_MEASUREMENTS.csv，QC子集不能覆盖/替换冻结PAA周期标签。所有figure均来自CSV，另有推荐路线SVG。
''')
 print('Seven reports and evidence matrix generated from executed results')
if __name__=='__main__':main()
