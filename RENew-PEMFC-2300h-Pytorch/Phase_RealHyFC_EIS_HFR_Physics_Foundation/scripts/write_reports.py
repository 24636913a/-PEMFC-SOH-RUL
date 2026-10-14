"""Describe completed measurements/diagnostics; no invented HFR or training table."""
import json,platform
import numpy as np
import pandas as pd
import scipy
from common import P,PROJECT,REPO,sha,js,cs,md

def run():
 data=json.loads((P/'outputs/CORE_ANALYSIS_COMPLETE.json').read_text());support=json.loads((P/'outputs/REAL_TRAIN_PHYSICAL_SUPPORT.json').read_text())
 fit=pd.read_csv(P/'outputs/EQUIVALENT_CIRCUIT_DIAGNOSTICS.csv');details=json.loads((P/'outputs/EQUIVALENT_CIRCUIT_FULL_PARAMETERS.json').read_text());quality=pd.read_csv(P/'outputs/EIS_DATA_QUALITY.csv')
 pol=pd.read_csv(P/'outputs/POLARIZATION_SOURCE_AND_CONDITIONS.csv');decision=json.loads((P/'outputs/RESEARCH_DECISION.json').read_text())
 R0_bound=sum(v['active_bounds'][0]!=0 for v in details.values());alpha_bound=sum(v['active_bounds'][3]!=0 or v['active_bounds'][6]!=0 for v in details.values())
 ref=fit.loc[fit.object.eq('REFERENCE_SINGLE_CELL')].set_index('family');a=ref.loc['Spatial_BoL_Jan2025'];b=ref.loc['Spatial_1150nom_Apr2025']
 numeric=[dict(event='2025-01-09 spatial reference00',R0_model_mohm=a.R0_effective_model_ohm*1000,area_scaled_model_candidate_ohm_cm2=a.ASR_effective_model_candidate_ohm_cm2,
  relative_fit_RMSE_percent=a.relative_complex_RMSE_percent,measured_HFR=False,certified_ASR=False),
 dict(event='2025-04-25 spatial reference00',R0_model_mohm=b.R0_effective_model_ohm*1000,area_scaled_model_candidate_ohm_cm2=b.ASR_effective_model_candidate_ohm_cm2,
  relative_fit_RMSE_percent=b.relative_complex_RMSE_percent,measured_HFR=False,certified_ASR=False)]
 cs('outputs/REPRESENTATIVE_DIAGNOSTIC_VALUES_NOT_HFR.csv',pd.DataFrame(numeric))
 js('configs/CPU_INTERFACE_AUDIT_IDENTITY.json',dict(code_sha256=sha(P/'scripts/qualify_interfaces.py'),no_formal_training_protocol=True,
  reason='Qualification-only analysis;all gates failed so no lambda/GPU experiment instantiated',
  source='Metadata3007 then non-Test Train2103 numeric rows',no_checkpoint_loaded=True,no_real_model_gradient_computed=True))
 js('outputs/CPU_ENVIRONMENT.json',dict(python=platform.python_version(),numpy=np.__version__,pandas=pd.__version__,scipy=scipy.__version__,
  torch_imported_by_stage=False,GPU_used=False,optimizer_scope='scipy fixed equivalent-circuit parameter identification only;no neural optimizer'))
 evidence=[
 dict(source='Fensterle et al. IJHE250(2026)156060',url='https://doi.org/10.1016/j.ijhydene.2026.156060',
  verified='Local original PDF methods pp2-3 and Figure1 p4 visually inspected',use='10-cell durability versus separate8-cell fresh sensitivity;nominal conditions and measurement objects',late_measurement_values_used=False),
 dict(source='Fensterle et al. Fuel Cells26(2026)e70057',url='https://onlinelibrary.wiley.com/doi/10.1002/fuce.70057',
  verified='Official full-text sections2.1/2.2.3/3.2 accessed',use='Nominal10-cell setup,5-cell reference,spatial local-current definitions,artifact-limited frequency domain and one R0+2R-CPE circuit',late_measurement_values_used=False),
 dict(source='Zahner EIS manual / official import documentation',url='https://doc.zahner.de/zahner_analysis/file_import/',
  verified='Official documentation accessed;export comments and configurable sensor meaning',use='Repeated initial-frequency branch and header-temperature limitations;comments are not certified physical state',late_measurement_values_used=False),
 dict(source='Zahner EIS manual',url='https://doc.zahner.de/manuals/eis.pdf',verified='Official manual accessed',use='ASCII/complex-data and acquisition conventions;no unsupported phase correction applied',late_measurement_values_used=False),
 dict(source='Project ReadMe_EIS_measurements.txt',url='local exact whitelist source',verified='Real content and LFS OID verified',use='Nominal current composition and reference-file00 identity;not a raw channel wiring map',late_measurement_values_used=False)]
 cs('inputs/PHYSICAL_METHOD_EVIDENCE_LEDGER.csv',pd.DataFrame(evidence))
 table='|真实测量事件|模型R0 / mΩ|R0×280 / Ωcm²|相对复数残差RMSE|\n|---|---:|---:|---:|\n'+''.join(f"|{r['event']}|{r['R0_model_mohm']:.6f}|{r['area_scaled_model_candidate_ohm_cm2']:.6f}|{r['relative_fit_RMSE_percent']:.6f}%|\n" for r in numeric)
 texts={
 '00_研究结论_通俗版.md':f'''# 实际结论
 **找到了真实、可用的电化学诊断数据，但尚未得到可直接用于TFT/SOH增强的已认证物理参数。**
 本轮读取65份核心EIS/空间参考谱，3620个实测点；201份独立新鲜BoL敏感性资料仅查头部；7份允许的名义极化曲线完成CPU描述分析。275个文件均为完整本地内容，与Git LFS对象SHA和原字节数全部一致，无下载。
 固定一个文献等效电路完成33次CPU参数拟合，3个预定初始点/谱；5个预定谱作R0 profile，未训练神经网络。以下是模型诊断值，**不是实测HFR或认证ASR，也不把两值之差叫老化增量**。
 {table}
 两类BoL不能混淆：2025年1月名义EIS有10个电池通道的证据；10–11月的新鲜BoL敏感性研究属于独立8-cell系列。名义BoL与1150h虽然10-cell配置相符，但没有原文件的不可变电堆编号及完整通道/时间绑定，纵向同堆资格仍CONDITIONAL。
 原论文确认高频寄生效应；最高频实部和48处表观高频过零交点不能直接认作HFR。正式HFR数0、认证ASR数0。物理Train共同适用域0；完整物理残差无法计算，记NA。
 状态：`EIS_ONLY_DESCRIPTIVE`、名义纵向`CONFIGURATION_CONDITIONAL`、`PHYSICS_TRAINING_NOT_QUALIFIED`。**没有GPU物理训练，没有新的SOH/RUL或准确率成绩。**
 有用成果是把真实测量、对象/身份问题、寄生影响和参数诊断落实到数据上。下一步可优先补同堆ID、接线/区域转换与去寄生校准，继续辨识欧姆项；目前不能把稀疏EIS直接拼入TFT或宣称SOH预测已验证。
 ''',
 '01_RealHyFC电化学数据资源清单.md':f'''# 已实际检查的资源
 exact whitelist共有275个路径：31份cell-EIS、34份spatial/reference-EIS、201份新鲜BoL敏感性头部、7份名义极化曲线、ReadMe及原论文。DATA_RESOURCE_INVENTORY.csv记录273份测量资源，SOURCE_CONTENT_MANIFEST.csv同时记录方法文件及全部SHA/字节。
 BoL核心cellEIS为两组各10通道（约50.5A与505A），1150h为10单电池加1复合参考；空间系列各16导出通道加1参考文件00。本地真实ASCII约5–7KB/谱，Git中是LFS指针；275个local内容均匹配LFS OID与大小。git-lfs3.7.1可用，没有执行fetch/pull/全库下载，也未打开混合zip。
 核心真实日历2025-01-08/09与04-24/25，独立新鲜敏感性头部2025-10-31至11-06。BoL名称不能代替物理堆/发布时间。频率、复数阻抗或模相、significance、elapsed时间、导出电压/电流齐备；没有精确durability OpHrs2、发布时刻、完整实际RH/P或有效界面状态。
 温度头部的0.x±x°C或unknown不视为75°C实际电化学温度：官方说明该字符串只有正确配置测温卡才有传感器意义。场景名义条件来自论文，不是每谱内部状态实测证明。
  低负载文件名按ReadMe一般规则相加会给60.4A，但实测头部为约50.495A，记录命名歧义，不用名称覆盖实测值；低负载图约0.18A/cm²由头部I/280得到。EIS_DURABILITY_CALENDAR_BRACKETING表只记录实际日历与前后已观测bin，不插值填造EIS OpHrs2或结果发布时间。
 禁止文件只枚举文件名和大小，写入EXCLUDED_TEST_FILE_METADATA_ONLY.csv；不读内容、不算Test全文件SHA、不解析2300h/2000h值。没有混入其他数据集。
 ''',
 '02_EIS测量对象与电堆身份审查.md':'''# 身份分层
 旧N1报告保留，不作改写。本轮补充按“1月名义BoL”和“10–11月BoL敏感性”拆分的证据，避免把旧报告关于独立8-cell敏感性的结论扩大到所有BoL文件。
 原IJHE方法页明确10-cell durability和独立8-cell fresh-sensitivity。[原论文](https://doi.org/10.1016/j.ijhydene.2026.156060)；名义BoL方法的前作说明10-cell及CDDM位置，且实际1月cell-EIS有10个正电压通道。[前作](https://onlinelibrary.wiley.com/doi/10.1002/fuce.70057)。这些支持同配置，但原始文件无物理电堆serial，仍不宣布SAME_STACK_VERIFIED。
 新鲜8-cell敏感性系列对durability10-cell属DIFFERENT_STACK_CONFIRMED；其晚于老化实验的测量不可在早期预测起点作校准。名义BoL对1150h标CONFIGURATION_CONDITIONAL，同物理堆绑定仍UNKNOWN；不能计算同堆老化归因。
 cell-EIS1150的文件00为文献所述CDDM上方5-cell复合参考，不是整个10-cell阻抗，也没有除5/除10移作单电池ASR。空间文件00是参考单电池；01–16涉及局部电流响应，而不是16块独立MEA。
 文献图给出7入口、2中央、7出口的区域定义；但所发布文件没有独立接线表证明导出index与图中区域或U1–U11的一一关系。CHxx和unit329xx是仪器通道/设备ID，不是电堆serial。未用电压接近来猜U通道，不翻转空间负实部，不用全280cm²作为局部区面积。
 逐对身份状态、证据和限制见EIS_STACK_IDENTITY_MATRIX.csv，逐通道见MEASUREMENT_CHANNEL_AND_UNIT_MAP.csv。
 ''',
 '03_真实EIS与HFR_ASR分析.md':f'''# 谱与实轴截距
 65谱3620点，无非有限复数点；46或60点原顺序全部保留。文件声明Columns7而表含Number索引及7数值列，实际8列已严格校验。模相按导出角度转换Z=|Z|exp(iφ)，矩形实虚部原样保留；没有相位翻转、取绝对值或正电阻补值。
 扫描从约1kHz升到5kHz再降至0.1Hz，重复频率是同次测量的往返，不是独立重复电池。最大复数重复频率相对差约{quality.max_intrascan_revisit_relative_difference.max()*100:.3f}%。32个空间导出通道出现负实部，保留并标记为原始传递/寄生解释尚未闭合。
 原作§3.2确认cell高频几何/接线寄生，因此本轮固定只对非重复下降分支0.1–1026Hz作一种R0+2R-CPE诊断，未搜索其他频带/电路。空间局部通道未作物理参数拟合；原作还特别指出局部高频伪影，不能套单电池规则。
 >=1000Hz共48处表观实轴过零（24个谱各2扫），保存端点、频率、significance和两扫差。交点没有去寄生证据，全部APPARENT_ONLY；最高频Z'也只作可观测特征。正式HFR提取停止，正式值0，不将外推R0冒充实测截距。
 {table}
 面积换算只给单电池/参考单电池模型候选，单位R[Ω]×A[cm²]=Ωcm²；这些不是认证ASR。复合5-cell与32个局部通道均不作面积归一化。没有任意经验比例或除cell数量。
 ''',
 '04_极化曲线与物理参数联系.md':f'''# 允许的极化资料
 实际读取BoL/300/600/850/1150/1500/1750七份文件的必要列，保留10个活动电压通道均值、实际j、温压与供气统计、扫描方向。稳定setpoint>=120记录的middle50是描述算子，不追加健康标签或重新选参考电流。重复升降负载与逐秒记录不是独立寿命重复。
 主SOH定义仍Vpol(j*=1.0)/0.6989533305168152，辅助1.6未重选；FROZEN_REFERENCE_SOH_UNCHANGED.csv直接导入已有7个参考。已有1150至1500保持率回升保留，不强制每时刻单调，也不能仅凭回升认定具体恢复机理。
 极化名义气体/冷却条件与EIS名义协议有联系，但EIS电流主要约1.8，而SOH参考固定1.0；EIS1150测量4月24/25，极化1150实际4月18，不能当作同一时刻。原CSV有OpHrs而无OpHrs2，例如1150文件导出OpHrs约1415h，1750文件约2135h，均是本轮允许Development文件的另一时钟，不能当耐久老化年龄或新Test结果。
 本轮全时段活动10-cell均值与导出U.S.AveCell最大差{pol.max_active_cell_average_difference_V.max()*1000:.3f}mV，主要是否由更新时序/瞬态导致未证实；描述曲线明确使用活动通道均值，没有据此改旧SOH参考。
 j×模型R0×280只说明该模型系列项的数量级，不能解释完整V：E_rev、活化、浓差及可逆含水影响仍未知。两个参考00分别得到面积尺度候选0.044089/0.049639Ωcm²，并未计算“多少SOH下降由欧姆解释”的比例，因为同堆、同cell、同工况和时间不成立。
 ''',
 '05_物理参数可辨识性及不确定性.md':f'''# 真正CPU辨识
 33谱固定同一电路，每谱3个预定初始点，合计99次最小二乘；5个预定代表谱各7个R0固定profile点，合计35次nuisance优化。模型参数7个，仅用于电化学谱诊断，没有拟合长期退化动力学或网络。输出每谱完整参数、log参数Jacobian奇异值、边界、初值稳定范围及逐点残差。
 相对复数拟合RMSE范围{fit.relative_complex_RMSE_percent.min():.6f}%–{fit.relative_complex_RMSE_percent.max():.6f}%，是阻抗拟合残差，不是TFT电压RMSE。最大Jacobian条件数约{fit.jac_condition.max():.1f}，有{R0_bound}个R0下界解。{alpha_bound}个谱至少一个CPE指数到1；指数1是理想电容的合法边界，不等价于所有R0都失效，但提示不能把常规对称参数误差条当完整物理不确定性。
 R0 profile记录固定范围0.5/0.75/0.9/1/1.1/1.25/1.5倍、其余参数优化后的cost。频率点与往返相关，且寄生/接线/状态有系统误差，profile不是95%置信区间，也未将文献的<0.3%仪器幅值估计当总误差；该仪器估计未包含线缆效应。
 谱拟合可辨识的有效电阻不自动等于独立膜损伤参数。τ、CPE指数、过程分配、范围截断和状态变化会影响解释；三初值同解仅为优化稳定，不能认证不可逆机理。
 ''',
 '06_与TFT动态双路径模型的结合资格.md':f'''# 两条路线
 路线A已实际完成：源身份、真实谱、固定电路诊断、单位和极化关系核对。它提供有用的电化学描述信息，但不能作同堆纵向SOH退化归因或未来SOH预测。
 路线B为PHYSICS_TRAINING_NOT_QUALIFIED。原F2原生TFT、因果多尺度及历史融合、48→24、14槽位完全保留；未加载/修改checkpoint，未拼EIS特征，未重启P1S健康门控。
 新防火墙先解析前3007条必要身份元数据，再确认Development选中0–3004；仅解析Train前2103条工况数值。边界中的1条test记录只看身份元数据，未解析其电压/工况。Train最大真实h1310，平均j接近1.8的窗0个；虽{support['instantaneous_max_j_ge1_75_bins']}窗含瞬时高负载，整半小时均值不能等同稳态EIS。平均j约0.18的17窗也没有共同内部状态认证。
 ASR若以后合格，也只闭合欧姆项；不能把E_rev和其他极化设0。未来实际j/T/RH/P仍未知，不能偷偷给模型输入；训练期教师还需独立完整物理结构、参数发布时间、purged时间隔离及同信息/容量非物理对照。
 96h Ridge三折、P1S/M2/AHC校准与旧负面结论保留，全六锚点回顾性参数未进入早期预测。
 ''',
 '07_软物理约束CPU资格实验.md':'''# 已执行的资格项
 检查测量身份、参数profile/Jacobian、单位、真实Train2103窗、相邻频率往返与寄生、预测起点信息及同信息控制接口。
 唯一条件候选是EIS锚定欧姆项的完整电化学电压残差：L=[(Vhat−(E_rev−eta_act−j ASR−eta_conc))/sV]²，sV沿用0.13680318714138442V。这里没有省掉未知的三项，也没有给真实Train补造教师电压。
 合成夹具的解析/有限差分输出及ASR梯度一致，仅证明数学接口。真实Train适用域0、物理残差/损失/参数梯度贡献均不可计算NA，不记0，不把合成参数当实测ASR，也未为拿梯度而加载原F2作无意义重复审查。
 资格矩阵保存逐项FAIL及2103行剔除原因：HFR/ASR身份、内部工况、发布时间/真实clock、完整方程与未来信息不齐。没有用Validation调参数/λ，没有lambda搜索。尚无独立物理教师，因此无法实例化同信息非物理训练对照。
 ''',
 '08_受控GPU结果_若资格通过.md':'''# 未启动正式物理训练
 全部关键资格未成立，GPU runs0，新增网络参数0，原checkpoint未加载或改动。没有物理增强权重、训练预测或日志可报告。
 动态电压RMSE/MAE、12h末端/尾部改善、物理残差降低、三seed稳定性比较均NA；没有生成虚构成绩表。33个CPU电化学拟合与99+35次最小二乘不是GPU模型训练。
 未来如补齐独立参数/工况/时间资格，才冻结唯一候选、权重和预算并做42/52/62配对；本轮不自动进入下一轮。
 ''',
 '09_SOH_RUL研究条件与下一步.md':'''# 回答研究问题
 RealHyFC有可实际使用的真实EIS，但目前用途是描述和条件性模型辨识。名义BoL与1150h的同堆serial、通道与实际运行时间尚未认证；独立8-cell新鲜敏感性与10-cell durability必须分开。测量谱、拟合参数、当前性能参考、未来SOH与不可逆损伤不是同一概念。
 优先补三类信息：原作者的同堆serial及名义/敏感性文件归属；PAD/原U通道/CDDM区域和局部电流增益/面积校准表；同设置短路/开路/线缆去嵌入及重复稳态HFR/电流中断测量。再同步记录诊断OpHrs2、结果发布时间、温湿压及扫描/恢复状态。
 值得保留真实谱资料与受寄生控制的系列电阻辨识方向；没有资格晋级任何物理训练方法。停止最高频Z'冒充HFR、区域R乘全面积、复合参考任意除cell数、两个未知堆作老化轨迹、ASR单项替代全电压以及普通辅助拟合改名PINN。
 论文目前可保留整模型动态多步电压预测和96h P1S短期代理预测的已有Development证据；门控独立增益、健康门控增益、SOH桥和物理训练创新仍未成立。
 SOH需足够同步且工况可比的健康参考与可逆恢复分离；RUL另需合法失效阈值、实际失效或删失及长期预测验证。没有把记录终止当失效，没有97%或其他RUL训练。提交main后停止，不创建新模型。
 '''}
 for name,text in texts.items():md(name,text)
 md('01_EIS测量身份与可比性.md','# 身份审查入口\n\n完整结论见02_EIS测量对象与电堆身份审查.md及outputs/EIS_STACK_IDENTITY_MATRIX.csv。名义BoL/1150h为CONFIGURATION_CONDITIONAL，新鲜BoL敏感性/老化堆为DIFFERENT_STACK_CONFIRMED；不宣称同堆老化。')
 md('README.md','''# RealHyFC EIS/HFR物理基础研究
 真实数据CPU分析完成：65谱3620点、33固定电路拟合、5组profile、201敏感性header及7名义极化。EIS_ONLY_DESCRIPTIVE；PHYSICS_TRAINING_NOT_QUALIFIED。
 顺序：run_data_analysis.py freeze/run（已完成拒绝重复拟合），qualify_interfaces.py（已完成），verify_results.py，render_figures.py，write_reports.py。所有模型及历史实验只读。
 数据来源exact whitelist与LFS content/OID校验齐全。无Test数值、无GPU、无新SOH或RUL。方法支持及读取程度见PHYSICAL_METHOD_EVIDENCE_LEDGER.csv。
 ''')
 md('reports/README.md','报告位于阶段根目录00–09。figures为真实测量/模型诊断图，不是HFR实测或物理模型增益。全部原始路径、日历与对象在inputs/outputs的来源和身份表。')
 print('10 required reports and source/identity handoff written')
if __name__=='__main__':run()
