"""Chinese handoff derived from real CSV/JSON, never substitutes planned scores."""
from common import *
def table(d):
 def val(x):
  if pd.isna(x):return 'NA'
  return f'{x:.6f}' if isinstance(x,(float,np.floating)) else str(x)
 return '| '+' | '.join(d.columns)+' |\n| '+' | '.join(['---']*len(d.columns))+' |\n'+'\n'.join('| '+' | '.join(val(v) for v in r)+' |' for r in d.itertuples(index=False,name=None))
def main():
 d=json.loads((P/'outputs/DECISION.json').read_text());u=json.loads((P/'outputs/DIAGNOSTIC_RECEIPT.json').read_text());v=json.loads((P/'outputs/INDEPENDENT_VERIFICATION.json').read_text())
 cfg=json.loads((P/'configs/FROZEN_PROTOCOL.json').read_text());cover=read(P/'outputs/COVERAGE_BY_LOAD.csv');err=read(P/'outputs/RETENTION_ERROR_AND_SENSITIVITY.csv')
 pred=read(P/'outputs/CONDITIONAL_RETENTION_DIAGNOSTIC.csv');matched=pred.loc[pred.seed.eq(42)&pred.conditional_retention_supported]
 b=read(P/'data_contracts/DEVELOPMENT.csv');ref=read(P/'data_contracts/EARLY_REFERENCE_POOL.csv');aux=json.loads((P/'outputs/AUXILIARY_QUALIFICATION.json').read_text())
 timeline=matched.drop_duplicates('target_row');first=str(timeline.target_first_calendar.min());last=str(timeline.target_last_calendar.max())
 # Fixed12h endpoint only: descriptive rank relationships, no inferential p-values.
 e=read(P/'outputs/FIXED_12H_ENDPOINT_DIAGNOSTIC.csv');e=e.loc[e.seed.eq(42)].copy();e['j']=e.target_row.map(b.actual_current_density_A_cm2_mean)
 relationships=[]
 for outcome in ['actual_voltage_V','conditional_voltage_retention_observed_percent','CR_mean_percent']:
  for x in ['target_ophrs_h','j']:
   relationships.append(dict(seed=42,horizon_h=12,n=len(e),outcome=outcome,exposure=x,descriptive_Spearman=float(e[outcome].rank().corr(e[x].rank())),inference='No significance or degradation attribution'))
 cs('outputs/DESCRIPTIVE_LOAD_TIME_RELATIONSHIPS.csv',pd.DataFrame(relationships))
 md('01_已有实验与本轮差异.md',f'''# 实际结论：B，部分可行

冻结F2能够产生有真实数据支持的回顾性条件电压保持率；这还不是BoL归一化健康指标、在线SOH或参考极化SOH预测。统计工况接近并不证明物理测量等价。

历史直接桥接结论 `TFT_TO_SOH_INSUFFICIENT_SUPPORT` 保持不变。以前研究动态预测到极化参考SOH的合法事件连接，本轮研究同类动态半小时电压在早期与后期工况共同支持下的相对变化，没有重新运行已失败的极化事件匹配。

本轮复用Phase H的F2三种子缓存，每个种子4920条查询、205个起点、251个不同目标窗；总计14760条查询不是14760个独立健康样本。源自Development Validation，模型选择使用过该Validation，不能称为独立Test。

48历史步→24未来步，0.5h分辨率；Raw14输入、Train冻结Top10掩码、历史电压EWMA α=0.3均保留。恢复公式为原始未平滑起点电压加已恢复伏尺度的缓存ΔV，没有再次逆标准化。缓存truth由float32差分形成，与原始float64目标的恢复最大差{v['raw_voltage_float32_truth_roundoff_max_V']:.12g} V；身份本身没有错位。

读取严格由前3007条元数据和1819.997738h界限决定，再仅解析入选3005条非Test记录的特征。排除记录的电压、工况等数值未解析。35个历史共用字段、105175个单元格逐位一致，不用粗容差；详见outputs/READ_RULE_EXACT_COMPARISON.csv。禁止整表读取与整混合文件哈希。

已审阅并登记：原TFT→SOH阶段、Phase F结构、Phase H多种子协议及训练源码、Ridge96结果、一体化阶段README及参数时间资格。32个源文件含F2/F阶段checkpoint和M2/Ridge参数均保留SHA256及Git blob。未重新训练、未推理、未改旧模型、未读Test数值或使用GPU。
''')
 md('02_同工况匹配资格分析.md',f'''# 同工况的统计支持与物理资格分开

实际存在电流密度均值/波动/极值、负载切换/运行点切换、零电流比例，以及阳极/阴极/冷却温度、湿度、压力、流量及计量比等工况记录。冻结Raw14输入主要是电压与电流负载统计；温湿压等仅作回顾性资格分析，没有加入TFT。逐字段身份见data_contracts/INPUT_VERSUS_AUDIT_CONDITIONS.csv。

早期参考规则在比较电压之前冻结：Train中第一次300h极化开始之前、窗口有效、无内部日历断点、RH均值0–100%、25项工况有限的全部记录。得到{d['early_reference_bins']}个半小时窗，覆盖{d['early_reference_first_h']:.1f}–{d['early_reference_last_h']:.1f}真实运行小时，涉及{d['early_reference_complete_cycles']}个完整物理周期；并非24条独立寿命轨迹。日期{ref.first_datetime.min()}至{ref.last_datetime.max()}。这是早期Development参照，不是经认证的动态BoL。极化BoL测量也不能替代这些不同动态工况的BoL电压。

逐项容差固定为有效Train该字段IQR的0.5倍；IQR=0时要求数值相等；零电流是否存在的类别也必须一致。全部25项同时满足。此容差依据Train分布，是统计共同支持诊断，不是假设仪器精度或已证明的物理容差；所以物理认证数量仍为0。完整数值见configs/TRAIN_ONLY_CALIPER_SCALES.csv。没有以电压大小、趋势或结果误差挑选参考。

主分析须至少3个参考窗、来自至少3个完整物理周期；同周期先均值，再周期等权汇总，避免长周期被重复加权。参考均值、周期SD、极值及每一匹配对工况差异均保存；周期SD不是独立样本置信区间。

{table(cover.loc[cover.caliper_IQR.eq(.5)])}

145/251={145/251*100:.2f}%仅为探索性统计参考支持率；232/251={232/251*100:.2f}%电流负载匹配不能替代温湿压共同支持。145个主要来自目标阶段{matched.target_ophrs_h.min():.1f}–{matched.target_ophrs_h.max():.1f}h，日历{first}至{last}。低平均电流<0.5 A/cm²的50个目标无一满足主支持规则，不能把中高负载结论外推到它们。

预先固定的敏感性，不用于重选规则：

{table(cover.loc[cover.load.eq('ALL'),['caliper_IQR','targets','load_only_any','joint_any','joint_3cycle_supported']])}

从13变为145再188说明覆盖强依赖匹配定义。只比较聚合统计也不能保证负载顺序、瞬态、水合状态与气体内部状态相同；不能据此宣称物理同工况已认证。
''')
 md('03_TFT电压型健康指标实验.md',f'''# 可以计算什么，不能计算什么

认证的 `HI_V=100*V_pred/V_BoL` 合格样本为0，输出HI_BoL_percent全部NA。本轮从未把任意动态电压除以0.6989533305168152 V。

在145个具有统计共同支持的目标上，计算明确改名的 `CR_V=100*V_pred/V_early`，即相对早期Development的条件电压保持率。它只是一项回顾性探索性健康代理，未证明测量等价，不能称标准SOH。每个种子有2871条重叠查询；完整逐点电压、分母与比值见outputs/CONDITIONAL_RETENTION_DIAGNOSTIC.csv。

误差定义为对同一早期分母的 `100*(V_pred-V_actual)/V_early`，是预测电压误差向比值的传播，不是对真实SOH标签的误差。

{table(err.loc[err.caliper_IQR.eq(.5),['seed','n','MAE','RMSE','P95','MAX']])}

上述单位均为条件保持率百分点。Seed42比值范围{u['seed42_ratio_min_percent']:.6f}%–{u['seed42_ratio_max_percent']:.6f}%；189条预测超过100%，不裁剪、不强制单调。固定12h终点有117个支持查询，相邻支持点出现59次上升，既可能来自恢复/工况/参考分母变化，也可能来自预测误差，不能当成不可逆损伤被修复的证据。原始动态电压全查询Seed62仍有307.985mV最大误差，条件支持子集较好的尾部不能掩盖全缓存异常。

参考分母在主分析0.690754–0.736441V，参考周期数3–17。周期SD中位数{u['reference_cycle_SD_median_mV']:.6f}mV，最大{u['reference_cycle_SD_max_mV']:.6f}mV。Seed42采用周期均值或周期中位数的最大比值差{u['seed42_mean_vs_median_max_abs_pp']:.6f}百分点；参考周期最小/最大值敏感性范围宽度中位数{u['seed42_reference_range_width_median_pp']:.6f}、最大{u['seed42_reference_range_width_max_pp']:.6f}百分点。这是分母敏感性，不是置信区间。

误差传播导数：dCR/dV_pred=100/V_early（百分点/V），dCR/dV_early=−100*V_pred/V_early²。以周期SD作局部扰动尺度，Seed42线性敏感性中位数{u['seed42_reference_SD_linear_sensitivity_median_pp']:.6f}百分点；不是新的物理损失或有效训练梯度宣称。

完整敏感性、负载分组、24步误差和原始电压保持基线分别见REFERENCE_SENSITIVITY_POINTWISE.csv、CONDITIONAL_RETENTION_LOAD_DIAGNOSTIC.csv、CONDITIONAL_RETENTION_HORIZON_ERRORS.csv、CONDITIONAL_RETENTION_BASELINE_METRICS.csv。原始电压保持基线跨负载极差，不能把胜过它解释为健康识别成功。

固定12h终点的时间/电流秩相关仅作描述，不检验显著性，不把相关性当退化真值：

{table(pd.DataFrame(relationships)[['outcome','exposure','n','descriptive_Spearman']])}

原始电压与电流的秩相关约−0.997；采用早期同统计工况参照后，实测条件保持率与电流仍约−0.738，与运行时间约−0.046。预测比值与时间约+0.174。这说明归一化没有排除明显的负载混淆，也没有给出清晰的退化轨迹证据。B判定仅支持条件电压诊断的可计算性和探索价值，不表示健康趋势已被验证。

严格在线SOH为0：未来实际工况仅作事后匹配，无可认证的提前给定工况计划；且冻结缓存checkpoint选择曾使用Validation标签至{b.iloc[2403].last_datetime}，晚于全部缓存起点。回顾性缓存误差仍保留其原Development Validation身份，不伪装成前瞻部署。
''')
 md('04_周期级性能预测辅助分析.md',f'''# 固定状态观测只能辅助解释

代码保持P1S名称以便复现，论文表述为“固定状态电压观测值”和“周期级性能保持率预测”。Ridge96既不输入TFT，也不形成第二主干，不重训、不融合、不求加权平均。

复用三折冻结结果，共H1 50、H2 47、H3 44条Ridge预测，共141条。H1/H2/H3是未来1/2/3个完整物理周期；真实运行提前量分别约9.52–9.89、19.04–19.41、28.57–28.93小时，日历时间还会受停机影响，不能改称1/2/3小时。

{table(read(P/'outputs/FROZEN_CYCLE_EQUAL_FOLD_METRICS.csv'))}

上表是mV-equivalent，不能与动态电压mV或CR百分点直接比较。H2第一折最大误差恶化事实保留：Persistence1.147497→Ridge1.202287mV-equivalent。

按真实运行小时检查目标测量落入动态预测窗，得到{aux['potential_time_landing_rows']}条跨度记录，涉及{aux['unique_physical_target_cycles']}个不同目标物理周期。并不等于33个独立事件：不同H会重复同一目标。原始cycle ID、目标时间、半小时窗、固定电流及动态电流波动逐项保存在CYCLE_FORECAST_TIME_LANDING_DIAGNOSTIC.csv。

这是时间落窗诊断，未把可用时间冒充精确测量时钟。固定状态选择性稳定dwell电压不同于半小时动态平均，缓存也没有登记相同的预测起点与观测算子。合格同目标同起点评价对为0；不计算方向获胜数或两分支趋势相关成绩。固定状态电压可说明局部性能变化，但不能验证动态比值为SOH。

F4仅在两块独立纵轴展示Development时间背景与覆盖差异，不拟合共同趋势、不插值、不同分母不共用健康数值标尺。
''')
 md('05_参考SOH对照及科学结论.md',f'''# 参考极化SOH结论保持

`SOH_ref=V_pol(j=1.0 A/cm²)/0.6989533305168152` 定义不变。7次BoL、300、600、850、1150、1500、1750 Development事件的历史结果、日历测量区间、条件资格和可用时间缺口复制为只读证据；见REFERENCE_SOH_PRIOR_QUALIFICATION_UNCHANGED.csv与SEVEN_POLARIZATION_TIME_CONDITIONS.csv。

没有重复运行已被否决的事件匹配，没有插值SOH或重建标签，没有用未来参考标签选择分母。原资格中共同事件数0，精确耐久运行时钟/结果发布时间与工况观测算子仍缺少认证。新动态比值不能绕过这些障碍，状态仍为 `REFERENCE_SOH_COMPARISON_NOT_QUALIFIED`，SOH MAE/RMSE为NA。

三种电压定义不同：动态半小时均值、固定状态稳定dwell值、极化j=1.0参考值。负载组成、温湿压、扫载顺序、瞬态与可逆恢复都可能影响电压。温湿压统计接近不足以断言不可逆退化相同。

本轮最终判定B：部分可行，限于统计共同支持下的回顾性条件电压比值。实测比值仍有明显电流相关（约−0.738），时间趋势很弱（约−0.046），健康意义尚未独立验证。不能宣称已建立BoL归一化HI、严格在线健康预测、物理增强训练、参考SOH精度改善或真实RUL能力。

独立核验恢复14760条查询、753个目标×阈值组合、11426条匹配对；8613个合法比值由50位Decimal独立计算，最大差{v['Decimal_ratio_max_abs_difference_pp']}百分点。32个历史源文件SHA不变。报告、CSV和图的身份由ARTIFACT_SHA256.csv索引。
''')
 md('06_下一步决策.md',f'''# 本阶段收口，不扩展模型

1. RealHyFC可以开展已记录聚合工况的回顾性比较，但尚不能认证完整物理状态相同。
2. 冻结主规则有145/251个目标支持，来自{matched.target_ophrs_h.min():.1f}–{matched.target_ophrs_h.max():.1f}运行小时；每个种子2871条重叠查询，不能当独立寿命样本。低负载组主支持为0。
3. 双路径TFT可形成条件电压保持率，Seed42传播MAE约0.639581百分点；这不是对真实SOH的准确率。实测比值仍与电流明显相关（约−0.738），与时间关系弱（约−0.046），尚无清晰退化轨迹证据。BoL指标合格数0。
4. 仅为事后探索性诊断，严格在线样本0。模型选择时间和未来计划缺失分别构成独立障碍。
5. 周期级性能预测保留为辅助参考；33条运行小时落窗记录不具等价测量与共同预测起点，合法趋势评分0对，不强行关联。
6. 距参考SOH至少缺：同对象、相同动态/固定协议的可认证BoL工况—电压参照；负载顺序与环境稳定/水合状态的可比性证据；预测起点已知的未来计划；冻结模型后未参与选择的合法前瞻Development回放区间；与参考极化测量精确连接的物理时钟、结果可用时间及相同观测算子。

最小后续动作是补齐/审查上述测量协议与计划可用时间证据，先建立共同观测算子和可比性门槛；若现有授权资料不能补齐，就保留动态电压预测与条件比值的科学边界，不通过改阈值、重新优化TFT或追加RUL来制造SOH成绩。此处是决策建议，本轮未自动执行新Phase。

论文可用：冻结TFT动态预测结果、统计匹配覆盖与失败负载区间、条件比值和分母/误差传播诊断；使用“探索性Development回顾分析”的身份。不能写成标准SOH估计已验证。无需新网络，也不能把普通平滑命名为物理约束。
''')
 md('README.md','''# 双路径TFT电压型SOH指标可行性

实际结论B：回顾性条件电压保持率有统计共同支持；BoL/在线/参考SOH资格均未成立。

先读06_下一步决策.md及02_同工况匹配资格分析.md。全部结果仅Development，原阶段与权重只读。

现有结果复算（无需读取原混合数据或训练）：

```powershell
$env:PYTHONUTF8='1'
D:\\ruanjian\\python3.11.9\\python.exe -B scripts/verify_independent.py
D:\\ruanjian\\python3.11.9\\python.exe -B -m unittest discover -s tests -v
```

工作目录为本Phase。Python3.11，numpy/pandas/SciPy/PyYAML/matplotlib；不需要torch。prepare.py→freeze.py→analyze.py→diagnostics.py是一次性数据读取与分析，完成标志阻止覆盖；不要在现有成果上重复运行。原始重现只能在独立干净副本按顺序执行。verify_independent.py从原冻结源与本阶段CSV独立复算，不导入分析函数。render_figures.py仅用CSV绘图；write_reports.py从实际CSV/JSON生成中文报告；quality_gate.py保留每次测试/验证失败证据，seal.py封存SHA和代码/图片索引。

主规则SHA见configs/FROZEN_PROTOCOL.sha256；来源Git blob和SHA见data_contracts/SOURCE_SHA256.csv；全成果SHA见outputs/ARTIFACT_SHA256.csv。图源/代码见data_contracts/FIGURE_SOURCE_INDEX.csv。无Test数值读取、无新模型推理训练、无GPU。
''')
 print('Six reports and README written from actual results')
if __name__=='__main__':main()
