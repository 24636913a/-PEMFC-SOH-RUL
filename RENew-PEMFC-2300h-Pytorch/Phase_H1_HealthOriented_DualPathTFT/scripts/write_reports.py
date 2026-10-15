"""Actual executed experiment handoff. Negative results are retained in full."""
from common import *
def table(d):
 def val(x):
  if pd.isna(x):return 'NA'
  return f'{x:.6f}' if isinstance(x,(float,np.floating)) else str(x)
 return '| '+' | '.join(d.columns)+' |\n| '+' | '.join(['---']*len(d.columns))+' |\n'+'\n'.join('| '+' | '.join(val(x) for x in r)+' |' for r in d.itertuples(index=False,name=None))
def main():
 d=json.loads((P/'outputs/FINAL_DECISION.json').read_text());v=json.loads((P/'outputs/INDEPENDENT_VERIFICATION.json').read_text());ref=json.loads((P/'outputs/REFERENCE_QUALIFICATION.json').read_text())
 counts=rd(P/'outputs/SAMPLE_COUNTS.csv');eq=rd(P/'outputs/PRIMARY_EQUAL_FOLD_METRICS.csv');fold=rd(P/'outputs/FOLD_THREE_SEED_MEANS.csv');runs=rd(P/'outputs/FOLD_SEED_METRICS.csv');global_=rd(P/'outputs/POOLED_GLOBAL_TAIL_AND_BIAS.csv');tr=rd(P/'outputs/TRAINING_AND_OVERFIT_DIAGNOSTIC.csv');st=rd(P/'outputs/THREE_SEED_STABILITY.csv')
 ix=eq.set_index('model');samples=rd(P/'data_contracts/USED_SAMPLES.csv');oldcommit=json.loads((P/'outputs/PRETRAINING_COMMIT.json').read_text())['commit']
 controls=eq.loc[eq.model.isin(['H-F2','Persistence','Ridge96_control'])];gainridge=100*(ix.loc['Persistence','MAE_mV']-ix.loc['Ridge96_control','MAE_mV'])/ix.loc['Persistence','MAE_mV']
 md('01_新健康预测任务与数据协议.md',f'''# H1固定状态电压事件预测：真实完成

正式判定 `{d['status']}`。完成27组独立初始化事件模型实验，每组Train内部选择epoch，再独立全Train重拟合，共54次真实GPU拟合；另15次CPU Ridge拟合。不是只做可行性审查。

目标严格为raw_cycle(k+1)的原冻结middle_50周期代表单体电压，0.789229989 A/cm²状态，10个有效单体均值的稳定窗口中位数再按周期汇总。每周期一个标签，只使用141个冻结合法周期；上一阶段新增5个审查周期不进入标签。训练标签是原始V，经Train-only中心化/尺度化，不以SOH百分比训练。

从上一阶段37/13、50/6、56/22候选交集开始，重新在周期确认后的起点构造48×14历史，未移动fold边界。实际保留：

{table(counts)}

内选为最后ceil15%Train的连续时间块；innerFit剔除1个过晚标签/折。inner模型标签至少早于首selection起点1h，outerTrain标签至少早于首Validation起点1h，再检查真实计算耗时。所有184条Train/Validation记录、inner/purge键及原始动态历史均保存，EXCLUDED_SAMPLES.csv为空而不是悄悄扩增候选。

本轮严格等待原周期完成确认：观察下一段15||14、15||0两run锚点结束。起点比旧最后P1S测量可用时刻晚0.74–0.75运行小时；到目标末次测量的实际提前量{samples.forecast_actual_measurement_end_lead_h.min():.5f}–{samples.forecast_actual_measurement_end_lead_h.max():.5f}h，不再冒称原9.52–9.89h，也不是下一小时。

这时目标raw(k+1)的初始锚点已发生，但全部目标固定状态电压观测尚未发生。因此是“周期确认后的H1固定状态事件预测”，不宣称整个目标周期开始前就已预测。确认锚点是起点已知信息，实际future lead/目标时间/未来温湿压不进入模型。

N1读取先前3007条元数据按1819.997738h筛选，再只解析被证明保留的3005条Development特征；不读排除行Test特征，不整表读取/哈希。不重新解析高频原始文件。历史权重、Ridge/M2/参考SOH不改变。
''')
 md('02_双路径TFT事件输出实现.md',f'''# 单事件查询，保留完整TFT处理流程

新模型位于models/event_tft.py，继承只读的CanonicalStrictHistoryTFT、DualPathStrictHistoryTFT、RegimeAwareDualPathTFT。保留VSN、native encoder/decoder LSTM、因果多尺度CNN、5→8→1历史工况门控、静态增强、解释性注意力、position-wise GRN及pre_output_gate_norm。

输入仍[B,48,14]。历史相对位置仍−47..0除以24，编码半小时历史。新的decoder只有一个全零事件查询，14个物理占位量及坐标均为0，不是0.5h提前量，也不是未来真实时间/电流。H1目标类型由这一个固定任务确定。

事件表示取native pre_output_gate_norm后的[B,1,64]状态，经原生同形状Linear64→1读出，输出[B,1]标准化绝对V。没有平均原24步预测、没有24半小时位置改名为周期、没有插值层、没有用普通LSTM/MLP替换TFT。模型类仍使用完整native forward。

H-F0/H-F1/H-F2可训练参数分别323483/332379/332436，输出层均65参数。相同种子共享native模块与读出头初始化严格一致，F1/F2共享CNN初始化一致。F1固定1:1，F2仅将其换为原历史工况门控。

历史动态任务Top10曾用较晚数据选出，不能回填早期fold。新任务保留同一Spearman-at-origin绝对相关Top10规则及强制meanV，但在innerFit或全Train相应集合内重新计算，按索引稳定处理并列；同fold同阶段三模型使用相同mask/scaler。EWMA α=0.3仅作用历史meanV。输入scaler拟合未滤波训练历史，与原处理合同一致。

新目标标准化用对应训练标签的V均值/标准差；原动态ΔV尺度不适用这一绝对V任务，理由在训练前登记。没有新Optuna、核/门控/层数/目标搜索。

预训练工程验证覆盖全部native模块真实调用、非零梯度、CNN前缀因果性、[B,1]形状、共享初始化、合成AdamW一步以及GPU保存重载精确一致。旧checkpoint从未load用于初始化，只做字节保护。

冻结协议SHA为{sha(P/'configs/FROZEN_PROTOCOL.json')}，工程及协议先提交推送 `{oldcommit}`，随后才开始真实训练。原协议、训练关键代码SHA与数据SHA仍一致。
''')
 md('03_HF0_HF1_HF2消融结果.md',f'''# 事件级消融：H-F2未取得三折平均优势

主指标是三个fold等权MAE，fold内先平均三个种子；不按验证点数量加权，不选择最佳seed，不集成预测。RMSE/P95等同样报告fold/seed分项与均值，不能把均值最大误差当全实验最大值。

{table(eq.loc[eq.model.str.startswith('H-'),['model','MAE_mV','RMSE_mV','MAE_SOH_pp','RMSE_SOH_pp']])}

逐折三种子均值：

{table(fold.loc[fold.model.str.startswith('H-'),['fold','model','MAE_mV','RMSE_mV','P95_mV','MAX_mV']])}

H-F2在第2/3折比F0/F1略低，但第一折MAE1.979903mV，比F0的1.874015及F1的1.654022更差，导致等权总体没有优势。不能只报告两个获益fold。

种子等权三折MAE的样本SD：F0约0.166216、F1约0.100659、F2约0.309451mV；F2更不稳定。所有27次结果如下，P95/max均是该run实际尾部：

{table(runs.loc[runs.seed.ge(0),['fold','model','seed','n','MAE_mV','RMSE_mV','P95_mV','MAX_mV']])}

所有inner采用冻结AdamW lr0.0009605449687795866、weight_decay6.95942232671908e−6、batch32、max450/patience40、梯度clip1、原hidden64/continuous32/dropout0.1；真实选中epoch范围{int(tr.selected_epoch.min())}–{int(tr.selected_epoch.max())}。内选共{int(tr.inner_optimizer_epochs.sum())}epoch，完整Train重拟合共{int(tr.final_refit_epochs.sum())}epoch。外层Validation从未用于选择epoch。

真实学习门控LSTM权重大致0.467–0.510，见REAL_F2_GATE_VALUES.csv；确有可训练门控和有效梯度，但本任务没有观察到其稳定性能增量。这不是所有任务上门控无效的证明。
''')
 md('04_与Persistence和Ridge96对照.md',f'''# 简单保持显著优于本轮健康TFT

所有模型使用完全相同的41个H1验证目标（13/6/22）。Persistence取已完成周期的最新固定状态代表电压。新阶段Ridge96在相同Train键上独立重拟合，固定五项96h历史特征、alpha[0.0001,0.01,1,100]和Train内部时间选择；原冻结Ridge未改动。

{table(controls[['model','MAE_mV','RMSE_mV','MAE_SOH_pp','RMSE_SOH_pp']])}

H-F2的MAE是Persistence的{d['F2_MAE_mV']/d['Persistence_MAE_mV']:.4f}倍，改善率{d['F2_improvement_vs_Persistence_percent']:.4f}%，也就是明显恶化，而不是低于5%门槛的一点点收益。三折均未赢Persistence，三个种子的等权结果也均未赢。

同任务Ridge96等权MAE比Persistence低{gainridge:.4f}%；它说明轻量周期趋势在这组样本上有实际价值。但Ridge用96h固定状态历史，TFT用24h动态统计，不能把二者误差差异全部归因于网络结构。结构贡献只由F0/F1/F2的共同输入消融判断。

原动态F2的7.2454mV是另一个负载波动更大的目标，不能与本轮1.003mV直接排名。本轮原始电压MAE约0.124mV的保持基线，反映了目标本身非常稳定；不能忽略它而只称神经网络误差很小。

全部新Ridge系数、scaler、alpha网格和训练键在models/RIDGE96_CONTROL_PARAMETERS.json，15次CPU拟合包括每fold4个inner alpha和1个final拟合。独立闭式解复算预测差为{v['closed_form_Ridge_max_abs_difference_V']:.3g}V。历史冻结成绩不作为本轮成绩替身。
''')
 md('05_电压型SOH预测结果.md',f'''# SOH_V代理可计算，但没有预测收益

只在预测电压之后转换：SOH_V=100×V/0.71383023635。真实对照使用同一冻结周期电压定义，模型不以百分比训练。这个量是固定状态电压型SOH代理，不是j1.0极化标准SOH。

{table(eq[['model','MAE_SOH_pp','RMSE_SOH_pp','P95_SOH_pp']])}

H-F2三折等权MAE{ix.loc['H-F2','MAE_SOH_pp']:.6f}百分点、RMSE{ix.loc['H-F2','RMSE_SOH_pp']:.6f}百分点；Persistence为{ix.loc['Persistence','MAE_SOH_pp']:.6f}和{ix.loc['Persistence','RMSE_SOH_pp']:.6f}，Ridge为{ix.loc['Ridge96_control','MAE_SOH_pp']:.6f}和{ix.loc['Ridge96_control','RMSE_SOH_pp']:.6f}。百分比变换不会凭空改变模型排名。

原参考的前10次测量到171.82091h可用；再考虑完整周期确认，真正确认到{ref['reference_first10_complete_cycle_confirmation_h']:.5f}h，日历{ref['reference_first10_complete_cycle_confirmation_calendar']}。全部本轮预测起点均在其后，允许因果地使用该冻结参照；仍不自动认证为BoL。

不裁剪100%以上预测，不强制单调，不改恢复标签。H-F2的全实验真实最大绝对误差为{global_.set_index('model').loc['H-F2','MAX_mV']:.6f}mV，折算{global_.set_index('model').loc['H-F2','MAX_SOH_pp']:.6f}百分点；这比平均各run的最大值更能显示尾部。全局P95和有偏性见POOLED_GLOBAL_TAIL_AND_BIAS.csv，属于依赖查询的描述统计，不替代等权主指标。

变化方向及滞后诊断见LAG_AND_CHANGE_DIAGNOSTICS.csv和F5。第一折H-F2各种子预测变化绝对值中位约1.43–2.65mV，而真实相邻变化中位仅0.145mV，方向一致率约15–23%；主要表现是偏差/变化幅度不可靠，不能只解释成固定时间滞后。没有对不规则周期序列拟合一个虚构的滞后小时数。
''')
 md('06_参考极化SOH与物理一致性分析.md',f'''# 代理预测与标准SOH资格分开

SOH_ref=V_pol(j1.0)/0.6989533305168152定义不变，复用7个Development事件，不读取2000/2300h Test值。固定0.78923状态与j1.0极化电压不是同一观测，不直接相减作误差，也不将动态电压除以极化分母。

沿用已有精确H1事件键，1150/1500/1750名义事件有{ref['exact_existing_retrospective_H1_links']}个关联目标周期。原M2_NC过去仿射系数及过去参考事件列表已核查，但参考结果发布时间未认证，排程属于回顾性，测量算子/工况不同；这不是三个严格在线标准SOH验证事件。

因此标准参考SOH合格配对为0，状态REFERENCE_SOH_COMPARISON_NOT_QUALIFIED，新标准SOH MAE/RMSE为NA。本轮未新拟合校准，也未为制造成绩计算这些不合格配对的标准SOH误差。已有M2正负结论均保留。

真实目标电流、温度、湿度、压力见VALIDATION_TARGET_MEASUREMENT_CONDITIONS.csv。训练范围之外预测、非正电压、超过100%代理及恢复组误差见RANGE_AND_RECOVERY_CHECKS.csv、RECOVERY_STAGE_ERRORS.csv。超出训练电压范围不自动判为违反电化学；保持原始输出，不事后裁剪修正成绩。

固定电流仍存在热/湿状态、负载前史和可逆恢复影响，不能把周期电压下降直接等同不可逆退化。没有PINN、弱物理MLP、新物理损失或RUL训练，也没有宣称凭代理预测就解决真实EOL验证。
''')
 md('07_论文级实验结论.md',f'''# 正式判定：{d['status']}

这轮已经实际完成指定训练，并非再次停在资格审查。工程、时间和样本门槛通过，模型能够输出下一周期固定状态电压，但没有实际预测优势。

冻结成功门槛是三折等权MAE相对Persistence改善≥5%、F2优于F0/F1整体、至少两fold优于各对照、至少两seed优于Persistence且尾部无>10%恶化。未看完结果改门槛。实际F2对Persistence恶化708.93%，尾部门槛失败；判为NO_GAIN。

F2第2/3折有相对结构小优势，但第一折更差，种子SD也更大。不能用两个fold或某一seed冒充三折稳健结构贡献。标准参考SOH仍没有合格验证成绩。

建议不将这条健康TFT升级为论文主要正面性能实验。可以保留为完整、可重复的任务迁移负结果或消融/局限证据。原动态TFT仍为主模型，固定状态Ridge96作为辅助参考；不因负结果增加层数、搜索新目标或继续叠加物理损失。

当前也不建议在这一无收益的H1分支上继续训练SOH物理网络。需要先有独立可观测、起点已知、可识别且具有有效梯度的物理约束及同信息对照，不能靠物理名义掩盖未解决的预测价值问题。本轮到此停止，不自动扩展H2/H3/其他骨干或RUL。

独立复算：184个样本的身份/时间、6份Train-only处理器、27个最终checkpoint、15次Ridge闭式验证、451条预测、33组指标及50位Decimal代理公式。27个checkpoint同GPU重放均bit-exact。CPU/GPU最大差{v['crossdevice_max_abs_difference_V']:.12g}V，2个CPU精度探针超过原2e−7V门槛；保留失败，不放宽门槛，以同设备精确重放确认复现。数据身份检查完全精确，不能用跨设备容差掩盖身份问题。

真实训练在2026年运行，模型参数历史可用性通过asof回放预算检查；不声称2025年已在线部署。实际单组完整训练{tr.total_fit_seconds.min():.3f}–{tr.total_fit_seconds.max():.3f}s，远小于原标签到首验证起点的预留日历间隔。输出时间及完整证明见每run COMPLETION.json。
''')
 md('08_通俗中文总结.md',f'''# 已训练完成：这轮健康TFT没有胜过简单方法

实际27组实验、54次GPU拟合（27次内选epoch+27次全Train重新初始化拟合），另15次CPU Ridge拟合；三个fold验证分别13/6/22个真实周期。

{table(eq[['model','MAE_mV','RMSE_mV','MAE_SOH_pp','RMSE_SOH_pp']])}

1. 能预测下一raw周期的固定状态电压，但输入仍是24h动态统计，当前没有优于保持法的价值证据。
2. F2总体不优于F0/F1；第二三折的小优势被第一折恶化抵消，种子波动更大。
3. F2的MAE约为Persistence的8.09倍，不能升级为健康预测主要正面实验。
4. 新阶段同任务Ridge96约0.110807mV，比保持法改善约{gainridge:.2f}%；其96h信息与TFT24h不同，不能直接用它证明网络结构优劣。
5. F2电压型SOH代理MAE约0.140506百分点、RMSE约0.151556百分点；保持法约0.017369/0.020920，Ridge约0.015523/0.018866百分点。
6. 预测变化及恢复阶段误差均保留，没有强制单调。第一折预测幅度明显偏大，尚不能可靠反映健康变化。
7. 该代理不是认证BoL/极化SOH，标准参考SOH合格配对仍为0。
8. 推荐保留本轮完整负结果，主线继续采用原动态TFT＋周期Ridge辅助；不自动追加物理网络、超参数搜索或RUL。

起点严格等待周期确认，目标周期初始锚点此时已发生，但目标固定状态测量全部在未来。新提前量约8.76–9.13运行小时，不是原测量起点的9.52–9.89h，也不是下一小时。

图表见figures/F1_H1_ablation_and_baselines.png、F2_real_voltage_and_TFT_predictions.png、F3_voltage_SOH_proxy.png、F4_seed_and_fold_stability.png、F5_change_tracking_and_controls.png。所有图来自真实逐点CSV。
''')
 md('README.md',f'''# H1健康导向事件级双路径TFT：完成

正式判定：{d['status']}。先读08_通俗中文总结.md。27个登记实验全部完成，54个真实GPU拟合pass（内选epoch及独立全Train重拟合），15次CPU Ridge拟合；不再重训已完成runID。

固定141个原P1S middle50标签，目标rawID+1。完整周期确认后预测单事件原始V，输入48×14，保留原native TFT/CNN/门控/注意力，单个全零事件查询输出[B,1]。原架构、checkpoint、Ridge和M2只读，无Test数值访问或高频原始重读。

预训练协议及工程验证已先提交{oldcommit}；configs/FROZEN_PROTOCOL.json及其SHA不变。正式权重在outputs/checkpoints，各run日志/选择依据在outputs/training_logs。ignored outputs/runtime保存本机恢复状态，不是新增独立实验；正式inner/final权重均交付。

复核（本Phase目录，Python3.11 + torch2.6.0+cu124/pytorch-forecasting1.8.0/numpy/pandas/scipy/sklearn/matplotlib/PyYAML）：

```powershell
$env:PYTHONUTF8='1'
D:\\ruanjian\\python3.11.9\\python.exe -B -m unittest discover -s tests -v
D:\\ruanjian\\python3.11.9\\python.exe -B scripts/verify_independent.py
```

独立程序不优化模型；重放Train-only处理、闭式Ridge、27权重及全部指标。CPU跨设备精度失败保留，以同GPU严格逐位重放进一步核验；需要可用CUDA设备，不把CPU误差阈值放宽掩盖问题。先验证，再按scripts/render_figures.py从CSV重画。readiness是历史asof仿真，不是声称2025年真实部署。

一次性流程prepare_data→engineering→freeze_protocol→预训练Git提交→train→baselines→analyze_results→reference_qualification→verify_independent→result_diagnostics→render_figures→write_reports。现有完成标志阻止重复拟合；不要在原目录删除标志重跑。
''')
 print('Eight reports generated from real results; negative result retained')
if __name__=='__main__':main()
