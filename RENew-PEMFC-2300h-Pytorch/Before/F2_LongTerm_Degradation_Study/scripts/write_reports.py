"""Generate the five requested reports from verified real outputs, no invented scores."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from audit_data import HERE,csv,dump,sha

def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def mdtable(frame):
    columns=list(frame.columns)
    def fmt(v):
        if isinstance(v,(float,np.floating)):return f'{v:.6f}'
        return str(v).replace('|','/')
    return '| '+' | '.join(columns)+' |\n|'+'|'.join(['---']*len(columns))+'|\n'+'\n'.join('| '+' | '.join(fmt(v) for v in row)+' |' for row in frame.itertuples(index=False,name=None))+'\n'

def run():
    a=HERE/'outputs/qualification_20261020_v2';p=Path(read(HERE/'outputs/ACTIVE_PILOT.json')['path'])
    verify=read(p/'INDEPENDENT_VERIFICATION.json');assert verify['PASS']
    decision=read(p/'PILOT_DECISION.json');assert not decision['expand_to_six_seeds']
    counts=pd.read_csv(a/'WINDOW_COUNTS.csv');summary=read(a/'P0_SUMMARY.json');target=read(a/'TARGET_SUMMARY.json')
    diagnostics=pd.read_csv(a/'TARGET_DIAGNOSTICS.csv');metrics=pd.read_csv(HERE/'tables/ALL_PILOT_METRICS.csv')
    vm=metrics[metrics.partition=='VALIDATION'].copy();models=vm[vm.model!='Persistence']
    files=sorted(p.glob('*/COMPLETED.json'));completed=[read(f) for f in files]
    reports=HERE/'reports';reports.mkdir(exist_ok=True)
    hist=len(read(a/'HISTORY_BEFORE.json'));missing=len(read(a/'PREEXISTING_MISSING_TRACKED_FILES.json'))
    c=counts[counts.history_steps==48].pivot(index='horizon_h',columns='partition',values='legal_origins').reset_index()
    c=c.rename(columns={'horizon_h':'跨度h','TRAIN':'Train起点','VALIDATION':'Validation起点','HOLDOUT':'后留起点_仅元数据'})
    doc=f'''# 00 长期预测可行性与数据连续性审计

目前支持24/48小时的探索性代理预测试运行，不支持在原7:1:2划分下开展有充分验证样本的96/168/336小时正式对照。没有通过桥接断点、插值、借后留标签或移动边界制造样本。

源文件为 Phase_N1_Data_Foundation/outputs/causal_feature_tables/causal_features_30min.csv，实际3843行，零基索引不变。半小时记录结束运行小时73.0–2257.0，3818行有效，25行无效；22个原始 temporal_segment_id，细化后18个有效连续片段。原始分段逐条见 SEGMENT_SUMMARY.csv；实际连续片段见 CONTINUOUS_VALID_PIECES.csv。最长片段380行，首末观测相隔189.5h。48步历史覆盖23.5h，最多剩166h可向后预测，因此168h即便改变分区也无法在当前连续片段中形成完整任务。缺失不能凭行号绕过。

固定边界：Train[0,2690)，73.0–1622.0h；Validation[2690,3074)，1622.5–1854.0h；后留[3074,3843)，1854.5–2257.0h。后留只读取时钟、有效性、分段、覆盖和启停标志等元数据以计数，没有读取其电压/工况数值建立目标、选跨度或评价模型。

相邻记录有7处运行时钟非0.5h，9处日历邻接不合法；最大相邻运行小时差94.5h。日历时间和运行时钟不是同一种时间，不能把运行小时缺失量误写为停机的日历时长。有效性、原始分段、运行钟与日历邻接独立检查；每一候选起点全部剔除原因保存在 L*_H*_*_EXCLUDED.csv。

## 48步历史下的真实样本数量

{mdtable(c)}
24h Validation为170起点、8160查询、264个唯一目标时间、5条互不重叠的目标轨迹；48h为98起点、9408查询、193个唯一目标时间、2条互不重叠目标轨迹，且全部来自同一个连续片段。这不是98条独立寿命轨迹。96h的2起点共享193个目标时间，只构成1条不重叠轨迹。

在系统统计和目标分析之前冻结了工程资格：Train至少100起点、Validation至少20起点，且分别至少3和2条互不重叠目标轨迹。这不是统计功效证明。先前为确认字段做过CSV表头和第0行预览，该行是无效Train记录；配置identity中“before source outcome inspection”应严格限定为“系统数值分析之前”，不能泛称此前一个数值都未看过。原注册不改写，补充说明保存为PREREGISTRATION_SCOPE_CLARIFICATION.json。没有提前查看Validation或后留指标来决定门槛。资格不使用后留成绩或数量选取跨度。48/96/192步历史全部核对，45个配置的合法窗口和唯一目标统计均已独立复算。192步历史的48h Validation虽有36起点，却只有1条不重叠目标轨迹，同样不合格。正式试运行固定48步历史，以保留更多同信息样本，没有按训练结果搜索历史长度。

## 不可逆退化和寿命资格

本表证明时间连续性，不证明不可逆损伤。运行状态、恢复与长期趋势还需分开。原始RealHyFC论文方法页4明确是达到研究目标后结束并全面表征；既有页8的特定工况500mV限值线索不构成本任务真实EOL。原文位置、SHA及边界见 literature/DESIGN_TRANSFER.md 和 SOURCE_INDEX.json。

## 替代协议建议（未执行）

若以后坚持96h，需另行批准更长的Validation区，使其中至少包含两个可以承载24h历史+96h未来的真实连续片段，同时保留更早Train与标签隔离；应仅按时钟和覆盖冻结边界。168h/336h不能靠移动7:1:2边界解决，需真实连续观测支持或另行论证不同的非等间隔物理测量任务。本轮没有更换协议，也没有启动其他Phase。
'''
    (reports/'00_长期预测可行性与数据连续性审计.md').write_text(doc,encoding='utf-8')
    doc=f'''# 01 长期退化指标定义与有效性

单纯平滑电压未通过资格；Train-only工况校正后的因果趋势通过数值筛查，可以试运行，但仍是探索性统计性能代理，不能认定为真实SOH或已分离的不可逆退化。

## 两个预先固定的定义

A_i = (1/48) Σ_(k=0..47) V_(i-k)，单位V/单电池。仅当48个样本全部有效、时钟连续且没有跨段时输出；不足48行记NA。只用过去，不用居中/双向/SG/全序列分解，也不跨断点。

B_raw,i = V_i − (z_i − z_ref)^T β，B_i = trailing_mean_48(B_raw)。z为电流密度均值/标准差、零电流占比、冷却液出口温度、阴极入口RH及压力。β是2673个有效Train观测拟合的带截距OLS的斜率，先按Train均值/标准差数值标准化；z_ref取Train各变量中位数，完整系数与单位见 B_OLS_PARAMETERS.csv。

z_ref只确定残差的加性坐标原点，不是BoL，也不声称这些中位数构成实际同时测得的恒定负载状态。所预测ΔB中该固定原点完全抵消。B不是极化曲线反演电压，不是物理方程，不用于除以BoL得到SOH。遗漏工况、恢复与统计模型误差仍可保留在残差里。

要求协变量在Train单变量范围内且有限，避免外推；这是必要条件，不能代替联合同工况证明。额外以Train留一近邻距离P95检查联合支持，Validation 355/381个有效观测通过（93.18%）。没有依据Validation误差调整阈值。OLS矩阵满秩7/7，条件数64.407588，不代表物理参数可辨识。校正模型不含未来健康锚点。

## 真实诊断结果

{mdtable(diagnostics[['target','partition','target_rows','label_fraction','p95_minus_p5_mV','step_rms_mV','rho_trailing_zero','rho_trailing_current']])}
Validation中，A与过去48bin零电流比例的Spearman相关为−0.855904，超过预先设定的绝对0.3门槛，说明平均电压仍受运行状态混合显著影响。B最大绝对负载相关为0.258241，通过该筛查；Train连续标签2062个、Validation266个。B的Validation逐步RMS为0.033544mV，而同位置原始电压为217.228443mV，说明平滑和工况校正确实大幅改变了任务难度，不能拿其低RMSE同旧原始动态电压比赛。

B的Validation P95−P5仅1.284312mV；波动小本身不证明健康信息丰富。必须同时比较保持法、终点误差、ΔB的R²，不能仅凭接近1的绝对电压R²宣称成功。B有效连续片段中约44.3%的Validation相邻变化为正，没有强制单调。两个可分析Validation片段拟合斜率分别−0.003778及−0.028600mV/h，是描述性局部性能变化，不能归因于不可逆损伤。

24h目标的前47个提前量因尾随窗口会部分含起点之前已经知道的样本，可能使短提前量更容易；24h终点的48个标签成分均位于起点后，48h终点代表未来24–48h的观测均值。按终点另算误差，避免用平滑重叠夸大长期能力。未来实际工况只能在观测到达后用冻结Train校正公式生成评价标签，绝不进入预测器。

结论：A不合格；B仅取得统计性能趋势试运行资格，没有取得标准SOH、同工况物理健康或不可逆退化验证资格。源数据只读前3074行生成目标；后留标签未构造。独立伪逆+累积和重算与原程序差异上限2e−12V，NA身份严格相同，数值精度差异不掩盖时间错位。
'''
    (reports/'01_长期退化指标定义与有效性.md').write_text(doc,encoding='utf-8')
    doc='''# 02 长期预测正式协议

完整冻结配置：configs/FROZEN_TRAINING_PROTOCOL.json。它在任何新训练开始前写入；每个新运行保存配置、代码、输入、标签、起点、目标、初始化和checkpoint身份。元数据资格先于目标数值检查，目标资格先于训练。

采用24h短对照、48h主探索跨度；96/168/336h不进入训练。固定48步历史，14槽位原始Top10掩码、历史因果EWMA alpha=.3；Train合法历史拟合scaler；预测目标为ΔB=B(t+h)−B(t)。目标标准化仅用本跨度Train的ΔB标准差，不减均值。两模型共享完全相同的输入张量、标签、窗口键、随机批次顺序和MSE公式，未来实测协变量不作为输入。

F2保留原生TFT、VSN/GRN、LSTM编码/解码、Attention、因果多尺度CNN、工况门控与凸组合，未采用上轮残差融合/学习率候选。只把build_input及forward_history的decoder长度、相对未来坐标及输出形状由24改为48/96，参数量仍332436。6个原模型类除这两处接口方法之外AST逐类严格一致；不把整个LSTM作用等同于融合g。

独立普通LSTM为14→64单层LSTM、最后隐藏状态线性输出48/96步，参数量分别23600/26720；单层内部Dropout不启用，未伪称两者正则化结构/容量完全相同。两模型不同容量是公平比较的限制，不把整体差异归结为CNN或门控的单独贡献。

共享AdamW lr=.0009605449687795866，weight_decay=6.95942232671908e−6，batch32，max450，patience40，梯度裁剪1，MSE；F2 dropout=.1。Validation标准化MSE改进至少1e−12保存best，连续40epoch未改进停止。从头初始化模型及空optimizer；没有加载历史权重续训。

首先Seed202×两模型×两跨度=4次pilot，预算最多1800epoch。预先要求：Validation查询RMSE和MAE均至少优于零ΔB保持法5%，ΔB R²>0，且固定终点RMSE不比保持法差。只有主48h两个模型均合格才扩大到六种子；该保守门槛在训练前冻结，不按48h成绩重写。若通过，原计划独立24次正式训练（6种子×2模型×2跨度），pilot不混为正式六种子成绩。本轮未触发扩展。

普通CUDA配置保留已验证设置：cudnn deterministic=True、benchmark=False，matmulTF32=False、cudnnTF32=True，strict algorithms=False。原生TFT插值反向的严格确定性限制仍在；设置种子不能保证逐位重复。没有反复重跑较好种子。运行身份保存真实每epoch权重SHA、Loss、梯度范数、优化器步数、最佳epoch。

主指标是ΔB的float64查询加权RMSE/MAE/R²，单位mV；另外保存绝对B的R²、唯一目标时间等权RMSE（先对同一目标所有查询的平方误差求均值，再对目标时间等权平均开根号，非先集成预测）、固定末步RMSE、P95/P99/max。所有重叠和极端查询保留。

后留区已经历史查看，不能称独立盲测。本轮pilot和选择未读取后留电压/工况目标数值；只有全部合格正式模型冻结后原计划进行一次描述性后留评价。因主跨度pilot失败，本轮不构造后留标签、不预测后留、不输出后留成绩。
'''
    (reports/'02_长期预测正式协议.md').write_text(doc,encoding='utf-8')
    resulttable=vm[['horizon_h','model','RMSE_mV','MAE_mV','R2_delta','terminal_RMSE_mV','target_time_equal_RMSE_mV','P99_mV','max_abs_mV']].sort_values(['horizon_h','model'])
    runstable=pd.DataFrame([dict(model=r['model'],跨度h=r['horizon_steps']*.5,参数量=r['parameters'],最佳epoch=r['best_epoch'],停止epoch=r['stop_epoch'],优化器更新=r['optimizer_steps'],训练及保存秒=r['wall_training_s']) for r in completed])
    gates=pd.read_csv(p/'PILOT_QUALIFICATION.csv')
    gate_stats=[]
    for h in [24,48]:
        g=pd.read_csv(p/f'F2_seed202_{h}h/VALIDATION_GATE.csv')
        gate_stats.append(dict(跨度h=h,n=len(g),g均值=g.g.mean(),g标准差=g.g.std(ddof=0),g最小=g.g.min(),g最大=g.g.max()))
    doc=f'''# 03 F2与LSTM长期预测对照结果

实际完成4次新GPU训练，283个epoch、10543次真实反向传播及AdamW更新；无GPU训练失败，没有使用旧权重或缓存代替。48h主跨度两个模型均未通过预先冻结的扩展规则，停止六种子及后留评价。以下均为Seed202的探索性pilot，不能表述为六种子平均/稳定优势。

## Validation真实结果

单位为工况校正趋势代理的mV，R²对应ΔB。保持法为B_pred(t+h)=B(t)。

{mdtable(resulttable)}
24h F2较保持法RMSE降低36.33%，LSTM降低43.21%；F2比LSTM RMSE高12.10%。48h F2比保持法RMSE高41.15%，LSTM比保持法高0.45%；两者ΔB R²均为负，固定末步误差也比保持法更大。F2比LSTM约高40.52%。不能凭24h的正面结果替48h主任务取得资格，也没有缩短主任务后追跑六种子。

{mdtable(runstable)}

Train与Validation、逐提前量、尾部、原始逐点CSV全部保留。48h F2 Train RMSE .178084mV、Validation .290407mV；LSTM分别.179752和.206661mV。F2训练Loss持续下降、Validation最佳在12epoch后恶化，支持“当前设置对本代理长跨度的泛化不足”这一解释，不足以单独证明某个融合部件有因果缺陷。F2主48h末步 .466142mV，LSTM .380866mV，保持法 .280296mV。F7同时展示首个共同起点和两模型平均SSE最差起点，未挑选有利窗口。

Validation局部B趋势比若干Train片段更平缓；48h固定起点图可见模型在部分窗口预测下降幅度过大。然而目标是统计残差平滑值，只有一个主跨度连续验证片段，不能把这种偏差直接解释为真实损伤预测偏差。按时间和Train-P75工况阈值的诊断见 DESCRIPTIVE_STATE_AGE_ERRORS.csv；极小分组的1–2条查询明确保留样本数，不能据此作可靠的恢复阶段结论。

## 门控真实行为

{mdtable(pd.DataFrame(gate_stats))}
门控数值确有小幅变化（图F10统一放大到.47–.49显示，门控完整定义域仍为0–1）。独立从种子重建初始化状态与运行初始SHA完全一致，并证实CNN、门控、LSTM编码/解码和其余TFT参数组均发生了真实参数更新，见 INDEPENDENT_PARAMETER_UPDATES.csv。这是参数参与证据，不是独立收益消融。本轮没有门控关闭/重新训练消融，不能由g相关性证明门控独立改善或加重误差。本任务不叠加融合改造。

## 核验与停止

45种窗口配置独立重算一致；A/B由另一种伪逆及前缀和实现复算，NA位置严格相同，浮点最大差异见 INDEPENDENT_VERIFICATION.json；4个best checkpoint在Train与Validation全部逐点推理回放严格相等，无粗略容差。梯度有限且非零、参数确有变化、空optimizer启动、每轮Early Stopping状态与最佳epoch均复核。

六种子均值/标准差、96h/168h/336h训练成绩、后留预测和RUL误差全部NA（未执行），原因已明确。没有虚构六种子图或未来轨迹。12组PNG/PDF由真实CSV和scripts/render_figures.py生成，主要图索引见README。
'''
    (reports/'03_F2与LSTM长期预测对照结果.md').write_text(doc,encoding='utf-8')
    doc=f'''# 04 长期预测最终研究总结

## 通俗结论

这份数据能做一定跨度的性能趋势探索，但当前固定时间划分不足以支持可靠的96小时及更长退化预测。我们真实训练了24小时和48小时两种任务，完成F2与普通LSTM各两次Seed202训练。24小时有可预测信息；48小时没有达到预先设定的有效学习门槛，因此及时停止，没有用更多种子碰运气。

1. **数据能支持多久？** 3843条记录并不等于一段不中断的1921.5小时。最长有效连续片段仅189.5h，保留24h输入后最多166h未来。96h验证仅2起点，168/336h零起点。24/48h可试运行，但主48h验证只来自1个连续片段，并非长期健康验证已充分。
2. **实际预测了多少小时？** 从固定起点直接输出未来24h和48h，期间未重新输入未来实测值。没有完成96/168h训练。48h试运行失败也是本轮完整结果，不以必须取得正面收益为完成条件。
3. **与旧12h任务有什么不同？** 旧任务预测原始动态电压ΔV；本轮预测Train工况校正后、24h因果均值代理的ΔB。其波动范围、平滑、标签和跨度不同。本轮0.1–0.3mV误差不能宣称比旧几十mV模型提升百倍，更不能直接改名SOH精度。
4. **F2超过LSTM了吗？** 没有。Validation 24h：F2 .131186mV、LSTM .117023mV；48h：F2 .290407mV、LSTM .206661mV。F2分别高约12.10%和40.52%。这只是一个预先固定种子的pilot，无六种子稳定性结论。
5. **主要问题是什么？** 原始负载混合影响显著，平滑本身不能变成健康标签；统计校正后的目标虽降低了负载相关性，仍不证明不可逆损伤。连续验证覆盖不足、目标动态范围很小、48h预测偏差和普通CUDA数值波动限制了可靠结论。F2训练好转而Validation变差，是泛化风险信号，不能仅由此认定CNN或门控失效。
6. **现在能做SOH/RUL吗？** 该统计参考并非BoL、B不是已验证SOH，也没有找到与当前目标一致的真实EOL时刻及触发记录。原文记录研究性终止，不可把最后采样当失效。RUL不训练、不报精度；最多保留长期性能趋势可行性研究与可能右删失的资料缺口。
7. **哪些能用于论文？** 可如实使用连续性审计、因果/Train-only目标方法、两模型同窗口pilot、完整负面长跨度结果及科学边界，标为探索性单电池证据。不能写成F2长期退化优于LSTM、六种子稳定提升、不可逆损伤识别、验证SOH/RUL或跨电池泛化。

## 本轮停止位置与下一步

P0完成，24/48h合格、96h及以后不合格；P1完成，A否决、B仅为统计性能代理；P2冻结；P3完成4次真实GPUpilot；P4完成实际结果复算及12组图；主48h门槛失败，六种子和后留评价未执行；P5真实RUL资格不足，未训练。

下一步最小证据缺口是**可比运行状态下、跨真实连续验证片段的健康观测与更长验证支持**，而非再搜索骨干。可先只读核实这些统计残差与已许可固定状态真实观测的时间身份和工况关系；若不能证明跨工况的稳定健康含义，不扩大长期神经训练。96h若需改变时间分区，先独立审批并仅按元数据冻结；168h在当前分箱连续性约束下不能硬做。本轮不自动执行这些建议。

## 保存与审计

所有新增内容仅在 F2_LongTerm_Degradation_Study。历史{hist}个现存追踪文件建立前后SHA保护；另外{missing}个原先缺失的依赖/缓存文件单独登记，不恢复、不删除、不掩盖。首轮审计在保护快照碰到一个既有缺失DLL，未读数据即失败，证据保留；工程修复只改本轮快照程序为显式登记缺失。PDF读取环境缺少pypdf，改用已有PyMuPDF只读核对，无模型改变。交互验收首轮因最终报告尚未生成失败，原始日志保留；完整报告生成后复验。4次GPU训练无工程失败。

完整数据结果、代码、checkpoint、训练日志、SHA及独立复算均保留。交互入口 F2_LongTerm_Interactive.py 默认只读，防止重复已完成的一次性研究；切为MODE='pilot'才会新建目录并从头训练，不使用旧权重。真实IPython逐单元图像显示通道另有验收记录，不把它冒称已经在VS Code GUI中人工验收。

最终科研状态：**LONG_TERM_PILOT_NOT_QUALIFIED_FOR_EXPANSION**。这是按预先冻结停止规则完成本轮研究，非训练未执行或伪造成功。
'''
    (reports/'04_长期预测最终研究总结.md').write_text(doc,encoding='utf-8')
    (HERE/'README.md').write_text('''# RealHyFC 长期趋势研究：24/48h试运行完成，停止扩大训练

先读 [通俗总结](reports/04_长期预测最终研究总结.md)。本轮真实运行4次Seed202 GPU训练，原生F2未超过LSTM；48h均未通过扩展门槛，六种子/后留/RUL均未执行。条件校正趋势是探索性统计代理，不是SOH或不可逆退化标签。

## VS Code

打开 F2_LongTerm_Interactive.py，选择项目Python3.11/CUDA，使用每个`# %%`的Run Cell或Run All Cells。默认MODE='analyze'只读展示真实CSV和图像；改为'pilot'才在新目录从头运行4个模型，显示实际Epoch/Loss和新曲线。不得为筛选较好成绩反复运行。该入口不import有顶层训练副作用的旧脚本。

## 文件索引

- configs：数据/目标资格规则与后续训练协议，两者均先于对应数据检查/训练冻结。
- scripts/audit_data.py：3843行时钟/有效性；45个窗口配置及全部剔除记录。
- scripts/audit_targets.py：仅前3074行构建目标，Train-only OLS；无后留数值。
- scripts/native_models.py：完整原生F2与普通LSTM定义，不依赖旧项目Python模块。
- scripts/train_longterm.py --pilot：真实GPU训练、反向传播、best保存；每次新目录。
- scripts/verify_independent.py：独立时钟/标签/指标复算与checkpoint逐点回放；--history检查历史保护。
- scripts/render_figures.py：基于CSV生成全部PNG/PDF；scripts/write_reports.py生成五份报告。
- scripts/verify_interactive.py：真实IPython代码单元及PNG显示负载验收，非VS Code GUI截图。
- outputs/qualification_20261020_v1：源数据读取前的缺失历史DLL错误，不删除。
- outputs/qualification_20261020_v2：正式连续性/目标审计、注册SHA、历史保护与失败说明。
- outputs/ACTIVE_PILOT.json：本轮4次训练目录身份；其子目录保留权重、逐点预测、日志及输入/标签/窗口SHA。
- tables：各指标、逐点预测及原始绘图数据。没有六种子平均/后留成绩。
- literature：已上传原文的设计借鉴、SHA及RealHyFC研究终止元数据。

## 图像用途（300dpi PNG + PDF + CSV）

|图|用途|
|---|---|
|F1|完整记录时钟、连续有效段与冻结边界，后留仅元数据|
|F2|短中长期候选的合法起点与不重叠目标覆盖，未训练跨度不补成绩|
|F3|真实原始电压、A/B因果趋势，Train+Validation，不跨断点|
|F4|A/B与历史零电流混合的相关性，非因果识别|
|F5|两模型与保持法24/48h真实pilot指标|
|F6|固定模型每一步预测误差|
|F7|首个共同起点与最差两模型平均SSE起点，完整轨迹不拼接|
|F8|仅固定末步的误差随真实目标运行小时变化|
|F9|所有重叠查询残差，保留尾部|
|F10|本轮F2的真实历史门控，不宣称独立因果收益|
|F11|本轮四次训练Loss，不回放历史训练|
|F12|真实目标运行状态与末步误差，未来状态仅事后诊断|

## 复核命令

从本目录执行 `python -B -m unittest discover -s tests -v`、`python -B scripts/verify_independent.py`。图表复绘不训练：`python -B scripts/render_figures.py`。只在需要新的预登记研究时执行`python -B scripts/train_longterm.py --pilot`；它不会续训或覆盖任何旧权重。

本研究不同于旧12h原始动态电压任务，RMSE不能跨目标直接排名。后留已历史查看，非独立盲测；本轮没有后留推理。没有真实EOL标签，不训练真实RUL。历史文件不删除、不覆盖、不移动。
''',encoding='utf-8')
    dump(HERE/'outputs/FINAL_RESEARCH_STATUS.json',dict(status='LONG_TERM_PILOT_NOT_QUALIFIED_FOR_EXPANSION',
        trained_models=len(completed),epochs=sum(x['stop_epoch'] for x in completed),optimizer_steps=sum(x['optimizer_steps'] for x in completed),
        six_seed_training_executed=False,holdout_outcome_rows_read=0,holdout_inference_executed=False,real_RUL_training_executed=False,
        completed_horizons_h=[24,48],rejected_horizons_h=[96,168,336],results_scope='Seed202 exploratory validation pilot only',
        GPU_training_failures=0,metrics_verified=True,checkpoint_replays_exact=True))
    print('Five reports and README generated from verified actual outputs')

if __name__=='__main__':run()
