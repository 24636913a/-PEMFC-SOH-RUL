"""Report saved qualification and historical summaries; no model or real-data loader."""
import csv, hashlib, json
from pathlib import Path

PHASE = Path(__file__).resolve().parents[1]
PROJECT = PHASE.parent
P3 = PROJECT / 'Phase_P3_Full450_Epoch_Sensitivity'

def load(p):
    return json.loads(p.read_text(encoding='utf-8'))

def put(name, content):
    p = PHASE / name
    if p.exists():
        raise RuntimeError('PRESERVE_EXISTING_DELIVERABLE:' + name)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open('w', encoding='utf-8', newline='\n') as f:
        f.write(content.strip() + '\n')

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def table(rows, columns):
    answer = ['|' + '|'.join(columns) + '|', '|' + '|'.join(['---'] * len(columns)) + '|']
    for row in rows:
        answer.append('|' + '|'.join(str(row[k]) for k in columns) + '|')
    return '\n'.join(answer)

def main():
    decision = load(PHASE / 'outputs/QUALIFICATION_DECISION.json')
    activity = load(PHASE / 'outputs/TRAIN_PHYSICS_ACTIVITY.json')
    tests = load(PHASE / 'outputs/SYNTHETIC_TEST_RESULTS.json')
    assert not decision['formal_training_admitted']
    assert len(activity) == 9 and all(r['violations'] == 0 for r in activity)
    assert tests['returncode'] == 0
    history = []
    source_manifest = {}
    for seed in (42, 52, 62):
        for role in ('best', 'last450'):
            p = P3 / f'outputs/training_logs/P3-G2-Seed{seed}/{role}_result.json'
            r = load(p)
            source_manifest[str(p)] = sha(p)
            history.append({k: r[k] for k in ('run_id', 'seed', 'checkpoint_role', 'selected_epoch',
                'RMSE_mV', 'MAE_mV', 'endpoint_12h_RMSE_mV', 'P95_abs_error_mV',
                'P99_abs_error_mV', 'max_abs_error_mV', 'checkpoint', 'checkpoint_sha256',
                'validation_prediction', 'prediction_sha256')})
    with (PHASE / 'outputs/HISTORICAL_P3_G2_METRICS.csv').open('x', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(history[0]))
        writer.writeheader(); writer.writerows(history)
    p = P3 / 'outputs/P3_ALL_HORIZON_METRICS.csv'
    source_manifest[str(p)] = sha(p)
    with p.open(encoding='utf-8', newline='') as f:
        reader = csv.DictReader(f)
        cols = reader.fieldnames
        horizons = [r for r in reader if r['model'] == 'G2']
    assert len(horizons) == 144
    with (PHASE / 'outputs/HISTORICAL_P3_G2_24STEP_METRICS.csv').open('x', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=cols); w.writeheader(); w.writerows(horizons)
    # These are report inputs, separate from the pre-execution model/data manifest.
    sources = [
        'Phase_F_TFT_DualPath_RegimeFusion/README.md',
        'Phase_H_Determinism_MultiSeed_Stability/README.md',
        'Phase_P2_Final_Gate_Ablation_Training/reports/P2_FINAL_RESEARCH_DECISION.md',
        'Phase_P_Final_Innovation_and_Experiment_Planning/P_PHYSICS_LITE_CANDIDATE_REVIEW.md',
        'Phase_P_Final_Innovation_and_Experiment_Planning/P_PHYSICS_LITE_EQUATION_AND_INFORMATION_AUDIT.md',
        'Phase_PAA_P1_StateToken_PreAnchor_Alignment/reports/FINAL_P1S_PREANCHOR_AUDIT.md',
        'Phase_PAV_PAA_Anchor_Validity_Qualification/reports/FINAL_PAA_VALIDITY_QUALIFICATION.md',
        'Phase_AHC_Sparse_Anchor_Affine_Health_Calibration/reports/FINAL_AFFINE_CALIBRATION_CHARACTERIZATION.md',
        'Phase_AHC_Sparse_Anchor_Affine_Health_Calibration/outputs/B2_CHRONOLOGICAL_PREFIX_EVALUATION.csv',
        'Phase_PHI_P1S_B2_Health_Index_Consolidation/reports/P1S_B2_HEALTH_INDICATOR_CONSOLIDATION_REPORT.md',
        'Phase_CTA_P1S_B2_Causal_Target_Audit/reports/P1S_B2_CAUSAL_TARGET_AUDIT_REPORT.md',
        'Phase_CTAE_P1S_Reference_Availability_Erratum/reports/CTA_REFERENCE_AVAILABILITY_ERRATUM.md',
        'Phase_CNP1S_Condition_Normalized_P1S_Development_Study/reports/CONDITION_NORMALIZED_P1S_DEVELOPMENT_STUDY.md',
        'Phase_CMRF_H3_H6_OriginConditioned_P1S_Forecasting/reports/H3_H6_ORIGIN_CONDITIONED_FORECASTING_REPORT.md',
        'Phase_PSDM_Probabilistic_Degradation_State_Model_Feasibility/reports/PSDM_FEASIBILITY_REPORT.md',
        'Phase_SOHRUL_Evidence_and_Roadmap/01_EXISTING_HEALTH_RESEARCH_EVIDENCE.md',
        'Phase_SOHRUL_Evidence_and_Roadmap/03_SOH_AND_HI_FEASIBILITY.md',
        'Phase_SOHRUL_Evidence_and_Roadmap/04_RUL_AND_EOL_FEASIBILITY.md',
        'Phase_K_Paper_Claims_and_SOH_RUL_Feasibility/reports/K_EOL_AND_CENSORING_FEASIBILITY.md',
        'Phase_RUL90_Data_Feasibility/FINAL_RUL90_FEASIBILITY_REPORT.md',
    ]
    for name in sources:
        p = PROJECT / name; source_manifest[str(p)] = sha(p)
    put('outputs/REPORT_SOURCE_MANIFEST.json', json.dumps(source_manifest, ensure_ascii=False, indent=2))
    short_history = [{**r, 'RMSE_mV': f"{r['RMSE_mV']:.6f}", 'MAE_mV': f"{r['MAE_mV']:.6f}",
        'endpoint_12h_RMSE_mV': f"{r['endpoint_12h_RMSE_mV']:.6f}",
        'P99_abs_error_mV': f"{r['P99_abs_error_mV']:.6f}"} for r in history]
    ht = table(short_history, ['seed','checkpoint_role','selected_epoch','RMSE_mV','MAE_mV','endpoint_12h_RMSE_mV','P99_abs_error_mV'])
    at = table([{**r, 'minimum_predicted_voltage_V': f"{r['minimum_predicted_voltage_V']:.9f}",
        'maximum_predicted_voltage_V': f"{r['maximum_predicted_voltage_V']:.9f}"} for r in activity],
        ['seed','role','minimum_predicted_voltage_V','maximum_predicted_voltage_V','violations','physics_loss_dimensionless'])
    put('00_MORNING_SUMMARY_CN.md', '''
# 中断恢复后的研究交接（2026-10-14）

**损失函数实现正确，但本次物理候选没有获得正式训练资格。没有新增预测成绩，也没有重复训练。**

1. 已完成什么：继承中断前的4份脚本、1份测试和14项通过记录；恢复后冻结来源身份，完成三个种子×初始化/最佳/450轮的真实Train前向检查，并完成本目录九份交接报告及历史指标CSV整理。没有重写历史阶段、原始数据或checkpoint。
2. 软物理是否成功：可以正确计算和反向传播；在真实Train预测上，9次检查共298,080个值的非负约束惩罚均为0。因此尚未成功成为有实际作用的训练项，不能作为已证明的论文创新。
3. 有没有新成绩：只有新的资格审计结果，没有新训练RMSE/MAE。历史P3 G2 Best均值RMSE约7.501584 mV，Last约14.726629 mV；这些是旧结果。本次无法评价软物理版本是否更好。
4. SOH/97%还差什么：SOH需要仅用起点前锚点的校准与下一锚点检验；97%需要可靠轨迹预测、运行情景和区间跨越评价。目前只有稀疏参考点，缺独立寿命及精确失效记录。
5. 下一件事：先按05方案复用AHC已保存前缀预测，补齐Persistence和年龄趋势对照，核验简单SOH估计是否稳定有用；本轮仅设计，不执行拟合。

停止原因有两项：非负约束过弱且实际不激活；没有认证适用于未来未知工况的非平凡电压上下界。没有将1.23 V或0.5 V直接设为全局边界，没有事后缩窄范围或追加权重搜索。

优先阅读[正式研究决定](07_FINAL_RESEARCH_DECISION.md)、[SOH方案](05_SOH_NEXT_EXPERIMENT.md)、[97%方案](06_RUL97_THRESHOLD_PLAN.md)。执行证据见[资格结果](outputs/QUALIFICATION_DECISION.json)和[Git审计](08_EXECUTION_AND_GIT_AUDIT.md)。
''')
    put('01_SOFT_PHYSICS_METHOD.md', '''
# 软物理方法与适用条件

研究模型仍为**TFT动态双路径模型（TFT-based Dynamic Dual-Path Model）**，使用原F2及P3 G2；保留原生TFT路径、因果多尺度路径、历史工况动态融合、48×14到24步DeltaV预测。没有修改backbone、门控或参数数目。

## 唯一候选和单位

`Vhat(t+h) = V_origin_raw + s * DeltaVhat_std(t+h)`。

DeltaV目标仅除以冻结目标尺度；不加输入电压scaler均值。起点用未EWMA的原始单体均值，不能用滤波输入替代。`s=0.13680318714138442 V`，来自原Train窗口未滤波电压std，不是新估计的未来DeltaV std。

通用实现：`Lphys = mean(ReLU((Vmin-Vhat)/s)^2 + ReLU((Vhat-Vmax)/s)^2)`；未注册某侧边界时省略该项。`Ltotal=MSE(DeltaVhat_std,DeltaV_std)+0.01*Lphys`。两项无量纲。

本次唯一实际筛查候选是`Vmin=0 V, Vmax=None`。lambda=0.01在真实前向检查前冻结；未开展另一阈值、公式或权重搜索。API合成测试的1.0 V只是测试夹具，绝不是RealHyFC注册上限。

## 物理依据及为什么未获资格

DOE/NETL《Fuel Cell Handbook》第7版§2.2、表2-2（印刷页2-4至2-6）说明可逆电位随温度和反应物/产物状态改变；1.229 V对应298 K、液态水标准条件，不能自动作为所有动态工况上限。[官方手册](https://www.netl.doe.gov/sites/default/files/netl-file/FCHandbook7.pdf)

只有已认证正常发电域内，非负电压才可作为很弱的合理性筛查。反转实验报告说明燃料/空气匮乏下可以出现负电压；这不是“不可能测量”。[Liang等，Behavior of PEMFC in starvation](https://www.sciencedirect.com/science/article/pii/S037877530501089X)（原始实验论文；本轮仅获取摘要，全文请求403，未声称全文核验）。

本项目未来工况未知，也缺未来界面分压、水状态及故障域认证。入口历史均值不能保证未来状态，当前未认证统一非平凡边界；这是一项针对当前信息条件的研究判断，不是证明理论上永远不存在可行域。历史0.5 V是特定测量/极化语境限值，不能移作动态均值电压下限。

## 活性与数值筛查

冻结工程筛查要求：Train全部合法1380起点；初始化及至少一个保存角色激活比例≥0.001且loss>1e-12；加权物理项/数据MSE≤0.1。这些是工程准入标准，不是物理界限。本次9项全部激活比例=0，即使只要求任意非零也失败；没有用Validation选择规则。

单侧ReLU平方区间内梯度为0，越界有限梯度直接作用于预测输出。合成梯度有效并不证明真实适用性。判定：`SOFT_PHYSICS_NOT_QUALIFIED`，具体含`PHYSICS_LOSS_INACTIVE`；停止正式训练。

保留Phase P对历史重建、未认证极化斜率、数据教师和短期单调性的否定证据，见[原审查](../Phase_P_Final_Innovation_and_Experiment_Planning/P_PHYSICS_LITE_CANDIDATE_REVIEW.md)。本轮不推翻该结论。
''')
    put('02_IMPLEMENTATION_AND_TESTS.md', '''
# 实现、测试和信息边界

中断前已完成14项测试，`outputs/SYNTHETIC_TEST_RESULTS.json`记录2026-10-14 01:11（北京时间），returncode=0、无真实数据加载。恢复后保留原记录，没有重复运行或覆盖。冻结manifest包含这些原脚本和测试SHA。

|要求|实际证据|结论|
|---|---|---|
|前向模型与G2一致|直接调用原gate_ablation_models的G2，无新网络wrapper；原类合成前向及状态不变检查|通过|
|lambda=0等于MSE|数值与梯度逐元素相同；真实G2参数梯度逐项比较|通过|
|单位恢复正确|使用raw起点加s×标准化DeltaV，测试排除输入mean|通过|
|区间内不惩罚|合成区间内部loss和梯度都为0|通过|
|越界梯度|人工上下越界均有限、非零、方向正确|通过，仅合成|
|真实Train非零作用|3 seed×3角色、每次33120预测值|失败，全部为0|
|未来输入隔离|predict_delta仅历史输入；decoder前14变量为0；推理未增加未来工况|通过|
|参数/输出身份|332436可训练参数，[B,24]输出|通过|

真实资格检查复现1380合法Train窗口、冻结Top10 mask、历史电压EWMA(0.3)、原scaler；与P3的原始/预处理x、y、起点、目标行、scaler及Development对象指纹逐一一致。三个种子初始化权重、训练初始RNG和第一轮采样顺序匹配。六个checkpoint文件SHA、模型/seed/协议元数据核验通过。

仅执行初始化Train模式带dropout前向，以及best/last的Train eval前向；没有optimizer、反向更新、新Validation推理。约17秒使用GPU作只读前向审计，不能称GPU正式训练。旧最佳checkpoint历史上由Validation选出，本轮不利用其成绩寻找物理公式；本次全Train的零活性证据已足以停止。

## 读写边界

数据reader严格`nrows=3007`后过滤`bin_end_ophrs2_h < 1819.997738`得到3005行，再先切Train `[0,2103)`才构造窗口。允许的有界Development对象读取包含Validation段，但没有为资格判断构造或评价Validation窗口。没有读取全混合文件计算SHA。

生产脚本使用审计hook限制仓库路径、禁止历史写入、禁止N1全量/3005替代读取、禁止其它历史CSV；合成防线测试覆盖拒绝、指纹后缀不变/合法前缀变化、源哈希篡改拒绝。hook是本进程工程防线，不是整个操作系统的安全沙箱；报告读取另有来源清单。

完整记录：[Train身份](outputs/TRAIN_DATA_AUTHORITY.json)、[运行访问日志](outputs/RUNTIME_DATA_ACCESS_AUDIT.json)、[冻结来源](outputs/SOURCE_AND_CODE_MANIFEST.json)、[控制台](outputs/QUALIFICATION_CONSOLE_20261014.log)。原历史数据、模型、协议和checkpoint未修改。
''')
    put('03_TRAINING_RESULTS.md', f'''
# 实际执行结果

**正式训练未启动：0个run、0 epoch、0 optimizer step。**没有软物理模型checkpoint、Validation预测缓存或新RMSE/MAE，不将任何旧成绩改名为新结果。三个计划seed均保留，均因同一候选未准入而未训练；不是训练失败后重跑。

真实Train资格审计如下（每行1380起点×24步）：

{at}

9次检查共298,080个预测值，惩罚、违规程度和激活比例全部为0。最小预测约0.001021683 V（seed62初始化）；大于0，不能人为把下界提高以触发loss。训练权重比例=0只说明无数值失衡，也说明没有附加学习信号。

资格审计耗时{decision['qualification_time_s']:.3f}秒，GPU为RTX 3070 Laptop，Python3.11.9/PyTorch2.6.0+cu124；完整环境见QUALIFICATION_DECISION.json。初始设备观察约52°C，无Python训练占用。没有持续训练资源曲线，因为没有正式训练。

判定不是“无改善”：正确类别是**候选未获资格**，没有公平新旧实验可评价。没有为了完成图表要求绘制不存在的新模型成绩；保存的是实际审计表和明确标注的历史汇总。
''')
    put('04_ABLATION_AND_COMPARISON.md', f'''
# 历史对照及可比性

历史P3已完成G0/G1/G2×42/52/62九次450轮，本轮没有重训。仅整理G2已有结果JSON和已有24步汇总CSV，不重新评价Validation预测。

{ht}

全精度及P95/最大误差/checkpoint和预测哈希见[历史六角色CSV](outputs/HISTORICAL_P3_G2_METRICS.csv)，24步见[历史horizon CSV](outputs/HISTORICAL_P3_G2_24STEP_METRICS.csv)。这里每个seed的best/last是同一次历史训练的两个角色，不是六次新训练。

原TFT是F0；F1固定双路径；P3 G0是固定0.5融合、G1全局学习静态融合、G2历史工况动态融合。G0不是原生TFT，G1不是“原F1”训练成绩的同义名。H/P2/P3训练预算、停止和选优口径不同，不能混为同条件成绩。

本轮通过的条件：G2结构与参数、初始化权重/初始RNG、第一epoch采样、窗口/目标/scaler、Top10/EWMA、P3代码与保存权重身份。**未执行新450轮训练，因此后续随机序列、逐epoch采样及运行预算一致性尚无实证；CUDA也不承诺bitwise复现。**若有合格新候选，历史G2可作为注册预算下的配对基线，不能凭本次初始化检查宣布已完成严格新旧实验。

已有三seed Best RMSE均值约7.501584 mV，Last约14.726629 mV；P3 G0/G1 Best均值约6.781101/6.801681 mV。P2/P3没有证明动态门控独立稳定优于两对照，本轮不改这一结论。

软物理版：全部未训练，RMSE/MAE/24步/端点/P95/P99/extreme/违规减少均为**未评价**；不填0，不推断等效或改善。Train旧模型非负筛查全0不能推广为Validation/Test不存在物理违规，更不能声称完整电化学一致性。
''')
    put('05_SOH_NEXT_EXPERIMENT.md', '''
# SOH已有证据与下一项最小实验

## 冻结量和已有成果

参考定义`SOH_V(t_i)=V(j*=1.0 A/cm²,t_i)/V(j*,BoL)`；辅助j*=1.6仅敏感性。沿用既有nominal 10-cell per-cell定义、BoL authority及曲线来源，不混接独立8-cell EIS。Development normal参考为BoL、300/600/850/1150/1500/1750七点；条件可比性仍有前提，不生成逐小时真实SOH。

P1S固定token `221||221.200 A`；dwell≥120s，primary middle_50，last_25仅已冻结敏感性，active10-cell均值的中位数、周期内再汇总。前10个complete周期暖启动，`Vref=0.71383023635 V`在171.82091h后可得；首个合法目标COMPLETE_011在181.34366h，之后131个因果观测。

|来源|真实已保存证据|当前含义|
|---|---|---|
|PAA/PAV|六点preanchor时间先于极化；3/6 median来源在共同支持内|有方向兼容，不代表全部状态可比|
|AHC|六锚点LOAO MAE0.0004768699；严格前三个后续锚点MAE0.0007646870|简单affine有有限支持，LOAO非在线证据|
|PHI|全六点B2：1.1988114938324186R−0.2090362463118213；141密集代理|全锚点校准回顾性；拟合一致性不是预测精度|
|CTA/CTAE|全序列SG/isotonic/PCHIP会改写过去；raw有35次回升|在线用raw与当时已知参考，不能硬单调|
|CNP1S|OLS/Ridge LOAO约0.00036947/0.00043175，总gate未全过|不再挑公式救回条件归一化|
|CMRF|H6 TimesNet0.716930 vs Persistence0.604104 mV-equivalent|复杂点预测未优于强基线|
|PSDM|H6 MAE0.707559，CRPS0.000665065 vs GP0.000631906|物理退化率与概率效用未成立|

这些是历史结果，来源SHA在REPORT_SOURCE_MANIFEST.json。PAA/PAV、AHC、PHI、CTA/CTAE、CNP1S、CMRF和PSDM报告均已阅读；不重建raw、参考或P1S。

## 明天只做一项：前缀校准到下一稀疏参考的SOH估计

任务是**有P1S测量时估计当前参考工况健康proxy**，区别于提前预测未来健康。只用保存Development数据；在参考可得后用raw R，不使用全锚点B2或全序列平滑。

固定三次前缀：已获得300/600/850参考→检验1150；加入1150→检验1500；加入1500→检验1750。以参考结果实际发布时刻为准；若只有nominal availability，明确降为nominal chronology replay，不冒充真实在线。已核实`Phase_AHC_Sparse_Anchor_Affine_Health_Calibration/outputs/B2_CHRONOLOGICAL_PREFIX_EVALUATION.csv`包含完全相同三前缀：1150/1500/1750预测分别0.9715868347291073、0.9740582586162393、0.9728474747386423。明天先复用该缓存，补齐Persistence和年龄趋势对照，不重复affine拟合；本轮仅核对既存值，未计算新对照成绩。

候选只用`Yhat=aR+b, a>=0`的CPU最小二乘，与已有AHC规则一致，不搜权重。每次仅当时已知锚点拟合；目标R由冻结preanchor interval ID/测量窗口取值，先保证测量结束时可得；不能替换为更合适的新dwell。1750目标即使历史上已暴露，当前也只作为既有Development回放，非独立验证。

对照：最近合法SOH参考Persistence；全部既往参考的普通线性年龄趋势；raw R原样作为尺度不等同的诊断对照。目标锚点nominal age只作当前估计时间坐标，不输入未来真实lead或未来工况。若操作年龄与nominal诊断小时不一致，保留身份而不伪造精确OpHrs2。

输出3个锚点逐点AE、有符号误差、MAE、参数、数据/可获得时间身份、in/out-of-support（保留全部样本）。共同BoL不确定性不当作独立误差；small-n不声称显著性。描述性波动区间不冒充95%概率覆盖。

判断：在全部三点保留的条件下，affine若未稳定优于Persistence和年龄趋势，停止复杂SOH模型并优先补独立同参考条件诊断；若有信号，也只支持同堆proxy估计。不要以三个点论证跨堆真实SOH。今晚没有拟合、校准更新或新SOH网络训练。
''')
    put('06_RUL97_THRESHOLD_PLAN.md', '''
# 自定义健康阈值下的条件剩余时间预测方案

**主阈值冻结为SOH_V=0.97，使用j*=1.0 A/cm²和原BoL。**它是观察历史后由用户指定的探索性性能阈值，不是官方EOL，不能称已验证实际故障RUL。96%、98%只报敏感性，不能择优替代97%。本次只设计，没有拟合、RUL训练或新增Test读取。

## 已知事件与不确定性

指令引用既有历史审计：1750h约97.1785%，2000h约96.9497%，2300h约96.9999%。本轮不重新打开后两点极化/健康目标文件，末位不能视为测量无误差。历史RUL90/道路报告支持晚期接近97%及EOL未建立；精确百分数按用户提供历史摘要引用，不自行重算。

|事件|可以说什么|
|---|---|
|首次观测低于阈值|当前nominal检查点中2000h首次低于97%|
|真实首次跨越|1750→2000有相邻上/下阈值证据；若连续及该段此前未穿越再恢复，候选区间为(1750,2000]|
|持续低于|2000与2300两次低值不能证明中间始终低于；持续性未确认|
|恢复/波动|2300比2000高且极接近97%，不能把微小差距解释成可靠永久故障|

稀疏检查点无法排除1750之前短暂跨越又恢复，所以(1750,2000]是给定“无更早未观测跨越”假设的名义事件区间，而非精确首次故障时间。没有跨越证据的期间不能无条件宣称完整连续未跨越。

## 未来运行假设与预测对象

主情景：未来沿用起点前已知运行计划/历史Development的duty-cycle支持域，参考测试条件不变，没有额外维护或恢复操作；未提供计划时只能声明“历史工况分布延续”情景。输出单位为operating hours，不等同wall-clock；循环到小时必须用起点前计划或过去估计并传递不确定性，不输入未来实际lead。超出支持域的负载/湿度/停机/维护变更只作情景敏感性。

输入起点t前已发布SOH参考及因果P1S观测；目标为未来参考工况健康保持率H(t+u)，不是动态负载电压/BoL。P1S的97%与SOH_V的97%不等价，不能直接用raw R穿越当SOH事件；前缀affine若资格不足，主路线仅稀疏SOH参考。

## 最小可执行设计（后续阶段）

先在合法Development以nominal起点850/1150/1500h作滚动回放，预测下一参考点健康；固定方法为Persistence及全部起点前参考的OLS线性趋势，第三项只在05校准合格时加入前缀P1S-affine加过去96h趋势。没有网络训练、权重搜索或重新拟合PSDM。

OLS健康轨迹`Hhat(t+u)=H_last+b*u`（时间偏移以最后已知参考时刻计算）；不强制b<0，也不裁剪恢复。H_last>0.97且b<0时线性情景crossing可解析得到；b≥0则预测期内无跨越。Persistence高于阈值时不跨越。公式给出的是情景模型解，不能作为观测精确事件标签。

每个起点冻结输入、可获得时间、方法与未来情景后再计算；禁止未来锚点反校准。先评价下一锚点MAE和逐点误差，三点太少不产生可靠寿命概率覆盖。Development未观测97%穿越，不能用它训练精确crossing标签或宣称验证了97%剩余时间。

主报告外推到起点后250 operating hours；250是设计上限，不是已验证可预测距离。只在该域内给有限crossing时间，远期需求另报长距离情景、误差积累和模型错设，不用无界线性根作为精确RUL。使用过去96h的P1S slope需足够观测、prefix replay及健康映射适用；不足则不输出该候选。

## 区间评价、未跨越和恢复

预测主事件`tau97=inf{u>=0:Hhat(t+u)<0.97}`，严格低于；若恰等于阈值单列边界。起点已低于则“起点已满足”，不是负剩余时间。未在250h域跨越，报“250h内未预测跨越”，不赋0或2300−t。只有模型延伸条件下才写tau>250，不能当真实寿命删失证据。

针对相邻观测给出事件区间(L,U]，点预测的距离为：区间内0，区间外距最近边界；这是区间相容距离，不是精确时间MAE。未来若有足够独立事件和合法分布，可评价P(L<T<=U)及区间删失似然，不能由PHI的±1.96描述带直接导出可信寿命概率。

另列“连续两次合格参考观测低于”的确认事件：确认时间为第二次测量，穿越仍是区间；两次间有无持续低于未知。轨迹可恢复时记录每次进入/离开及首次进入，主首次跨越不因恢复重置；持续跨越另以固定诊断节奏与运行记录审定，不事后改定义。

2000/2300已历史解盲，不再充当确认性独立Test。未来验证需要新授权且独立的数据、测试节奏、工况计划和测量不确定性。只增加同堆重叠窗口不能增加独立失效数。没有真实故障记录前称**自定义健康阈值下的条件剩余时间预测**。
''')
    put('07_FINAL_RESEARCH_DECISION.md', '''
# 最终研究决定

**当前软物理候选停止；现有模型和成果保留；下一步优先核验简单SOH前缀估计。**

软物理分类为SOFT_PHYSICS_NOT_QUALIFIED：工程实现与合成梯度通过，真实Train作用为零，非平凡可行域未认证。不是SOFT_PHYSICS_USEFUL，也不是已完成训练后的SOFT_PHYSICS_NO_CLEAR_GAIN。不能把新创新称为已证有效，不能删除Phase P否定结论。

保留：TFT动态双路径模型原结构、所有F/H/P/P2/P3 checkpoint、已保存历史指标及本次14测试/9次Train审计证据。G2新增loss不导致改名固定双路径。P2/P3动态门控独立增益未支持的结论保持。

停止：本候选450epoch训练、另一上下界、第二lambda、额外seed、backbone搜索、门控调整、复杂SOH/RUL新网络、晚期目标读取及确认性Test重启。没有需要继续监控的训练进程；不会自动启动下一轮研究。

继续（供负责人明天实施）：05中的CPU前缀校准身份核验及下一锚点SOH proxy估计，先复用AHC已有同任务缓存；06中的97%情景/区间预测设计，待短期预测和数据资格成立才执行。

若未来有独立物理包络认证与真实训练活性，需另立新阶段和协议，保留本阶段结论，不能重写此候选或重用失败Run ID。若需要真实RUL主张，则先补明确失效准则及独立寿命资料。
''')
    put('08_EXECUTION_AND_GIT_AUDIT.md', '''
# 执行恢复与Git审计

## 起点现场（北京时间2026-10-14 09:27起）

本地main HEAD及缓存origin/main均为8997b8c881f206680235361b36f892f0edc2af62。git ls-remote两次连接重置；随后认证GitHub CLI branches/main请求成功，实时远程SHA相同。没有拉取、合并或覆盖其它成果。

中断前本目录已有`.gitignore`、4脚本、1测试、SYNTHETIC_TEST_RESULTS.json及忽略的matplotlib缓存；最后文件更新时间01:11。没有协议/资格决策/训练ledger/checkpoint/九份报告，本阶段尚未提交。进程查询未发现python训练进程；nvidia-smi只有显示应用。本轮从资格冻结继续，未重新运行已通过的合成测试。

仓库历史tracked diff与暂存区起初为空；另有多个旧N2阶段node_modules未跟踪依赖文件，保持原样、没有加入本阶段提交。恢复期间未修改历史tracked文件。提交范围只能是本阶段目录。

## 实际后续执行

1. --freeze只执行一次，SHA=39003814b9af339765b90177cac8a4bf1586f65f1e90ba28aadaac9a66ff4ed8。
2. --qualify只执行一次；约17秒GPU前向检查、9个seed/role组合，0 optimizer step。控制台记录保留；没有技术失败或重启。
3. 仅候选准入失败，formal_runs_started=0。没有运行--run，没有本阶段训练成绩或新权重。
4. 报告生成只读资格JSON、历史指标汇总和文档；新增REPORT_SOURCE_MANIFEST记录其身份，独立于执行前冻结manifest。新增报告脚本不参与模型/资格执行，不修改执行身份。
5. 用phase-local .gitattributes固定LF，避免core.autocrlf=true在提交时改变冻结脚本/JSON哈希；自检须验证工作区与Git index字节身份一致。

原执行脚本把BASE设为参考SHA，因本次本地/实时远程确实相同才freeze成功；未来仓库更新不能修改BASE并重冻结本次协议。Windows绝对源路径用于本机身份审计；迁移机器应单独审定路径/环境，不宣称直接可移植复现。

## 读写与资料限制

唯一真实时序表读取为nrows=3007有界Development读取，过滤3005，再仅1380 Train窗口。对象指纹非混合全文件SHA。checkpoint读取6个旧G2 best/last，仅Train推理。无新Validation推理、Assessment/Test、raw、晚期极化/健康目标；2000/2300百分数只引用指令里的既有审计摘要。

SOH报告读取范围见REPORT_SOURCE_MANIFEST；历史Test曝光边界仍继承，不称全部资料从未暴露。文献查询只补适用条件，未下载RealHyFC数据。NETL官方手册已读；starvation原始实验论文全文403，只使用检索摘要证据。

## 可复现入口与成果保护

`python -B scripts/run_soft_physics_qualification.py --freeze`、`--qualify`均为一次性入口，已有协议/决策会拒绝重跑。`--run`没有在物理资格失败时放行的训练worker。已完成结果应阅读/验证，不删除后重跑。报告生成脚本也拒绝覆盖已有报告。

最终提交前自检保存在outputs/RECOVERY_FINAL_AUDIT.json；实际提交/推送状态另存忽略的outputs/GIT_TRANSPORT_RECEIPT.local.json，并可由Git提交历史与实时远程验证，交接回复给出实际SHA。网络失败时保留本地提交、不强推、不重做训练。报告不写入自身commit SHA，避免为回填SHA制造重复提交。
''')
    put('README.md', '''
# TFT动态双路径模型软物理候选资格审计

完成中断恢复；正式训练0次，候选未获资格。先读00_MORNING_SUMMARY_CN.md。

执行前代码/输入身份：configs/DDP_SOFT_PHYSICS_FROZEN_PROTOCOL.json及.sha256。
历史G2对照与本阶段Train资格结果分开保存。禁止删除原协议、测试记录或决策后重跑。
本阶段没有SOH/RUL拟合，没有下一轮自动研究任务。
''')
    print('Created nine reports, two historical CSVs and report-source manifest; no model execution.')

if __name__ == '__main__':
    main()
