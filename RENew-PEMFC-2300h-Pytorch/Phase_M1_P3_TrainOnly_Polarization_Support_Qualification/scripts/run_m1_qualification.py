"""Train-only evidence audit. No raw loader, model, forward, fitting or pairing fallback."""
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd
from m1_core import (BASE, RESEARCH, CONDITIONS, N1, metadata_hash, digest_bytes,
                     require, load_train, eligible_metadata, no_voltage_view, AUX_NONVOLTAGE)

PHASE = Path(__file__).resolve().parents[1]
REPO = PHASE.parents[1]

def write(relative, value):
    p=PHASE/relative; p.parent.mkdir(parents=True,exist_ok=True)
    if isinstance(value,pd.DataFrame):
        value.to_csv(p,index=False,lineterminator='\n')
    elif isinstance(value,str):
        p.write_text(value,encoding='utf-8')
    else:
        p.write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf-8')

def protocol(name):
    p=PHASE/'configs'/f'{name}.yaml'
    sha=digest_bytes(p.read_bytes())
    require(sha==(PHASE/'configs'/f'{name}.sha256').read_text().strip(),'PROTOCOL_HASH_MISMATCH')
    return json.loads(p.read_text(encoding='utf-8')),sha

def preflight():
    access,sha_a=protocol('M1_FROZEN_ACCESS_PROTOCOL')
    support,sha_s=protocol('M1_FROZEN_SUPPORT_PROTOCOL')
    subprocess.run(['git','merge-base','--is-ancestor',BASE,'HEAD'],cwd=REPO,check=True)
    stage0=json.loads((PHASE/'outputs/M1_STAGE0_FREEZE_AUDIT.json').read_text(encoding='utf-8'))
    require(stage0['pass_'] and not stage0['real_data_read'],'STAGE0_NOT_PASSED')
    for relative,sha in access['execution_code_sha256'].items():
        require(digest_bytes((PHASE/relative).read_bytes())==sha,'EXECUTION_IDENTITY_CHANGED:'+relative)
    allowed=[r['path'] for r in access['static_source_allowlist']]
    for row in access['static_source_allowlist']:
        require(metadata_hash(REPO,row['path'],allowed)==row['sha256'],'SOURCE_IDENTITY_CHANGED:'+row['path'])
    return access,support,sha_a,sha_s

def main():
    access,support,sha_a,sha_s=preflight()
    start=datetime.now(timezone.utc).isoformat()
    # This is the FIRST real numeric access of M1, after freeze + synthetic tests.
    frame,boundary=load_train(REPO,access)
    write('outputs/M1_RUNTIME_ACCESS_AUDIT.json',boundary)
    mask,eligibility=eligible_metadata(frame)
    eligible_count=int(mask.sum())
    eligibility.append(dict(stage='raw_dwell_and_state_certification',denominator_before=eligible_count,
         exclusions='',remaining='',evidence_level='NOT_IDENTIFIABLE_NO_RAW_CERTIFICATE; not a numeric zero'))
    write('outputs/M1_PAIR_ELIGIBILITY_SUMMARY.csv',pd.DataFrame(eligibility))
    conditions=[]
    for col in CONDITIONS:
        for name,lo,hi in [('calibration',0,1051),('evidence',1051,2103)]:
            values=frame[col].iloc[lo:hi].to_numpy(float)
            finite=np.isfinite(values)
            good=values[finite]
            conditions.append(dict(source_field=col,domain=name,source_rows=f'[{lo},{hi})',
                rows=hi-lo,finite_rows=int(finite.sum()),finite_fraction=float(finite.mean()),
                minimum=float(np.min(good)) if len(good) else None,
                maximum=float(np.max(good)) if len(good) else None,
                IQR=float(np.percentile(good,75)-np.percentile(good,25)) if len(good) else None,
                support_status='MARGINAL_AVAILABILITY_ONLY; JOINT_DWELL_SUPPORT_NOT_IDENTIFIED'))
    write('outputs/M1_CONDITION_SUPPORT_REPORT.csv',pd.DataFrame(conditions))
    view=no_voltage_view(frame)
    # Actual bounded-Train column lineage perturbation. Not a model forward.
    changed=frame.copy()
    voltage=[col for col in changed if 'voltage' in col.lower()]
    changed.loc[:,voltage]=changed.loc[:,voltage]*7+3
    invariant=view.equals(no_voltage_view(changed))
    require(invariant,'AUX_VOLTAGE_LINEAGE_TEST_FAILED')
    isolation=dict(auxiliary_view='PROPOSED_AND_COLUMN_LINEAGE_TESTED; no model built',
      visible_columns=AUX_NONVOLTAGE,visible_voltage_columns=[],
      all_times_voltage_excluded=True,raw_cells_excluded=True,exported_average_excluded=True,
      EWMA_and_all_voltage_descendants_excluded=True,neighbor_voltage_excluded=True,
      matched_condition_voltage_excluded=True,pretrained_latent_excluded=True,
      age_cycle_curve_identifiers_excluded=True,train_scaler_voltage_reference_excluded=True,
      original_f2_view_modified=False,nonphysical_control_requires_same_view=True,
      bounded_train_voltage_perturbation_invariance_pass=invariant,
      real_label_dwell_to_aux_window_source_join='NOT_EXECUTED: raw dwell identity unavailable',
      isolation_scientifically_qualified=False,reason='Complete actual-label time/source lineage not established',
      status='P3_AUXILIARY_TARGET_LEAKAGE_RISK_UNRESOLVED')
    write('outputs/M1_AUXILIARY_ISOLATION_AUDIT.json',isolation)
    summary=dict(phase=PHASE.name,base_commit=BASE,execution_head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip(),
      start_utc=start,access_protocol_sha256=sha_a,support_protocol_sha256=sha_s,
      train_rows=2103,train_age_bin_start_h=boundary['start_h'],train_age_bin_end_h=boundary['end_h'],
      train_f2_legal_windows=1380,necessary_bin_candidates=eligible_count,
      calibration_necessary_bin_candidates=int(mask.iloc[:1051].sum()),
      evidence_necessary_bin_candidates=int(mask.iloc[1051:].sum()),
      complete_14_condition_bin_rows=int(np.isfinite(frame[CONDITIONS].to_numpy(float)).all(axis=1).sum()),
      unique_train_temporal_segments=int(frame.temporal_segment_id.nunique()),
      independently_verified_stable_dwells=None,available_legal_low_high_pairs=None,
      certified_pairs=0,certified_pairs_interpretation='zero certified, NOT evidence of zero physically available pairs',
      joint_dwell_support='NOT_IDENTIFIABLE',material_state_comparable='NOT_ESTABLISHED',
      polarization_order_evidence='NOT_COMPUTED_ON_UNQUALIFIED_BINS',
      neighbor_pair_fraction=None,cross_age_pair_fraction=None,recovery_pair_fraction=None,
      pair_deltaV_distribution=None,pair_violation_rate=None,repeat_noise=None,scan_hysteresis=None,
      amplitude_supervision_qualified=False,
      structural_voltage_free_view_pass=True,actual_auxiliary_label_lineage_qualified=False,
      constant_output_order_loss=0.,constant_output_excluded_by_order_only=False,
      quasi_steady_status='QUASI_STEADY_STATE_UNVERIFIABLE',
      simultaneous_issues=['raw dwell duration/stabilization/scan unverified',
        'joint condition support at dwell level unverified','membrane-water/material/recovery state comparability unresolved',
        'raw file Train-exact record/byte cutoff not independently established; historical PAA reads broader than F2 Train',
        'actual label-dwell source/time join unavailable despite a safe voltage-free prospective view',
        'independent amplitude evidence/anti-j-only-curve control not established'],
      final_status='P3_QUASI_STEADY_OR_STATE_COMPARABILITY_UNRESOLVED',
      next_independent_physics_f2_experiment_allowed=False,
      physics_informed_historical_auxiliary_learning_candidate_qualified=False,
      model_trained=False,model_forward=False,model_instantiated=False,
      raw_data_accessed=False,polarization_content_accessed=False,
      validation_accessed=False,assessment_accessed=False,test_accessed=False,
      full_mixed_source_hash=False,SOH_constructed=False,RUL=False,
      original_phase_M_status='NO_ADMISSIBLE_PINN_CONSTRAINT_YET',
      scientific_not_technical_failure=True,seed='NOT_APPLICABLE',automatic_next_stage=False)
    write('outputs/M1_SUPPORT_SUMMARY.json',summary)
    source_lines=['# M1 来源身份台账','',f'基准：`{BASE}`。静态来源以 SHA256 固定；混合 N1 仅计算允许 Train 列/行逻辑 fingerprint，不计算完整文件 hash。',
       '', '|来源|读取类别|SHA256|最后修改提交|','|---|---|---|---|']
    for r in access['static_source_allowlist']:
        source_lines.append(f"|{r['path']}|{r['access_class']}|{r['sha256']}|{r['last_modified_commit']}|")
    source_lines.extend(['',f"混合源：`{N1}`，物理数据记录 `[0,2103)` 与 F2 Train 行一致。允许 fingerprint：`{boundary['allowed_train_fingerprint_sha256']}`。",
     '权重只读 bytes，未反序列化、未建立模型。静态历史程序只读源码，未 import 或执行。',
     'raw、polarization、RCR 数值 catalog、PAA 全量测量、旧 prediction 缓存均未打开。'])
    write('evidence/M1_SOURCE_IDENTITY_LEDGER.md','\n'.join(source_lines)+'\n')
    write('evidence/M1_TRAIN_BOUNDARY_AUDIT.md',f'''# F2 Train 身份与边界审计

技术边界：PASS。F2 行 `[0,2103)`；bin_start 起点 {boundary['start_h']} h，最后 bin_end {boundary['end_h']} h。这是聚合 bin 的身份，非原始最早观测年龄。

1. N1 `finalize_aggregate_table()` 在输出前按 bin_start 全局排序，未按 valid 或 split 删除行。
2. C `load_development()` 仅按 bin_end<1819.997738 过滤并 reset_index，没有删 invalid 行。E/F/H 继承同一顺序与 `[0,2103)`。
3. 本轮不调用旧 C/E loader，也不读3007行。无buffer FileIO只获取header和2103个完整CSV逻辑记录，quoted newline不当作记录边界。没有第2104条记录/下一字节探测。
4. 运行验证所有消费行都满足旧age过滤，所以源prefix→F2 Train是恒等映射；检查严格增序/唯一start、0.5h宽度、首末冻结边界、segment整数/增序与原1380个合法48→24窗口身份。没有重算segment。
5. 行序/筛选证明绑定 N1/C/E 源码hash与E split manifest。后续执行须重新验证这些hash；不借完整N1 Git blob证明内容。

物理读取bytes={boundary['bytes_consumed']}。读后仅retain显式列。fingerprint=`{boundary['allowed_train_fingerprint_sha256']}`，只含2103个Train记录的SAFE_COLUMNS。禁止使用完整源大小/mtime作为完整性证明。

raw来源另外不合格：原PAA默认上限1719.31h、chunk解析、全RCR表读取都超过本轮F2 Train。原120s/middle50%规则只能作为历史语义，不能直接执行。没有可信的每raw文件Train截止byte/record清单，本轮未读任何raw。不能读边界后的第一条raw记录来发现停止点。

Validation、Assessment、Test数值行读取：NO。
''')
    write('evidence/M1_AUXILIARY_VOLTAGE_LINEAGE.md','''# 辅助电压完整血缘

raw有效10cells(U1–U11排除U6) → instantaneous cell mean / cross-cell std / range / min/max → bin统计；exported U.S.AveCell为并行电压来源。

这些观测 → EWMA或其他邻接/滚动缓存 → 后续历史槽位；以及matched_condition_voltage、运行点电压reference、P1S/retention/B2/latent，都可能携带标签信息。仅mask cell_voltage_mean不足：其余cell通道、min/max、range和EWMA后继仍可复制。

本轮拟议独立辅助view采用显式9个NON-voltage Raw14通道，所有时间的所有电压血缘及voltage-derived cache都不暴露。没有仅删除当前dwell电压后保留污染EWMA的做法。age/cycle/curve ID不在可见列。真实Train列值扰动和synthetic raw/exported/EWMA perturbation均调用生产view验证不变。

这是一项充分保守的列级隔离设计，不是已经构造完成的真实dwell训练集。真实label dwell的source/time身份尚未恢复，不能声称完成完整标签窗口联接。原F2主输入/Top10/EWMA/scaler均不修改；未来同容量非物理对照必须采用同一辅助view。未建立辅助网络、编码器或训练任务。
''')
    write('reports/M1_PHYSICAL_MEASUREMENT_SEMANTICS.md','''# 物理测量语义

依据：N1 dictionary/config/build code、G-STOP PHYSICAL_RELATION_AUDIT、PAA protocol/selection code、PAV PAA_CLAIM_BOUNDARY、CSHSE incident、Phase M identifiability。具体源hash见来源台账。

|来源|量与状态|本轮可证/不能互换的原因|
|---|---|---|
|durability bin|有效10cells动态运行30min mean V；cell_std是瞬时跨cell离散度时间mean|不是120s稳定dwell；cellstd不是时间噪声；没有raw时间顺序/扫描方向证据|
|PAA stable P1S|固定221||221.200，raw run>=120s，middle50% dwell voltage median，再interval median|固定单负载token不能提供低/高j曲线；本轮不执行更宽Development读器、不复用全表数值|
|nominal polarization|characterization curve、per-cell V与actual j|不能混用运行bin的负载均值；温湿压/hold/扫描/配置需独立确认，本轮未读取真实曲线|
|reference-condition voltage|条件标准化诊断健康anchor|不是动态bin，也不是已验证连续SOH|

j.S.act单位A/cm²、I.S.act单位A；历史G-STOP支持约280cm²比例，但这不是极化关系证据。durability stack10cell，U6候选异常已排除；BoL短stack配置不能无核查拼接。

传感器语义：T.Si/So.A/C是入口/出口温度，不是反应界面温度；p.Si/So是kPag表压，不是反应气体界面分压；RH为入口湿度，不等于膜含水；F单位nl/min，stoichiometry为无量纲。材料/界面慢状态、恢复/滞后不由这些bin mean闭合。

未证实：实际raw dwell长度/稳定化、极化扫描方向/hold、Air-Air或其他恢复干预的Train逐事件身份、膜水与材料状态可比。`QUASI_STEADY_STATE_UNVERIFIABLE`。运行load transitions==0仅表示不存在相邻>=5A跳变，不能证明不存在慢漂移或热/水滞后。
''')
    write('reports/M1_JOINT_SUPPORT_AND_STABILITY.md',f'''# 联合支持与稳定域

真实F2 Train：2103 bins，{summary['unique_train_temporal_segments']}个原始temporal segments；14工况同时有限的bin数{summary['complete_14_condition_bin_rows']}。按冻结必要条件剩余{eligible_count}个bin，校准/证据子域分别{summary['calibration_necessary_bin_candidates']}/{summary['evidence_necessary_bin_candidates']}。逐步分母和排除数见M1_PAIR_ELIGIBILITY_SUMMARY.csv。

**这些都是bin可用性，不是独立稳定dwell数量或联合匹配pair数量。** 每变量完整率/范围见M1_CONDITION_SUPPORT_REPORT.csv，仅是边际数据审计，不能把同时有限或range overlap叫joint matching支持。

校准行[0,1051)、证据[1051,2103)事先按时间隔离；无随机pair split。拟议caliper/最小Δj/时间距离/非共享dwell规则已在support protocol冻结，未因电压结果修改。它们为PROJECT_ANALYSIS_DESIGN，不是仪器精度或通用物理容差；还需原始稳定与状态证据才能执行配对。

未运行真实matching，因此neighbor-state/cross-aging/intervention/recovery pair比例、材料状态、扫描/hold一致性都为NOT_IDENTIFIABLE而非0。没有把早期低j/晚期高j拼成曲线。真正独立稳定dwell与pair数未知；0 certified pairs只指本轮尚未认证，不指数据里没有合法pair。
''')
    write('reports/M1_POLARIZATION_ORDER_EVIDENCE.md','''# 历史极化次序证据

冻结后进行了Train身份、bin必要条件和工况字段完整性审查。稳定dwell/同状态门槛未通过，未计算未认证bin的ΔV_pair、违约率或排序相关性。重复测量噪声、扫描/滞后差异、独立段/批次极化关系以及caliper敏感性均NOT_RUN。没有按观察电压下降挑pair。

理论参照：DOE/EG&G Fuel Cell Handbook 7th ed. §2.5，https://www.netl.doe.gov/sites/default/files/netl-file/FCHandbook7.pdf；本轮沿用Phase M已经核实的一手来源与适用边界，不请求任何项目数据。

同T/反应活动度/膜水材料状态、正j与合法稳态域内，Tafel/欧姆/浓差模型给出条件负斜率；不同动态状态之间不保证该次序。条件理论不是本数据的新经验验证，不约束未来24步unknown actual，不转换成时间单调SOH。

`P3_POLARIZATION_SUPPORT_NOT_ESTABLISHED`作为证据结论，最终科学状态采用协议允许的quasi-steady/state unresolved。没有模型性能或收益主张。
''')
    write('reports/M1_TARGET_LEAKAGE_PREVENTION.md','''# 辅助目标泄漏防线

CSHSE事故说明provenance操作也属于访问范围：即使不参与模型，混合文件whole hash仍越界。本轮明确只读F2 Train2103记录，hash仅静态白名单，限定source fingerprint与whole-source SHA不同。

辅助目标隔离必须覆盖raw cells、exported average、mean/min/max/std/range、voltage-derived统计、EWMA后继、邻接平滑与重复缓存。生产no_voltage_view用显式9非电压列而非黑名单；真实Train电压列与synthetic EWMA/raw/exported值扰动均不改变view。来源依赖皆为I/j/operation_point/count，不以V构造。

时间/源范围仍需独立恢复实际label dwell并证明窗口先于/不含label测量；本轮没有恢复raw身份，完整实际隔离资格未通过。不得根据低loss推断隔离成功。未来control采用同view，主F2不改。本阶段不创建任何auxiliary model。
''')
    write('reports/M1_CONSTANT_COLLAPSE_AND_IDENTIFIABILITY.md','''# 常数塌缩与可辨识性

L_order=max(0,q_high−q_low)²。任意常数q=C均给0损失。固定下降线q(j)=C−aj(a>=0)也为0，能忽略工况和材料状态；同曲线身份记忆及电压后继复制也可制造虚假成功。生产order_loss与合成测试明确展示常数/下降线/反向线。

非退化监督需要合法独立实测幅度或独立半经验校准，明确V/mV单位，标签从visible view彻底排除，校准/证据时间隔离、无共享dwell，不使用未来actual。幅度项可在数学上排除常数，但必须至少存在可验证的幅度变化及跨j/条件证据；单调次序不能证明这些。

本轮没有已认证dwell或独立幅度证据，故`P3_NONDEGENERATE_SUPERVISION_NOT_ESTABLISHED`。不是把未测膜水设常数，不拟合j0/ASR/jlim，也不运行神经网络。不能由拟议safe view和一个可写公式推出任务已经可学习。

最小支持floor为每时间域20个无共享dwell pairs、3负载水平、2独立段，均为冻结项目设计而非充分物理证明。未来若执行，需报告常数预测误差、j-only线误差、condition/state忽略、curve ID记忆与voltage-copy控制；不得先看结果再换criterion。本轮仅作数学/数据资格审查。
''')
    write('reports/M1_FINAL_P3_RESEARCH_DECISION.md',f'''# Phase M1 最终P3研究决策

最终状态：`{summary['final_status']}`。Train访问技术防线通过；科学资格未通过。**不允许进入独立physics-F2训练实验。** Phase M `NO_ADMISSIBLE_PINN_CONSTRAINT_YET`、P1/P2及F2冻结均不修改。

## 十二项回答

1. 真正独立稳定dwell有多少？**未知**。2103为Train bins，{eligible_count}为必要metadata候选，均非dwell数量。
2. 合法低/高j pairs多少？可用物理pairs未知；**本轮0 certified**，未强配未认证bins。
3. T/RH/P/flow是否联合支持？只有边际完整率与同时有限bin审查。dwell层共同支持未证实。
4. 材料状态可比吗？未证实，膜水/热迟滞/恢复/扫描/干预仍是缺口。
5. 经验极化关系成立吗？未证实；未在不合格pairs上算次序来宣称成立。
6. 超过30min聚合能力吗？是。raw稳定/hold/扫描/持续时间必须额外核查；当前raw-exact Train访问身份尚无独立证明，未越界读取。
7. 全电压血缘已隔离吗？保守全时段voltage-free view的列级血缘与扰动测试PASS；实际dwell source/time join尚未建立，**完整资格未通过**。
8. 独立aux input view是什么？固定9个非电压Raw14通道，不含age/cycle/curve ID，原F2不改，未来非物理control同view；未建立模型。
9. 幅度监督能排除常数吗？理论上需独立幅度；本轮实测幅度资格未证实。order-only已证明有常数0-loss，不合格。
10. 未知变量？界面气体活动度/膜水、材料状态、稳定化/扫描/hold、干预恢复身份、逐raw源Train截止映射。
11. 最低科学资格满足吗？**NO**。联合稳定支持、同状态、实际隔离和非退化幅度均不能升级。
12. 可进入独立物理F2实验设计吗？**NO，当前只有有条件补证问题，禁止启动训练。** 未自动创建补充阶段。

## 同时存在的阻断与解释

这不是Train技术访问失败：源顺序/滤选身份、冻结1310h边界和1380窗口都一致。是quasi-steady及state/comparability未证实；同时真实joint support、实际label-time血缘与独立amplitude都未闭合。不用一个容易修好的hash问题掩盖科学缺口。

不把未知写成不存在，不因bin候选少就否定所有历史极化，不把多pair当独立实验。不作新SOH/RUL标签、模型性能、future12h收益或严格PDE-PINN结论。

P3后续若获新的补证授权，首先需要可信Train-only raw测量身份/截断、稳定扫描与状态证据，之后才可能构造非复制幅度任务。**本轮不继续执行；是否进一步补证由人工研究决策决定。**

deep-research证据分级使理论、历史PAA条件方向稳健性与本轮bin统计严格分离。报告参照用户提供的协议式中文风格；AI辅助代码及证据整理，不声称完整系统综述、多人审稿或真实模型实验。
''')
    write('README.md','''# Phase M1 — Train-only历史极化物理支持资格审查

范围：仅P3，F2 Train [0,2103)。冻结主F2与历史M不变。不训练/forward，不读Validation/Assessment/Test。

运行：`python scripts/freeze_m1_protocol.py`（首次、真实数据前，合成测试+协议hash）；`python scripts/run_m1_qualification.py`（独立新进程）；`python -m unittest discover -s tests -v`。

两个.yaml为JSON语法的有效YAML，内容由hash绑定。再运行审查仅核验基准祖先和冻结code/source哈希，不重写协议。任何访问/身份失败即fail closed。

结果先读reports/M1_FINAL_P3_RESEARCH_DECISION.md。必要bin数量不是dwell/pair数量，未识别的量保留null。当前科学状态为P3_QUASI_STEADY_OR_STATE_COMPARABILITY_UNRESOLVED，不允许physics-F2训练。
''')
    allowed=[r['path'] for r in access['static_source_allowlist']]
    unchanged=all(metadata_hash(REPO,r['path'],allowed)==r['sha256'] for r in access['static_source_allowlist'])
    require(unchanged,'SOURCE_POST_AUDIT_IDENTITY_CHANGED')
    require(digest_bytes((PHASE/'configs/M1_FROZEN_ACCESS_PROTOCOL.yaml').read_bytes())==sha_a and
            digest_bytes((PHASE/'configs/M1_FROZEN_SUPPORT_PROTOCOL.yaml').read_bytes())==sha_s,'POST_PROTOCOL_CHANGED')
    files={str(p.relative_to(PHASE)).replace('\\','/'):digest_bytes(p.read_bytes())
           for directory in ['reports','evidence','outputs'] for p in sorted((PHASE/directory).glob('*')) if p.is_file()}
    write('outputs/M1_FINAL_INTEGRITY_AUDIT.json',dict(technical_audit='PASS',source_identity_unchanged=unchanged,
        access_protocol_sha256=sha_a,support_protocol_sha256=sha_s,
        protocol_frozen_before_first_real_read=True,strict_train_access_pass=True,
        synthetic_test_log='evidence/M1_SYNTHETIC_TEST_LOG.txt',
        full_mixed_source_hash=False,read_ahead=False,validation_accessed=False,assessment_accessed=False,test_accessed=False,
        raw_accessed=False,polarization_accessed=False,models_imported=False,model_forward=False,training=False,
        scientific_status=summary['final_status'],physics_f2_training_allowed=False,
        outputs_sha256=files,end_utc=datetime.now(timezone.utc).isoformat()))
    print(json.dumps(summary,indent=2,ensure_ascii=False))

if __name__=='__main__':
    try:
        main()
    except Exception as error:
        write('outputs/M1_TECHNICAL_STOP.json',dict(final_status='P3_TRAIN_ACCESS_BLOCKED',
              exception_type=type(error).__name__,reason=str(error),training=False,automatic_retry=False))
        raise
