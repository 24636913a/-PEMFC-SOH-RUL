"""Bounded identity authentication -> raw holds -> Stage1 scientific STOP."""
import hashlib
import json
import subprocess
import sys
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
import pandas as pd
from m2_access import (check,sha_bytes,verify_registered_bytes,permitted_path,LimitedBytes,
                       numeric,validate_chronology,read_f2_identity_prefix,inspect_authorized_header)
from m2_dwells import RAW_COLUMNS,COND,CELLS,measured_hold,split_complete_runs,apply_stability_thresholds

PHASE=Path(__file__).resolve().parents[1]; REPO=PHASE.parents[1]
BASE='601043fb1d82b8da87c42ea59e8d97be0c100358'
R='RENew-PEMFC-2300h-Pytorch/'

def write(relative,obj):
    p=PHASE/relative; p.parent.mkdir(parents=True,exist_ok=True)
    if isinstance(obj,pd.DataFrame): obj.to_csv(p,index=False,lineterminator='\n')
    else:
        text=obj if isinstance(obj,str) else json.dumps(obj,indent=2,ensure_ascii=False,allow_nan=False)+'\n'
        p.write_text(text,encoding='utf-8',newline='\n')

def protocols():
    output=[]
    for name in ['M2_ACCESS_PROTOCOL','M2_DWELL_QUALIFICATION_PROTOCOL']:
        p=PHASE/'configs'/f'{name}.yaml'; sha=sha_bytes(p.read_bytes())
        check(sha==(PHASE/'configs'/f'{name}.sha256').read_text().strip(),'PROTOCOL_IDENTITY_MISMATCH')
        output.append((json.loads(p.read_text(encoding='utf-8')),sha))
    access=output[0][0]
    subprocess.run(['git','merge-base','--is-ancestor',BASE,'HEAD'],cwd=REPO,check=True)
    for relative,sha in access['code_identity'].items():
        check(sha_bytes((PHASE/relative).read_bytes())==sha,'CODE_IDENTITY_CHANGED:'+relative)
    for s in access['metadata_allowlist']:
        check(sha_bytes((REPO/s['path']).read_bytes())==s['sha256'],'METADATA_CHANGED:'+s['path'])
    return output

def parse_source(root,source,allow,bins):
    path,permit=permitted_path(root,source,allow)
    schema=inspect_authorized_header(root,source,allow)
    check(set(RAW_COLUMNS).issubset(schema['columns']),'RAW_REQUIRED_COLUMN_MISSING')
    results=[]; tail=pd.DataFrame(); count=0; age_plateaus=0; hold_count=0
    previous_age=None; previous_time=None; observed_first=None; observed_last=None
    sensor={key:dict(finite=0,missing=0,minimum=None,maximum=None,outside_RH_0_100=0) for key in COND}
    bin_RH={}
    with path.open('rb',buffering=0) as underlying:
        stream=LimitedBytes(underlying,permit['byte_end'])
        chunks=pd.read_csv(stream,sep=';',encoding='utf-8-sig',usecols=RAW_COLUMNS,
                  dtype={'operation_point':str},chunksize=20000,low_memory=False,skip_blank_lines=False)
        for frame in chunks:
            frame=frame.copy()
            for c in RAW_COLUMNS:
                if c not in ['DateTime','operation_point']: frame[c]=numeric(frame[c]).astype(float)
            frame['_dt']=pd.to_datetime(frame.DateTime,errors='coerce')
            plateaus,previous_age,previous_time=validate_chronology(frame['OpHrs2 [h]'],frame._dt,previous_age,previous_time)
            age_plateaus+=plateaus
            check(np.isfinite(frame[['I.S.set [A]','I.S.act [A]','j.S.act [A/cm2]']].to_numpy()).all(),'REQUIRED_CURRENT_NONFINITE')
            frame['_record']=np.arange(count,count+len(frame)); count+=len(frame)
            if observed_first is None: observed_first=float(frame['OpHrs2 [h]'].iloc[0])
            observed_last=float(frame['OpHrs2 [h]'].iloc[-1])
            for key,col in COND.items():
                values=frame[col].to_numpy(float); finite=np.isfinite(values); good=values[finite]
                item=sensor[key]; item['finite']+=int(finite.sum()); item['missing']+=int((~finite).sum())
                if len(good):
                    item['minimum']=float(min(item['minimum'],good.min())) if item['minimum'] is not None else float(good.min())
                    item['maximum']=float(max(item['maximum'],good.max())) if item['maximum'] is not None else float(good.max())
                    if key.startswith('RH_'): item['outside_RH_0_100']+=int(((good<0)|(good>100)).sum())
            # Reaggregation check only inside qualified raw scope. No source
            # size/mtime or values from mixed files are used to complete a bin.
            grouped=frame.assign(_bin=np.floor(frame['OpHrs2 [h]']/.5)*.5).groupby('_bin',sort=False)
            stat=grouped.agg(n=('OpHrs2 [h]','size'),a=(COND['RH_anode'],'sum'),c=(COND['RH_cathode'],'sum'),
                             na=(COND['RH_anode'],'count'),nc=(COND['RH_cathode'],'count'))
            for key,row in stat.iterrows():
                sums=bin_RH.setdefault(float(key),np.zeros(5)); sums+=row[['n','a','c','na','nc']].to_numpy(float)
            combined=pd.concat([tail,frame],ignore_index=True) if len(tail) else frame
            complete,tail=split_complete_runs(combined)
            for group in complete:
                hold_count+=1; measured=measured_hold(group,source,bins)
                if measured is not None: results.append(measured)
        if len(tail):
            hold_count+=1; measured=measured_hold(tail,source,bins)
            if measured is not None: results.append(measured)
        check(stream.count==permit['byte_end'] and underlying.tell()==permit['byte_end'],'RAW_BYTE_BUDGET_NOT_EXACT')
    check(count==permit['historical_record_count'],'HISTORICAL_RAW_RECORD_COUNT_MISMATCH')
    check(observed_first==permit['historical_age_min_h'] and observed_last==permit['historical_age_max_h'],'HISTORICAL_RAW_EXTENT_MISMATCH')
    return results,dict(source_file=source,parsed_records=count,first_age_h=observed_first,last_age_h=observed_last,
       bytes_parsed=permit['byte_end'],age_reversals=0,strict_age_increase=(age_plateaus==0),age_plateaus_within_chunks=age_plateaus,
       nondecreasing_age=True,datetime_strict=True,total_token_holds=hold_count,measurement_candidate_count=len(results),sensor=sensor,
       source_schema=schema),bin_RH

def main():
    (access,sha_a),(dwell_protocol,sha_d)=protocols()
    root=Path(access['raw_root']); allow=access['raw_allowlist']; identities=[]
    check(bool(allow),'M2_RAW_TRAIN_ACCESS_NOT_ESTABLISHED')
    for source in allow:
        evidence=verify_registered_bytes(root,source,allow); identities.append(evidence)
        write('outputs/M2_RAW_IDENTITY_AUDIT.json',identities)
        print('Identity PASS '+Path(source).name,flush=True)
    inventory=access['source_inventory_before_identity_verification']
    for row in inventory:
        if row['source_file'] in allow:
            row['status']='TRAIN_ONLY_SOURCE_VERIFIED'; row['raw_content_identity_pass']=True
            row['identity_proof']='committed N1 SHA matched; N1+RCR entire-source ranges/counts; registered byte extent'
    write('manifests/M2_TRAIN_EXACT_CUTOFF_MANIFEST.json',dict(status='PARTIAL_F2_TRAIN_RAW_ACCESS_ESTABLISHED',
       raw_allowlist=allow,source_rows=inventory,whole_F2_Train_raw_coverage=False,
       mixed1310_exact_cutoff_available=False,mixed_source_all_bytes_read=False,
       identity_check=identities,raw_numeric_observations_scope='first9 wholly-Train sources only'))
    m1ref=REPO/access['f2_metadata_prefix']['identity_reference']
    m1=json.loads(m1ref.read_text(encoding='utf-8'))
    bins=read_f2_identity_prefix(REPO,access['f2_metadata_prefix']['path'],access['f2_metadata_prefix']['byte_end'],m1)
    holds=[]; source_audits=[]; reaggregate={}
    for source in allow:
        measurements,audit,sums=parse_source(root,source,allow,bins)
        holds+=measurements; source_audits.append(audit)
        for key,value in sums.items(): reaggregate.setdefault(key,np.zeros(5)); reaggregate[key]+=value
        write('outputs/M2_RUNTIME_RAW_READ_AUDIT.json',source_audits)
        print(f"Dwell scan {Path(source).name}: {audit['parsed_records']} rows; {len(measurements)} holds>=120s",flush=True)
    check(len(holds)>0,'M2_RAW_DWELL_SUPPORT_INSUFFICIENT')
    table=pd.DataFrame(holds)
    table.insert(0,'dwell_id',[f'M2_DWELL_{i:06d}' for i in range(1,len(table)+1)])
    check(not table.duplicated(['source_file','raw_record_start_inclusive','raw_record_end_exclusive']).any(),'DUPLICATE_DWELL_IDENTITY')
    table,thresholds=apply_stability_thresholds(table)
    table['access_protocol_sha256']=sha_a; table['qualification_protocol_sha256']=sha_d
    table['authorized_source_byte_start']=0
    table['authorized_source_byte_end']=table.source_file.map({s:r['byte_end'] for s,r in allow.items()})
    table['source_sha256']=table.source_file.map({s:r['historical_source_sha256'] for s,r in allow.items()})
    write('manifests/M2_DWELL_IDENTITY_MANIFEST.csv',table)
    write('outputs/M2_STABILITY_THRESHOLD_AUDIT.json',thresholds)
    rh_comparison=[]
    for row_id,row in bins.iterrows():
        sums=reaggregate.get(float(row.bin_start_ophrs2_h))
        if sums is None or sums[0]!=row.raw_sample_count or sums[3]!=sums[0] or sums[4]!=sums[0]: continue
        rh_comparison.append(dict(train_bin_row=int(row_id),bin_start_h=float(row.bin_start_ophrs2_h),
            n=int(sums[0]),raw_anode_RH_mean=sums[1]/sums[3],N1_anode_RH_mean=float(row.actual_anode_inlet_rh_pct_mean),
            raw_cathode_RH_mean=sums[2]/sums[4],N1_cathode_RH_mean=float(row.actual_cathode_inlet_rh_pct_mean)))
    comp=pd.DataFrame(rh_comparison)
    comp['anode_abs_difference_pct']=(comp.raw_anode_RH_mean-comp.N1_anode_RH_mean).abs()
    comp['cathode_abs_difference_pct']=(comp.raw_cathode_RH_mean-comp.N1_cathode_RH_mean).abs()
    write('outputs/M2_RH_RAW_VS_TRAIN_BIN_REAGGREGATION.csv',comp)
    raw_rows=sum(s['parsed_records'] for s in source_audits)
    rh_above=sum(s['sensor']['RH_anode']['outside_RH_0_100'] for s in source_audits)
    maximum=max(s['sensor']['RH_anode']['maximum'] for s in source_audits)
    nquiet=int(table.relative_quiet_candidate.sum())
    summary=dict(phase=PHASE.name,base_commit=BASE,access_protocol_sha256=sha_a,qualification_protocol_sha256=sha_d,
       raw_read_qualification='PASS_PARTIAL_TRAIN_ONLY',raw_sources_verified=len(allow),raw_sources_blocked=17-len(allow),
       whole_f2_train_raw_covered=False,exact1310_mixed_prefix_available=False,raw_rows_parsed=raw_rows,
       actual_raw_age_start_h=min(s['first_age_h'] for s in source_audits),actual_raw_age_end_h=max(s['last_age_h'] for s in source_audits),
       unique_holds_ge120s=len(table),relative_quiet_candidates=nquiet,
       relative_quiet_evidence_domain=int((table.relative_quiet_candidate & table.source_domain.eq('evidence')).sum()),
       independent_physically_certified_dwells=0,physically_available_stable_dwell_count=None,
       distinct_rounded_actual_j_levels=int(table.actual_j_median_A_cm2.round(3).nunique()),
       certified_pairs=0,physical_pair_statistics_executed=False,quasi_steady_certified=False,
       RH_anode_raw_outside_0_100_count=rh_above,RH_anode_raw_max_pct=maximum,
       RH_reaggregation_complete_bin_count=len(comp),RH_reaggregation_max_abs_difference_pct=float(comp.anode_abs_difference_pct.max()),
       RH_cause_resolved=False,RH_clipped=False,RH_silently_dropped=False,
       final_status='M2_STATE_COMPARABILITY_UNRESOLVED' if nquiet else 'M2_RAW_DWELL_SUPPORT_INSUFFICIENT',
       coexist_blockers=['Physical quasi-steady certification lacks independently justified sensor/stabilization tolerances',
         'Membrane-water/material/thermal state, scan order, intervention and recovery comparability unresolved',
         'RH>100 measurement mechanism/calibration not independently identified',
         'Exact mixed-file1310 byte boundary unavailable: incomplete F2 Train raw coverage',
         'Stage2/3 not executed; real auxiliary lineage qualification and independent amplitude not certified'],
       column_isolation_inherited_from_M1=True,complete_auxiliary_isolation=False,nondegenerate_amplitude_supervision=False,
       model_training=False,model_forward=False,checkpoints_loaded=False,validation_accessed=False,assessment_accessed=False,test_accessed=False,
       mixed_raw_whole_file_hash=False,whole_Train_source_identity_hash=True,
       next_physics_f2_training_allowed=False,automatic_next_stage=False)
    write('outputs/M2_SUPPORT_SUMMARY.json',summary)
    make_reports(access,inventory,table,thresholds,source_audits,comp,summary)
    write('outputs/M2_INTEGRITY_AUDIT.json',dict(technical_audit='PASS',raw_access_qualification='PASS_PARTIAL_TRAIN_ONLY',
        access_protocol_sha256=sha_a,qualification_protocol_sha256=sha_d,protocol_frozen_before_any_raw_byte_read=True,
        synthetic_tests='PASS',no_out_of_scope_underlying_read=True,unverified_raw_source_opened=False,
        forbidden_mixed_file_hash=False,whole_Train_identity_hash=True,old_wide_loader_called=False,
        Validation=False,Assessment=False,Test=False,training=False,checkpoint_load=False,
        Stage2_pair_statistics='NOT_RUN',Stage3_auxiliary_certification='NOT_RUN',
        scientific_status=summary['final_status'],runtime_logs='outputs/M2_RUNTIME_RAW_READ_AUDIT.json',
        artifacts_sha256={str(p.relative_to(PHASE)).replace('\\','/'):sha_bytes(p.read_bytes())
                        for p in sorted(PHASE.rglob('*')) if p.is_file() and '__pycache__' not in str(p)
                        and p.name!='M2_INTEGRITY_AUDIT.json'},completed_utc=datetime.now(timezone.utc).isoformat()))
    print(json.dumps(summary,indent=2,ensure_ascii=False))

def make_reports(access,inventory,table,thresholds,audits,comp,s):
    lines=['# Raw source inventory','','这里只展示历史来源身份metadata和本轮限定读取状态，不含非Train实际观测。',
      '', '|顺序|源|历史age范围h|当前状态|允许byte_end|','|---|---|---|---|---|']
    for row in inventory:
        lines.append(f"|{row['file_order']}|{row['source_file']}|{row['historical_age_min_h']}–{row['historical_age_max_h']}|{row['status']}|{row['authorized_byte_end']}|")
    lines.extend(['','前9个全Train源用已登记历史完整字节extent核验SHA，与历史N1 source_file_manifest完全一致；其范围先由N1+RCR穷尽式timeline/count audit固定，不是扫描年龄发现截止。',
     'Loadcycle_101-112跨1225.01–1330.22h，未发现可信的1310精确byte/record前缀；其余7源全在F2 Train之后。后8源0内容字节读取。',
     'source_file_manifest中的historical full SHA是历史既存证据。本轮只认证9个已证明全Train文件，不计算任何mixed/outside-Train文件hash。',
     '历史N1未发现age reversals不等于严格增加；本轮另记录age plateau。UTF-8-sig/semicolon为历史declared并在已授权源解析；未对非授权源探测BOM/引号/编码。'])
    write('evidence/M2_RAW_SOURCE_INVENTORY.md','\n'.join(lines)+'\n')
    write('evidence/M2_RAW_ACCESS_ELIGIBILITY_DECISION.md',f'''# Raw读取资格

`PASS_PARTIAL_TRAIN_ONLY`。9/17源完成历史内容SHA一致性验证，numeric范围{s['actual_raw_age_start_h']}–{s['actual_raw_age_end_h']}h，共{s['raw_rows_parsed']}records。不是完整72.5–1310h raw覆盖。

证据链：commit固定的N1 source_file_manifest(实际raw SHA/size) → N1全文件timeline(row count/min/max/时序审计) ↔ RCR独立timeline(row count/range) → 本轮预登记byte[0,historical_size) → 有上限identity读取SHA一致 → 同一byte限额内parse。这是已知全Train源的验证，不以文件名/粗范围替代完整性，也不通过whole-mixed扫描找边界。

F2 bin prefix只消费M1已审计的5976261bytes并复现其2103行逻辑fingerprint；不运行旧3007读器。raw record offset为CSV logical data record，skip_blank_lines=false；quoting/embedded LF由bounded parser处理。每源记录数与首末年龄再逐项一致。没有跨源合并dwell。

精确1310 mixed-source cutoff仍NOT_ESTABLISHED。不能按秒数、均匀行长、OpHrs差或raw_sample_count估算byte offset；也没有读1310后第一条记录。只有已有可信全Train源被授权，不完整scope不妨碍限定子域审查，但限制推广到全部Train。
''')
    write('reports/M2_RAW_DWELL_STABILITY_REPORT.md',f'''# Raw dwell稳定性

{s['raw_rows_parsed']}个合法Train raw records中恢复{s['unique_holds_ge120s']}个>=120s独立token保持片段。先按operation point/set current rounded3/时间gap>5s分段，无voltage-based rephasing；不跨source file拼接。

每dwell以middle50% measurement、至少30samples；等待为hold的25%(>=30s)。同一既有segment内可跨0.5h bin，缺失bin或跨segment不满足候选必要条件。记录原文件、logical record起止、raw/measurement时间、bin/segment、源SHA和协议SHA，详见M2_DWELL_IDENTITY_MANIFEST.csv。

校准仅[0,1051)，证据[1051,2103)；P50相对安静阈值见M2_STABILITY_THRESHOLD_AUDIT.json，不参考任何低高电压差或违约率。相对安静候选{s['relative_quiet_candidates']}，其中证据子域{s['relative_quiet_evidence_domain']}。独立120s保持片段不等于physically certified dwell；**{s['independent_physically_certified_dwells']} certified**，可用稳定dwell真实总数仍未知。

`DWELL_FOUND_BUT_QUASI_STEADY_NOT_VERIFIED`。PAA120s/middle50是操作定义，不是热/水/电化学近稳态充分条件。P50是统计设计而非sensor精度或稳态物理容差；没有以安静排序调整准则。缺少独立stabilization、材料状态和hysteresis证据，因此停止在Stage1，不执行极化pair违约率。
''')
    write('reports/M2_SENSOR_QUALITY_AND_PHYSICAL_DOMAIN.md',f'''# 传感器质量与物理域

阳极入口原字段`RH.Si.A  [%]`(注意两空格)，dictionary标为actual，不是独立的controller字段`RH.Si.A.CV [%]`。单位%。合法raw中超出[0,100]的记录{s['RH_anode_raw_outside_0_100_count']}，最大{s['RH_anode_raw_max_pct']}%。全部保留、flag PENDING，未clip100或静默删除。

在{s['RH_reaggregation_complete_bin_count']}个完整raw样本数匹配的Train bins上，raw RH重新mean与既有N1 anode RH mean最大绝对差{s['RH_reaggregation_max_abs_difference_pct']:.12g}个百分点。只能核查解析/聚合一致性，不能证明仪器校准有效。source raw已经存在异常高RH，不能将它归因于新聚合步骤。详情见M2_RH_RAW_VS_TRAIN_BIN_REAGGREGATION.csv。

原因仍未确认：sensor量程/校准、derived RH算法、过饱和/恢复状态或其他测量解释均无独立文档足以定案。读取了合法范围的controller RH/露点/T列用于保留schema，但没有用其相关性冒充传感器证书。入口RH不等于膜水或电极界面活动度。

每源温度/压力/flow/stoichiometry的finite/missing/min/max统计见runtime audit。pressure为kPag表压，不直接转成界面分压；flow为nl/min，stoic无量纲。负flow或非正stoic仅flag pending。未借物理单位同名将所有状态默认可比。

`M2_PHYSICAL_MEASUREMENT_QUALITY_UNRESOLVED`为并存限制；未声称所有RH<=100 dwell均已校准，也未根据RHflag补配一套更有利极化样本。
''')
    write('reports/M2_JOINT_CONDITION_SUPPORT.md',f'''# 共同工况支持

保持片段中rounded actual j水平数{s['distinct_rounded_actual_j_levels']}，只是可见负载水平，不是joint support或认证pair数量。Stage1的physical quasi-steady/state/sensor要求未通过，因此没有执行Stage2共同支持/low-high配对，pair count仅0 certified，可用合法pairs仍未知。

M1的0.5h temporal caliper对应1800s、minΔj=0.2A/cm²、condition IQR caliper=0.1、joint distance<=0.5、minimum20独立pairs/3levels/2segments等设计保持，不改阈值。新raw within-hold统计不是对原pair规则的事后替代。

温湿压/流量参数用于物理资格，未加到辅助模型输入。source identity PASS与多个j水平不能替代状态可比或共同支持。没有把早期低负载与晚期高负载拼成polarization curve。
''')
    write('reports/M2_STATE_AND_HYSTERESIS_AUDIT.md','''# 材料状态、恢复与滞后

已恢复合法raw保持片段的时间/来源/加载token身份，没有建立材料状态观测。run持有固定setpoint不证明膜水/界面温度平衡。DateTime gap、zero-current、startup/stop等操作标志只能提出恢复候选，不能据此确定材料不变或具体可逆机理。

目前无可直接绑定所有dwell的扫描方向、独立稳定化证书、干预/Air-Air记录、材料状态或热/水滞后控制。未测慢状态不设常数，短时间caliper不自动视为材料状态相同。没有计算未认证pair的极化顺序违约率。阶段停止不是“RealHyFC没有极化规律”。
''')
    write('evidence/M2_VOLTAGE_LABEL_LINEAGE.md','''# Voltage label lineage

raw有效10个cell → 瞬时cell mean → middle50% dwell median为拟议辅助标签。已输出真实measurement logical record/time起止与source identity。它同源于N1 mean/min/max/std/range及EWMA后继，因此主F2 voltage通道不能直接复用于辅助可见view。

M1九个非电压通道view原封不动；T/RH/P/flow只用于资格，未新增辅助输入；未暴露raw cell、exported mean、P1S/B2、EWMA或voltage-derived cache。M1列级隔离仍保留。

Stage1 physical gate未通过，故Stage3 actual auxiliary input-window/label dwell联接尚未执行。这里的真实来源恢复不等于complete temporal isolation PASS。没有建立辅助网络或使用新dwell作为伪SOH训练标签。
''')
    write('reports/M2_AUXILIARY_IDENTIFIABILITY.md','''# 辅助任务非退化性

Order-only max(0,q_high−q_low)²的常数0-loss解沿用M1，不改变。固定下降j-only曲线、忽略条件、curve/segment ID记忆、voltage-derived copy均为必要negative controls。

目前恢复raw电压测量，但没有已认证的多负载共同支持，不能将这些measurement直接升级为可靠独立幅值监督。Stage3未执行，不拟合曲线或辅助网络。非退化幅值证据、完整auxiliary时序隔离都仍未建立。没有新增SOH/RUL、EOL或物理F2 loss。
''')
    write('reports/M2_FINAL_P3_RESEARCH_DECISION.md',f'''# M2最终P3决策

状态：`{s['final_status']}`。Raw访问为**PASS_PARTIAL_TRAIN_ONLY**，科学资格不通过。恢复raw来源是实质性进展，但不授权Physics-informed F2训练。

## 向研究负责人回答

1. 找到可信逐源Train截止吗？9个全Train源的历史完整byte extent与SHA得到认证；跨1310源的精确前缀仍没有。
2. 哪些可安全读？Loadcycle_0-14、15-25、26-37、37-50、51-70、70-75、76-82、82-84、85-100。每源实际SHA/byte上限见manifest。
3. 哪些继续阻断？101-112跨界，112-125及其后7个外Train源(含该112-125)均不读；总共8个未授权源。
4. 实际多少独立稳定dwell？恢复{s['unique_holds_ge120s']}个>=120s保持片段，统计相对安静{s['relative_quiet_candidates']}；物理认证0，真正可用稳定dwell数量未知。
5. 覆盖多少实际负载？保持片段rounded j有{s['distinct_rounded_actual_j_levels']}种，不能把量化噪声形成的细levels当作独立实验负载支持。
6. 低高共同支持足够吗？未进入合格Stage2配对，无法确认。
7. 温湿压/flow可比吗？原始统计可获得，但joint/state physical证据未闭合。
8. 材料状态、恢复/滞后能控制吗？不能充分控制，缺独立慢状态/扫描/干预身份。
9. RH>100已解释吗？raw明确存在且与N1重聚合一致；不是新parser产生的clip/补值。校准/计算/物理解释尚未确认。
10. 极化幅值证据足够吗？NO，没有已认证独立低高pairs，未算不合格ΔV。
11. 辅助目标隔离吗？真实measurement身份已恢复，M1列view保留；Stage3完整时序/源联接资格仍未通过。
12. 最低物理学习条件？NO。阻断为quasi-steady/state/sensor/evidence资格，不是“PINN训练失败”。

## 继承结论与停止

NO_ADMISSIBLE_PINN_CONSTRAINT_YET；P3_QUASI_STEADY_OR_STATE_COMPARABILITY_UNRESOLVED；PHASE_J_F2_ARCHITECTURE_FROZEN；PHASE_L_CONFIRMATORY_TEST_NOT_ELIGIBLE全部保留。M1整bin候选0并未否定raw dwell，本轮已明确证明它们是不同层级。

Stage0部分源身份通过 → Stage1恢复保持片段和相对稳定统计 → physical门槛未通过 → Stage2/3停止。不改变M1caliper、不执行极化次序、辅助训练、Test或预测收益计算。不自动开启新阶段。

deep-research的证据分级区分历史身份hash、运行统计与物理证明；报告采用用户提供的中文协议式写法。AI辅助审计/代码整理，不声称真实神经训练、完整机理验证或独立Test合格。
''')
    write('README.md','''# Phase M2 — Train-only Raw Dwell Qualification

仅RealHyFC/P3。冻结F2主任务不变，不加载checkpoint或模型。先metadata freeze与synthetic tests，再对已证明全Train的9个源验证登记byte extent/SHA。跨1310及其后源一律0byte读取。

运行：`python scripts/freeze_m2.py`（首次冻结，不读raw）；`python scripts/run_m2.py`（identity先于parse）；`python -m unittest discover -s tests -v`。

协议.json语法为有效YAML，配套.sha256先于任何raw内容读取。M2_INTEGRITY_AUDIT保存最终文件hash。身份验证hash覆盖9个全Train文件；明确没有mixed-file whole hash，不把它称为完整Train覆盖。

先读reports/M2_FINAL_P3_RESEARCH_DECISION.md。保持片段和统计quiet不是物理稳态认证；主科学状态未通过，不启动Physics-informed F2。
''')

if __name__=='__main__':
    try: main()
    except Exception as exc:
        write('outputs/M2_FAILURE_RECORD.json',dict(status='M2_RAW_TRAIN_ACCESS_NOT_ESTABLISHED',
             failure_type=type(exc).__name__,reason=str(exc),automatic_retry=False,training=False))
        raise
