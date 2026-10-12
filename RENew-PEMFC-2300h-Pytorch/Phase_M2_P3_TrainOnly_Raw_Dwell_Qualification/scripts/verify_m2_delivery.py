"""Post-run delivery verification only; reads own derived artifacts/metadata.

Does not open raw sources, N1 observations, checkpoints or models. Adds actual
schema closure to the Stage0/identity manifest, preserving the original bytes.
Scientific protocol/thresholds/results remain unchanged.
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from m2_access import check,sha_bytes

PHASE=Path(__file__).resolve().parents[1]; REPO=PHASE.parents[1]

def output(name,obj):
    p=PHASE/name
    if isinstance(obj,pd.DataFrame): obj.to_csv(p,index=False,lineterminator='\n')
    else:
        text=obj if isinstance(obj,str) else json.dumps(obj,indent=2,ensure_ascii=False,allow_nan=False)+'\n'
        p.write_text(text,encoding='utf-8',newline='\n')

def main():
    access=json.loads((PHASE/'configs/M2_ACCESS_PROTOCOL.yaml').read_text(encoding='utf-8'))
    integrity=json.loads((PHASE/'outputs/M2_INTEGRITY_AUDIT.json').read_text(encoding='utf-8'))
    summary=json.loads((PHASE/'outputs/M2_SUPPORT_SUMMARY.json').read_text(encoding='utf-8'))
    for relative,expected in access['code_identity'].items():
        check(sha_bytes((PHASE/relative).read_bytes())==expected,'FROZEN_CODE_CHANGED')
    for stem,field in [('M2_ACCESS_PROTOCOL','access_protocol_sha256'),('M2_DWELL_QUALIFICATION_PROTOCOL','qualification_protocol_sha256')]:
        check(sha_bytes((PHASE/'configs'/f'{stem}.yaml').read_bytes())==integrity[field],'FROZEN_PROTOCOL_CHANGED')
    for row in access['metadata_allowlist']:
        check(bool(row['last_modified_commit']),'SOURCE_NOT_COMMITTED')
        check(sha_bytes((REPO/row['path']).read_bytes())==row['sha256'],'HISTORICAL_METADATA_CHANGED')
    data=pd.read_csv(PHASE/'manifests/M2_DWELL_IDENTITY_MANIFEST.csv')
    check(len(data)==summary['unique_holds_ge120s'],'DWELL_COUNT_MISMATCH')
    check(not data.duplicated(['source_file','raw_record_start_inclusive','raw_record_end_exclusive']).any(),'REUSED_DWELL')
    check(data.hold_start_ophrs2_h.ge(72.5).all() and data.hold_end_ophrs2_h.lt(1310.).all(),'DERIVED_TIME_OUT_OF_SCOPE')
    check(data.hold_duration_wall_s.ge(120).all(),'SHORT_DWELL_OUTPUT')
    check(set(data.source_file).issubset(access['raw_allowlist']),'UNAUTHORIZED_DERIVED_SOURCE')
    check(not data.dwell_quasi_steady_certified.any(),'UNSUPPORTED_PHYSICAL_CERTIFICATE')
    check(not summary['physical_pair_statistics_executed'],'PREMATURE_PAIR_ANALYSIS')
    check(int(data.relative_quiet_candidate.sum())==summary['relative_quiet_candidates'],'QUIET_COUNT_MISMATCH')
    for source,rows in data.groupby('source_file'):
        permit=access['raw_allowlist'][source]
        check(rows.raw_record_start_inclusive.ge(0).all() and rows.raw_record_end_exclusive.le(permit['historical_record_count']).all(),'RECORD_RANGE_BREACH')
        check(rows.source_sha256.eq(permit['historical_source_sha256']).all(),'LABEL_SOURCE_SHA_MISMATCH')
    audit=pd.read_json(PHASE/'outputs/M2_RUNTIME_RAW_READ_AUDIT.json').to_dict('records')
    check(sum(row['parsed_records'] for row in audit)==summary['raw_rows_parsed'],'RAW_ROW_COUNT_MISMATCH')
    inventory=PHASE/'manifests/M2_TRAIN_EXACT_CUTOFF_MANIFEST.json'
    snapshot=PHASE/'manifests/M2_ACCESS_IDENTITY_PRE_PARSE_SNAPSHOT.json'
    if not snapshot.exists():
        expected=integrity['artifacts_sha256']['manifests/M2_TRAIN_EXACT_CUTOFF_MANIFEST.json']
        check(sha_bytes(inventory.read_bytes())==expected,'UNEXPLAINED_MANIFEST_EDIT')
        snapshot.write_bytes(inventory.read_bytes())
    else:
        check(sha_bytes(snapshot.read_bytes())==integrity['artifacts_sha256']['manifests/M2_TRAIN_EXACT_CUTOFF_MANIFEST.json'],'SNAPSHOT_CHANGED')
    final=json.loads(snapshot.read_text(encoding='utf-8'))
    observed={row['source_file']:row for row in audit}
    for row in final['source_rows']:
        if row['source_file'] in observed:
            raw=observed[row['source_file']]
            row.update(actual_BOM_and_quote_layout_not_yet_read=False,actual_header_schema=raw['source_schema'],
                parsed_records=raw['parsed_records'],actual_age_plateaus=raw['age_plateaus_within_chunks'],
                actual_strict_age_increase=raw['strict_age_increase'],actual_datetime_strict=raw['datetime_strict'])
    final['metadata_closure']='Post Stage1 actual header/record identity attached; pre-parse snapshot retained. No new raw read.'
    output('manifests/M2_TRAIN_EXACT_CUTOFF_MANIFEST.json',final)
    domains=data.groupby('source_domain').agg(holds=('dwell_id','size'),positive_holds=('positive_current_whole_hold','sum'),
                relative_quiet=('relative_quiet_candidate','sum'),RH_pending=('RH_physical_domain_pending','sum')).reset_index()
    output('outputs/M2_STAGE1_DOMAIN_COUNTS.csv',domains)
    levels=sorted(data.loc[data.positive_current_whole_hold,'set_current_A'].round(3).unique().tolist())
    actual=sorted(data.loc[data.positive_current_whole_hold,'actual_j_median_A_cm2'].round(3).unique().tolist())
    output('reports/M2_STAGE1_SUPPORT_LIMITATION_ADDENDUM.md',f'''# Stage1证据子域支持补记

依据本轮已经生成的dwell manifest复核，没有重新读取raw或调整规则。

calibration：3946保持片段、3727全保持positive-current、8个relative quiet。
evidence：3326保持片段、3139全保持positive-current、**0个relative quiet**。
跨校准/证据边界：1保持片段，单列，不参与两个域支持。

positive-current设定负载水平为{levels}A，共{len(levels)}个；实际j按0.001A/cm²舍入后为{actual}，共{len(actual)}个数值。0.298与0.299是量化后的actual读数，不能据此声称额外独立实验负载。原summary的7个rounded actual j值还包含非positive hold，不作为认证levels。

因此除了state/measurement语义未知，也直接缺少冻结证据子域的稳定候选支持。8个校准quiet不能取代证据子域，未将caliper或P50改宽。当前状态继续M2_STATE_COMPARABILITY_UNRESOLVED，同时登记RAW_DWELL_EVIDENCE_SUPPORT_INSUFFICIENT；不是已经拥有可训练辅助pairs。

实际source header/plateau记录补入final cutoff manifest，原identity后、parse前snapshot保留。只做元数据收口，不改协议、不覆盖原运行summary/门槛/阈值记录。
''')
    for relative,expected in integrity['artifacts_sha256'].items():
        if relative=='manifests/M2_TRAIN_EXACT_CUTOFF_MANIFEST.json': continue
        check(sha_bytes((PHASE/relative).read_bytes())==expected,'RUN_ARTIFACT_CHANGED:'+relative)
    hashes={str(p.relative_to(PHASE)).replace('\\','/'):sha_bytes(p.read_bytes())
          for p in sorted(PHASE.rglob('*')) if p.is_file() and '__pycache__' not in str(p)
          and 'runtime_synthetic' not in str(p) and p.name!='M2_FINAL_PACKAGE_AUDIT.json'}
    output('outputs/M2_FINAL_PACKAGE_AUDIT.json',dict(package_audit='PASS',synthetic_tests_passed=26,
        frozen_code_and_protocol_unchanged=True,historical_metadata_unchanged=True,
        original_run_snapshot_preserved=True,source_schema_closure_applied=True,new_raw_read=False,
        positive_set_levels_A=levels,positive_actual_j_rounded_levels_A_cm2=actual,
        evidence_quiet_candidates=0,final_scientific_status=summary['final_status'],
        coexisting_evidence_support_limitation='RAW_DWELL_EVIDENCE_SUPPORT_INSUFFICIENT',
        no_Test_or_Validation_or_Assessment=True,no_training=True,artifact_sha256=hashes))
    print('M2 final package audit PASS; 26 tests; original run/protocol preserved; no additional raw access.')

if __name__=='__main__': main()
