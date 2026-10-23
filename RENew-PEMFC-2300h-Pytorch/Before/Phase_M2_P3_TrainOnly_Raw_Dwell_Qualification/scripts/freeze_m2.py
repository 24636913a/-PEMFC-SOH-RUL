"""Metadata-only Stage0 plan freeze. No raw content open in this program."""
import csv
import json
import subprocess
import sys
from datetime import datetime,timezone
from pathlib import Path
import yaml
from m2_access import check,sha_bytes

PHASE=Path(__file__).resolve().parents[1]; REPO=PHASE.parents[1]
R='RENew-PEMFC-2300h-Pytorch/'
BASE='601043fb1d82b8da87c42ea59e8d97be0c100358'
M1=R+'Phase_M1_P3_TrainOnly_Polarization_Support_Qualification/'
N1=R+'Phase_N1_Data_Foundation/'
RCR=R+'Phase_RCR_Raw_Resolution_Cycle_Reconstruction_Audit/'
STATIC=[N1+'outputs/audits/source_file_manifest.csv',N1+'outputs/audits/loadcycle_source_timeline.csv',
 RCR+'outputs/RAW_TIMELINE_AUDIT.csv',RCR+'SOURCE_MANIFEST.yaml',
 N1+'scripts/phase_n1_build_foundation.py',N1+'scripts/phase_n1_common.py',N1+'scripts/phase_n1_verify.py',
 N1+'configs/phase_n1_config.yaml',N1+'outputs/causal_feature_tables/causal_feature_dictionary.csv',
 RCR+'scripts/run_raw_cycle_reconstruction_audit.py',
 M1+'configs/M1_FROZEN_ACCESS_PROTOCOL.yaml',M1+'configs/M1_FROZEN_SUPPORT_PROTOCOL.yaml',
 M1+'outputs/M1_RUNTIME_ACCESS_AUDIT.json',M1+'outputs/M1_AUXILIARY_ISOLATION_AUDIT.json',
 M1+'evidence/M1_SYNTHETIC_TEST_LOG.txt',M1+'reports/M1_JOINT_SUPPORT_AND_STABILITY.md',
 M1+'reports/M1_PHYSICAL_MEASUREMENT_SEMANTICS.md',M1+'reports/M1_FINAL_P3_RESEARCH_DECISION.md',
 M1+'outputs/M1_CONDITION_SUPPORT_REPORT.csv',M1+'scripts/m1_core.py',
 R+'Phase_M_F2_PINN_Physics_Feasibility/reports/M_FINAL_PHYSICS_RESEARCH_DECISION.md',
 R+'Phase_M_F2_PINN_Physics_Feasibility/reports/M_EQUATION_IDENTIFIABILITY_AUDIT.md',
 R+'Phase_G_STOP_Physics_Consistent_Future_Conditions/PHYSICAL_RELATION_AUDIT.md',
 R+'Phase_PAA_P1_StateToken_PreAnchor_Alignment/configs/PREREGISTERED_PAA_PROTOCOL.yaml',
 R+'Phase_PAA_P1_StateToken_PreAnchor_Alignment/scripts/run_p1_state_token_preanchor_audit.py',
 R+'Phase_PAA_P1_StateToken_PreAnchor_Alignment/SOURCE_MANIFEST.yaml',
 R+'Phase_PAV_PAA_Anchor_Validity_Qualification/docs/PAA_CLAIM_BOUNDARY.md',
 R+'Phase_L_Test_Eligibility_and_Reproducibility_Freeze/reports/L_TEST_EXPOSURE_FORENSIC_REPORT.md',
 R+'Phase_L_Test_Eligibility_and_Reproducibility_Freeze/manifests/L_FROZEN_MODEL_MANIFEST.json',
 R+'Phase_AHC_TV_One_Time_Late_Life_Test_Protocol/configs/AHC_TV_RUNTIME_SOURCE_REGISTRY.yaml',
 R+'Phase_AHC_TV_One_Time_Late_Life_Test_Protocol/scripts/ahc_tv_raw_alignment_core.py',
 R+'Phase_CHM_Cycle_Aligned_Sparse_Anchor_Health_Model/configs/CHM_SOURCE_ALLOWLIST.yaml',
 R+'Phase_MQ_RealHyFC_Mechanism_Qualification/configs/ATTEMPT_2_LOCAL_SOURCE_ALLOWLIST.yaml',
 R+'Phase_RHA_Raw_Operation_Point_Health_Alignment/SOURCE_MANIFEST.yaml',
 R+'Phase_RHA_P1_Posthoc_Single_Point_Identifiability/SOURCE_MANIFEST.yaml']

def write(relative,value):
    p=PHASE/relative; p.parent.mkdir(parents=True,exist_ok=True)
    payload=value if isinstance(value,str) else json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n'
    p.write_text(payload,encoding='utf-8',newline='\n')
    return sha_bytes(p.read_bytes())

def main():
    check(subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip()==BASE,'BASE_COMMIT_CHANGED')
    check(subprocess.check_output(['git','branch','--show-current'],cwd=REPO,text=True).strip()=='main','MAIN_REQUIRED')
    check(not (PHASE/'configs/M2_ACCESS_PROTOCOL.yaml').exists(),'PROTOCOL_ALREADY_FROZEN')
    changes=subprocess.check_output(['git','diff','--name-only',BASE],cwd=REPO,text=True).splitlines()
    check(not changes,'TRACKED_TREE_NOT_CLEAN')
    test=subprocess.run([sys.executable,'-m','unittest','discover','-s',str(PHASE/'tests'),'-v'],cwd=PHASE,text=True,capture_output=True)
    write('evidence/M2_SYNTHETIC_FIREWALL_TEST_LOG.txt',test.stdout+test.stderr)
    check(test.returncode==0,'SYNTHETIC_FIREWALL_FAILURE')
    sources=[]
    for relative in STATIC:
        p=REPO/relative; check(p.is_file(),'MISSING_METADATA:'+relative)
        sources.append(dict(path=relative,sha256=sha_bytes(p.read_bytes()),
          last_modified_commit=subprocess.check_output(['git','log','-1','--format=%H','--',relative],cwd=REPO,text=True).strip(),
          access_class='STATIC_CODE_OR_SOURCE_IDENTITY_METADATA_OR_HISTORICAL_REPORT',runtime_import=False))
    registry=yaml.safe_load((REPO/(RCR+'SOURCE_MANIFEST.yaml')).read_text(encoding='utf-8'))
    raw_root=Path(registry['raw_source']['root'])
    with (REPO/(N1+'outputs/audits/source_file_manifest.csv')).open(encoding='utf-8-sig',newline='') as f:
        hashes={row['relative_path']:row for row in csv.DictReader(f) if row['source_role']=='durability_loadcycle_csv'}
    with (REPO/(N1+'outputs/audits/loadcycle_source_timeline.csv')).open(encoding='utf-8-sig',newline='') as f:
        n1={row['filename']:row for row in csv.DictReader(f)}
    with (REPO/(RCR+'outputs/RAW_TIMELINE_AUDIT.csv')).open(encoding='utf-8-sig',newline='') as f:
        rcr={row['source_file']:row for row in csv.DictReader(f)}
    allow={}; inventory=[]
    for order,entry in enumerate(registry['raw_source']['files'],1):
        relative=entry['path']; path=raw_root/relative; name=Path(relative).name
        h=hashes[relative]; t=n1[name]; old=rcr[relative]
        age_min=float(t['ophrs2_min_h']); age_max=float(t['ophrs2_max_h'])
        present=path.is_file(); size=path.stat().st_size if present else None
        whole_train=72.5<=age_min<=age_max<1310.
        agreement=(int(t['raw_rows'])==int(old['raw_rows_seen'])==int(old['development_rows_retained']) and
                    float(old.get('development_age_start_h') or -1)==age_min and
                    float(old.get('development_age_end_h') or -1)==age_max)
        item=dict(source_file=relative,file_order=order,historical_age_min_h=age_min,historical_age_max_h=age_max,
            historical_record_count=int(t['raw_rows']),historical_size_bytes=int(h['size_bytes']),
            current_metadata_size_bytes=size,size_agrees=size==int(h['size_bytes']),present=present,
            historical_source_sha256=h['sha256'],historical_modified_time_utc=h['modified_time_utc'],
            n1_age_reversals=int(t['ophrs2_reversal_count']),n1_timestamp_reversals=int(t['timestamp_reversal_count']),
            n1_duplicate_timestamps=int(t['duplicate_timestamp_count']),strict_age_increase_proven=False,
            historical_record_to_byte_cutoff_1310=None,historical_prefix_sha256_1310=None,
            encoding_historical='utf-8-sig',separator_historical=';',actual_BOM_and_quote_layout_not_yet_read=True)
        if not present:
            item.update(status='SOURCE_UNAVAILABLE',authorized_byte_end=0,reason='Known historical path unavailable')
        elif whole_train and agreement and size==int(h['size_bytes']):
            allow[relative]=dict(preauthorized_domain='ENTIRE_SOURCE_PROVEN_TRAIN',byte_start=0,byte_end=int(h['size_bytes']),
                historical_age_min_h=age_min,historical_age_max_h=age_max,historical_record_count=int(t['raw_rows']),
                historical_source_sha256=h['sha256'],n1_rcr_range_and_count_agree=agreement,historical_manifest_committed=True)
            item.update(status='IDENTITY_UNRESOLVED',authorized_byte_end=int(h['size_bytes']),
                 reason='Qualified for bounded identity verification using committed full-source hash + two exhaustive historical timeline/count audits; numeric parsing requires hash PASS')
        else:
            scope='CROSSES_1310_H' if age_min<1310<=age_max else 'ENTIRELY_OUTSIDE_F2_TRAIN'
            item.update(status='MIXED_SOURCE_NOT_AUTHORIZED',authorized_byte_end=0,scope=scope,
                 reason='No independently frozen exact1310 prefix byte/hash identity; filename/size/time alone never authorizes raw content')
        inventory.append(item)
    check(len(inventory)==17,'RAW_SOURCE_REGISTRY_COUNT_MISMATCH')
    m1=json.loads((REPO/(M1+'outputs/M1_RUNTIME_ACCESS_AUDIT.json')).read_text())
    support=json.loads((REPO/(M1+'configs/M1_FROZEN_SUPPORT_PROTOCOL.yaml')).read_text(encoding='utf-8'))
    code={str(p.relative_to(PHASE)).replace('\\','/'):sha_bytes(p.read_bytes())
         for d in ['scripts','tests'] for p in sorted((PHASE/d).glob('*.py'))}
    access=dict(phase=PHASE.name,base_commit=BASE,frozen_utc=datetime.now(timezone.utc).isoformat(),
      status='FROZEN_BEFORE_RAW_IDENTITY_READ_OR_STATISTICS',raw_root=str(raw_root),
      raw_allowlist=allow,source_inventory_before_identity_verification=inventory,
      interpretation='Full-file Train extent already established by historical content-bound metadata; bounded byte hash authenticates it, not discovering cutoff by scan. No mixed-source hash or boundary probe.',
      exact_1310_mixed_source_prefix_found=False,partial_train_source_coverage_permitted=True,
      f2_train_rows=[0,2103],f2_bin_start_h=72.5,f2_bin_end_h=1310.,f2_legal_windows=1380,
      f2_metadata_prefix=dict(path=N1+'outputs/causal_feature_tables/causal_features_30min.csv',byte_end=m1['bytes_consumed'],
                              identity_reference=M1+'outputs/M1_RUNTIME_ACCESS_AUDIT.json'),
      metadata_allowlist=sources,code_identity=code,synthetic_firewall_tests='PASS',
      prohibited=['raw sources absent from allowlist','raw next-record boundary discovery','mixed raw hash/scan','unbounded raw read',
                   'Validation/Assessment/Test values','old wide readers','checkpoint loading','any model training/forward'],
      raw_whole_file_hash_policy='ONLY already-qualified wholly-Train source byte extents; forbidden for mixed sources',
      no_raw_values_read_by_freeze=True)
    dwell=dict(phase=PHASE.name,frozen_before_raw_read=True,
      discovery='operation_point + set current rounded3; split gaps>5s/source boundary; never merge across files',
      hold_wall_duration_s_min=120,window='middle50%; settling wait=25% duration, at least30s',minimum_window_samples=30,
      cross_bin_allowed='only same frozen continuous Train segment, no missing intermediate bin',
      calibration_rows=[0,1051],evidence_rows=[1051,2103],domain_crossing_holds_excluded_from_formal_support=True,
      statistical_quiet_thresholds='deterministic calibration P50 of eight within-dwell metrics; floor1e-8; no outcome/violation tuning',
      statistical_quiet_is_physical_quasi_steady=False,
      qualification_requires=['sensor domain/units','calibration accuracy or independently justified stabilization',
                             'membrane water/thermal/material comparability','scan/loading hysteresis/recovery control'],
      RH_gt100_policy='retain and flag PENDING; do not clip or silently drop; do not infer oversaturation or calibration cause',
      M1_matching_design=support['prospective_matching'],M1_minimum_support=support['minimum_support'],
      time_conversion=dict(max_temporal_distance_h=.5,max_temporal_distance_s=1800,min_delta_j_A_cm2=.2),
      auxiliary_view=support['isolation']['visible_channels'],no_added_T_RH_P_or_voltage_inputs=True,
      order_or_pair_amplitude_calculated_only_after_all_physical_gates=True,
      negative_controls=['constant voltage','fixed descending j-only line','ignore conditions','curve-ID memorization','voltage-copy'],
      stop='Stage1 failure -> no Stage2 pair selection or Stage3 auxiliary certification; no model training; no automatic next phase')
    for name,value in [('M2_ACCESS_PROTOCOL',access),('M2_DWELL_QUALIFICATION_PROTOCOL',dwell)]:
        sha=write('configs/'+name+'.yaml',value)
        write('configs/'+name+'.sha256',sha+'\n')
    write('manifests/M2_TRAIN_EXACT_CUTOFF_MANIFEST.json',dict(status='FROZEN_IDENTITY_CHECK_PENDING',
      mixed1310_exact_cutoff_available=False,source_rows=inventory,raw_allowlist=allow,
      numeric_rows_parsed=0,byte_cutoffs_are_historical_whole_Train_sources_not_inferred_offsets=True))
    print('M2 Stage0 metadata freeze PASS; bounded identity candidates='+str(len(allow))+'; raw data not yet read.')

if __name__=='__main__': main()
