"""Stage 0: synthetic tests and identity freeze; never opens the N1 data file."""
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from m1_core import BASE, RESEARCH, N1, CONDITIONS, SAFE_COLUMNS, AUX_NONVOLTAGE, metadata_hash, digest_bytes, require

PHASE = Path(__file__).resolve().parents[1]
REPO = PHASE.parents[1]
STATIC = [
 RESEARCH+'/Phase_H_Determinism_MultiSeed_Stability/configs/H_FROZEN_PROTOCOL.yaml',
 RESEARCH+'/Phase_TFT_FullFeature_CausalFilter_Comparison/outputs/E_SPLIT_MANIFEST.json',
 RESEARCH+'/Phase_TFT_FullFeature_CausalFilter_Comparison/outputs/E_COMMON_SCALER.json',
 RESEARCH+'/Phase_TFT_FullFeature_CausalFilter_Comparison/scripts/run_e_comparison.py',
 RESEARCH+'/Phase_TFT_SingleSplit_Feature_Filter_Ablation/scripts/run_c_ablation.py',
 RESEARCH+'/Phase_N1_Data_Foundation/scripts/phase_n1_build_foundation.py',
 RESEARCH+'/Phase_N1_Data_Foundation/scripts/phase_n1_common.py',
 RESEARCH+'/Phase_N1_Data_Foundation/configs/phase_n1_config.yaml',
 RESEARCH+'/Phase_N1_Data_Foundation/outputs/causal_feature_tables/causal_feature_dictionary.csv',
 RESEARCH+'/Phase_N1_Data_Foundation/outputs/causal_feature_tables/causal_feature_availability.csv',
 RESEARCH+'/Phase_G_STOP_Physics_Consistent_Future_Conditions/PHYSICAL_RELATION_AUDIT.md',
 RESEARCH+'/Phase_PAA_P1_StateToken_PreAnchor_Alignment/configs/PREREGISTERED_PAA_PROTOCOL.yaml',
 RESEARCH+'/Phase_PAA_P1_StateToken_PreAnchor_Alignment/scripts/run_p1_state_token_preanchor_audit.py',
 RESEARCH+'/Phase_PAV_PAA_Anchor_Validity_Qualification/docs/PAA_CLAIM_BOUNDARY.md',
 RESEARCH+'/Phase_CSHSE_Causal_SameCycle_Health_State_Estimation/reports/INVALID_PROTOCOL_BREACH_RECORD.md',
 RESEARCH+'/Phase_M_F2_PINN_Physics_Feasibility/reports/M_EQUATION_IDENTIFIABILITY_AUDIT.md',
 RESEARCH+'/Phase_M_F2_PINN_Physics_Feasibility/reports/M_FINAL_PHYSICS_RESEARCH_DECISION.md',
 RESEARCH+'/Phase_M_F2_PINN_Physics_Feasibility/evidence/M_PHYSICS_LITERATURE_LEDGER.md',
 RESEARCH+'/Phase_M_F2_PINN_Physics_Feasibility/evidence/M_SOURCE_IDENTITY_MANIFEST.json',
 RESEARCH+'/Phase_L_Test_Eligibility_and_Reproducibility_Freeze/manifests/L_FROZEN_MODEL_MANIFEST.json',
]
WEIGHTS = [RESEARCH+f'/Phase_H_Determinism_MultiSeed_Stability/outputs/checkpoints/H-F{f}-Seed{s}_best.pt'
           for f in [0,1,2] for s in [42,52,62]]

def dump(relative, obj):
    path = PHASE/relative
    path.parent.mkdir(parents=True, exist_ok=True)
    # JSON is a strict YAML subset, intentionally parsed without custom tags.
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False)+'\n', encoding='utf-8')
    return digest_bytes(path.read_bytes())

def main():
    head = subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip()
    require(head == BASE, 'BASE_COMMIT_CHANGED')
    require(subprocess.check_output(['git','branch','--show-current'],cwd=REPO,text=True).strip()=='main','MAIN_REQUIRED')
    changed = subprocess.check_output(['git','diff','--name-only',BASE],cwd=REPO,text=True).splitlines()
    require(all(x.startswith(str(PHASE.relative_to(REPO)).replace('\\','/')+'/') for x in changed), 'HISTORICAL_TRACKED_CHANGE')
    require(not (PHASE/'configs/M1_FROZEN_ACCESS_PROTOCOL.yaml').exists(), 'PROTOCOL_ALREADY_FROZEN')
    run = subprocess.run([sys.executable,'-m','unittest','discover','-s',str(PHASE/'tests'),'-v'],
                         cwd=PHASE, capture_output=True, text=True)
    (PHASE/'evidence').mkdir(exist_ok=True)
    (PHASE/'evidence/M1_SYNTHETIC_TEST_LOG.txt').write_text(run.stdout+run.stderr,encoding='utf-8')
    require(run.returncode==0,'STAGE0_SYNTHETIC_FIREWALL_FAILED')
    allowed = STATIC+WEIGHTS
    entries = []
    prior = json.loads((REPO/STATIC[-2]).read_text(encoding='utf-8'))['sources']
    for rel in allowed:
        sha = metadata_hash(REPO,rel,allowed)
        if rel in prior:
            require(sha == prior[rel]['sha256'], 'HISTORICAL_IDENTITY_CHANGED:'+rel)
        last = subprocess.check_output(['git','log','-1','--format=%H','--',rel],cwd=REPO,text=True).strip()
        entries.append(dict(path=rel,sha256=sha,last_modified_commit=last,
                            access_class='WEIGHT_BYTES_ONLY_NO_DESERIALIZATION' if rel in WEIGHTS else 'STATIC_METADATA_CODE_DOC',
                            runtime_imported=False))
    split = json.loads((REPO/STATIC[1]).read_text(encoding='utf-8'))['segments']['train']
    require(split['rows']==[0,2103] and split['start_h']==72.5 and split['end_h']==1310 and split['legal_windows']==1380,
            'F2_SPLIT_AUTHORITY_MISMATCH')
    # Exact static identity chain, frozen BEFORE reading any physical N1 record.
    n1 = (REPO/STATIC[5]).read_text(encoding='utf-8')
    c = (REPO/STATIC[4]).read_text(encoding='utf-8')
    e = (REPO/STATIC[3]).read_text(encoding='utf-8')
    require('pd.DataFrame(rows).sort_values("bin_start_ophrs2_h").reset_index(drop=True)' in n1,
            'SORTED_EXPORT_PROOF_MISSING')
    require('prefix.loc[prefix["bin_end_ophrs2_h"] < 1819.997738].reset_index(drop=True)' in c,
            'ORIGINAL_ROW_FILTER_PROOF_MISSING')
    require('dev = c.load_development(cfg)' in e or 'c.load_development' in e, 'E_LOADER_PROOF_MISSING')
    execution = {str(p.relative_to(PHASE)).replace('\\','/'):digest_bytes(p.read_bytes())
                 for folder in ['scripts','tests'] for p in sorted((PHASE/folder).glob('*.py'))}
    freeze_time = datetime.now(timezone.utc).isoformat()
    access = dict(phase=PHASE.name,base_commit=BASE,frozen_at_utc=freeze_time,
                  status='FROZEN_BEFORE_REAL_TRAIN_STATISTICS', mixed_source=N1,record_count=2103,
                  train_rows=[0,2103],train_start_h=72.5,train_end_h=1310.,
                  validation_rows_prohibited=[2103,2404],assessment_rows_prohibited=[2404,3005],
                  read_method='unbuffered FileIO read(1); CSV logical records; no next-byte probe',
                  row_identity_proof='N1 sorted export + C age-only filter + E unchanged row slices; runtime all first 2103 survive filter; boundaries and 1380 windows agree',
                  retain_columns=SAFE_COLUMNS,
                  prohibited_fields=['train_operation_point_voltage_reference','matched_condition_voltage','global_B2','SOH','RUL'],
                  prohibited_reads=['whole N1 hash/blob/read','nrows=3007','raw/electrochemical data without exact Train row/byte map','model forward/fit/train'],
                  raw_access_enabled=False,polarization_content_access_enabled=False,
                  header_schema_from='N1 dictionary; physical header read with bounded reader only',
                  static_source_allowlist=entries,execution_code_sha256=execution,
                  synthetic_firewall_tests='PASS before physical N1 read',seed='NOT_APPLICABLE_NO_RANDOM_PROCESS')
    support = dict(phase=PHASE.name,frozen_at_utc=freeze_time,
      task='P3 qualification only; original F2 48x14 ->24 DeltaV unchanged',
      calibration_rows=[0,1051],evidence_rows=[1051,2103],
      calibration_rule='condition-only IQR scale; std fallback; zero scale -> exact equality. No outcome-based adjustment',
      bin_screen='coverage>=0.8; valid; j>0; zero fraction=0; no transition; one op; no start/shutdown/internal gap; not gap-return; all conditions finite',
      bin_screen_certifies_dwell=False,
      mandatory_dwell_evidence='raw source identity/time interval + verified stabilization, duration and scan/hold protocol; not 30min means',
      conditions=CONDITIONS,
      mandatory_state_evidence=['same 10-cell configuration','same operating protocol','neighbor temporal segment',
        'scan direction or independently supported absence of hysteresis','no characterization/intervention/recovery',
        'material-state proximity','membrane-water/thermal-state comparability'],
      prospective_matching=dict(max_temporal_distance_h=.5,min_delta_j_A_cm2=.2,
        per_variable_caliper_IQR_fraction=.1,joint_standardized_distance_max=.5,
        threshold_basis='PROJECT_ANALYSIS_DESIGN: adjacent 0.5h bins, conservative 0.2 A/cm2 separation; not sensor precision or universal electrochemical tolerance. Physical certification still mandatory',
        calibration_outcome_access=False,matching='deterministic joint calipers then nearest without dwell reuse; ties source row IDs',
        sensitivity_IQR_fractions=[.05,.2],sensitivity_never_selects_primary=True),
      minimum_support=dict(nonshared_dwell_pairs_per_domain=20,distinct_load_levels=3,independent_segments=2,
        basis='PROJECT_ANALYSIS_DESIGN feasibility floor, not power/significance or proof of physics'),
      trigger_for_real_voltage_pair_statistics='all raw dwell, state, joint support, independent amplitude and isolation prerequisites PASS',
      empirical_metrics=['pair DeltaV V/mV distribution','violation fraction','unique dwells/segments','repeat measurement noise','hysteresis/scan','segment and batch summaries','no overlapping-pair independence claim'],
      isolation=dict(auxiliary_view='all-time voltage-free 9 nonvoltage channels, no age/curve/cycle identity',
        visible_channels=AUX_NONVOLTAGE,all_voltage_descendants_excluded=True,
        original_f2_input_unchanged=True,nonphysical_control_same_view=True,
        data_lineage_scope='synthetic + schema; no encoder or auxiliary model instantiated'),
      amplitude_supervision='independent measured dwell voltages with labels excluded from auxiliary view, calibration/evidence time isolated, no shared dwell; not certified until qualified raw pair evidence',
      constant_checks=['constant voltage','fixed decreasing j-only line','ignore conditions/state','memorize curve identity','voltage descendant copy'],
      stops=['Train firewall failure -> P3_TRAIN_ACCESS_BLOCKED','missing quasi-steady/state evidence -> no pair relation computation',
             'isolation or amplitude failure -> no training eligibility','no automatic next phase'],
      allowed_final_statuses=['P3_TRAIN_SUPPORT_AND_AUXILIARY_ISOLATION_QUALIFIED','P3_COMMON_SUPPORT_INSUFFICIENT',
        'P3_QUASI_STEADY_OR_STATE_COMPARABILITY_UNRESOLVED','P3_AUXILIARY_ISOLATION_FAILED',
        'P3_NONDEGENERATE_SUPERVISION_NOT_ESTABLISHED','P3_TRAIN_ACCESS_BLOCKED','P3_SCIENTIFIC_QUALIFICATION_UNRESOLVED'])
    for name,obj in [('M1_FROZEN_ACCESS_PROTOCOL',access),('M1_FROZEN_SUPPORT_PROTOCOL',support)]:
        sha = dump('configs/'+name+'.yaml',obj)
        (PHASE/'configs'/f'{name}.sha256').write_text(sha+'\n',encoding='utf-8')
    dump('outputs/M1_STAGE0_FREEZE_AUDIT.json',dict(pass_=True,real_data_read=False,
         frozen_at_utc=freeze_time,execution_code_sha256=execution,synthetic_test_exit_code=run.returncode))
    print('M1 Stage0 frozen; synthetic firewall PASS; real data not yet read.')

if __name__=='__main__':
    main()
