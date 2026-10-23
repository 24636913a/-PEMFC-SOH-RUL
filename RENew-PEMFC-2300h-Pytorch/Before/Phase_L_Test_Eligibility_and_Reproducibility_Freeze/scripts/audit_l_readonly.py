"""Phase L: guarded document/identity audit. Never imports a historical runner.

JSON serialization is used for YAML 1.2 protocol compatibility. Numeric metric
helpers below accept synthetic records only; they never open prediction caches.
"""
import argparse
import ast
import csv
import hashlib
import importlib.metadata
import json
import math
import platform
import statistics
import subprocess
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

PHASE = Path(__file__).resolve().parents[1]
REPO = PHASE.parents[1]
ROOT = 'RENew-PEMFC-2300h-Pytorch/'
BASE = '1fe9ce87ecc33ac297ff0e4a14cd4ca7a8ada3a6'
STATUS = 'PHASE_L_CONFIRMATORY_TEST_NOT_ELIGIBLE'
CONFIG = PHASE / 'configs/L_AUDIT_ACCESS_PROTOCOL.yaml'
DENIED_OPERATIONS = {'model_forward', 'model_load', 'train', 'test_numeric_read',
                     'formal_unblind', 'mixed_source_read', 'mixed_source_hash',
                     'mixed_source_git_blob', 'checkpoint_write', 'new_test_metric'}

# These are vetted small documents, static source or metadata-only ledgers.
# Rich late-life reports, numeric tables, raw sources and prediction arrays are absent.
SOURCES = [
 ('N1_CODE', 'Phase_N1_Data_Foundation/scripts/phase_n1_build_foundation.py', 'static_source'),
 ('N1_META', 'Phase_N1_Data_Foundation/outputs/audits/foundation_run_metadata.json', 'range_count_and_development_only_audit'),
 ('RUL_CODE', 'Phase_RUL90_Data_Feasibility/scripts/run_rul90_feasibility.py', 'static_source'),
 ('RUL_LEDGER', 'Phase_RUL90_Data_Feasibility/RUN_AUDIT.json', 'metadata_only_execution_ledger'),
 ('AHC_LEDGER', 'Phase_AHC_TV_One_Time_Late_Life_Test_Protocol/runtime/AHC_TV_ONE_SHOT_EXECUTION_LEDGER.json', 'metadata_only_execution_ledger'),
 ('AHC_RUN', 'Phase_AHC_TV_One_Time_Late_Life_Test_Protocol/scripts/run_ahc_tv_one_shot.py', 'static_source'),
 ('AHC_STATE', 'Phase_AHC_TV_One_Time_Late_Life_Test_Protocol/scripts/ahc_tv_one_shot_ledger.py', 'static_source'),
 ('AHC_P', 'Phase_AHC_TV_One_Time_Late_Life_Test_Protocol/scripts/ahc_tv_stage_p.py', 'static_source'),
 ('AHC_RAW', 'Phase_AHC_TV_One_Time_Late_Life_Test_Protocol/scripts/ahc_tv_raw_alignment_core.py', 'static_source'),
 ('PM_CODE', 'Phase_AHC_TV_Target_Blind_Alignment_Postmortem/scripts/run_target_blind_alignment_postmortem.py', 'static_source'),
 ('PM_REPORT', 'Phase_AHC_TV_Target_Blind_Alignment_Postmortem/reports/AHC_TV_TARGET_BLIND_ALIGNMENT_FAILURE_POSTMORTEM.md', 'target_blind_alignment_flags_and_counts_only'),
 ('DRIFT_CODE', 'Phase_AHC_TV_Late_Life_Operating_Condition_Support_Drift_Audit/scripts/run_late_life_support_drift_audit.py', 'static_source'),
 ('DRIFT_README', 'Phase_AHC_TV_Late_Life_Operating_Condition_Support_Drift_Audit/README.md', 'scope_document'),
 ('ELTE_CODE', 'Phase_ELTE_Exploratory_Late_Life_Target_Evaluation/scripts/run_elte_one_shot.py', 'static_source'),
 ('ELTE_LEDGER', 'Phase_ELTE_Exploratory_Late_Life_Target_Evaluation/runtime/ELTE_ONE_SHOT_LEDGER.json', 'metadata_only_execution_ledger'),
 ('ELTE_EXPOSURE', 'Phase_ELTE_Exploratory_Late_Life_Target_Evaluation/runtime/PROJECT_TARGET_EXPOSURE.json', 'exposure_flags_only'),
 ('CSHSE_INCIDENT', 'Phase_CSHSE_Causal_SameCycle_Health_State_Estimation/reports/INVALID_PROTOCOL_BREACH_RECORD.md', 'incident_document_no_numeric_results'),
 ('H_CONFIG', 'Phase_H_Determinism_MultiSeed_Stability/configs/H_FROZEN_PROTOCOL.yaml', 'development_protocol'),
 ('H_REPORT', 'Phase_H_Determinism_MultiSeed_Stability/reports/H_FINAL_CONCLUSION.md', 'saved_development_report'),
 ('H_COMMON', 'Phase_H_Determinism_MultiSeed_Stability/scripts/h_common.py', 'static_source'),
 ('H_TRAIN', 'Phase_H_Determinism_MultiSeed_Stability/scripts/run_h_training.py', 'static_source'),
 ('TFT', 'Phase_TFT_Original_Baseline/models/canonical_tft.py', 'static_source'),
 ('F_CONFIG', 'Phase_F_TFT_DualPath_RegimeFusion/configs/F0_F2_EXECUTION_PROTOCOL.yaml', 'development_protocol'),
 ('F_SCHEMA', 'Phase_F_TFT_DualPath_RegimeFusion/models/frozen_feature_schema.py', 'static_source'),
 ('F_CONV', 'Phase_F_TFT_DualPath_RegimeFusion/models/causal_multiscale_branch.py', 'static_source'),
 ('F1', 'Phase_F_TFT_DualPath_RegimeFusion/models/dualpath_strict_history_tft.py', 'static_source'),
 ('F2', 'Phase_F_TFT_DualPath_RegimeFusion/models/regime_aware_dualpath_tft.py', 'static_source'),
 ('E_CODE', 'Phase_TFT_FullFeature_CausalFilter_Comparison/scripts/run_e_comparison.py', 'static_source'),
 ('E_SCALER', 'Phase_TFT_FullFeature_CausalFilter_Comparison/outputs/E_COMMON_SCALER.json', 'frozen_train_only_scaler'),
 ('C_CODE', 'Phase_TFT_SingleSplit_Feature_Filter_Ablation/scripts/run_c_ablation.py', 'static_source'),
 ('IR_COMMON', 'Phase_IR_Validation_Reload_Parity_Audit/scripts/ir_common.py', 'static_source'),
 ('IR_REPORT', 'Phase_IR_Validation_Reload_Parity_Audit/reports/IR_FINAL_RECOMMENDATION.md', 'saved_development_report'),
 ('I_METRIC', 'Phase_I_Frozen_Checkpoint_Assessment/scripts/i_metrics.py', 'static_source'),
 ('I2_COMMON', 'Phase_I2_Frozen_Assessment_Evaluation/scripts/i2_common.py', 'static_source'),
 ('I2_RUN', 'Phase_I2_Frozen_Assessment_Evaluation/scripts/run_i2_assessment.py', 'static_source'),
 ('I2_REPORT', 'Phase_I2_Frozen_Assessment_Evaluation/reports/I2_FINAL_SCIENTIFIC_CONCLUSION.md', 'saved_development_report'),
 ('J_ID', 'Phase_J_F2_Architecture_and_Gate_Mechanism_Audit/outputs/checkpoint_identity/J_NINE_CHECKPOINT_IDENTITY.json', 'checkpoint_identity_metadata_only'),
 ('J_REPORT', 'Phase_J_F2_Architecture_and_Gate_Mechanism_Audit/reports/J_FINAL_ARCHITECTURE_DECISION.md', 'saved_development_report'),
 ('K_ID', 'Phase_K_Paper_Claims_and_SOH_RUL_Feasibility/outputs/K_SOURCE_MANIFEST.json', 'identity_and_development_prefix_metadata'),
 ('K_REPORT', 'Phase_K_Paper_Claims_and_SOH_RUL_Feasibility/reports/K_FINAL_RESEARCH_RECOMMENDATION.md', 'scope_and_development_claims'),
]
ALLOW = {str((REPO / (ROOT + p)).resolve()) for _, p, _ in SOURCES}


class FirewallError(RuntimeError):
    pass


def guard_operation(operation):
    if operation in DENIED_OPERATIONS:
        raise FirewallError('L_OPERATION_PROHIBITED:' + operation)


def guard_path(path, mode='document_read', checkpoint_paths=()):
    p = Path(path).resolve()
    if mode == 'write':
        if not p.is_relative_to(PHASE) or p.suffix == '.pt':
            raise FirewallError('L_HISTORICAL_OR_CHECKPOINT_WRITE_PROHIBITED')
        return p
    if mode == 'checkpoint_hash' and str(p) in checkpoint_paths:
        return p
    if mode == 'document_read' and (str(p) in ALLOW or p.is_relative_to(PHASE)):
        if p.suffix in {'.pt', '.npz', '.parquet'} or p.name == 'causal_features_30min.csv':
            raise FirewallError('L_NUMERIC_SOURCE_PROHIBITED')
        return p
    raise FirewallError('L_SOURCE_NOT_ALLOWLISTED:' + str(p))


def read_document(path):
    return guard_path(path).read_bytes()


def sha(data):
    return hashlib.sha256(data).hexdigest()


def verify_digest(path, expected, mode='document_read', checkpoint_paths=()):
    p = guard_path(path, mode, checkpoint_paths)
    if not p.is_file():
        raise FirewallError('L_REQUIRED_EVIDENCE_MISSING:' + str(p))
    # Only vetted documents and registered checkpoint bytes, never mixed data.
    h = hashlib.sha256()
    with p.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    observed = h.hexdigest()
    if observed != expected:
        raise FirewallError('L_SOURCE_IDENTITY_CHANGED:' + str(p))
    return observed


def require_evidence(records, required):
    missing = sorted(set(required) - set(records))
    if missing:
        raise FirewallError('L_REQUIRED_EVIDENCE_MISSING:' + ','.join(missing))


def git(*args):
    return subprocess.check_output(['git', *args], cwd=REPO).decode('utf-8').strip()


def write_json(relative, value):
    p = guard_path(PHASE / relative, 'write')
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8', newline='\n')


def write_csv(relative, rows):
    if not rows:
        raise FirewallError('L_EMPTY_EVIDENCE_TABLE')
    p = guard_path(PHASE / relative, 'write')
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open('w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator='\n')
        w.writeheader()
        w.writerows(rows)


def freeze():
    if git('rev-parse', 'HEAD') != BASE:
        raise FirewallError('L_FREEZE_BASE_CHANGED')
    if git('diff', '--name-only') or git('diff', '--cached', '--name-only'):
        raise FirewallError('L_TRACKED_TREE_DIRTY_BEFORE_FREEZE')
    entries = []
    for sid, rel, scope in SOURCES:
        path = ROOT + rel
        p = guard_path(REPO / path)
        if not p.is_file():
            raise FirewallError('L_REQUIRED_SOURCE_MISSING:' + path)
        data = read_document(p)
        # Safe, explicitly allowlisted small text only. No numeric mixed blob access.
        blob = subprocess.check_output(['git', 'show', BASE + ':' + path], cwd=REPO)
        if data.replace(b'\r\n', b'\n') != blob.replace(b'\r\n', b'\n'):
            raise FirewallError('L_HISTORICAL_TEXT_CHANGED:' + path)
        entries.append({'id': sid, 'path': path, 'scope': scope, 'sha256': sha(data),
                        'base_blob_sha256': sha(blob), 'git_blob': git('rev-parse', BASE + ':' + path),
                        'source_last_commit': git('log', '-1', '--format=%H', BASE, '--', path),
                        'line_ending_contract': 'local byte SHA; base text equality permits CRLF-to-LF only'})
    ids = json.loads(read_document(REPO / (ROOT + dict((a, b) for a, b, _ in SOURCES)['J_ID'])))
    k = json.loads(read_document(REPO / (ROOT + dict((a, b) for a, b, _ in SOURCES)['K_ID'])))
    prior = {r['identity']: r for r in k['checkpoint_identities']}
    checkpoints = []
    for model in ['F0', 'F1', 'F2']:
        for seed in [42, 52, 62]:
            key = f'H-{model}-Seed{seed}'
            r = ids[key]
            if r['sha256'] != prior[key]['expected_sha256'] or r['path'] != prior[key]['path']:
                raise FirewallError('L_J_K_CHECKPOINT_AUTHORITY_MISMATCH:' + key)
            checkpoints.append({'identity': key, 'model': model, 'seed': seed,
                                'path': r['path'], 'sha256': r['sha256'],
                                'parameter_count': r['parameter_count'], 'best_epoch': r['best_epoch'],
                                'source_authorities': ['J_ID', 'K_ID']})
    cfg = {'phase': PHASE.name, 'base_commit': BASE, 'scope': 'READ_ONLY_DOCUMENTS_IDENTITY_AND_SYNTHETIC_TESTS',
           'sources': entries, 'checkpoints': checkpoints, 'denied_operations': sorted(DENIED_OPERATIONS),
           'mixed_files': 'ALL CONTENT/HASH/GIT-BLOB ACCESS PROHIBITED; no N1 prefix needed in L',
           'development_boundary_h': 1819.997738, 'no_model_import': True,
           'task': {'name': 'PEMFC_DYNAMIC_MULTISTEP_DELTA_V_FORECASTING', 'resolution_h': .5,
                    'lookback': 48, 'horizon': 24, 'input_slots': 14,
                    'top10_indices': [0, 3, 13, 11, 4, 9, 6, 7, 10, 8], 'voltage_ewma_alpha': .3},
           'evaluation': {'primary': 'forecast_query_weighted_RMSE_mV',
                          'sensitivity': 'target_time_equal_weight_RMSE_mV', 'seeds': [42, 52, 62],
                          'seed_summary': 'arithmetic_mean_and_sample_SD_ddof1', 'quantiles': 'linear',
                          'all24_steps_retained': True, 'seed_selection': 'PROHIBITED'},
           'window': {'allow_preorigin_development_context': True, 'future_targets_all_in_evaluation_region': True,
                      'same_segment': True, 'continuity_h': .5, 'continuity_atol': 1e-6,
                      'future_real_covariates_input': 'PROHIBITED'},
           'state_machine': {'real_state': 'BLOCKED', 'real_authorization': False,
                             'reason': 'known late-life exposure; no qualified independent candidate; real I/O isolation unresolved'},
           'code_identity': {p.relative_to(PHASE).as_posix(): sha(read_document(p))
                             for folder in ['scripts', 'tests'] for p in sorted((PHASE / folder).glob('*.py'))},
           'limitations': ['not an OS-wide I/O monitor', 'not exhaustive proof of all historical process I/O',
                          'file inventory is metadata, not evidence of execution',
                          'historical full-file checksum breach is not automatically supervised-label leakage']}
    write_json('configs/L_AUDIT_ACCESS_PROTOCOL.yaml', cfg)
    checksum = sha(read_document(CONFIG))
    p = guard_path(CONFIG.with_suffix('.sha256'), 'write')
    p.write_text(checksum + '  L_AUDIT_ACCESS_PROTOCOL.yaml\n', encoding='utf-8', newline='\n')
    print('L_PROTOCOL_FROZEN', checksum)


def synthetic_metrics(rows, synthetic=False):
    if not synthetic:
        raise FirewallError('L_NEW_REAL_METRIC_PROHIBITED')
    keys = [(r['origin'], r['step']) for r in rows]
    if len(set(keys)) != len(keys) or not rows:
        raise FirewallError('L_SYNTHETIC_QUERY_KEY_INVALID')
    groups, steps = defaultdict(list), defaultdict(list)
    errors, truth = [], []
    for r in rows:
        if r['target'] != r['origin'] + r['step'] or not 1 <= r['step'] <= 24:
            raise FirewallError('L_SYNTHETIC_TARGET_KEY_INVALID')
        e = r['prediction'] - r['truth']
        if not math.isfinite(e):
            raise FirewallError('L_SYNTHETIC_NONFINITE')
        errors.append(e)
        truth.append(r['truth'])
        groups[r['target']].append(e * e)
        steps[r['step']].append(e * e)
    if set(steps) != set(range(1, 25)) or any(sum(r['origin'] == origin for r in rows) != 24 for origin, _ in keys):
        raise FirewallError('L_SYNTHETIC_INCOMPLETE_HORIZONS')
    def quantile(v, q):
        v = sorted(v)
        position = (len(v) - 1) * q
        lo, hi = math.floor(position), math.ceil(position)
        return v[lo] + (v[hi] - v[lo]) * (position - lo)
    sse = math.fsum(e * e for e in errors)
    denom = math.fsum((v - statistics.mean(truth)) ** 2 for v in truth)
    absv = [abs(e) * 1000 for e in errors]
    target_sse = sorted((math.fsum(v) for v in groups.values()), reverse=True)
    return {'forecast_query_weighted_RMSE_mV': math.sqrt(sse / len(errors)) * 1000,
            'target_time_equal_weight_RMSE_mV': math.sqrt(statistics.mean(statistics.mean(v) for v in groups.values())) * 1000,
            'MAE_mV': statistics.mean(absv), 'R2': 1 - sse / denom if denom > 0 else None,
            'horizon_RMSE_mV': [math.sqrt(statistics.mean(steps[h])) * 1000 for h in range(1, 25)],
            'endpoint_12h_RMSE_mV': math.sqrt(statistics.mean(steps[24])) * 1000,
            'P95_mV': quantile(absv, .95), 'P99_mV': quantile(absv, .99), 'max_abs_error_mV': max(absv),
            'max_target_SSE_share': target_sse[0] / sse if sse else None,
            'top5_target_SSE_share': math.fsum(target_sse[:5]) / sse if sse else None,
            'origins': len({r['origin'] for r in rows}), 'queries': len(rows), 'unique_targets': len(groups)}


def synthetic_seed_summary(rmse):
    if set(rmse) != {42, 52, 62}:
        raise FirewallError('L_ALL_THREE_SEEDS_REQUIRED')
    v = [rmse[s] for s in [42, 52, 62]]
    return {'arithmetic_mean_RMSE': statistics.mean(v), 'sample_SD': statistics.stdev(v),
            'pooled_RMSE_equal_query_counts_only': math.sqrt(statistics.mean(x * x for x in v))}


def validate_window(metadata, origin, region_start, region_end):
    first, last = origin - 47, origin + 24
    if first < 0 or last >= len(metadata):
        raise FirewallError('L_WINDOW_LENGTH_INVALID')
    b = metadata[first:last + 1]
    if any(not r['valid'] or not r['finite_schema'] for r in b):
        raise FirewallError('L_WINDOW_INVALID_METADATA')
    if len({r['segment'] for r in b}) != 1:
        raise FirewallError('L_WINDOW_SEGMENT_CROSSING')
    if any(abs(b[j + 1]['end_age'] - b[j]['end_age'] - .5) > 1e-6 for j in range(71)):
        raise FirewallError('L_WINDOW_TIME_GAP_OR_DUPLICATE')
    if any(abs(r['end_age'] - r['start_age'] - .5) > 1e-6 for r in b):
        raise FirewallError('L_BIN_PHYSICAL_WIDTH_INVALID')
    if not region_start <= metadata[origin]['end_age'] <= region_end:
        raise FirewallError('L_ORIGIN_REGION_CROSSING')
    # Fixed boundary rule: history can precede evaluation, all future bins wholly within it.
    if any(r['start_age'] < region_start or r['end_age'] > region_end for r in metadata[origin + 1:last + 1]):
        raise FirewallError('L_TARGET_REGION_CROSSING')
    return True


def transition(state, target, eligible=False, frozen=False, synthetic_pass=False):
    if target in {'AUTHORIZED_ONE_SHOT', 'PREDICTIONS_FROZEN_BEFORE_TARGET_SCORING', 'TARGET_SCORING_COMPLETED'}:
        guard_operation('formal_unblind')
    allowed = {'NOT_AUTHORIZED': 'SOURCE_ELIGIBILITY_REVIEWED',
               'SOURCE_ELIGIBILITY_REVIEWED': 'SCIENTIFIC_PROTOCOL_FROZEN',
               'SCIENTIFIC_PROTOCOL_FROZEN': 'RUNTIME_SYNTHETIC_VERIFIED',
               'RUNTIME_SYNTHETIC_VERIFIED': 'READY_FOR_EXPLICIT_AUTHORIZATION'}
    if target in {'BLOCKED', 'CLOSED'}:
        return target
    if allowed.get(state) != target or not eligible or not frozen or (target == 'READY_FOR_EXPLICIT_AUTHORIZATION' and not synthetic_pass):
        raise FirewallError('L_STATE_PREREQUISITE_NOT_MET')
    return target


def static_audit():
    forbidden_imports = {'torch', 'pandas', 'numpy', 'optuna', 'yaml'}
    forbidden_calls = {'fit', 'forward', 'predict', 'train', 'backward', 'load_state_dict', 'read_csv', 'read_parquet'}
    for p in (PHASE / 'scripts').glob('*.py'):
        tree = ast.parse(read_document(p).decode('utf-8'))
        for n in ast.walk(tree):
            if isinstance(n, (ast.Import, ast.ImportFrom)):
                names = [a.name.split('.')[0] for a in n.names] if isinstance(n, ast.Import) else [(n.module or '').split('.')[0]]
                if set(names) & forbidden_imports:
                    raise FirewallError('L_FORBIDDEN_IMPORT')
            if isinstance(n, ast.Call):
                name = n.func.attr if isinstance(n.func, ast.Attribute) else n.func.id if isinstance(n.func, ast.Name) else ''
                if name in forbidden_calls:
                    raise FirewallError('L_FORBIDDEN_CALL:' + name)
    return {'pass': True, 'scope': 'Phase L production scripts only', 'historical_code_executed': False}


def run():
    cfg = json.loads(read_document(CONFIG))
    verify_digest(CONFIG, read_document(CONFIG.with_suffix('.sha256')).decode().split()[0])
    if cfg['base_commit'] != BASE or subprocess.run(['git', 'merge-base', '--is-ancestor', BASE, 'HEAD'], cwd=REPO).returncode:
        raise FirewallError('L_BASE_NOT_ANCESTOR')
    sources = cfg['sources']
    require_evidence({s['id']: s for s in sources}, [s[0] for s in SOURCES])
    if {s['path'] for s in sources} != {ROOT + p for _, p, _ in SOURCES}:
        raise FirewallError('L_SOURCE_ALLOWLIST_CHANGED')
    for r in sources:
        verify_digest(REPO / r['path'], r['sha256'])
    for rel, expected in cfg['code_identity'].items():
        verify_digest(PHASE / rel, expected)
    cps = cfg['checkpoints']
    if {(r['model'], r['seed']) for r in cps} != {(m, s) for m in ['F0', 'F1', 'F2'] for s in [42, 52, 62]} or len(cps) != 9:
        raise FirewallError('L_NINE_MODEL_SET_INVALID')
    cp_paths = {str((REPO / r['path']).resolve()) for r in cps}
    cp_before = [verify_digest(REPO / r['path'], r['sha256'], 'checkpoint_hash', cp_paths) for r in cps]
    docs = {r['id']: read_document(REPO / r['path']).decode('utf-8') for r in sources}
    rul, ahc, elte = (json.loads(docs[k]) for k in ['RUL_LEDGER', 'AHC_LEDGER', 'ELTE_LEDGER'])
    if not (rul['full_data_target_accessed'] and rul['test_seal_for_future_model_selection'] == 'compromised'
            and ahc['status'] == 'ABORTED_AFTER_TEST_EXECUTION_STARTED' and not ahc['target_unblinded']
            and elte['targets_unblinded'] and elte['completed']):
        raise FirewallError('L_EXPOSURE_FACT_NOT_REPRODUCED')
    ev = [
      ('N1', 'N1_CODE;N1_META', 'all durability age span; final region >=1819.997738 h', 'raw voltage/current and aggregate features',
       'automated full-source timeline/aggregation; cell-mask selection Train+Validation only', 'YES underlying future dynamic voltage',
       'dataset construction/counts; no Test model metric confirmed here', 'no selection demonstrated by N1 alone',
       'builder and completed foundation metadata', 'full numerical preprocessing is distinct from decision feedback', 'human review of each aggregate not established; target blindness not proved', 'UNKNOWN_ELIGIBILITY'),
      ('RUL90', 'RUL_CODE;RUL_LEDGER', 'full 2300 h including late region', 'raw voltage/current; full polarization/health descriptive outputs',
       'full-data descriptive audit; recovery reused existing summaries', 'YES same underlying dynamic voltage and health targets',
       'descriptive statistics and recommended research direction; no forecast Test metrics', 'project-level direction feedback; F2-specific tuning link not proved',
       'successful ledger + full-stage writer/code', 'test seal explicitly compromised before later research route decisions', 'complete human/process decision ancestry not recoverable', 'NOT_ELIGIBLE_AS_CONFIRMATORY_TEST'),
      ('AHC-TV', 'AHC_LEDGER;AHC_RUN;AHC_P;AHC_RAW', 'all 17 raw Loadcycle files; pre-2000/2300 use', '10 active cell voltages/current/conditions; polarization DateTime metadata',
       'raw full columns parsed before first Stage-P alignment exception', 'YES dynamic voltage; NO polarization target unblinding in this run',
       '2000 alignment inconclusive; no frozen paired predictions/target scores', 'one-shot consumed; failure motivated separate diagnostics',
       'aborted ledger plus executed Stage-P path and postmortem', 'Stage-P abort does not imply no raw voltage reading', 'no executed raw full-file checksum asserted; per-checkpoint 2300 formal prediction not confirmed', 'TARGET_OR_METRIC_EXPOSED'),
      ('AHC postmortem', 'PM_CODE;PM_REPORT;AHC_P', 'full raw sources; intervals before 2000/2300 characterization', 'P1S raw-derived voltage/retention and conditions',
       '17 full raw sources; reused original raw reader and P1S implementation', 'YES underlying dynamic voltage; polarization Y blind at this time',
       '2000 support 0/3; 2300 support 2/3; next support-drift audit recommended', 'health alignment route posterior diagnostics; not proof F2 hyperparameter tuning',
       'persisted target-blind report + reader trace', 'polarization-blind is not dynamic-voltage-blind', 'exact Test origin-target overlap not enumerated because numeric sources forbidden', 'POSTHOC_SELECTION_INFLUENCED'),
      ('AHC support drift', 'DRIFT_CODE;DRIFT_README', 'cached late P1S intervals for 2000/2300', 'operating conditions plus P1S_voltage_V/retention',
       'documented scope/code specifies unbounded cached-table read with voltage columns; independent execution I/O ledger not recovered', 'cached voltage in executed-path schema; no nominal polarization parser in this stage',
       'support drift diagnostics, not F2 Test RMSE', 'health protocol follow-up; direct F2 selection link unknown',
       'scope document + _require_target_blind_table schema', 'a label named target-blind can still read another task voltage target', 'numeric drift report excluded this round; historical causal influence incompletely logged', 'TARGET_OR_METRIC_EXPOSED'),
      ('ELTE', 'ELTE_CODE;ELTE_LEDGER;ELTE_EXPOSURE', '2000/2300 characterization; late P1S intervals', 'polarization health targets, health predictions and error metrics',
       'paired targets opened after exploratory prediction freeze', 'YES polarization health Y; not a direct F2 DeltaV metric',
       'health errors/results produced; values deliberately not read in L', 'separate exploratory protocol after AHC support failure; confirms health-route reuse',
       'completed consumed ledger and project exposure flags', 'later target opening supersedes earlier historical polarization-blind statement', 'no F2-specific target scoring or training established by this evidence', 'NOT_ELIGIBLE_AS_CONFIRMATORY_TEST'),
      ('CSHSE incident', 'CSHSE_INCIDENT', 'whole mixed 30min feature file beyond Development prefix', 'whole-file byte checksum',
       'full-file integrity read; previous run invalidated', 'byte access YES; Test-label extraction not confirmed',
       'protocol repair and clean rerun mandated', 'access protocol changed; no verified Test-label-based model tuning',
       'preserved incident record', 'DATA_ACCESS_BOUNDARY_BREACH not TEST_LABEL_LEAKAGE', 'no OS-level forensic log of past process', 'TARGET_BLIND_BUT_FEATURE_EXPOSED'),
      ('H/I-R/I2/J/K', 'H_CONFIG;H_REPORT;IR_REPORT;I2_REPORT;J_REPORT;K_REPORT', 'Development <1819.997738 h', 'saved Development metrics and model identity',
       'prefix-only experiments/reused Development Assessment; L documents only', 'NO new late-region target access established by these selected sources',
       'known Assessment selection bias and frozen F2 research recommendation', 'Development selection/reuse explicit; late-region independent eligibility requires wider history',
       'frozen reports and static code', 'nine-model freezing does not restore previously exposed late data', 'absence in these reports cannot establish all-process no-Test history', 'UNKNOWN_ELIGIBILITY'),
    ]
    by_id = {r['id']: r for r in sources}
    rows = []
    for phase, ids, region, dtype, readrange, future, feedback, influence, evidence, fact, unknown, classification in ev:
        selected = [by_id[k] for k in ids.split(';')]
        rows.append({'phase': phase, 'commit': ';'.join(sorted({r['source_last_commit'] for r in selected})),
                     'file_path': ';'.join(r['path'] for r in selected), 'time_region': region, 'data_type': dtype,
                     'actual_read_range': readrange, 'contains_future_target': future, 'result_feedback': feedback,
                     'selection_influence': influence, 'evidence_source': ids + ': ' + evidence,
                     'confirmed_fact': fact, 'unknown': unknown, 'classification': classification})
    write_csv('evidence/L_HISTORICAL_TEST_EXPOSURE_REGISTRY.csv', rows)
    # Filename-only cross-project inventory: no historical report/data blob opened here.
    inventory = []
    for path in git('ls-files', '--', ROOT).splitlines():
        lower = path.removeprefix(ROOT).lower()
        if any(t in lower for t in ['test', 'holdout', 'late_life', 'late-life', '2000', '2300', 'firewall', 'ledger', 'exposure', 'prefix']):
            inventory.append({'path': path, 'content_reviewed_in_L': path in {r['path'] for r in sources},
                              'review_status': 'vetted_allowlisted_source' if path in {r['path'] for r in sources} else 'metadata_only_NOT_EVIDENCE_OF_EXECUTION',
                              'numeric_content_opened': False})
    write_csv('evidence/L_CROSS_PROJECT_PATH_INVENTORY.csv', inventory)
    manifest = {'base_commit': BASE, 'protocol_sha256': sha(read_document(CONFIG)), 'sources': sources,
                'numeric_raw_or_test_artifacts_read': [], 'historical_code_imported': False,
                'source_tier': 'executed ledgers + static traces stronger than names or self-report alone',
                'inventory_count': len(inventory), 'history_complete': False}
    write_json('manifests/L_SOURCE_MANIFEST.json', manifest)
    write_json('manifests/L_FROZEN_MODEL_MANIFEST.json', {'base_commit': BASE, 'models': [dict(r, identity_pass=True,
               loaded_as_model=False, changed=False, future_evaluation_set=True) for r in cps],
               'Repeat_A_B_included': False, 'best_seed_selected': False, 'count': 9})
    historical_env = {'python': '3.11.9', 'torch': '2.6.0+cu124', 'pytorch_forecasting': '1.8.0',
                     'CUDA': '12.4', 'cuDNN': 90100, 'GPU': 'NVIDIA GeForce RTX 3070 Laptop GPU',
                     'batch_size': 32, 'CPU_threads': 1, 'cudnn_deterministic': True,
                     'cudnn_benchmark': False, 'deterministic_algorithms': False, 'TF32_matmul': False,
                     'TF32_cudnn': True, 'strict_bitwise_determinism_claimed': False}
    installed = {'python': platform.python_version(), 'platform': platform.platform()}
    for name in ['torch', 'pytorch-forecasting', 'numpy', 'pandas']:
        try:
            installed[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            installed[name] = 'NOT_INSTALLED'
    write_json('manifests/L_REPRODUCIBILITY_MANIFEST.json', {'base_commit': BASE,
               'protocol_sha256': sha(read_document(CONFIG)), 'code_and_config': [r for r in sources if r['scope'] in {'static_source', 'development_protocol', 'frozen_train_only_scaler'}],
               'historical_environment_authority': historical_env, 'installed_metadata_without_model_import': installed,
               'current_GPU_runtime_probed': False, 'current_historical_inference_parity_tested': False,
               'inference': {'path': 'original H Path A -> e.predict -> c.predict', 'CUDA_construct': True,
                  'load_state_dict': 'strict', 'model_eval': True, 'grad_context': 'torch.no_grad',
                  'requires_grad': 'preserve original flags', 'batch_size': 32,
                  'original_I_predict_frozen': 'PROHIBITED; observed semantics differ'},
               'common_scaler_authority': by_id['E_SCALER'], 'scaler_refitted': False,
               'external_library_code_hashes': 'preserved as historical H_CONFIG records; no new library content execution',
               'full_new_runtime_ready': False})
    reports = ['L_TEST_EXPOSURE_FORENSIC_REPORT.md', 'L_TASK_LEVEL_EVALUATION_ELIGIBILITY.md',
               'L_ENVIRONMENT_AND_CODE_IDENTITY.md', 'L_FROZEN_METRIC_SPECIFICATION.md',
               'L_MULTISEED_REPORTING_CONTRACT.md', 'L_WINDOW_AND_INFORMATION_SET_CONTRACT.md',
               'L_ONE_SHOT_TEST_STATE_MACHINE_PLAN.md', 'L_PAPER_REPRODUCIBILITY_PACKAGE.md',
               'L_FINAL_EVALUATION_READINESS_DECISION.md']
    for name in reports:
        p = PHASE / 'reports' / name
        if not p.is_file() or not read_document(p).strip():
            raise FirewallError('L_DELIVERY_MISSING:' + name)
    test_result = json.loads(read_document(PHASE / 'outputs/L_SYNTHETIC_TEST_RESULTS.json'))
    if not test_result['passed'] or test_result['tests_run'] < 12:
        raise FirewallError('L_SYNTHETIC_TEST_FAILURE')
    changed = git('status', '--porcelain', '--untracked-files=all').splitlines()
    for line in changed:
        # Git quotes Unicode paths: evaluate full historical differences separately below.
        if line[:2] != '??' and not PHASE.relative_to(REPO).as_posix() in line[3:].strip('"'):
            raise FirewallError('L_UNRELATED_TRACKED_CHANGE')
    outside = [p for p in git('diff', '--name-only', BASE, '--', ROOT).splitlines()
               if not p.startswith(PHASE.relative_to(REPO).as_posix() + '/')]
    if outside:
        raise FirewallError('L_HISTORICAL_DIFF_CHANGED')
    cp_after = [verify_digest(REPO / r['path'], r['sha256'], 'checkpoint_hash', cp_paths) for r in cps]
    for r in sources:
        verify_digest(REPO / r['path'], r['sha256'])
    final = {'phase': PHASE.name, 'base_commit': BASE, 'execution_head': git('rev-parse', 'HEAD'),
             'utc': datetime.now(timezone.utc).isoformat(), 'audit_status': 'PASS', 'research_status': STATUS,
             'protocol_sha256': sha(read_document(CONFIG)), 'sources_verified': len(sources),
             'nine_checkpoints_before_after_equal': cp_before == cp_after, 'models_loaded': 0,
             'model_forward_calls': 0, 'training_calls': 0, 'new_validation_assessment_inference': False,
             'test_numeric_read': False, 'mixed_source_content_read': False, 'mixed_source_hash': False,
             'mixed_source_git_blob_read': False, 'new_Test_metric': False, 'seed_selected': False,
             'historical_files_changed': False, 'checkpoint_changed': False, 'new_SOH_RUL_labels': False,
             'static_audit': static_audit(), 'synthetic_tests': test_result,
             'source_inventory_scope': 'tracked filenames only; unreviewed contents deliberately excluded',
             'all_process_IO_globally_monitored': False, 'history_complete': False,
             'monitoring_limit': 'guarded L runner + static checks + explicit agent tool trace; not OS-wide interception or proof of all past I/O',
             'future_state': 'BLOCKED', 'real_Test_authorized': False, 'independent_candidate_identified': False,
             'stop': 'no Test/no new inference/no next phase'}
    write_json('outputs/L_FINAL_READONLY_INTEGRITY_AUDIT.json', final)
    write_json('outputs/L_FINAL_DECISION.json', {'status': STATUS, 'candidate_region': 'RealHyFC >=1819.997738 h',
               'classification': 'NOT_ELIGIBLE_AS_CONFIRMATORY_TEST', 'scope': 'existing shared late-life region only',
               'facts': ['RUL90 full Test voltage/current access and compromised seal', 'AHC raw voltage use and posterior support diagnosis',
                         'ELTE later exploratory health target/error exposure'],
               'F2_Test_supervised_training_proved': False, 'direct_F2_hyperparameter_influence_proved': False,
               'reason': 'shared candidate targets and diagnostic feedback already known; posterior decision influence cannot be excluded',
               'unidentified_external_region': 'UNKNOWN_ELIGIBILITY', 'explicit_authorization_readiness': False})
    print(json.dumps(final, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--freeze', action='store_true')
    args = parser.parse_args()
    if args.freeze:
        freeze()
    else:
        run()
