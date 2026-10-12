"""Closed-list document audit. No numerical prediction or model execution."""
import argparse
import csv
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from o_catalog import RECORDS

PHASE = Path(__file__).resolve().parents[1]
REPO = PHASE.parent.parent
ROOT = 'RENew-PEMFC-2300h-Pytorch/'
BASE = '9723ed75910594155b4c0ae4888d9bbe8519ec11'

# Exact static paths only; no directory-wide hashing. Manifest metadata that
# refers to weights is allowed, but the referenced weight path is never opened.
STATIC = [
 'Phase_TFT_Original_Baseline/models/canonical_tft.py',
 'Phase_F_TFT_DualPath_RegimeFusion/models/causal_multiscale_branch.py',
 'Phase_F_TFT_DualPath_RegimeFusion/models/dualpath_strict_history_tft.py',
 'Phase_F_TFT_DualPath_RegimeFusion/models/frozen_feature_schema.py',
 'Phase_F_TFT_DualPath_RegimeFusion/models/regime_aware_dualpath_tft.py',
 'Phase_F_TFT_DualPath_RegimeFusion/configs/F0_F2_EXECUTION_PROTOCOL.yaml',
 'Phase_F_TFT_DualPath_RegimeFusion/outputs/implementation_audit/F_MODEL_PARAMETER_COUNTS.json',
 'Phase_H_Determinism_MultiSeed_Stability/scripts/h_common.py',
 'Phase_H_Determinism_MultiSeed_Stability/configs/H_FROZEN_PROTOCOL.yaml',
 'Phase_H_Determinism_MultiSeed_Stability/reports/H_FINAL_CONCLUSION.md',
 'Phase_H_Determinism_MultiSeed_Stability/reports/H_SAME_SEED_REPEATABILITY.md',
 'Phase_H_Determinism_MultiSeed_Stability/outputs/metrics/H_VALIDATION_METRICS.csv',
 'Phase_H_Determinism_MultiSeed_Stability/outputs/metrics/H_MULTISEED_STATISTICS.csv',
 'Phase_TFT_FullFeature_CausalFilter_Comparison/scripts/run_e_comparison.py',
 'Phase_TFT_FullFeature_CausalFilter_Comparison/outputs/E_COMMON_SCALER.json',
 'Phase_J_F2_Architecture_and_Gate_Mechanism_Audit/outputs/checkpoint_identity/J_NINE_CHECKPOINT_IDENTITY.json',
 'Phase_J_F2_Architecture_and_Gate_Mechanism_Audit/outputs/architecture/J_PARAMETER_COUNTS.csv',
 'Phase_J_F2_Architecture_and_Gate_Mechanism_Audit/outputs/fixed_gate_diagnostic/J_VALIDATION_METRICS.csv',
 'Phase_J_F2_Architecture_and_Gate_Mechanism_Audit/outputs/fixed_gate_diagnostic/J_FIXED_GATE_COMPARISON.csv',
 'Phase_J_F2_Architecture_and_Gate_Mechanism_Audit/outputs/gate_diagnostics/J_GATE_STATISTICS.csv',
 'Phase_J_F2_Architecture_and_Gate_Mechanism_Audit/outputs/integrity_audit/J_ORIGINAL_H_FULL_VALIDATION_CACHE_AUDIT.csv',
 'Phase_J_F2_Architecture_and_Gate_Mechanism_Audit/outputs/J_SUMMARY.json',
 'Phase_J_F2_Architecture_and_Gate_Mechanism_Audit/reports/J_FINAL_ARCHITECTURE_DECISION.md',
 'Phase_K_Paper_Claims_and_SOH_RUL_Feasibility/reports/K_FINAL_RESEARCH_RECOMMENDATION.md',
 'Phase_L_Test_Eligibility_and_Reproducibility_Freeze/manifests/L_FROZEN_MODEL_MANIFEST.json',
 'Phase_L_Test_Eligibility_and_Reproducibility_Freeze/reports/L_FINAL_EVALUATION_READINESS_DECISION.md',
 'Phase_L_Test_Eligibility_and_Reproducibility_Freeze/outputs/L_FINAL_READONLY_INTEGRITY_AUDIT.json',
 'Phase_I2_Frozen_Assessment_Evaluation/outputs/checkpoint_manifest/I2_CHECKPOINT_MANIFEST.json',
 'Phase_I2_Frozen_Assessment_Evaluation/outputs/metrics/I2_ASSESSMENT_METRICS.csv',
 'Phase_I2_Frozen_Assessment_Evaluation/outputs/metrics/I2_MULTISEED_SUMMARY.csv',
 'Phase_I2_Frozen_Assessment_Evaluation/outputs/diagnostics/I2_ALL_HORIZON_METRICS.csv',
 'Phase_M_F2_PINN_Physics_Feasibility/reports/M_FINAL_PHYSICS_RESEARCH_DECISION.md',
 'Phase_M1_P3_TrainOnly_Polarization_Support_Qualification/reports/M1_FINAL_P3_RESEARCH_DECISION.md',
 'Phase_M2_P3_TrainOnly_Raw_Dwell_Qualification/reports/M2_FINAL_P3_RESEARCH_DECISION.md',
 'Phase_N_Physics_Informed_F2_Method_Convergence/N_FINAL_TRAINING_READINESS_DECISION.md',
]
ALLOW_STATIC = frozenset(ROOT + p for p in STATIC)


def digest(payload):
    return hashlib.sha256(payload).hexdigest()


def allowed_path(relative):
    if relative not in ALLOW_STATIC:
        raise ValueError('O_SOURCE_NOT_ALLOWLISTED:' + relative)
    p = (REPO / relative).resolve()
    if not p.is_relative_to(REPO.resolve()) or p.suffix in ('.pt', '.npz', '.parquet'):
        raise ValueError('O_PROHIBITED_SOURCE')
    return p


def source_bytes(relative):
    return allowed_path(relative).read_bytes()


def verify_digest(payload, expected):
    if digest(payload) != expected:
        raise ValueError('O_SOURCE_HASH_MISMATCH')


def git(*args, binary=False):
    result = subprocess.run(['git', *args], cwd=REPO, check=True, capture_output=True)
    return result.stdout if binary else result.stdout.decode('utf-8').strip()


def write_json(path, obj):
    if not path.resolve().is_relative_to(PHASE.resolve()):
        raise ValueError('O_WRITE_OUTSIDE_PHASE')
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')


def identities():
    lp = ROOT+'Phase_L_Test_Eligibility_and_Reproducibility_Freeze/manifests/L_FROZEN_MODEL_MANIFEST.json'
    jp = ROOT+'Phase_J_F2_Architecture_and_Gate_Mechanism_Audit/outputs/checkpoint_identity/J_NINE_CHECKPOINT_IDENTITY.json'
    ip = ROOT+'Phase_I2_Frozen_Assessment_Evaluation/outputs/checkpoint_manifest/I2_CHECKPOINT_MANIFEST.json'
    l = json.loads(source_bytes(lp))['models']
    j = json.loads(source_bytes(jp))
    i = {x['H_run_id']: x for x in json.loads(source_bytes(ip))}
    expected = {f'H-{model}-Seed{seed}' for model in ('F0','F1','F2') for seed in (42,52,62)}
    if {x['identity'] for x in l} != expected or set(j) != expected or set(i) != expected:
        raise ValueError('O_NINE_IDENTITY_SET_MISMATCH')
    for x in l:
        for other in (j[x['identity']], i[x['identity']]):
            for field in ('path','sha256','model','seed','best_epoch','parameter_count'):
                if x[field] != other[field]:
                    raise ValueError('O_HISTORICAL_IDENTITY_MISMATCH:'+field)
    return {'historical_manifests_agree': True, 'identities': sorted(expected),
            'current_weight_bytes_verified': False, 'weight_files_opened': 0}


def export_pdf_ledger():
    # Reuse internal extraction. No model data, only user-supplied literature.
    inventory = json.loads((PHASE/'runtime/inventory.json').read_text(encoding='utf-8'))
    if len(inventory) != 21 or len(RECORDS) != 21:
        raise ValueError('O_PDF_COUNT_MISMATCH')
    fields = ['local_id','paper_id','source_path','sha256','pages','english_title','authors',
              'journal','publication_year_status','doi','first_page_verified','evidence_grade',
              'method_experiment_pages','figures_equations_tables','dataset','supplement_status',
              'verification_scope']
    rows = []
    for n, pid, title, authors, journal, year, doi, grade, sections, locations, dataset, supplement in RECORDS:
        inv = inventory[n-1]
        source = (REPO/inv['path']).resolve()
        if source.parent != (REPO/'参考论文').resolve() or source.suffix != '.pdf':
            raise ValueError('O_LITERATURE_ONLY')
        verify_digest(source.read_bytes(), inv['sha256'])
        if doi.startswith('10.') and doi.lower() not in inv['first_page_text'].lower():
            raise ValueError('O_FIRST_PAGE_DOI_MISMATCH:'+pid)
        rows.append(dict(zip(fields, [inv['local_id'],pid,inv['path'],inv['sha256'],inv['pages'],
           title,authors,journal,year,doi,True,grade,sections,locations,dataset,supplement,
           'core full text' if pid in {'P1','P2','P3','P4','P5','P6','P7'} else
           'selected method and experiment chapters' if pid.startswith('S') else 'first page only'])))
    out = PHASE/'evidence/O_PDF_IDENTITY_AND_SOURCE_LEDGER.csv'
    with out.open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return rows


def export_figure_index():
    """Filename-only inventory; do not read pixels or extract numeric values."""
    rows = []
    for name in ('Phase_H_Determinism_MultiSeed_Stability',
                 'Phase_J_F2_Architecture_and_Gate_Mechanism_Audit',
                 'Phase_I2_Frozen_Assessment_Evaluation'):
        folder = REPO/ROOT/name/'figures'
        for path in sorted(folder.rglob('*')):
            if path.suffix.lower() in ('.png','.pdf','.svg') and path.is_file():
                rows.append({'path':path.relative_to(REPO).as_posix(), 'source_phase':name,
                             'verification':'filename availability only; pixels/data not reread',
                             'reuse_condition':'must retain frozen H identity and Development/Assessment label; no confirmatory Test claim'})
    out = PHASE/'evidence/O_HISTORICAL_FIGURE_INDEX.csv'
    with out.open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['path','source_phase','verification','reuse_condition'])
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def audit(freeze=False):
    head = git('rev-parse','HEAD')
    git('merge-base','--is-ancestor',BASE,head)
    if git('branch','--show-current') != 'main':
        raise ValueError('O_MAIN_REQUIRED')
    changed = set(filter(None, git('diff','--name-only',BASE,head).splitlines()))
    prefix = PHASE.relative_to(REPO).as_posix()+'/'
    if any(not x.startswith(prefix) for x in changed):
        raise ValueError('O_HISTORY_CHANGED_SINCE_BASE')
    for diffargs in [('diff','--name-only'),('diff','--cached','--name-only')]:
        if any(not x.startswith(prefix) for x in git(*diffargs).splitlines()):
            raise ValueError('O_OUT_OF_SCOPE_LOCAL_CHANGE')
    snap_path = PHASE/'outputs/O_STATIC_SOURCE_MANIFEST.json'
    if freeze and snap_path.exists():
        raise ValueError('O_SOURCE_SNAPSHOT_ALREADY_FROZEN')
    current = []
    for relative in sorted(ALLOW_STATIC):
        data = source_bytes(relative)
        original = git('show', BASE+':'+relative, binary=True)
        if data.replace(b'\r\n',b'\n') != original.replace(b'\r\n',b'\n'):
            raise ValueError('O_SOURCE_DIFFERS_FROM_BASE:'+relative)
        current.append({'path':relative, 'file_sha256':digest(data),
                        'base_git_blob_sha256':digest(original), 'source_commit':BASE,
                        'comparison':'byte identity except CRLF/LF checkout normalization',
                        'purpose':'static code/config/report/saved aggregate evidence only'})
    if freeze:
        write_json(snap_path, {'base_commit':BASE, 'sources':current})
    else:
        frozen = json.loads(snap_path.read_text(encoding='utf-8'))
        if frozen['sources'] != current or frozen['base_commit'] != BASE:
            raise ValueError('O_SOURCE_SNAPSHOT_CHANGED')
    id_audit = identities()
    pdfs = export_pdf_ledger()
    figure_count = export_figure_index()
    expected_reports = [
      'O_01_TFT_FULLTEXT_COMPARISON','O_02_DUAL_PATH_AND_DILATED_CNN_COMPARISON',
      'O_03_PHYSICS_INFORMED_PRIOR_ART_COMPARISON','O_04_INFERENCE_INFORMATION_SET_AUDIT',
      'O_05_ARCHITECTURE_OVERLAP_MATRIX','O_06_F2_NOVELTY_ASSESSMENT',
      'O_07_F2_EXPERIMENT_EVIDENCE_AUDIT','O_08_REVIEWER_CRITICISM_AND_RESPONSES',
      'O_09_PAPER_TITLE_ABSTRACT_OUTLINE','O_10_FINAL_PUBLICATION_DECISION']
    docs = [PHASE/'reports'/f'{x}.md' for x in expected_reports] + [
      PHASE/'README.md', PHASE/'evidence/O_FULL_TEXT_EVIDENCE_LEDGER.md',
      PHASE/'evidence/O_F2_CODE_AND_CHECKPOINT_IDENTITY.md']
    for path in docs:
        if not path.is_file() or len(path.read_text(encoding='utf-8')) < 300:
            raise ValueError('O_REQUIRED_DOCUMENT_EMPTY_OR_MISSING:'+path.name)
    final_text = docs[9].read_text(encoding='utf-8')
    for status in ('INCREMENTAL_BUT_DEFENSIBLE','EVIDENCE_INCOMPLETE','READY_FOR_CONDITIONAL_DRAFT'):
        if status not in final_text:
            raise ValueError('O_FINAL_DECISION_MISSING:'+status)
    # Only filename inventory of already rendered literature images, no metrics.
    visual = [p.relative_to(PHASE).as_posix() for p in (PHASE/'runtime').rglob('*.png')]
    record = {
      'phase':PHASE.name, 'base_commit':BASE, 'execution_head':head,
      'utc':datetime.now(timezone.utc).isoformat(), 'document_audit':'PASS',
      'static_sources_verified':len(current), 'pdf_first_pages_verified':len(pdfs),
      'core_full_text_papers':['P1','P2','P3','P4','P5','P6','P7'],
      'selected_chapter_supplements':['S1','S2','S3','S4','S5'],
      'historical_figures_filename_inventory_count':figure_count,
      'rendered_literature_page_files':visual,
      'visual_check_note':'rendering inventory supports access provenance; reading attestations and claim locations are in the evidence ledger; existence alone does not prove comprehension',
      'checkpoint_identity':id_audit,
      'performance_numbers':'organized saved historical tables; not recomputed',
      'prediction_cache_read':False, 'raw_or_N1_content_read':False,
      'mixed_source_hash':False, 'model_loads':0, 'model_forward_calls':0,
      'training_calls':0, 'physics_parameter_fitting':False, 'test_numeric_read':False,
      'new_SOH_RUL_labels':False, 'new_performance_figures':False,
      'assessment_used_for_model_selection':False, 'historical_files_changed':False,
      'all_process_IO_globally_monitored':False,
      'access_statement_scope':'Phase O tool trace and closed-list document script; not a claim about every past run or OS-wide monitoring',
      'copyrighted_full_text_or_figures_in_git':'NO; runtime is ignored',
      'novelty_status':'INCREMENTAL_BUT_DEFENSIBLE', 'evidence_status':'EVIDENCE_INCOMPLETE',
      'writing_status':'READY_FOR_CONDITIONAL_DRAFT',
      'stop':'no training, new inference, Physics-F2, backbone search, manuscript submission'}
    test_result = PHASE/'outputs/O_DOCUMENT_GUARD_TEST_RESULTS.json'
    if test_result.exists():
        record['document_guard_tests'] = json.loads(test_result.read_text(encoding='utf-8'))
        if not record['document_guard_tests']['pass']:
            raise ValueError('O_DOCUMENT_GUARD_TESTS_FAILED')
    record['execution_code_sha256'] = {
        p.relative_to(PHASE).as_posix():digest(p.read_bytes())
        for p in (Path(__file__).resolve(), PHASE/'scripts/o_catalog.py',
                  PHASE/'scripts/o_pdf_tools.py', PHASE/'tests/test_o_document_guards.py')}
    write_json(PHASE/'outputs/O_READONLY_EXECUTION_AUDIT.json', record)
    print(json.dumps({k:record[k] for k in ('document_audit','static_sources_verified',
      'pdf_first_pages_verified','novelty_status','evidence_status','writing_status')}, ensure_ascii=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--freeze', action='store_true')
    audit(parser.parse_args().freeze)
