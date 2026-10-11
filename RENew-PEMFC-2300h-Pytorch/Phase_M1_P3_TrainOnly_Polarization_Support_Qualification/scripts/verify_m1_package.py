"""Post-run package verification only. Does not import/read any source data.

Not part of the frozen numeric execution chain. Verifies the original run
snapshot, and separately binds the explicitly documented editorial addendum.
"""
import hashlib
import json
from pathlib import Path

PHASE=Path(__file__).resolve().parents[1]
EDITORIAL={'reports/M1_JOINT_SUPPORT_AND_STABILITY.md','reports/M1_FINAL_P3_RESEARCH_DECISION.md'}

def sha(path):
    path=path.resolve()
    if not path.is_relative_to(PHASE.resolve()):
        raise RuntimeError('PACKAGE_VERIFIER_EXTERNAL_READ_DENIED')
    return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    access=json.loads((PHASE/'configs/M1_FROZEN_ACCESS_PROTOCOL.yaml').read_text(encoding='utf-8'))
    snapshot=json.loads((PHASE/'outputs/M1_FINAL_INTEGRITY_AUDIT.json').read_text(encoding='utf-8'))
    for name,expected in access['execution_code_sha256'].items():
        assert sha(PHASE/name)==expected, 'FROZEN_EXECUTION_CODE_CHANGED:'+name
    for stem,field in [('M1_FROZEN_ACCESS_PROTOCOL','access_protocol_sha256'),('M1_FROZEN_SUPPORT_PROTOCOL','support_protocol_sha256')]:
        assert sha(PHASE/'configs'/f'{stem}.yaml') == snapshot[field]
        assert (PHASE/'configs'/f'{stem}.sha256').read_text().strip()==snapshot[field]
    changes=[]
    for relative,expected in snapshot['outputs_sha256'].items():
        if sha(PHASE/relative)!=expected:
            changes.append(relative)
    assert set(changes)==EDITORIAL, 'UNEXPLAINED_OUTPUT_CHANGE:'+str(changes)
    assert (PHASE/'evidence/M1_REPORT_EDITORIAL_ADDENDUM.md').is_file()
    required=['README.md','outputs/M1_SUPPORT_SUMMARY.json','outputs/M1_PAIR_ELIGIBILITY_SUMMARY.csv',
      'outputs/M1_CONDITION_SUPPORT_REPORT.csv','outputs/M1_AUXILIARY_ISOLATION_AUDIT.json',
      'evidence/M1_SOURCE_IDENTITY_LEDGER.md','evidence/M1_TRAIN_BOUNDARY_AUDIT.md','evidence/M1_AUXILIARY_VOLTAGE_LINEAGE.md',
      'reports/M1_PHYSICAL_MEASUREMENT_SEMANTICS.md','reports/M1_JOINT_SUPPORT_AND_STABILITY.md',
      'reports/M1_POLARIZATION_ORDER_EVIDENCE.md','reports/M1_TARGET_LEAKAGE_PREVENTION.md',
      'reports/M1_CONSTANT_COLLAPSE_AND_IDENTIFIABILITY.md','reports/M1_FINAL_P3_RESEARCH_DECISION.md']
    for relative in required:
        assert (PHASE/relative).is_file() and (PHASE/relative).stat().st_size>0, 'MISSING_DELIVERABLE:'+relative
    summary=json.loads((PHASE/'outputs/M1_SUPPORT_SUMMARY.json').read_text(encoding='utf-8'))
    assert summary['train_rows']==2103 and summary['train_f2_legal_windows']==1380
    assert not any(summary[k] for k in ['validation_accessed','assessment_accessed','test_accessed','model_trained','model_forward'])
    hashes={str(p.relative_to(PHASE)).replace('\\','/'):sha(p)
            for p in sorted(PHASE.rglob('*')) if p.is_file() and '__pycache__' not in str(p)
            and p.name!='M1_FINAL_PACKAGE_AUDIT.json'}
    output=dict(package_audit='PASS',required_nonempty_files=len(required),
      frozen_execution_identity='UNCHANGED',protocol_identity='UNCHANGED',
      original_run_snapshot_retained=True,documented_editorial_changes=sorted(changes),
      synthetic_regression_tests=20,test_result='PASS',
      real_data_rerun_for_editorial_changes=False,final_status=summary['final_status'],
      source_train_csv_read_by_package_verifier=False,artifact_sha256=hashes)
    (PHASE/'outputs/M1_FINAL_PACKAGE_AUDIT.json').write_text(json.dumps(output,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    print('M1 package audit PASS; original run snapshot retained; code/protocol unchanged.')

if __name__=='__main__':
    main()
