"""Hash the delivered new phase after all independent checks, preserving failures."""
import json
from pathlib import Path
import pandas as pd
from audit_data import HERE,sha,csv,dump

def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def seal():
    out=HERE/'outputs';audit=out/'qualification_20261020_v2';run=Path(read(out/'ACTIVE_PILOT.json')['path'])
    required=[audit/'HISTORY_PROTECTION_FINAL.json',run/'INDEPENDENT_VERIFICATION.json',
        run/'PARAMETER_UPDATE_AUDIT.json',out/'INTERACTIVE_ACCEPTANCE.json',out/'FIGURE_VISUAL_QA.json']
    for p in required:assert read(p)['PASS'],p
    assert read(run/'PILOT_DECISION.json')['completed_fresh_training_runs']==4
    assert not read(run/'PILOT_DECISION.json')['expand_to_six_seeds']
    assert read(out/'FINAL_RESEARCH_STATUS.json')['holdout_outcome_rows_read']==0
    assert len(list((HERE/'reports').glob('*.md')))==5
    assert len(list((HERE/'figures').glob('*.png')))==12
    assert len(list((HERE/'figures').glob('*.pdf')))==12
    assert len(list(run.glob('*/best.pt')))==4
    for path in HERE.rglob('*.py'):compile(path.read_text(encoding='utf-8'),str(path),'exec')
    tests=(out/'FINAL_TESTS.txt').read_text(encoding='utf-8-sig')
    assert 'Ran 7 tests' in tests and 'OK' in tests
    excluded={out/'ARTIFACT_SHA256.csv',out/'FINAL_INTEGRITY.json'}
    rows=[]
    for p in sorted(HERE.rglob('*')):
        if p.is_file() and p not in excluded:
            assert p.stat().st_size<95*1024*1024,p
            rows.append(dict(path=p.relative_to(HERE).as_posix(),bytes=p.stat().st_size,SHA256=sha(p)))
    csv(out/'ARTIFACT_SHA256.csv',pd.DataFrame(rows))
    dump(out/'FINAL_INTEGRITY.json',dict(PASS=True,files_hashed=len(rows),
        sha256_manifest=sha(out/'ARTIFACT_SHA256.csv'),complete_checkpoints=4,fresh_training_runs=4,
        unit_tests_passed=7,independent_prediction_replay_exact=True,IPython_real_PNG_displays=16,
        scientific_status='LONG_TERM_PILOT_NOT_QUALIFIED_FOR_EXPANSION',
        historical_existing_files_unchanged=read(audit/'HISTORY_PROTECTION_FINAL.json')['files_hashed'],
        historical_preexisting_missing_files_recorded=read(audit/'HISTORY_PROTECTION_FINAL.json')['preexisting_missing_files'],
        holdout_outcome_rows_read=0,RUL_training=False,failures_preserved=True))
    print('DELIVERY INTEGRITY PASS:',len(rows),'hashed files')

if __name__=='__main__':seal()
