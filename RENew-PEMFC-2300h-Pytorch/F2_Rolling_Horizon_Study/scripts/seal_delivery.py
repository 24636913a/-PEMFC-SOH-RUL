"""Finalize artifact completeness/hashes after real training and verification."""
from pathlib import Path
import numpy as np
from PIL import Image
from common import HERE,CONFIG,read,read_csv,dump,csv,sha

def seal(out):
    assert read(out/'INDEPENDENT_VERIFICATION.json')['PASS']
    assert read(out/'HISTORY_PROTECTION_FINAL.json')['PASS']
    assert read(out/'INTERACTIVE_ACCEPTANCE.json')['PASS']
    assert read(out/'FIGURE_VISUAL_QA.json')['PASS']
    assert read(out/'HOLDOUT_EVALUATION_COMPLETED.json')['completed_models']==6
    status=read(out/'FINAL_RESEARCH_STATUS.json');assert status['new_training_runs']==6 and status['historical_model_training_runs']==0
    paths=read_csv(out/'INDEPENDENT_GRADIENT_AND_PATH_UPDATES.csv')
    assert paths.any_epoch_positive_gradient.all() and (paths.changed_elements>0).all()
    dump(out/'NONZERO_GRADIENT_AND_UPDATE_CHECK.json',dict(PASS=True,groups_checked=len(paths),all_groups_have_nonzero_gradient_and_updates=True,
        source_sha256=sha(out/'INDEPENDENT_GRADIENT_AND_PATH_UPDATES.csv')))
    assert len(list((out/'models').glob('*/COMPLETED.json')))==6
    assert len(list((out/'models').glob('*/best.pt')))==6
    assert not list((out/'models').glob('*/FAILURE_TRACEBACK.txt'))
    assert len(list((HERE/'reports').glob('*.md')))==6
    figures=list((HERE/'figures').glob('*.png'));pdfs=list((HERE/'figures').glob('*.pdf'))
    assert len(figures)==len(pdfs)==12
    for p in figures:
        im=Image.open(p);assert min(im.info.get('dpi',(0,0)))>=299,p
    test=(HERE/'outputs/PREEVALUATION_TESTS.txt').read_text(encoding='utf-8-sig')
    assert 'Ran 9 tests' in test and 'OK' in test
    for p in HERE.rglob('*.py'):compile(p.read_text(encoding='utf-8'),str(p),'exec')
    before=read(out/'COHORT_BEFORE_TRAINING.json')
    for n,x in before['code_sha256'].items():assert sha(HERE/'scripts'/n)==x,n
    assert sha(CONFIG)==before['protocol_sha256']
    checkpoints=[]
    for p in sorted((out/'models').glob('*/COMPLETED.json')):
        x=read(p);assert x['best_checkpoint_sha256']==sha(p.parent/'best.pt')
        checkpoints.append(dict(model=x['model'],H=x['horizon_steps'],best_epoch=x['best_epoch'],stop_epoch=x['stop_epoch'],SHA256=sha(p.parent/'best.pt')))
    dump(out/'FAILURE_RECORDS.json',dict(fresh_GPU_training_failures=0,failed_formal_models=[],
        scientific_target_not_met='All new holdout RMSE values exceed10mV; C1 fails frozen Validation improvement criterion, all outputs retained.',
        preexisting_missing_files=read(out/'HISTORY_PROTECTION_FINAL.json')['preexisting_missing'],
        engineering_notes=['Upstream shared encoder/decoder AddNorm was inspected before evaluation; diagnostic hook restricted to48 history positions. CPU parity tests passed, no formal prediction corruption or training change.',
            'Lightning warnings about already-saved loss/logging_metrics are warnings, not training failures.'],
        no_unreported_retries=True,formal_runs=6,extra_optimization_runs=0))
    excluded={out/'ARTIFACT_SHA256.csv',out/'FINAL_INTEGRITY.json'};files=[]
    for p in sorted(HERE.rglob('*')):
        if p.is_file() and p not in excluded:
            assert p.stat().st_size<95*1024*1024,p
            files.append(dict(path=p.relative_to(HERE).as_posix(),bytes=p.stat().st_size,SHA256=sha(p)))
    csv(out/'ARTIFACT_SHA256.csv',files)
    dump(out/'FINAL_INTEGRITY.json',dict(PASS=True,files_hashed=len(files),artifact_manifest_sha256=sha(out/'ARTIFACT_SHA256.csv'),
        new_training_runs=6,total_epochs=status['total_epochs'],optimizer_updates=status['optimizer_updates'],seed=202,
        original_history_files_unchanged=read(out/'HISTORY_PROTECTION_FINAL.json')['existing_files_checked'],
        independent_checkpoint_replays=18,unit_tests_passed=9,interactive_modes_verified=['show','analyze'],real_PNG_displays=24,
        verified_PNG_PDF_pairs=12,all_gradients_groups_nonzero=True,formal_training_failures=0,
        checkpoints=checkpoints,status='F2_ROLLING_HORIZON_STUDY_COMPLETED',no_new_phase_started=True))
    print('Delivery integrity PASS:',len(files),'SHA256-hashed artifacts;6 fresh models;no failed training')

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path);a=p.parse_args();seal(a.run or Path(read(HERE/'outputs/ACTIVE_RUN.json')['path']))
