"""Artifact ledger and old-result protection; does not train or select a model."""
from pathlib import Path
import json
import frozen_f2 as f
from io_utils import js,csv
from protocol import verify_history


def finalize(run):
    """Run only after the conditional experiment workflow has stopped."""
    status=f.read_json(run/'FINAL_DECISION.json')
    f.require(status['status'] in ['NO_QUALIFIED_UPGRADE','FINAL_EVALUATION_COMPLETE'],
              'EXPERIMENT_NOT_FINISHED')
    for source in [run/'CODE_BEFORE_A.json',run/'B_REGISTRATION.json']:
        saved=f.read_json(source)
        hashes=saved if source.name=='CODE_BEFORE_A.json' else saved['source_sha256']
        for name,digest in hashes.items():f.require(f.sha(f.HERE/name)==digest,'FINAL_FROZEN_MATH_CHANGED:'+name)
    records=[]
    for path in sorted((run/'models').glob('*/*/TRAINING_LEDGER.json')):
        ledger=f.read_json(path)
        f.require(ledger['status']=='TRAINED','INCOMPLETE_MODEL_AT_FINALIZATION')
        f.require(f.read_json(path.parent/'EVALUATION_AUDIT.json')['status']=='PASS','MODEL_AUDIT')
        f.require(f.sha(path.parent/'best_model.pt')==ledger['best_checkpoint_sha256'],'FINAL_CHECKPOINT_SHA')
        records.append(dict(stage=ledger['stage'],candidate=ledger['candidate'],seed=ledger['seed'],
            best_epoch=ledger['best_epoch'],stop_epoch=ledger['stop_epoch'],
            backward_calls=ledger['backward_calls'],optimizer_steps=ledger['optimizer_steps'],
            initialization_sha256=ledger['initialization_sha256'],
            checkpoint_sha256=ledger['best_checkpoint_sha256'],
            checkpoint=path.parent.relative_to(f.HERE).as_posix()+'/best_model.pt',
            historical_checkpoint_loaded=ledger['historical_checkpoint_loaded'],
            training_seconds=ledger['training_seconds']))
    f.require(len(records)<=27 and all(not r['historical_checkpoint_loaded'] for r in records),'FRESH_BUDGET')
    csv(run/'tables/ALL_FRESH_TRAININGS.csv',records)
    from artifact_index import build
    figure_receipt=build(run)
    verify_history(run)
    f.require(f.sha(f.SOURCE)==f.read_json(run/'PROTOCOL.json')['data_sha256'],'DATA_CHANGED')
    manifest=[]
    excluded={'ARTIFACT_SHA256.csv','FINAL_INTEGRITY.json'}
    for path in sorted(f.HERE.rglob('*')):
        if not path.is_file() or '__pycache__' in path.parts or path.name in excluded:
            continue
        manifest.append(dict(path=path.relative_to(f.HERE).as_posix(),bytes=path.stat().st_size,
                             SHA256=f.sha(path)))
    csv(run/'ARTIFACT_SHA256.csv',manifest)
    result=dict(status='PASS',fresh_trainings=len(records),artifact_files=len(manifest),
        artifact_manifest_SHA256=f.sha(run/'ARTIFACT_SHA256.csv'),
        historical_files_unchanged=f.read_json(run/'HISTORY_PROTECTION.json')['unchanged_SHA256_files'],
        figure_validation=figure_receipt,
        data_unchanged=True,checkpoints_verified=True,
        frozen_A_and_B_mathematical_source_SHA256_verified=True,
        scientific_status=status['status'],no_unsupported_training_claims=True)
    js(run/'FINAL_INTEGRITY.json',result)
    return result
