"""One-shot, pre-budgeted serial diagnostic runner; never resumes or replaces evidence."""
from pathlib import Path
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone

HERE=Path(__file__).resolve().parent
FOLDER=HERE.parent
REPO=FOLDER.parents[2]


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def save(path,obj):path.write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n',encoding='utf-8',newline='\n')


if __name__=='__main__':
    if hasattr(sys.stdout,'reconfigure'):sys.stdout.reconfigure(encoding='utf-8')
    receipt=HERE/'PROBE_EXECUTION.json'
    assert not receipt.exists(),'Inspect the existing probes; never overwrite or rerun to obtain agreement.'
    old=subprocess.check_output(['git','ls-files','-z','--',str(FOLDER.relative_to(REPO))],cwd=REPO).decode('utf-8').rstrip('\0').split('\0')
    protected={name:sha(REPO/name) for name in old}
    data=REPO/'RENew-PEMFC-2300h-Pytorch/Phase_N1_Data_Foundation/outputs/causal_feature_tables/causal_features_30min.csv'
    protected[data.relative_to(REPO).as_posix()]=sha(data)
    save(HERE/'HISTORY_BEFORE.json',protected)
    plan=dict(status='RUNNING',created_utc=datetime.now(timezone.utc).isoformat(),
        base_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip(),
        source_sha256=sha(FOLDER/'F2_Train_Interactive.py'),worker_sha256=sha(HERE/'probe_training_prefix.py'),
        fixed_budget=['original_a:6epochs','original_b:6epochs','strict_check:at most 1 step'],
        reason='Historical logs omit per-batch RNG/dropout/gradient/weight states; locate the missing evidence with two short prefixes.',
        seed=202,feature_mode='top10',formal_max_epochs=450,formal_patience=40,
        formal_protocol_unchanged=True,diagnostic_prefix_is_not_a_formal_retraining_or_score=True,
        no_historical_weights_or_predictions_loaded=True,no_holdout_scoring=True,completed=[])
    save(receipt,plan)
    try:
        for probe in ['original_a','original_b','strict_check']:
            with (HERE/(probe+'.log')).open('w',encoding='utf-8',newline='\n') as log:
                child=subprocess.Popen([sys.executable,'-B',str(HERE/'probe_training_prefix.py'),'--probe',probe],
                    cwd=FOLDER,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,encoding='utf-8',errors='strict',
                    env=dict(os.environ,PYTHONIOENCODING='utf-8'))
                for line in child.stdout:
                    log.write(line);log.flush();print(line,end='',flush=True)
                result=child.wait()
                assert result==0,(probe,result)
            plan['completed'].append(probe);save(receipt,plan)
        for name,digest in protected.items():assert sha(REPO/name)==digest,('HISTORY_CHANGED',name)
        plan.update(status='PASS',protected_files_unchanged=len(protected),finished_utc=datetime.now(timezone.utc).isoformat())
    except Exception as error:
        plan.update(status='FAILURE_PRESERVED',error=repr(error));raise
    finally:save(receipt,plan)
    print('PREFIX_PROBES_FINISHED_HISTORY_UNCHANGED',flush=True)
