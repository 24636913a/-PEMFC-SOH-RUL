"""Independent, fixed-budget launcher. Every invocation uses new GPU processes.

No torch import here: the workspace variable is passed before Python starts.
Run from VS Code terminal using the original Python 3.11/CUDA interpreter.
"""
from pathlib import Path
from datetime import datetime,timezone
import hashlib
import json
import os
import subprocess
import sys

HERE=Path(__file__).resolve().parent
FOLDER=HERE.parent
REPO=FOLDER.parents[2]


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def read(path):return json.loads(path.read_text(encoding='utf-8'))
def save(path,value):path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8',newline='\n')


def compare_prefixes(run):
    left,right=run/'strict_a',run/'strict_b'
    a=read(left/'INITIALIZATION.json');b=read(right/'INITIALIZATION.json')
    initial_equal=a==b
    traces=[]
    for folder in [left,right]:traces.append([json.loads(s) for s in (folder/'BATCH_TRACE.jsonl').read_text(encoding='utf-8').splitlines()])
    differences=[]
    for index,(u,v) in enumerate(zip(*traces)):
        different=[key for key in u if u[key]!=v[key]]
        if different:differences.append(dict(step=index+1,epoch=u['epoch'],batch=u['batch'],fields=different))
    epochs=[read(folder/'EPOCH_TRACE.json') for folder in [left,right]]
    loss_equal=True;epoch_state_equal=True
    for u,v in zip(*epochs):
        loss_equal &= all(u[key]==v[key] for key in ['train_standardized_MSE','validation_standardized_MSE'])
        epoch_state_equal &= {key:value for key,value in u.items() if key!='epoch_seconds'}=={key:value for key,value in v.items() if key!='epoch_seconds'}
    complete=[read(folder/'COMPLETE.json') for folder in [left,right]]
    passed=initial_equal and not differences and loss_equal and epoch_state_equal and all(len(x)==330 for x in traces) and all(len(x)==6 for x in epochs)
    result=dict(status='SHORT_STRICT_BITWISE_PASS' if passed else 'SHORT_STRICT_NOT_BITWISE',
        initial_equal=initial_equal,batches_each=list(map(len,traces)),epoch_loss_equal=bool(loss_equal),
        epoch_states_equal=bool(epoch_state_equal),all_batch_gradients_weights_optimizers_RNG_equal=not differences,
        first_unequal_batches=differences[:10],unequal_batches=len(differences),
        final_state_equal=complete[0]['final_state_sha256']==complete[1]['final_state_sha256'],
        final_optimizer_equal=complete[0]['final_optimizer_sha256']==complete[1]['final_optimizer_sha256'])
    save(run/'PAIR_COMPARISON.json',result)
    return passed


def execute(run,task):
    env=dict(os.environ,PYTHONIOENCODING='utf-8')
    if task=='benchmark_baseline':env.pop('CUBLAS_WORKSPACE_CONFIG',None)
    else:env['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
    command=[sys.executable,'-B',str(HERE/'deterministic_prefix.py'),'--task',task,'--run-dir',str(run)]
    with (run/(task+'.log')).open('w',encoding='utf-8',newline='\n') as log:
        child=subprocess.Popen(command,cwd=FOLDER,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,encoding='utf-8',errors='strict')
        for line in child.stdout:
            log.write(line);log.flush();print(line,end='',flush=True)
        return child.wait()


if __name__=='__main__':
    if hasattr(sys.stdout,'reconfigure'):sys.stdout.reconfigure(encoding='utf-8')
    run=HERE/'outputs'/(datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')+'_seed202_top10_strict')
    run.mkdir(parents=True,exist_ok=False)
    names=subprocess.check_output(['git','ls-files','-z','--',str(FOLDER.relative_to(REPO))],cwd=REPO).decode('utf-8').rstrip('\0').split('\0')
    protected={name:sha(REPO/name) for name in names}
    data=REPO/'RENew-PEMFC-2300h-Pytorch/Phase_N1_Data_Foundation/outputs/causal_feature_tables/causal_features_30min.csv'
    protected[data.relative_to(REPO).as_posix()]=sha(data)
    save(run/'HISTORY_BEFORE.json',protected)
    plan=dict(status='RUNNING',created_utc=datetime.now(timezone.utc).isoformat(),
        base_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip(),
        worker_sha256=sha(HERE/'deterministic_prefix.py'),launcher_sha256=sha(Path(__file__).resolve()),
        formal_program_sha256=sha(FOLDER/'F2_Train_Interactive.py'),seed=202,feature_mode='top10',
        fixed_budget={'strict_repeats':2,'diagnostic_epochs_each':6,'optimizer_updates_each':330,
            'speed_benchmark_only_if_pair_pass':{'configurations':2,'warmup_updates_each':5,'timed_updates_each':20}},
        runtime_changes={'CUBLAS_WORKSPACE_CONFIG':':4096:8','deterministic_algorithms':True,'warn_only':False},
        source_model_and_hyperparameters_unchanged=True,no_holdout_scoring=True,
        full_training_authorized_or_started=False,no_six_seed_training=True,completed=[])
    save(run/'PLAN.json',plan)
    save(HERE/'LATEST_RUN.json',{'run':run.relative_to(HERE).as_posix()})
    try:
        failed=False
        for task in ['strict_a','strict_b']:
            result=execute(run,task)
            if result!=0:
                failure=read(run/task/'FAILURE.json') if (run/task/'FAILURE.json').exists() else {'error':'See worker log'}
                plan.update(status='STRICT_PREFIX_FAILED_STOPPED',failed_task=task,failure=failure)
                failed=True;break
            plan['completed'].append(task);save(run/'PLAN.json',plan)
        if not failed:
            if compare_prefixes(run):
                plan['short_pair_passed']=True;save(run/'PLAN.json',plan)
                for task in ['benchmark_baseline','benchmark_strict']:
                    result=execute(run,task)
                    if result!=0:
                        plan.update(status='SPEED_DIAGNOSTIC_FAILED',failed_task=task);failed=True;break
                    plan['completed'].append(task)
                if not failed:plan.update(status='SHORT_PAIR_AND_SPEED_COMPLETED')
            else:plan.update(status='SHORT_PAIR_NOT_BITWISE_STOPPED',short_pair_passed=False)
        for name,digest in protected.items():assert sha(REPO/name)==digest,('HISTORY_CHANGED',name)
        assert sha(HERE/'deterministic_prefix.py')==plan['worker_sha256']
        plan.update(historical_files_unchanged=len(protected),finished_utc=datetime.now(timezone.utc).isoformat())
    except Exception as error:
        plan.update(status='ORCHESTRATION_FAILURE_PRESERVED',error=repr(error));raise
    finally:save(run/'PLAN.json',plan)
    print('DETERMINISM_VALIDATION_FINISHED',plan['status'],str(run),flush=True)
