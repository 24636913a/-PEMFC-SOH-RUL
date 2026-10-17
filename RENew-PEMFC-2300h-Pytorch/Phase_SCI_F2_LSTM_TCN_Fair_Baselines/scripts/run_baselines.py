"""Serial GPU controller. First two complete full formal pilots and pass audits."""
from baseline_common import *
import ctypes

def ledger():
    rows=[]
    for name,seed in RUNS:
        state=P/f'outputs/training_logs/{name}_seed{seed}/run_state.json';r=read(state) if state.exists() else dict(status='NOT_STARTED')
        rows.append({'model':name,'seed':seed,**r})
    js('outputs/execution_ledger/RUN_LEDGER.json',rows);return rows

def main():
    cfg();assert read(P/'outputs/audits/PREFLIGHT_REPORT.json')['status']=='PASS'
    # Temporary per-thread Windows execution state, restored on exit.
    ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)
    try:
        for i,(name,seed) in enumerate(RUNS):
            states=ledger();row=states[i]
            if row['status'] in ['FAILED','TRAINING']:raise RuntimeError('PREVIOUS_RUN_INTERRUPTED_OR_FAILED_NO_AUTOMATIC_RETRAIN')
            if i>=2:
                for pn,ps in RUNS[:2]:assert read(P/f'outputs/audits/{pn}_seed{ps}_AUDIT.json')['status']=='PASS'
            if row['status']!='COMPLETED':
                logfile=path(f'outputs/training_logs/{name}_seed{seed}/console.log')
                with logfile.open('w',encoding='utf-8',newline='\n') as stream:
                    completed=subprocess.run([sys.executable,'-B',str(P/'scripts/train_baseline.py'),'--model',name,'--seed',str(seed)],cwd=P,stdout=stream,stderr=subprocess.STDOUT)
                ledger();assert completed.returncode==0,('RUN_FAILED',name,seed)
            auditpath=P/f'outputs/audits/{name}_seed{seed}_AUDIT.json'
            if not auditpath.exists():subprocess.run([sys.executable,'-B',str(P/'scripts/audit_run.py'),'--model',name,'--seed',str(seed)],cwd=P,check=True)
            assert read(auditpath)['status']=='PASS'
            print('COMPLETED_AND_AUDITED',name,seed,flush=True)
        assert all(x['status']=='COMPLETED' for x in ledger());event('SIX_RUNS_COMPLETED')
    finally:ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)

if __name__=='__main__':
    try:main()
    except Exception as exc:ledger();js('outputs/execution_ledger/CONTROLLER_FAILURE.json',dict(error=repr(exc),traceback=traceback.format_exc()));raise
