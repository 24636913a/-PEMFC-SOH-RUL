"""Serial registered seed order, quality stopping, 12h safety budget, no reruns."""
from screen_common import *
from reliable_io import js
import ctypes,argparse
from datetime import datetime

def main(resume=False):
 c=cfg();lock=P/'outputs/execution_ledger/CONTROLLER.lock'
 lock.parent.mkdir(parents=True,exist_ok=True)
 with lock.open('x',encoding='utf-8') as f:f.write(str(os.getpid()))
 progress=P/'outputs/execution_ledger/PROGRESS.json';child=None;awake=False
 if progress.exists():
  assert resume,'EXISTING_EXECUTION_USE_EXPLICIT_RESUME_AFTER_REVIEW'
  state=json.loads(progress.read_text())
  assert not any(x['status']=='STARTED' for x in state['seeds'].values()),'UNRESOLVED_STARTED_RUN'
  wall=datetime.fromisoformat(state['start_UTC'].replace('Z','+00:00')).timestamp()
  assert time.time()-wall<c['maximum_wall_s'],'ORIGINAL_12H_BUDGET_EXHAUSTED'
 else:
  assert not resume
  wall=time.time()
  state=dict(status='RUNNING',start_UTC=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime(wall)),protocol_sha256=sha(P/'configs/frozen_seed_screening.json'),seeds={str(s):dict(seed=s,status='NOT_STARTED') for s in SEEDS})
 started=time.monotonic()-(time.time()-wall);state.update(status='RUNNING',PID=os.getpid())
 def save():
  state['elapsed_s']=time.monotonic()-started
  state['completed_seeds']=sum(x['status']=='COMPLETED' for x in state['seeds'].values())
  state['qualified_seeds']=[s for s in SEEDS if state['seeds'][str(s)].get('qualified')]
  js(progress.relative_to(P),state)
 try:
  if os.name=='nt':awake=bool(ctypes.windll.kernel32.SetThreadExecutionState(0x80000001));assert awake
  state['temporary_system_sleep_prevention']=awake;save();append_event(dict(event='CONTROLLER_STARTED',PID=os.getpid(),seed_order=SEEDS,resume=resume))
  for seed in SEEDS:
   if len(state['qualified_seeds'])>=3:state['stop_reason']='THREE_NEW_QUALIFIED_SEEDS_FOUND';break
   entry=state['seeds'][str(seed)]
   if entry['status']!='NOT_STARTED':continue
   if time.monotonic()-started>c['maximum_wall_s']-c['start_next_seed_reserve_s']:
    state['stop_reason']='12H_SAFETY_BUDGET_INSUFFICIENT_NEXT_SEED_RESERVE';break
   cfg();entry.update(status='STARTED',started_UTC=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()));save();append_event(dict(event='SEED_STARTED',seed=seed))
   folder=P/f'outputs/training_logs/seed_{seed}';folder.mkdir(parents=True,exist_ok=True)
   with (folder/'console.log').open('wb') as log:
    child=subprocess.Popen([sys.executable,'-B',str(P/'scripts/train_seed.py'),'--seed',str(seed)],cwd=REPO,env=dict(os.environ,PYTHONUTF8='1',PYTHONDONTWRITEBYTECODE='1'),stdout=log,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
    begin=time.monotonic();limit=min(c['per_seed_timeout_s'],c['maximum_wall_s']-(begin-started)-120)
    while child.poll() is None:
     if time.monotonic()-begin>limit:
      child.terminate()
      try:child.wait(timeout=30)
      except subprocess.TimeoutExpired:child.kill();child.wait(timeout=15)
      entry.update(status='TIMEOUT',watchdog_limit_s=limit);break
     time.sleep(1)
   code=child.returncode;child=None
   if entry['status']=='TIMEOUT':state['stop_reason']='WATCHDOG_TIMEOUT_REQUIRES_REVIEW';save();break
   if code:
    entry.update(status='FAILED',returncode=code);state['stop_reason']='TRAINING_FAILURE_REQUIRES_VALIDITY_REVIEW';save();append_event(dict(event='SEED_FAILED',seed=seed,returncode=code));break
   r=json.loads((P/f'outputs/metrics/seed_{seed}.json').read_text());assert r['status']=='COMPLETED'
   assert sha(P/r['checkpoint'])==r['checkpoint_sha256'] and sha(P/r['predictions'])==r['prediction_sha256']
   entry.update(status='COMPLETED',RMSE_mV=r['RMSE_mV'],endpoint_RMSE_mV=r['endpoint_RMSE_mV'],max_abs_error_mV=r['max_abs_error_mV'],best_epoch=r['best_epoch'],stop_epoch=r['stop_epoch'],training_time_s=r['training_time_s'],**qualification(r))
   ranking();save();append_event(dict(event='SEED_COMPLETED',seed=seed,**qualification(r),RMSE_mV=r['RMSE_mV']));print(f"Completed seed{seed}: {r['RMSE_mV']:.6f}mV {qualification(r)['selection_status']}",flush=True)
  save()
  if len(state['qualified_seeds'])>=3:state['status']='ROUND2_THREE_NEW_SEEDS_FOUND';state['stop_reason']='THREE_NEW_QUALIFIED_SEEDS_FOUND'
  elif state.get('stop_reason','') in ['TRAINING_FAILURE_REQUIRES_VALIDITY_REVIEW','WATCHDOG_TIMEOUT_REQUIRES_REVIEW']:state['status']='ROUND2_INTERRUPTED_OR_BLOCKED'
  else:state['status']='ROUND2_INSUFFICIENT_QUALIFIED_SEEDS'
  save();append_event(dict(event='TRAINING_SWEEP_TERMINAL',status=state['status'],completed=state['completed_seeds'],qualified=state['qualified_seeds']));ranking()
  with (P/'outputs/execution_ledger/finalization.log').open('wb') as log:
   result=subprocess.run([sys.executable,'-B',str(P/'scripts/finalize.py')],cwd=REPO,stdout=log,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
  state['finalization_returncode']=result.returncode;save()
 except BaseException as exc:
  if child is not None and child.poll() is None:child.terminate();child.wait(timeout=30)
  state.update(status='ROUND2_INTERRUPTED_OR_BLOCKED',exception=repr(exc),traceback=traceback.format_exc());save();append_event(dict(event='CONTROLLER_FAILED',exception=repr(exc)));raise
 finally:
  if awake:ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)
  state['temporary_sleep_prevention_restored']=True;save();lock.unlink(missing_ok=True)

if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('--resume',action='store_true');main(parser.parse_args().resume)
