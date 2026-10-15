"""Single-GPU serial controller, crash ledger, temporary sleep prevention, watchdog."""
from screen_common import *
import ctypes
def main():
 c=cfg();lock=P/'outputs/execution_ledger/CONTROLLER.lock';lock.parent.mkdir(parents=True,exist_ok=True)
 with lock.open('x',encoding='utf-8') as s:s.write(str(os.getpid()))
 progress=P/'outputs/execution_ledger/PROGRESS.json'
 assert not progress.exists(),'CONTROLLER_ALREADY_ENTERED_NO_RESTART'
 started=time.monotonic();wall=time.time();states={str(seed):dict(status='NOT_STARTED',seed=seed) for seed in SEEDS}
 state=dict(status='RUNNING',PID=os.getpid(),start_UTC=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime(wall)),protocol_sha256=sha(P/'configs/frozen_seed_screening.json'),seeds=states)
 def save():state['elapsed_s']=time.monotonic()-started;js(progress.relative_to(P),state)
 child=None;sleep_flag=False
 try:
  if os.name=='nt':
   observed=ctypes.windll.kernel32.SetThreadExecutionState(0x80000001);assert observed,'TEMPORARY_SLEEP_PREVENTION_FAILED';sleep_flag=True
  state['temporary_system_sleep_prevention']=sleep_flag;save();append_event(dict(event='CONTROLLER_STARTED',PID=os.getpid(),seed_order=SEEDS))
  for seed in SEEDS:
   elapsed=time.monotonic()-started
   if elapsed>c['maximum_wall_s']-c['start_next_seed_reserve_s']:
    state['stop_reason']='INSUFFICIENT_FIXED_ONE_HOUR_RESERVE';break
   cfg();entry=states[str(seed)];entry.update(status='STARTED',started_UTC=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()));save();append_event(dict(event='SEED_STARTED',seed=seed))
   folder=P/f'outputs/training_logs/seed_{seed}';folder.mkdir(parents=True,exist_ok=True)
   with (folder/'console.log').open('wb') as log:
    child=subprocess.Popen([sys.executable,'-B',str(P/'scripts/train_seed.py'),'--seed',str(seed)],cwd=REPO,env=dict(os.environ,PYTHONUTF8='1',PYTHONDONTWRITEBYTECODE='1'),stdout=log,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
    seed_start=time.monotonic();timeout=min(c['per_seed_timeout_s'],c['maximum_wall_s']-(seed_start-started)-c['summary_reserve_s'])
    while child.poll() is None:
     if time.monotonic()-seed_start>timeout:
      child.terminate()
      try:child.wait(timeout=30)
      except subprocess.TimeoutExpired:child.kill();child.wait(timeout=15)
      entry.update(status='TIMEOUT',watchdog_limit_s=timeout);break
     time.sleep(1)
   code=child.returncode;child=None
   if entry['status']=='TIMEOUT':
    append_event(dict(event='SEED_TIMEOUT',seed=seed,limit_s=timeout));state['stop_reason']='WATCHDOG_TIMEOUT_REQUIRES_CUDA_REVIEW';save();break
   if code:
    entry.update(status='FAILED',returncode=code);append_event(dict(event='SEED_FAILED',seed=seed,returncode=code))
    failure=folder/'failure.json';text=failure.read_text() if failure.exists() else (folder/'console.log').read_text(errors='replace')[-10000:]
    # An unknown failure cannot certify CUDA/data validity. Stop safely, no retry.
    state['stop_reason']='TRAINING_FAILURE_REQUIRES_VALIDITY_REVIEW';state['failure_evidence']=text;save();break
   result=json.loads((P/f'outputs/metrics/seed_{seed}.json').read_text());assert result['status']=='COMPLETED'
   assert sha(P/result['checkpoint'])==result['checkpoint_sha256'] and sha(P/result['predictions'])==result['prediction_sha256'],'POSTSEED_ARTIFACT_IDENTITY_MISMATCH'
   entry.update(status='COMPLETED',RMSE_mV=result['RMSE_mV'],best_epoch=result['best_epoch'],stop_epoch=result['stop_epoch'],training_time_s=result['training_time_s'])
   ranking();save();append_event(dict(event='SEED_COMPLETED',seed=seed,RMSE_mV=result['RMSE_mV']));print(f'Completed seed{seed}: {result["RMSE_mV"]:.6f}mV',flush=True)
  cfg();completed=sum(r['status']=='COMPLETED' for r in states.values())
  state['completed_seeds']=completed;state['status']='OVERNIGHT_F2_SEED_SCREENING_COMPLETED' if completed==len(SEEDS) else 'OVERNIGHT_F2_SEED_SCREENING_PARTIAL' if completed else 'OVERNIGHT_F2_SEED_SCREENING_BLOCKED';save()
  append_event(dict(event='TRAINING_SWEEP_TERMINAL',status=state['status'],completed=completed));ranking()
  # Finalization script only reads saved predictions/checkpoints; no new fit.
  finalizer=P/'scripts/finalize.py'
  if finalizer.exists():
   with (P/'outputs/execution_ledger/finalization.log').open('wb') as log:
    final=subprocess.run([sys.executable,'-B',str(finalizer)],cwd=REPO,stdout=log,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
   state['finalization_returncode']=final.returncode;save()
 except BaseException as exc:
  if child is not None and child.poll() is None:child.terminate();child.wait(timeout=30)
  state.update(status='OVERNIGHT_F2_SEED_SCREENING_BLOCKED',exception=repr(exc),traceback=traceback.format_exc());save();append_event(dict(event='CONTROLLER_FAILED',exception=repr(exc)));raise
 finally:
  if sleep_flag:ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)
  state['temporary_sleep_prevention_restored']=True;save();lock.unlink(missing_ok=True)
if __name__=='__main__':main()
