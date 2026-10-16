"""Terminal-only audits, figures/reports and safe publication; never fit."""
from screen_common import *
from reliable_io import js

def run(script,*args):
 result=subprocess.run([sys.executable,'-B',str(P/'scripts'/script),*args],cwd=REPO)
 assert result.returncode==0,(script,result.returncode)

def main():
 state=json.loads((P/'outputs/execution_ledger/PROGRESS.json').read_text())
 assert state['status']!='RUNNING' and not any(x['status']=='STARTED' for x in state['seeds'].values())
 if state['status']=='ROUND2_INTERRUPTED_OR_BLOCKED':
  entered=[int(s) for s,r in state['seeds'].items() if r['status']!='NOT_STARTED']
  last=max(entered) if entered else 'none'
  js(f'outputs/audits/INTERRUPTION_REVIEW_SEED_{last}.json',dict(status='REVIEW_REQUIRED_NOT_PUBLISHED_AS_COMPLETE',stop_reason=state.get('stop_reason'),completed=state['completed_seeds'],qualified=state['qualified_seeds'],original_start_UTC=state['start_UTC'],new_training_started=False,optimizer_steps=0))
  print('INTERRUPTION_PRESERVED_FOR_VALIDITY_REVIEW; no duplicate partial commit',flush=True)
  return
 result=subprocess.run([sys.executable,'-B','-m','unittest','discover','-s',str(P/'tests'),'-v'],cwd=REPO,capture_output=True)
 (P/'outputs/audits/FINAL_TESTS.log').write_bytes(result.stdout+result.stderr)
 js('outputs/audits/FINAL_TESTS.json',dict(returncode=result.returncode,optimizer_steps=0));assert result.returncode==0
 run('protect_history.py');run('audit_round2.py');run('summarize_round2.py');run('generate_figures.py')
 js('outputs/audits/FINALIZATION.json',dict(status='PASS',completed=state['completed_seeds'],qualified=state['qualified_seeds'],optimizer_steps=0))
 with (P/'outputs/execution_ledger/publish.log').open('wb') as log:
  proc=subprocess.Popen([sys.executable,'-B',str(P/'scripts/publish_results.py')],cwd=REPO,stdout=log,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
  for _ in range(120):
   time.sleep(.5)
   if b'PUBLISHER_TEMPORARY_POWER_GUARD_READY' in (P/'outputs/execution_ledger/publish.log').read_bytes():break
   assert proc.poll() is None,'PUBLISHER_EXITED_EARLY'
  else:raise RuntimeError('PUBLISHER_AWAKE_HANDOFF_FAILED')

if __name__=='__main__':
 try:main()
 except Exception as exc:js('outputs/audits/FINALIZATION_FAILURE.json',dict(status='FAILED',exception=repr(exc),traceback=traceback.format_exc()));raise
