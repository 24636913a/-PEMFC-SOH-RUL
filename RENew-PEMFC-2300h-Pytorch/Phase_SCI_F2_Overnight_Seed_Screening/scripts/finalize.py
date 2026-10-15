"""After sweep only: tests, independent replay, plots, reports, publication handoff."""
from screen_common import *
from execution_paths import progress_path
from reliable_io import js
def main():
 state=json.loads((progress_path()).read_text());assert state['status'].startswith('OVERNIGHT_')
 folder=P/'outputs/audits';folder.mkdir(exist_ok=True)
 tests=subprocess.run([sys.executable,'-B','-m','unittest','discover','-s',str(P/'tests'),'-v'],cwd=REPO,capture_output=True)
 (folder/'FINAL_TESTS.log').write_bytes(tests.stdout+tests.stderr);js('outputs/audits/FINAL_TESTS.json',dict(returncode=tests.returncode,tests=12,optimizer_steps=0))
 assert tests.returncode==0,'FINAL_TESTS_FAILED'
 for script in ['protect_history.py','audit_seed_results.py']:
  subprocess.run([sys.executable,'-B',str(P/'scripts'/script)],cwd=REPO,check=True)
 r=ranking()
 if len(r):subprocess.run([sys.executable,'-B',str(P/'scripts/generate_paper_figures.py')],cwd=REPO,check=True)
 subprocess.run([sys.executable,'-B',str(P/'scripts/summarize_results.py')],cwd=REPO,check=True)
 js('outputs/audits/FINALIZATION.json',dict(status='PASS',completed=len(r),no_optimizer_steps=True,publication='Publisher waits until controller finally restores temporary sleep prevention'))
 publish=P/'scripts/publish_results.py'
 with (P/'outputs/execution_ledger/publish.log').open('wb') as log:
  proc=subprocess.Popen([sys.executable,'-B',str(publish)],cwd=REPO,stdout=log,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
  if os.name=='nt':
   for _ in range(300):
    if 'PUBLISHER_TEMPORARY_POWER_GUARD_READY' in (P/'outputs/execution_ledger/publish.log').read_text(errors='replace'):break
    if proc.poll() is not None:raise RuntimeError('PUBLISHER_STARTUP_FAILED')
    time.sleep(.1)
   else:raise RuntimeError('PUBLICATION_POWER_HANDOFF_TIMEOUT')
 print('PUBLISHER_DISPATCHED '+str(proc.pid),flush=True)
if __name__=='__main__':
 try:main()
 except Exception as exc:js('outputs/audits/FINALIZATION_FAILURE.json',dict(status='BLOCKED',exception=repr(exc),traceback=traceback.format_exc()));raise
