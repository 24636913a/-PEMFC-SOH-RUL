"""Seal quiescent phase, commit only its files and attempt ordinary safe main push."""
import os,ctypes
_publish_awake=False
if __name__=='__main__' and os.name=='nt':
 _publish_awake=bool(ctypes.windll.kernel32.SetThreadExecutionState(0x80000001))
 assert _publish_awake,'PUBLICATION_TEMPORARY_POWER_GUARD_FAILED'
 print('PUBLISHER_TEMPORARY_POWER_GUARD_READY',flush=True)
from screen_common import *
from execution_paths import progress_path
from reliable_io import js
def git(*args,timeout=120):return subprocess.check_output(['git',*args],cwd=REPO,timeout=timeout)
def main():
 statepath=progress_path()
 for _ in range(120):
  state=json.loads(statepath.read_text())
  if state.get('temporary_sleep_prevention_restored'):break
  time.sleep(1)
 else:raise RuntimeError('CONTROLLER_NOT_QUIESCENT_NO_SEAL')
 time.sleep(2)
 assert json.loads((P/'outputs/audits/FINALIZATION.json').read_text())['status']=='PASS'
 cfg();prefix=P.relative_to(REPO).as_posix()+'/'
 assert git('branch','--show-current').decode().strip()=='main'
 assert not git('diff','--cached','--name-only','-z'),'PREEXISTING_INDEX_NOT_EMPTY'
 changed=git('status','--porcelain','-z').decode('utf-8').split('\0')
 assert all(not r or r[3:].startswith(prefix) for r in changed),'OTHER_UNCOMMITTED_CHANGES_NO_COMMIT'
 # Recorded as-of this inventory: no unverified backup claims. Successful Git
 # push receipt later proves tracked checkpoint/prediction delivery separately.
 inventory=[];partial_epochs={}
 for weight in (P/'outputs/checkpoints').glob('*.pt'):
  import re
  hit=re.search(r'seed_(\d+)',weight.name)
  if hit and not (P/f'outputs/metrics/seed_{hit.group(1)}.json').exists():
   partial_epochs[int(hit.group(1))]=torch.load(weight,map_location='cpu',weights_only=True).get('best_epoch')
 for folder in ['outputs/checkpoints','outputs/predictions','outputs/training_logs']:
  for f in sorted((P/folder).rglob('*')):
   if not f.is_file():continue
   parts=f.relative_to(P).parts;seed=None
   import re
   match=re.search(r'seed_(\d+)',f.as_posix())
   if match:seed=int(match.group(1))
   metricfile=P/f'outputs/metrics/seed_{seed}.json';met=json.loads(metricfile.read_text()) if metricfile.exists() else {}
   inventory.append(dict(path=f.relative_to(P).as_posix(),local_absolute_path=str(f),bytes=f.stat().st_size,sha256=sha(f),seed=seed,best_epoch=met.get('best_epoch',partial_epochs.get(seed)),artifact_status='COMPLETED' if met.get('status')=='COMPLETED' else 'FAILED_OR_PARTIAL_NOT_RANKED',
    backed_up_at_inventory_creation=False,Git_delivery_planned=f.name!='batch_events.jsonl' and f.suffix!='.writing',backup_note='Incomplete atomic write, local-only and unqualified' if f.suffix=='.writing' else 'Full batch journals local-only; never backed up by this task' if f.name=='batch_events.jsonl' else 'Git push pending; verify final receipt'))
 cs('outputs/LOCAL_ARTIFACT_INVENTORY.csv',pd.DataFrame(inventory))
 # Imported or locally developed code and every deliverable get file hashes.
 cs('outputs/audits/FINAL_CODE_SHA256.csv',pd.DataFrame([dict(path=f.relative_to(P).as_posix(),sha256=sha(f)) for f in sorted((P/'scripts').glob('*.py'))]+[dict(path=f.relative_to(P).as_posix(),sha256=sha(f)) for f in sorted((P/'tests').glob('*.py'))]))
 exclude={'outputs/ARTIFACT_SHA256.csv','outputs/SEAL_RECEIPT.json','outputs/GIT_RECEIPT.local.json','outputs/execution_ledger/publish.log','outputs/execution_ledger/CONTROLLER.lock'}
 artifacts=[]
 for f in sorted(P.rglob('*')):
  if not f.is_file() or '__pycache__' in f.parts or f.suffix in ['.pyc','.writing']:continue
  path=f.relative_to(P).as_posix()
  if path in exclude or f.name=='batch_events.jsonl':continue
  artifacts.append(dict(path=path,bytes=f.stat().st_size,sha256=sha(f)))
 cs('outputs/ARTIFACT_SHA256.csv',pd.DataFrame(artifacts));js('outputs/SEAL_RECEIPT.json',dict(status='SEALED',artifacts=len(artifacts),bytes=sum(r['bytes'] for r in artifacts),
  manifest_sha256=sha(P/'outputs/ARTIFACT_SHA256.csv'),scientific_protocol_sha256=sha(P/'configs/frozen_seed_screening.json'),
  repair_sha256=sha(P/'configs/LOGGER_REPAIR_AMENDMENT.json'),local_only_full_journals_indexed=True,no_Test=True))
 git('add','--',prefix);names=[n for n in git('diff','--cached','--name-only','-z').decode('utf-8').split('\0') if n]
 assert names and all(n.startswith(prefix) for n in names),'INDEX_SCOPE_MISMATCH'
 check=subprocess.run(['git','diff','--cached','--check'],cwd=REPO,capture_output=True)
 assert check.returncode==0,check.stdout.decode(errors='replace')+check.stderr.decode(errors='replace')
 for row in artifacts:
  index=git('show',':'+prefix+row['path']);assert hashlib.sha256(index).hexdigest()==row['sha256'],('INDEX_BYTES_MISMATCH',row['path'])
 git('commit','-q','-m','Complete registered dynamic F2 seed screening with full-pool and selection audits',timeout=180)
 head=git('rev-parse','HEAD').decode().strip();receipt=dict(status='LOCAL_COMMIT_COMPLETE',commit=head,phase=P.name,execution_progress=progress_path().relative_to(P).as_posix(),only_new_phase_committed=True,proxy_settings_changed=False)
 js('outputs/GIT_RECEIPT.local.json',receipt)
 push=subprocess.run(['git','push','origin','main'],cwd=REPO,capture_output=True,timeout=900)
 receipt.update(push_returncode=push.returncode,push_stdout=push.stdout.decode(errors='replace'),push_stderr=push.stderr.decode(errors='replace'))
 if push.returncode:receipt['status']='LOCAL_COMMIT_PUSH_FAILED'
 else:
  remote=git('ls-remote','origin','refs/heads/main',timeout=60).decode().split()[0];assert remote==head,'PUSH_REMOTE_IDENTITY_MISMATCH'
  receipt.update(status='PUSH_VERIFIED',remote_main=remote,tracked_checkpoints_predictions_backed_up_to_GitHub=True,local_only_batch_journals_backed_up=False)
 receipt['worktree_clean']=not bool(git('status','--porcelain'));js('outputs/GIT_RECEIPT.local.json',receipt);print(json.dumps(receipt,ensure_ascii=False),flush=True)
if __name__=='__main__':
 try:main()
 except Exception as exc:
  f=P/'outputs/GIT_RECEIPT.local.json';prior=json.loads(f.read_text()) if f.exists() else {}
  prior.update(status='PUBLISH_FAILED_PRESERVED',execution_progress=progress_path().relative_to(P).as_posix(),exception=repr(exc),traceback=traceback.format_exc(),proxy_settings_changed=False);js('outputs/GIT_RECEIPT.local.json',prior);raise
 finally:
  if _publish_awake:
   ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)
   f=P/'outputs/GIT_RECEIPT.local.json'
   if f.exists():
    prior=json.loads(f.read_text());prior['publisher_temporary_sleep_guard_restored']=True;js('outputs/GIT_RECEIPT.local.json',prior)
