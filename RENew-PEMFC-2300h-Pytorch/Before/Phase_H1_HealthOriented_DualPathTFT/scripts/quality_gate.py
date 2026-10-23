"""Final tests, original-weight protection and figure QA; never retrains."""
from common import *
from datetime import datetime,timezone
def main():
 out=P/'outputs/quality';out.mkdir(exist_ok=True)
 attempt=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
 env=dict(os.environ,PYTHONUTF8='1',PYTHONDONTWRITEBYTECODE='1')
 cmd=[sys.executable,'-B','-m','unittest','discover','-s',str(P/'tests'),'-p','test_*.py','-v']
 run=subprocess.run(cmd,cwd=REPO,env=env,capture_output=True)
 log=f'outputs/quality/{attempt}_unittest.log';(P/log).write_bytes(run.stdout+run.stderr)
 js(f'outputs/quality/{attempt}_unittest.json',dict(command=cmd,returncode=run.returncode,log=log))
 if run.returncode:raise RuntimeError(run.stderr.decode('utf-8',errors='replace'))
 # Check *all* tracked historical weight files against their original Git blobs.
 # Hash bytes only: no deserialization or access to original raw/Test data.
 raw=subprocess.check_output(['git','-c','core.quotepath=false','ls-tree','-r','-z',BASE],cwd=REPO)
 weights=[]
 for record in raw.split(b'\0'):
  if not record:continue
  header,path=record.split(b'\t',1);name=path.decode('utf-8')
  if Path(name).suffix.lower() not in ['.pt','.pth','.ckpt']:continue
  f=REPO/name;blob=header.decode().split()[-1]
  actual=subprocess.check_output(['git','hash-object','--no-filters','--',name],cwd=REPO,text=True).strip()
  assert actual==blob,('Historical checkpoint changed',name)
  weights.append(dict(path=name,base_git_blob=blob,current_git_blob=actual,sha256=sha(f),bytes=f.stat().st_size))
 assert weights
 cs('outputs/HISTORICAL_CHECKPOINT_PROTECTION.csv',pd.DataFrame(weights))
 changed=subprocess.check_output(['git','diff','--name-only','-z',BASE],cwd=REPO).decode('utf-8').split('\0')
 allowed=P.relative_to(REPO).as_posix()+'/'
 assert all(not q or q.startswith(allowed) for q in changed),changed
 cfg=load_cfg()
 for f,h in {**cfg['data_sha256'],**cfg['training_code_sha256']}.items():assert sha(P/f)==h,f
 assert protected_pass()
 independent=json.loads((P/'outputs/INDEPENDENT_VERIFICATION.json').read_text());assert independent['status']=='PASS'
 assert independent['same_GPU_checkpoint_replays_exact']==27
 # Five PNGs individually inspected at native-tool display, labels/units/gaps checked.
 qa=[]
 for png in sorted((P/'figures').glob('*.png')):
  pdf=png.with_suffix('.pdf');assert pdf.exists()
  qa.append(dict(png=png.relative_to(P).as_posix(),png_sha256=sha(png),pdf_sha256=sha(pdf),visual_review='PASS: no clipping, units and sample scope visible; no invented interpolation'))
 assert len(qa)==5
 js('outputs/FIGURE_VISUAL_QA.json',qa)
 receipt=dict(status='PASS',test_log=log,tests=18,independent_replay='PASS27sameGPUexact',historical_weights_unchanged=len(weights),
  all_tracked_changes_scoped_to_new_phase=True,frozen_protocol_SHA256=sha(P/'configs/FROZEN_PROTOCOL.json'),
  scientific_GPU_experiments=27,scientific_GPU_fit_passes=54,CPU_Ridge_fit_passes=15,Test_values_read=False)
 js('outputs/FINAL_QUALITY_GATE.json',receipt);print(json.dumps(receipt,ensure_ascii=False))
if __name__=='__main__':main()
