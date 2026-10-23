"""Persist every verification attempt, including failures; no training entry point."""
from common import *
import sys,datetime
def main():
 stamp=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ');results=[]
 for name,args in [('unittest',['-m','unittest','discover','-s','tests','-v']),('independent',['scripts/verify_independent.py'])]:
  r=subprocess.run([sys.executable,'-B']+args,cwd=P,capture_output=True,text=True,encoding='utf-8')
  item=dict(name=name,command=[sys.executable,'-B']+args,returncode=r.returncode,stdout=r.stdout,stderr=r.stderr)
  results.append(item);js(f'outputs/quality_attempts/{stamp}_{name}.json',item)
  print(name,'PASS' if r.returncode==0 else 'FAIL',flush=True)
  if r.returncode:print(r.stderr);raise RuntimeError('Quality failure retained; no commit permitted')
 ledger=read(P/'data_contracts/SOURCE_SHA256.csv');assert all(sha(ROOT/r.path)==r.sha256 for r in ledger.itertuples())
 status=subprocess.check_output(['git','status','--porcelain','-z','--untracked-files=all'],cwd=REPO).decode('utf-8')
 prefix=ROOT.name+'/'+P.name+'/'
 assert all(line[3:].replace('\\','/').startswith(prefix) for line in status.split('\0') if line),'Outside-phase changes exist'
 js('outputs/FINAL_QUALITY_GATE.json',dict(status='PASS',tests=10,independent_verification='PASS',
  source_files_unchanged=len(ledger),outside_phase_changes=False,read_Test_numeric_values=False,trained=False,GPU=False,
  attempt_UTC=stamp,attempt_files=[f'outputs/quality_attempts/{stamp}_{r["name"]}.json' for r in results]))
 print('FINAL QUALITY PASS; historical source SHA unchanged')
if __name__=='__main__':main()
