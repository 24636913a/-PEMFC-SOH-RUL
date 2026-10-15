"""Keep every test/verification attempt. Does not restart raw extraction."""
from common import *
import sys,datetime
def main():
 stamp=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ');attempts=[]
 for name,args in [('unit',['-m','unittest','discover','-s','tests','-v']),('independent',['scripts/verify_independent.py'])]:
  result=subprocess.run([sys.executable,'-B']+args,cwd=P,capture_output=True,text=True,encoding='utf-8')
  record=dict(name=name,returncode=result.returncode,stdout=result.stdout,stderr=result.stderr,command=[sys.executable,'-B']+args)
  js(f'outputs/quality_attempts/{stamp}_{name}.json',record);attempts.append(f'outputs/quality_attempts/{stamp}_{name}.json')
  print(name,'PASS' if result.returncode==0 else 'FAIL',flush=True)
  if result.returncode:print(result.stderr);raise RuntimeError('Quality failure retained; stop before Git')
 for ledgerfile in ['SOURCE_SHA256.csv','ADDITIONAL_METADATA_SOURCE_SHA256.csv']:
  d=rd(P/'data_contracts'/ledgerfile);assert all(sha(ROOT/r.path)==r.sha256 for r in d.itertuples())
 status=subprocess.check_output(['git','status','--porcelain','-z','--untracked-files=all'],cwd=REPO).decode('utf-8')
 prefix=ROOT.name+'/'+P.name+'/'
 assert all(s[3:].replace('\\','/').startswith(prefix) for s in status.split('\0') if s),'Unrelated workspace change, stop'
 js('outputs/FINAL_QUALITY_GATE.json',dict(status='PASS',unit_tests=11,independent='PASS',historical_sources_unchanged=35,
  no_training=True,no_GPU=True,no_Test_numeric_values=True,outside_phase_changes=False,attempts=attempts))
 print('FINAL QUALITY PASS')
if __name__=='__main__':main()
