"""Safe orchestration. Completed experiment -> verify only; fresh copy -> controlled run."""
import subprocess,sys,json,os
from pathlib import Path
P=Path(__file__).resolve().parents[1]
def invoke(script,*args):subprocess.run([sys.executable,'-B',str(P/'scripts'/script),*args],cwd=P.parent,check=True)
def run():
 os.environ['PYTHONUTF8']='1'
 if (P/'outputs/RESUME_COMPLETE.json').exists():invoke('verify_independent.py');return
 if not (P/'configs/FROZEN_PROTOCOL.json').exists():invoke('run_experiment.py','freeze')
 if not (P/'outputs/RUN_LEDGER.json').exists():
  with (P/'outputs/CPU_RUN_LOG.txt').open('w',encoding='utf-8') as log:r=subprocess.run([sys.executable,'-B',str(P/'scripts/run_experiment.py'),'run'],cwd=P.parent,stdout=log,stderr=subprocess.STDOUT)
  if r.returncode and 'INSUFFICIENT_INNER_LABEL_ISOLATION' not in (P/'outputs/CPU_RUN_LOG.txt').read_text(encoding='utf-8'):raise RuntimeError('Unexpected failure; inspect without blind restart')
 invoke('resume_and_finalize.py');invoke('qualify_and_summarize.py');invoke('verify_independent.py');invoke('write_reports.py');invoke('render_figures.py')
 print('Fresh reproduction complete; paper export requires tests and separate visual review')
if __name__=='__main__':run()
