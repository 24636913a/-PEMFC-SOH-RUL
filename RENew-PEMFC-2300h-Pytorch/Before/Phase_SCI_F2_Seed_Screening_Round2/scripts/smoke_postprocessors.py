"""CPU/in-memory check using completed real Round2 caches, never fake seed results."""
from screen_common import *
from reliable_io import js
from unittest.mock import patch
from contextlib import redirect_stdout
from types import SimpleNamespace
import io
import summarize_round2 as summaries
import generate_figures as figures
from matplotlib.figure import Figure
from monitor_progress import read_shared

def main():
 state=json.loads(read_shared(P/'outputs/execution_ledger/PROGRESS.json'))
 completed=[s for s in SEEDS if state['seeds'][str(s)]['status']=='COMPLETED']
 assert completed,'WAIT_FOR_FIRST_ACTUAL_COMPLETED_SEED'
 rows=[dict(json.loads((P/f'outputs/metrics/seed_{s}.json').read_text())) for s in completed]
 rank=pd.DataFrame([dict(r,**qualification(r)) for r in rows]).sort_values(['RMSE_mV','seed']).reset_index(drop=True);rank.insert(0,'rank',np.arange(1,len(rank)+1))
 horizon=pd.DataFrame([dict(seed=r['seed'],**x) for r in rows for x in r['horizon_metrics']])
 tables={'outputs/ROUND2_HORIZON_INDEPENDENT.csv':horizon,'outputs/ROUND2_SEED_RANKING.csv':rank};reports={};objects={};buffers={}
 # Only the in-memory copy is terminal for the report generator. Live ledger is never written.
 snapshot=dict(state,status='CPU_POSTPROCESSOR_SMOKE_ONLY')
 real_read=Path.read_text;real_rd=rd;real_stat=Path.stat;real_save=Figure.savefig
 def read(path,*a,**kw):
  if path==P/'outputs/execution_ledger/PROGRESS.json':return json.dumps(snapshot)
  return real_read(path,*a,**kw)
 def write_csv(name,df):tables[str(name)]=df.copy()
 def load(path):
  path=Path(path)
  if path.is_relative_to(P) and path.relative_to(P).as_posix() in tables:return tables[path.relative_to(P).as_posix()].copy()
  return real_rd(path)
 def savefig(fig,destination,*a,**kw):
  path=Path(destination);buffer=io.BytesIO();real_save(fig,buffer,*a,format=path.suffix[1:],**kw);buffers[path]=buffer.getvalue()
 def stat(path,*a,**kw):
  if path in buffers:return SimpleNamespace(st_size=len(buffers[path]))
  return real_stat(path,*a,**kw)
 def digest(path):
  path=Path(path)
  if path in buffers:return hashlib.sha256(buffers[path]).hexdigest()
  return sha(path)
 with patch.object(Path,'read_text',read),patch.object(summaries,'ranking',lambda:rank.copy()),patch.object(summaries,'rd',load),patch.object(summaries,'cs',write_csv),patch.object(summaries,'js',lambda name,obj:objects.update({name:obj})),patch.object(summaries,'report',lambda name,txt:reports.update({name:txt})),redirect_stdout(io.StringIO()):summaries.main()
 assert len(reports)==6 and len(tables['outputs/F2_SIX_CANDIDATE_SEEDS.csv'])==3+int(rank.qualified.sum())
 with patch.object(figures,'rd',load),patch.object(figures,'cs',write_csv),patch.object(figures,'js',lambda name,obj:objects.update({name:obj})),patch.object(figures,'sha',digest),patch.object(Figure,'savefig',savefig),patch.object(Path,'stat',stat),redirect_stdout(io.StringIO()):figures.main()
 assert len(buffers)==12 and all(len(b)>1000 for b in buffers.values())
 for name in ['F2_new_selected_fixed_origin','F5_all_candidates_fixed_origin']:
  df=tables['outputs/figure_data/'+name+'.csv']
  if len(df):assert df.groupby('candidate_seed').size().eq(24).all() and df.origin_row.nunique()==1
 result=dict(status='PASS',completed_real_seeds=completed,qualified_count=int(rank.qualified.sum()),reports_in_memory=len(reports),rendered_figures_in_memory=len(buffers),live_ledger_modified=False,formal_outputs_written=False,GPU_initialized=False,optimizer_steps=0)
 path=f'outputs/audits/CPU_POSTPROCESSOR_SMOKE_N{len(rank)}_Q{int(rank.qualified.sum())}.json'
 assert not (P/path).exists(),'SMOKE_ALREADY_DONE'
 js(path,result);print(json.dumps(result),flush=True)

if __name__=='__main__':main()
