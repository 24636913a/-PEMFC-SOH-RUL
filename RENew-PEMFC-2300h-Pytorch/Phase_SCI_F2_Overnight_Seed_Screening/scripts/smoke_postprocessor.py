"""CPU/in-memory smoke of current reports, including all completed Top5 horizons."""
from screen_common import *
from reliable_io import js
from unittest.mock import patch
import summarize_results as report
import statistics,math
def main():
 rank=rd(P/'outputs/F2_SEED_RANKING.csv');rows=[]
 for seed in rank.seed:
  d=rd(P/f'outputs/predictions/F2_seed_{seed}.csv')
  for step,g in d.groupby('horizon_step'):
   error=[float(a)-float(b) for a,b in zip(g.y_pred_delta_V,g.y_true_delta_V)]
   rows.append(dict(seed=int(seed),horizon_step=int(step),horizon_h=float(step*.5),RMSE_mV=1000*math.sqrt(math.fsum(x*x for x in error)/len(error)),MAE_mV=1000*math.fsum(abs(x) for x in error)/len(error),n=len(g)))
 horizon=pd.DataFrame(rows);original_rd=report.rd;tables={};texts={};objects={}
 def read(f):return horizon.copy() if Path(f).resolve()==P/'outputs/F2_HORIZON_METRICS.csv' else original_rd(f)
 with patch.object(report,'rd',side_effect=read),patch.object(report,'cs',side_effect=lambda k,v:tables.__setitem__(str(k),v.copy())),patch.object(report,'md',side_effect=lambda k,v:texts.__setitem__(str(k),v)),patch.object(report,'js',side_effect=lambda k,v:objects.__setitem__(str(k),v)):
  report.main()
 summary=objects['outputs/FINAL_SUMMARY.json'];assert len(texts)==7 and summary['completed']==len(rank)
 assert abs(summary['full_pool_RMSE_mean_mV']-statistics.mean(rank.RMSE_mV))<1e-12
 assert abs(summary['full_pool_RMSE_sample_std_mV']-statistics.stdev(rank.RMSE_mV))<1e-12
 assert len(tables['outputs/F2_TOP5_HORIZON_DIAGNOSTICS.csv'])==min(5,len(rank))
 assert len(tables['outputs/ALL_SEEDS_AT_GLOBAL_WORST_KEY.csv'])==len(rank)
 identity=tables['outputs/audits/RUN_IDENTITY_INDEX.csv'];assert identity.actual_worker.isin(['scripts/train_seed_v2.py','scripts/train_seed_v3.py']).all()
 js('outputs/audits/POST_IO_REPAIR_POSTPROCESSOR_SMOKE.json',dict(status='PASS',scope='Only cached completed data; all reports/tables held in RAM, no premature final report',completed_seeds=len(rank),reports=7,CSV_tables=len(tables),all_Top5_horizons_present=True,GPU_used=False,optimizer_steps=0,report_source_SHA256=sha(P/'scripts/summarize_results.py')))
 print('CURRENT_REPORTS_AND_TOP5_HORIZONS_SMOKE_PASS',flush=True)
if __name__=='__main__':main()
