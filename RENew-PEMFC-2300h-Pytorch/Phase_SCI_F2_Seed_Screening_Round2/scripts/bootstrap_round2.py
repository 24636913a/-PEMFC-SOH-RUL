"""One-time creation of Round2 wrappers; inherited training math stays exact."""
from pathlib import Path
P=Path(__file__).resolve().parents[1]
OLD=P.parent/'Phase_SCI_F2_Overnight_Seed_Screening'
for name in ['configs','outputs/audits','outputs/checkpoints','outputs/predictions','outputs/metrics','outputs/training_logs','outputs/execution_ledger','figures','reports','tests']:
 (P/name).mkdir(parents=True,exist_ok=True)

def write(name,s):
 f=P/name
 assert not f.exists(),f
 f.write_text(s,encoding='utf-8',newline='\n')

s=(OLD/'scripts/screen_common.py').read_text(encoding='utf-8')
s=s.replace("BASE='bca03706f06673e60b5ee5f2cfa82d33e17033db'", "BASE='69f79bd5b1286e03a7100950c164ade464858d45'\nOLD=ROOT/'Phase_SCI_F2_Overnight_Seed_Screening'")
start=s.index('SEEDS=');end=s.index('\n',start)
s=s[:start]+'SEEDS=list(range(201,251))'+s[end:]
a=s.index('def cfg():');b=s.index('\ndef data():',a)
s=s[:a]+'''def cfg():
 f=P/'configs/frozen_seed_screening.json'
 assert sha(f)==f.with_suffix('.sha256').read_text().strip(),'FROZEN_CONFIG_IDENTITY_MISMATCH'
 c=json.loads(f.read_text());assert c['seeds']==SEEDS
 for key in ['critical_code_sha256','data_sha256']:
  for path,digest in c[key].items():assert sha(P/path)==digest,(key,path)
 for path,digest in c['source_sha256'].items():
  assert sha(Path(path) if Path(path).is_absolute() else REPO/path)==digest,('HISTORICAL_IDENTITY_MISMATCH',path)
 assert c['thresholds']==dict(RMSE_mV=6.3,endpoint_RMSE_mV=10.0,max_abs_error_mV=50.0)
 assert c['maximum_wall_s']==43200 and c['qualified_stop_count']==3
 return c
''' +s[b:]
a=s.index('def ranking():')
s=s[:a]+'''def qualification(r):
 a=r['RMSE_mV']<=6.3
 b=r['endpoint_RMSE_mV']<=10.0 and r['max_abs_error_mV']<=50.0
 return dict(A_pass=a,B_pass=b,qualified=a and b,selection_status='QUALIFIED_HIGH_QUALITY_F2_SEED' if a and b else 'LOW_RMSE_WITH_TAIL_RISK' if a else 'NOT_QUALIFIED')

def ranking():
 rows=[]
 for seed in SEEDS:
  f=P/f'outputs/metrics/seed_{seed}.json'
  if f.exists():
   r=json.loads(f.read_text())
   if r['status']=='COMPLETED':rows.append(dict(r,**qualification(r)))
 d=pd.DataFrame(rows)
 if len(d):
  d=d.sort_values(['RMSE_mV','seed'],kind='stable').reset_index(drop=True)
  d.insert(0,'rank',np.arange(1,len(d)+1));cs('outputs/ROUND2_SEED_RANKING.csv',d)
  cs('outputs/ROUND2_QUALIFIED_SEEDS.csv',d[d.qualified].sort_values('seed'))
  cs('outputs/ROUND2_HORIZON_METRICS.csv',pd.DataFrame([dict(seed=int(r['seed']),**x) for r in rows for x in r['horizon_metrics']]))
 return d
'''
write('scripts/screen_common.py',s)
s=(OLD/'scripts/train_seed_v3.py').read_text(encoding='utf-8')
s=s.replace('from reliable_io import js,verify_IO_repair','from reliable_io import js,atomic_replace')
s=s.replace(' from repair_identity import verify_repair\n verify_repair()\n verify_IO_repair()\n','')
s=s.replace('os.replace(tmp,checkpoint)','atomic_replace(tmp,checkpoint)')
s=s.replace("  js(f'outputs/metrics/seed_{seed}.json',result);", "  result['step1_RMSE_mV']=float(1000*np.sqrt(np.mean((saved.loc[saved.horizon_step.eq(1),'y_pred_delta_V'].to_numpy()-saved.loc[saved.horizon_step.eq(1),'y_true_delta_V'].to_numpy())**2)))\n  result['horizon_metrics']=[dict(horizon_step=int(step),horizon_h=float(step*.5),RMSE_mV=float(1000*np.sqrt(np.mean((g.y_pred_delta_V.to_numpy()-g.y_true_delta_V.to_numpy())**2))),MAE_mV=float(1000*np.mean(abs(g.y_pred_delta_V.to_numpy()-g.y_true_delta_V.to_numpy()))),n=len(g)) for step,g in saved.groupby('horizon_step')]\n  js(f'outputs/metrics/seed_{seed}.json',result);")
write('scripts/train_seed.py',s)
write('.gitignore','__pycache__/\n*.pyc\n*.writing\noutputs/training_logs/*/batch_events.jsonl\noutputs/execution_ledger/CONTROLLER.lock\noutputs/execution_ledger/publish.log\noutputs/GIT_RECEIPT.local.json\n')
write('.gitattributes','* -text whitespace=blank-at-eol,blank-at-eof,space-before-tab,cr-at-eol\n*.pt binary\n*.npz binary\n*.png binary\n*.pdf binary\n')
s=(OLD/'scripts/protect_history.py').read_text(encoding='utf-8')
write('scripts/protect_history.py',s)
s=(OLD/'scripts/audit_delivery.py').read_text(encoding='utf-8')
write('scripts/audit_delivery.py',s)
print('ROUND2_WRAPPERS_CREATED')
