"""Independent stdlib/Decimal prediction replay, identities, metrics and claim boundaries."""
import csv,json,hashlib,math,ast
from decimal import Decimal,localcontext
from pathlib import Path
P=Path(__file__).resolve().parent;ROOT=P.parents[1]
def rows(name):
 with (P/name).open(encoding='utf-8',newline='') as f:return list(csv.DictReader(f))
def sha(q):
 h=hashlib.sha256()
 with q.open('rb') as f:
  for block in iter(lambda:f.read(1048576),b''):h.update(block)
 return h.hexdigest()
def key(r):return (int(r['H']),r['fold'],int(r['origin_raw_cycle_id']),int(r['target_raw_cycle_id']))
def run():
 if (P/'CLOSURE_MANIFEST.json').exists():
  manifest=json.loads((P/'CLOSURE_MANIFEST.json').read_text(encoding='utf-8'))
  for r in manifest['files']:assert sha(P/r['path'])==r['SHA256'],'SEALED_ARTIFACT_CHANGED '+r['path']
 freeze=json.loads((P/'H123_FREEZE.json').read_text(encoding='utf-8'));legal=rows('H123_LEGAL_SAMPLES.csv');pred=rows('H123_PREDICTIONS.csv');metrics=rows('H123_FOLD_METRICS.csv');source=rows('SOURCE_INDEX.csv');checks=[]
 def require(test,msg):assert test,msg;checks.append(msg)
 require(len(legal)==616 and len(pred)==282 and len(metrics)==18,'exact original record counts')
 require(len({key(r) for r in legal})==len(legal),'legal sample identities unique')
 require(len({key(r)+(r['method'],) for r in pred})==len(pred),'prediction keys unique')
 legalmap={key(r):r for r in legal};params={(p['H'],p['fold']):p for p in freeze['final_parameter_records']};D=Decimal;maxdiff=D(0)
 with localcontext() as ctx:
  ctx.prec=60
  for r in pred:
   original=legalmap[key(r)];require(original['role']=='validation','prediction uses exact legal Validation key')
   for col in ['origin_calendar','target_calendar','origin_cycle_interval_id','target_cycle_interval_id','target_R']+freeze['features']:require(r[col]==original[col],'prediction identity/feature columns exact')
   require(int(r['target_raw_cycle_id'])-int(r['origin_raw_cycle_id'])==int(r['H']),'physical cycle target origin+H')
   p=params[(int(r['H']),r['fold'])];v=D(r['origin_R'])
   if r['method']=='RIDGE96':v+=sum((D(r[f])-D(str(mu)))/D(str(sd))*D(str(beta)) for f,mu,sd,beta in zip(freeze['features'],p['mean'],p['scale'],p['coef']))+D(str(p['intercept']))
   diff=abs(v-D(r['predicted_R']));maxdiff=max(maxdiff,diff);require(diff<D('2e-15'),'Decimal60 saved-coefficient replay')
   require(abs((D(r['predicted_R'])-D(r['target_R']))*D('713.83023635')-D(r['error_mVeq']))<D('2e-10'),'retention to mV-equivalent conversion')
 for r in metrics:
  g=[p for p in pred if int(p['H'])==int(r['H']) and p['fold']==r['fold'] and p['method']==r['method']];e=[(float(p['predicted_R'])-float(p['target_R']))*713.83023635 for p in g];a=sorted(abs(v) for v in e);pos=.95*(len(a)-1);lo=math.floor(pos);hi=math.ceil(pos);q=a[lo]+(a[hi]-a[lo])*(pos-lo)
  require(len(g)==int(r['n']),'metric sample count')
  for k,v in [('MAE_mVeq',sum(a)/len(a)),('RMSE_mVeq',math.sqrt(sum(v*v for v in e)/len(e))),('P95_mVeq',q),('max_mVeq',max(a))]:require(abs(float(r[k])-v)<1e-9,'independent metric '+k)
 for r in rows('H123_EQUAL_FOLD_METRICS.csv'):
  g=[m for m in metrics if m['H']==r['H'] and m['method']==r['method']];require(len(g)==3,'three nonempty folds per formal mean');require(abs(sum(float(m['MAE_mVeq']) for m in g)/3-float(r['equal_fold_MAE_mVeq']))<1e-12,'equal fold mean not pooled')
 h2=[r for r in metrics if r['H']=='2' and r['fold']=='fold_1'];p=next(r for r in h2 if r['method']=='PERSISTENCE');r=next(r for r in h2 if r['method']=='RIDGE96');require(float(r['max_mVeq'])>float(p['max_mVeq']),'H2fold1 maximum-error degradation retained')
 for r in rows('H123_TIME_AND_CODE_CHECKS.csv'):
  require(float(r['outer_label_available_max_h'])<=float(r['first_validation_origin_available_h'])+1e-9,'outer label isolation');require(float(r['inner_label_max_h'])<=float(r['inner_selection_origin_min_h'])+1e-9,'inner label isolation')
 for r in source:require(sha((ROOT/r['path_from_research_root']).resolve())==r['SHA256'],'source SHA256 unchanged '+r['key'])
 links=rows('CONNECTION_MATRIX.csv');require(next(r for r in links if r['source_module']=='Physical_diagnostics' and r['target_module']=='H123_training_loss')['implementation']=='UNQUALIFIED_INTERFACE','physical training not represented as completed');require(next(r for r in links if r['target_module']=='Validated_true_RUL')['implementation']=='UNQUALIFIED_INTERFACE','true RUL not represented as validated')
 require(next(r for r in links if r['source_module']=='H1_H3_Ridge_P1S')['implementation']=='COMPLETED_CONDITIONAL','SOH bridge remains conditional')
 scope=json.loads((P/'LITERATURE_SCOPE.json').read_text());require(not scope['new_verified_EOL_event'] and not scope['Test_csv_read'],'EOL and Test boundaries')
 # No recovery curve opened; only authorized document/caches and zip central directories.
 ledger=json.loads((P/'READ_LEDGER.json').read_text());require(not any('afterAirAirStart.csv' in s for s in ledger),'next recovery experiment not performed');require(not any('Polcurve_Nominal_RHFC_2000h.csv' in s or 'Polcurve_Nominal_RHFC_2300h.csv' in s for s in ledger),'late polarization content not opened')
 require(freeze['new_model_fits']==0 and freeze['new_hyperparameter_selection']==0,'no new training or selection')
 source_keys={r['key'] for r in source}
 for r in rows('EVIDENCE_MATRIX.csv')+links:
  require(all(k in source_keys for k in r['source_keys'].split(';')),'every matrix evidence source indexed')
 for r in rows('EXISTING_FIGURE_AND_CODE_INDEX.csv'):
  require(sha(ROOT/r['image_path'])==r['image_SHA256'],'indexed image hash');require(sha(ROOT/r['plot_script'])==r['plot_script_SHA256'],'indexed plot code hash')
  for q,h in zip(r['CSV_paths'].split(';'),r['CSV_SHA256'].split(';')):require(sha(ROOT/q)==h,'indexed figure CSV hash')
 # Test identities/contract and all source-backed computation; no synthetic performance inputs.
 result={'status':'PASS','implementation':'stdlib CSV/JSON + Decimal60; no production predictor import, no fit','checks':len(checks),'unique_check_types':len(set(checks)),'prediction_rows':len(pred),'max_prediction_difference_R':str(maxdiff),'source_files_checked':len(source),'H2_fold1_tail_worsening_retained':True,'Test_numerics_read':False,'new_fits':0,'next_experiment_not_executed':True}
 (P/'INDEPENDENT_VERIFICATION.json').write_bytes((json.dumps(result,ensure_ascii=False,indent=2)+'\n').encode('utf-8'));print(json.dumps(result,ensure_ascii=False))
if __name__=='__main__':run()
