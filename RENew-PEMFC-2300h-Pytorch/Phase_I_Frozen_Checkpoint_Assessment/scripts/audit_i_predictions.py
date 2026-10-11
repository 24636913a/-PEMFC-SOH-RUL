"""Independent audit of all nine persisted caches; no neural model invocation."""
from i_common import *
from i_metrics import project_metrics,independent_metrics,diagnostics

def audit_all(cfg):
 truth=pd.read_csv(OUT/'preparation_audit/I_ASSESSMENT_KEY_TRUTH_AUTHORITY.csv',float_precision='round_trip')
 rows=[];checks=[]
 for spec,identity in zip(RUNS,cfg['checkpoints']):
  path=OUT/'predictions'/f"{spec['run_id']}_assessment.csv"
  result=read_json(OUT/'execution_ledger'/f"{spec['run_id']}_result.json")
  if not path.is_file() or path.stat().st_size==0 or digest(path)!=result['prediction_file_sha256']:raise RuntimeError('MISSING_OR_CHANGED_PREDICTION_CACHE:'+spec['run_id'])
  frame=pd.read_csv(check_read(path),float_precision='round_trip')
  frame=validate_predictions(frame,truth,spec,digest(PROTOCOL),identity['sha256'],cfg['input_fingerprints']['assessment_input_sha256'])
  a=project_metrics(frame);b=independent_metrics(frame)
  for key in a:
   difference=abs(float(a[key])-float(b[key]));stored=abs(float(a[key])-float(result['metrics'][key]))
   passed=difference<=cfg['metrics']['independent_audit_absolute_tolerance'] and stored<=cfg['metrics']['independent_audit_absolute_tolerance']
   checks.append({**spec,'metric':key,'project_value':a[key],'independent_value':b[key],'absolute_difference':difference,'stored_difference':stored,'pass':passed})
   if not passed:raise RuntimeError('INDEPENDENT_METRIC_MISMATCH:'+spec['run_id']+':'+key)
  if spec['model']=='F2':
   gate_path=OUT/'diagnostic'/f"{spec['run_id']}_gates.csv"
   gates=pd.read_csv(check_read(gate_path),float_precision='round_trip')
   if len(gates)!=378 or gates.origin_row.duplicated().any() or not np.array_equal(gates.origin_row,frame.origin_row.unique()) or not gates.gate.between(0,1).all() or not (gates.protocol_sha256==digest(PROTOCOL)).all() or not (gates.checkpoint_sha256==identity['sha256']).all():raise RuntimeError('PERSISTED_GATE_IDENTITY_MISMATCH')
   if digest(gate_path)!=result['gate_file_sha256']:raise RuntimeError('PERSISTED_GATE_HASH_CHANGED')
  summary,targets,horizons=diagnostics(frame)
  for table,name in [(targets,'target_SSE'),(horizons,'horizon')]:
   for key,value in spec.items():table[key]=value
   csv_write(OUT/'diagnostic'/f"{spec['run_id']}_{name}.csv",table)
  rows.append({**spec,**a,**summary,'records':len(frame),'origins':frame.origin_row.nunique(),'prediction_file_sha256':digest(path)})
 if len(rows)!=9 or sum(r['records'] for r in rows)!=81648:raise RuntimeError('NINE_RUN_COMPLETENESS_FAILURE')
 csv_write(OUT/'metrics/I_ASSESSMENT_METRICS.csv',pd.DataFrame(rows))
 csv_write(OUT/'metrics/I_INDEPENDENT_METRIC_AUDIT.csv',pd.DataFrame(checks))
 atomic_json(OUT/'metrics/I_ARTIFACT_INTEGRITY_AUDIT.json',{'pass':True,'runs':9,'prediction_records':81648,'keys_and_truth_exact':True,'independent_metric_checks':len(checks),'max_absolute_difference':max(r['absolute_difference'] for r in checks),'protocol_sha256':digest(PROTOCOL),'code_identity_sha256':cfg['code_identity']['sha256'],'formal_test_accessed':False})
 return pd.DataFrame(rows)

if __name__=='__main__':audit_all(load_protocol())
