"""Independently recalculate persisted queries and check all nine artifacts."""
import math
from i2_common import *

def compare_metrics(frame,tolerance=1e-10):
 project=project_metrics(frame); independent=independent_metrics(frame)
 rows=[]
 for key,value in project.items():
  difference=abs(float(value)-independent[key])
  if not math.isfinite(value) or difference>tolerance: raise RuntimeError('INDEPENDENT_METRIC_DISAGREEMENT:'+key)
  rows.append({'metric':key,'project':float(value),'independent':independent[key], 'absolute_difference':difference,'pass':True})
 return project,rows

def audit_cache(spec,cfg,truth):
 csvpath=OUT/'predictions'/(spec['run_id']+'_assessment.csv')
 rawpath=csvpath.with_suffix('.npz'); ap=csvpath.with_name(spec['run_id']+'_assessment_audit.json')
 for path in [csvpath,rawpath,ap]:
  if not path.is_file() or path.stat().st_size==0: raise RuntimeError('REQUIRED_CACHE_MISSING_OR_EMPTY:'+str(path))
 audit=read_json(ap)
 if digest(csvpath)!=audit['csv_sha256'] or digest(rawpath)!=audit['raw_npz_sha256']: raise RuntimeError('PREDICTION_FILE_HASH_MISMATCH')
 if any(audit.get(k)!=v for k,v in spec.items()) or audit['protocol_sha256']!=digest(PROTOCOL): raise RuntimeError('CACHE_AUDIT_IDENTITY_MISMATCH')
 if audit['checkpoint_sha256']!=cfg['checkpoints_by_run'][spec['run_id']]['sha256']: raise RuntimeError('CACHE_CHECKPOINT_IDENTITY_MISMATCH')
 verify_state(audit['state_audit'])
 if audit['state_dict_before_sha256']!=audit['state_dict_after_sha256']: raise RuntimeError('CACHE_STATE_CHANGED')
 f=validate_frame(read_csv(csvpath),truth,spec,cfg)
 with np.load(allowed(rawpath),allow_pickle=False) as raw:
  standard=raw['standardized_prediction']; voltage=raw['voltage_prediction']
  if standard.shape!=(378,24) or standard.dtype!=np.float32 or voltage.dtype!=np.float32: raise RuntimeError('RAW_OUTPUT_SHAPE_DTYPE_MISMATCH')
  if not np.array_equal(standard*cfg['target_scale_V'],voltage): raise RuntimeError('FROZEN_INVERSE_SCALE_MISMATCH')
  if not np.array_equal(voltage.ravel().astype(np.float64),f.y_pred_delta_V.to_numpy()): raise RuntimeError('RAW_CSV_FLOAT_MISMATCH')
 metrics,checks=compare_metrics(f,cfg['metrics']['audit_absolute_tolerance'])
 summary,targets,horizons=diagnostics(f)
 for df in [targets,horizons]:
  for k in ['run_id','model','seed']: df[k]=spec[k]
 return {**spec,**metrics,**summary,'predictions':len(f),'origins':f.origin_row.nunique(),
  'csv_sha256':audit['csv_sha256'],'raw_npz_sha256':audit['raw_npz_sha256']}, checks, targets,horizons

def audit_all(cfg):
 ledger=read_json(LEDGER)
 if len(ledger['completed_validation'])!=9 or len(ledger['completed_assessment'])!=9: raise RuntimeError('NINE_RUNS_NOT_COMPLETED')
 truth=read_csv(OUT/'inputs/I2_ASSESSMENT_KEY_TRUTH.csv'); rows=[];checks=[];reloads=[]
 for spec in RUNS:
  va=read_json(OUT/'validation_reload_audit'/(spec['run_id']+'_validation_audit.json'))
  if not va['difference_statistics']['array_equal'] or va['protocol_sha256']!=digest(PROTOCOL): raise RuntimeError('VALIDATION_GATE_NOT_EXACT')
  verify_state(va['state_audit'])
  vraw=OUT/'validation_reload_audit'/(spec['run_id']+'_validation.npz')
  if digest(vraw)!=va['raw_npz_sha256']: raise RuntimeError('VALIDATION_RAW_HASH_MISMATCH')
  with np.load(allowed(vraw),allow_pickle=False) as raw:
   if not np.array_equal(raw['standardized_prediction'],raw['reference_standardized']): raise RuntimeError('VALIDATION_RELOAD_NOT_EXACT')
  reloads.append({**spec,**va['difference_statistics'],'state_unchanged':True})
  row,check,target,horizon=audit_cache(spec,cfg,truth);rows.append(row)
  checks.extend({**spec,**x} for x in check)
  csv_write(OUT/'diagnostics'/(spec['run_id']+'_target_SSE.csv'),target)
  csv_write(OUT/'diagnostics'/(spec['run_id']+'_horizon.csv'),horizon)
 csv_write(OUT/'metrics/I2_ASSESSMENT_METRICS.csv',pd.DataFrame(rows))
 csv_write(OUT/'metrics/I2_INDEPENDENT_METRIC_AUDIT.csv',pd.DataFrame(checks))
 csv_write(OUT/'validation_reload_audit/I2_VALIDATION_RELOAD_SUMMARY.csv',pd.DataFrame(reloads))
 result={'pass':True,'runs':9,'validation_exact':9,'records':sum(r['predictions'] for r in rows),
  'metric_checks':len(checks),'max_metric_absolute_difference':max(r['absolute_difference'] for r in checks),
  'all_raw_CSV_predictions_exact':True,'keys_truth_age_equal':True,'state_entries_changed':0,
  'protocol_sha256':digest(PROTOCOL),'code_identity_sha256':cfg['code_identity']['sha256'],
  'full_N1_hash':False,'formal_test_accessed':False,'new_training':False,'gate_capture':'NOT_COLLECTED'}
 if result['records']!=81648: raise RuntimeError('TOTAL_RECORD_COUNT_MISMATCH')
 json_write(OUT/'metrics/I2_ARTIFACT_INTEGRITY_AUDIT.json',result)
 load_protocol();return result

if __name__=='__main__': print(audit_all(load_protocol()))
