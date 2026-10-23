"""Recompute persisted differences independently without any model inference."""
import math
from ir_common import *

def independent_statistics(actual,saved,scale):
 a=actual.astype(np.float64).ravel().tolist();s=saved.astype(np.float64).ravel().tolist();delta=[p-q for p,q in zip(a,s)];maximum=max(abs(v) for v in delta);rmse=math.sqrt(math.fsum(v*v for v in delta)/len(delta))
 return {'array_equal':bool(np.array_equal(actual,saved)),'unequal_elements':sum(p!=q for p,q in zip(a,s)),'unequal_fraction':sum(p!=q for p,q in zip(a,s))/len(a),'max_abs_difference_standardized':maximum,'mean_abs_difference_standardized':math.fsum(abs(v) for v in delta)/len(delta),'difference_RMSE_standardized':rmse,'max_abs_difference_mV':maximum*scale*1000,'difference_RMSE_mV':rmse*scale*1000}

def audit_all(cfg):
 rows=[];checks=[];comparisons=[]
 with np.load(allowed(OUT/'input_identity/IR_FIXED_VALIDATION_INPUT.npz'),allow_pickle=False) as z:origins=z['origins'].copy();truth=z['y'].copy()
 for identity in cfg['checkpoints']:
  outputs={}
  with np.load(allowed(REPO/identity['reference_path']),allow_pickle=False) as z:reference=z['saved'].copy()
  for path_id in PATHS:
   name=identity['H_run_id']+'_Path'+path_id;record=read_json(OUT/'path_comparisons'/(name+'.json'))
   if record['H_run_id']!=identity['H_run_id'] or record['path_id']!=path_id or record['checkpoint_sha256']!=identity['checkpoint_sha256'] or record['protocol_sha256']!=digest(PROTOCOL) or record['input_sha256']!=cfg['data_identity']['fixed_input_sha256']:raise RuntimeError('IR_DIFFERENCE_RECORD_IDENTITY_MISMATCH')
   scalar={k:v for k,v in record.items() if not isinstance(v,(list,dict))}
   if record['status']=='EXCEPTION':rows.append(scalar);continue
   archive=OUT/'path_comparisons'/(name+'.npz')
   if digest(archive)!=record['prediction_archive_sha256']:raise RuntimeError('IR_ARCHIVE_HASH_MISMATCH')
   with np.load(allowed(archive),allow_pickle=False) as z:actual=z['actual'].copy();saved=z['saved'].copy();difference=z['difference'].copy();origin_copy=z['origins'].copy()
   if not np.array_equal(saved,reference) or not np.array_equal(origin_copy,origins) or not np.array_equal(difference,actual.astype(float)-saved.astype(float)) or h.array_hash(actual)!=record['output_sha256']:raise RuntimeError('IR_ARCHIVE_ARRAY_IDENTITY_MISMATCH')
   if not record['finite']:rows.append(scalar);continue
   independent=independent_statistics(actual,saved,identity['target_scale_V'])
   for key,value in independent.items():
    same=value==record[key] if isinstance(value,(bool,int)) else abs(value-record[key])<=1e-12
    checks.append({'H_run_id':identity['H_run_id'],'path_id':path_id,'statistic':key,'stored':record[key],'independent':value,'pass':same})
    if not same:raise RuntimeError('IR_INDEPENDENT_DIFFERENCE_MISMATCH:'+key)
   max_position=np.unravel_index(np.abs(difference).argmax(),difference.shape)
   if record['max_difference_sample_index_zero_based']!=int(max_position[0]) or record['max_difference_horizon_step']!=int(max_position[1])+1:raise RuntimeError('IR_MAXIMUM_LOCATION_MISMATCH')
   state=read_json(OUT/'parameter_state_audit'/(name+'.json'))
   if not record['state_audit_pass'] or not state['all_state_entries_equal'] or not state['requires_grad_unchanged'] or not state['all_parameter_grad_None'] or state['state_sha256_before']!=state['state_sha256_after'] or any(state['operation_counters'].values()) or not all(entry['equal'] and entry['before_sha256']==entry['after_sha256'] for entry in state['state_entries']):raise RuntimeError('IR_READ_ONLY_STATE_AUDIT_FAILURE')
   scale=identity['target_scale_V'];current_RMSE=math.sqrt(math.fsum((p*scale-y)**2 for p,y in zip(actual.ravel().astype(float),truth.ravel().astype(float)))/768)*1000
   if abs(current_RMSE-record['fixed32_actual_RMSE_mV'])>1e-10:raise RuntimeError('IR_FIXED32_METRIC_RECALCULATION_FAILURE')
   outputs[path_id]=actual;rows.append(scalar)
  if 'A' in outputs:
   for path_id,actual in outputs.items():
    stats=independent_statistics(actual,outputs['A'],identity['target_scale_V']);comparisons.append({'model':identity['model'],'seed':identity['seed'],'comparison':path_id+'-A',**stats})
   for left,right in [('B','E'),('C','E'),('D','E')]:
    if left in outputs and right in outputs:comparisons.append({'model':identity['model'],'seed':identity['seed'],'comparison':left+'-'+right,**independent_statistics(outputs[left],outputs[right],identity['target_scale_V'])})
 if len(rows)!=45:raise RuntimeError('IR_MATRIX_INCOMPLETE')
 csv_write(OUT/'path_comparisons/IR_NINE_CHECKPOINT_FIVE_PATH_STATISTICS.csv',pd.DataFrame(rows));csv_write(OUT/'path_comparisons/IR_INDEPENDENT_DIFFERENCE_AUDIT.csv',pd.DataFrame(checks));csv_write(OUT/'path_comparisons/IR_PATH_TO_PATH_COMPARISONS.csv',pd.DataFrame(comparisons))
 json_write(OUT/'execution_logs/IR_OUTPUT_AUDIT.json',{'pass':True,'matrix_records':45,'compared_paths':sum(r['status']=='COMPARED' for r in rows),'exception_paths':sum(r['status']=='EXCEPTION' for r in rows),'independent_numeric_checks':len(checks),'protocol_sha256':digest(PROTOCOL),'Assessment_inference':False,'Test_accessed':False})
 return pd.DataFrame(rows)

if __name__=='__main__':audit_all(load_protocol())
