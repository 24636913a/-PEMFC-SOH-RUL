"""Prepare data identities, not predictions; freeze before one-shot evaluation."""
import subprocess
from i2_common import *

def freeze_inputs():
 assert_fresh()
 if PROTOCOL.exists(): raise RuntimeError('I2_PROTOCOL_ALREADY_FROZEN')
 head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip()
 if head!=BASE: raise RuntimeError('BASE_COMMIT_CHANGED')
 if subprocess.check_output(['git','diff','--name-only'],cwd=REPO,text=True).strip(): raise RuntimeError('TRACKED_TREE_DIRTY')
 ip=i.load_protocol(); rp=ir.load_protocol(); evidence=read_json(IR/'outputs/IR_SUMMARY.json')
 if evidence['path_exact_counts']['A']!=9: raise RuntimeError('IR_PATH_A_NOT_VERIFIED')
 if read_json(i.LEDGER)['state']!='BLOCKED': raise RuntimeError('I_NOT_BLOCKED')
 manifests=[]
 for old,spec in zip(i.RUNS,RUNS):
  _,_,manifest=i.verify_checkpoint(old, next(r for r in ip['checkpoints'] if r['run_id']==old['run_id']))
  manifest['run_id']=spec['run_id']; manifests.append(manifest)
 data=i.prepare_data()
 if data['fingerprints']!=ip['input_fingerprints']: raise RuntimeError('I2_I_DATA_IDENTITY_MISMATCH')
 # Exact reference keys/labels from I preparation; only Development content.
 authority=I/'outputs/preparation_audit/I_ASSESSMENT_KEY_TRUTH_AUTHORITY.csv'
 old=read_csv(authority)
 if not np.array_equal(old.to_numpy(),data['truth'].to_numpy()): raise RuntimeError('I2_I_TRUTH_AUTHORITY_MISMATCH')
 fixedpath=IR/'outputs/input_identity/IR_FIXED_VALIDATION_INPUT.npz'
 with np.load(allowed(fixedpath),allow_pickle=False) as fixed:
  if not np.array_equal(data['vx'][:32],fixed['x']): raise RuntimeError('I2_IR_FIXED_INPUT_MISMATCH')
 npz_write(OUT/'inputs/I2_INPUTS.npz',validation_x=data['vx'][:32],assessment_x=data['ax'])
 csv_write(OUT/'inputs/I2_ASSESSMENT_KEY_TRUTH.csv',data['truth'])
 json_write(OUT/'checkpoint_manifest/I2_CHECKPOINT_MANIFEST.json',manifests)
 json_write(OUT/'inputs/I2_DATA_IDENTITY.json',{k:v for k,v in data.items() if k not in ['vx','ax','truth']})
 json_write(OUT/'inputs/I2_NO_TRAINING_CODE_AUDIT.json',static_audit())
 sources={}
 for p in ALLOW:
  if p.exists():
   key=p.relative_to(REPO).as_posix() if p.is_relative_to(REPO) else str(p)
   sources[key]=digest(p)
 # Bind every historical input actually consumed and code dependency, including external PF sources.
 sources.update(ip['sources']); sources.update(rp['sources'])
 prepared={p.relative_to(REPO).as_posix():digest(p) for p in (OUT/'inputs').iterdir()}
 cfg={'phase':PHASE.name,'base_commit':BASE,'status':'FROZEN_BEFORE_NEW_ASSESSMENT_INFERENCE',
  'authorization':'USER_EXPLICIT_NINE_FROZEN_CHECKPOINT_DEVELOPMENT_ASSESSMENT',
  'code_identity':own_code(),'sources':sources,'prepared_inputs':prepared,
  'checkpoints':manifests,'checkpoints_by_run':{r['run_id']:r for r in manifests},
  'H_protocol_sha256':digest(h.PROTOCOL),'I_protocol_sha256':digest(i.PROTOCOL),'IR_protocol_sha256':digest(ir.PROTOCOL),
  'input_fingerprints':data['fingerprints'],'data':ip['data'],'task':ip['task'],'preprocessing':ip['preprocessing'],
  'target_scale_V':data['scale'],'inference':{'path':'H Path A','helper':'e.predict(model,x,32,device)',
   'construction':'h.model(...,cuda) then strict state load','context':'eval plus inherited torch.no_grad',
   'requires_grad':'preserve original True flags','batch_size':32,'dtype':'float32',
   'inverse_scale':'original float32 standardized array * frozen Python target_scale; retain resulting float32 V',
   'fixed_validation_gate':'np.array_equal(actual,checkpoint.fixed_validation_prediction), 9/9 before Assessment',
   'gate_capture':'NOT_COLLECTED_TO_PRESERVE_UNMODIFIED_H_PATH; no extra Assessment forwards'},
  'metrics':{'primary':'forecast-query-weighted RMSE (mV), all 9072 queries',
   'auxiliary':['MAE_mV','R2'],'diagnostics':['endpoint_RMSE_mV','P95_mV','P99_mV','max_abs_error_mV',
    'target_time_equal_RMSE_mV','horizon_RMSE','target_SSE','Top5_target_SSE_share','multiplicity'],
   'target_equal':'sqrt(mean_target(mean_query(squared_error)))*1000; NOT prediction ensemble',
   'audit_absolute_tolerance':1e-10,'aggregation':'equal-seed arithmetic mean; sample SD ddof=1; pooled descriptive separately',
   'paired':'F1-F0, F2-F1, F2-F0; negative means first/minuend lower RMSE; relative=(old-new)/old*100',
   'paired_wording_erratum':'User prose saying latter lower contradicts formulas; formulas authoritative',
   'scientific_rules':'all3 F2 better both=>GAIN; all3 better F1 only=>GATE_GAIN; mixed signs=>SEED_DEPENDENCE; otherwise NOT_TRANSFERRED; mean F2-F0>=0 adds NOT_TRANSFERRED'},
  'history_limits':HISTORY_LIMITS,'scope':{'training':False,'Test':False,'new_seed_selection':False,
   'scaler_refit':False,'architecture_change':False,'outlier_deletion':False,'prediction_ensemble':False},
  'stop_rules':['any identity change','any fixed Validation not array_equal','any worker failure: no retry',
   'no ranking until nine caches independently audited','no automatic next stage'],
  'one_shot':True}
 text_write(PROTOCOL,yaml.safe_dump(cfg,allow_unicode=True,sort_keys=False))
 text_write(SHA,digest(PROTOCOL)+'  '+PROTOCOL.name+'\n')
 json_write(OUT/'checkpoint_manifest/I2_SOURCE_MANIFEST.json',sources)
 load_protocol()
 print('I2_PROTOCOL_FROZEN',digest(PROTOCOL),'checkpoints',len(manifests),'Assessment origins',378,flush=True)

if __name__=='__main__': freeze_inputs()
