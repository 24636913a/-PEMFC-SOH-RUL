"""Freeze all dependencies before any new Assessment model prediction."""
import argparse
import ast
import subprocess
from i_common import *

def static_inference_audit():
 forbidden={'backward','step','fit','train','fit_condition','train_model','pretrain_audit','top10_scores','top10_scores'}
 calls=[]
 for path in sorted((PHASE/'scripts').glob('*.py')):
  tree=ast.parse(path.read_text(encoding='utf-8'))
  for node in ast.walk(tree):
   if isinstance(node,ast.Call):
    name=node.func.attr if isinstance(node.func,ast.Attribute) else node.func.id if isinstance(node.func,ast.Name) else ''
    if name in forbidden:raise RuntimeError('TRAINING_OR_RESELECTION_PATH:'+str(path)+':'+name)
    if name in ['predict_delta','regime_weight','load_state_dict']:calls.append({'file':path.name,'line':node.lineno,'call':name})
 return {'pass':True,'scope':'all Phase I production scripts; inherited modules imported read-only, only explicitly called bounded/preprocess/model APIs','forbidden_calls':sorted(forbidden),'inference_calls':calls}

def freeze():
 if PROTOCOL.exists() or LEDGER.exists():raise RuntimeError('I_ALREADY_FROZEN_NO_REWRITE')
 if subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip()!=BASE:raise RuntimeError('BASE_COMMIT_CHANGED')
 if subprocess.check_output(['git','branch','--show-current'],cwd=REPO,text=True).strip()!='main':raise RuntimeError('MAIN_REQUIRED')
 dirty=subprocess.check_output(['git','status','--porcelain','--untracked-files=no'],cwd=REPO,text=True)
 if dirty:raise RuntimeError('TRACKED_TREE_NOT_CLEAN')
 hc=h.load_protocol();hl=read_json(H/'outputs/audits/H_EXECUTION_LEDGER.json')
 if hl['status']!='PHASE_H_MULTISEED_VALIDATION_COMPLETED':raise RuntimeError('H_PRIMARY_EXPERIMENT_INCOMPLETE')
 identities=[]
 for spec in RUNS:
  if hl['runs'].get(spec['H_run_id'],{}).get('status')!='COMPLETED':raise RuntimeError('H_RUN_INCOMPLETE')
  _,_,item=verify_checkpoint(spec);identities.append(item)
 code_audit=static_inference_audit();data=prepare_data();runtime=configure_environment()
 sources={p.relative_to(REPO).as_posix():digest(p) for p in READ_ALLOWLIST if p.is_relative_to(REPO)}
 cfg={'phase':PHASE.name,'base_commit':BASE,'status':'FROZEN_BEFORE_NEW_ASSESSMENT_PREDICTIONS',
 'authorization':'USER_EXPLICIT_NINE_FROZEN_CHECKPOINT_DEVELOPMENT_ASSESSMENT',
 'only_execution_entry':'python -B scripts/run_i_assessment.py --execute',
 'H_protocol_sha256':h.file_hash(h.PROTOCOL),'H_code_identity_sha256':hc['code_identity']['sha256'],
 'scope':{'new_assessment_inference':True,'formal_test_access':'PROHIBITED','training':'PROHIBITED','backward':'PROHIBITED','optimizer_update':'PROHIBITED','checkpoint_reselection':'PROHIBITED','new_seed_selection':'PROHIBITED','scaler_refit':'PROHIBITED'},
 'data':{'read_nrows':3007,'filter':'bin_end_ophrs2_h < 1819.997738','development_rows':3005,'train_rows':[0,2103],'validation_rows':[2103,2404],'assessment_rows':[2404,3005],
 'input_history':'origin-47 through origin; history may precede Assessment start as in frozen C/F builder','targets':'origin+1 through origin+24; each target inside Assessment; same segment, valid, 0.5h continuity',
 'origins':378,'records_per_run':9072,'total_records':81648,'full_N1_hash':'PROHIBITED','historical_F_key_truth_tolerance_V':1e-15,'historical_F_age_tolerance_h':1e-9},
 'task':hc['task'],'preprocessing':hc['preprocessing'],'target_scale_V':data['scale'],'input_fingerprints':data['fingerprints'],
 'scaler_source':(e.PHASE/'outputs/E_COMMON_SCALER.json').relative_to(REPO).as_posix(),
 'checkpoints':identities,'excluded_checkpoints':['H-F0-Repeat-A_best.pt','H-F0-Repeat-B_best.pt'],
 'environment':runtime,'inference':{'model_mode':'eval','autograd':'torch.inference_mode','requires_grad':False,'batch_size':32,'weights_only_load':True,'device':'cuda','validation_reload':'first 32 frozen Validation origins, bitwise equality to checkpoint saved predictions'},
 'metrics':{'primary':'forecast_query_weighted_Assessment_RMSE_mV','formula':'sqrt(mean((prediction_V-true_V)^2))*1000','secondary':['MAE_mV','R2'],
 'diagnostics':['12h_endpoint_RMSE_mV','P95_absolute_error_mV','P99_absolute_error_mV','maximum_absolute_error_mV','target_time_equal_RMSE_mV','24_horizon_RMSE','per_target_SSE','top5_target_SSE_share','multiplicity','F2_gate_distribution'],
 'target_time_equal':'sqrt(mean_target(mean_query_for_target(squared_error_V)))*1000; NOT averaged-prediction ensemble',
 'percentile':'linear interpolation on all 9072 absolute errors','R2':'1-SSE/sum((truth-mean(truth))^2)','independent_audit_absolute_tolerance':1e-10,'error_concentration':'full target universe retained; per-target signed MSE contribution, no outlier-excluded reranking'},
 'paired':{'seeds':[42,52,62],'differences':['F1-F0','F2-F1','F2-F0'],'negative':'new/minuend model has lower RMSE','relative_improvement':'(reference_RMSE-new_RMSE)/reference_RMSE*100','SD':'ddof=1','aggregation':'arithmetic mean of 3 seed-specific metrics; pooled metric is separately labelled; no ensemble'},
 'scientific_decision':{'all3_F2_better_F0_and_F1':'F2_DEVELOPMENT_ASSESSMENT_MULTISEED_GAIN_OBSERVED','all3_F2_better_F1_not_all3_F0':'F2_GATE_GAIN_WITHOUT_NATIVE_SUPERIORITY','otherwise_any_comparison_improves_only1or2':'F2_ASSESSMENT_GAIN_SEED_DEPENDENT','otherwise':'F2_VALIDATION_GAIN_NOT_TRANSFERRED_TO_ASSESSMENT','additional_descriptive_labels':'seed-dependent if either count=1/2; non-transfer if count_F2_F0=0 or mean_F2_F0_difference>=0; ties not improvements'},
 'firewall':'only explicit frozen local artifacts and nrows=3007 filtered N1; no raw source, Test file/label/target or external N1 request',
 'stops':['checkpoint/data/code identity mismatch','reload mismatch','invalid keys/truth/age','nonfinite output','parameter/buffer update','missing prediction or gate file','independent metric mismatch','interrupted run; no automatic repeated Assessment'],
 'ledger_states':['PREPARED','ALL_CHECKPOINTS_VERIFIED','ASSESSMENT_RUNNING','ALL_PREDICTIONS_SAVED','METRICS_AUDITED','COMPLETED','BLOCKED'],
 'history_limits':['TIMESNET_FREEZE','BACKBONE_SEARCH_CLOSED','PHASE_F_ASSESSMENT_INHERITED_SELECTION_BIAS','C1_D0_E0_REPRODUCIBILITY_UNRESOLVED','F2_VALIDATION_MULTISEED_GAIN_OBSERVED'],
 'bias':'Reused Development Assessment; used in C/D/E/F, not independent Test; does not establish SOH/RUL or fair cross-protocol TimesNet superiority',
 'sources':sources,'code_identity':code_identity()}
 PROTOCOL.parent.mkdir(parents=True,exist_ok=True)
 with PROTOCOL.open('w',encoding='utf-8',newline='\n') as f:f.write(yaml.safe_dump(cfg,sort_keys=False,allow_unicode=True))
 with SHA.open('w',encoding='utf-8',newline='\n') as f:f.write(digest(PROTOCOL)+'  I_ASSESSMENT_PROTOCOL.yaml\n')
 atomic_json(OUT/'checkpoint_manifest/I_CHECKPOINT_MANIFEST.json',identities)
 csv_write(OUT/'preparation_audit/I_ASSESSMENT_KEY_TRUTH_AUTHORITY.csv',data['truth'])
 atomic_json(OUT/'preparation_audit/I_DATA_IDENTITY.json',{'pass':True,'fingerprints':data['fingerprints'],'prefix_access':data['prefix_access'],'historical_F_truth':data['historical_truth_audit']})
 atomic_json(OUT/'preparation_audit/I_NO_TRAINING_CODE_AUDIT.json',code_audit)
 atomic_json(OUT/'preparation_audit/I_PROTOCOL_FREEZE.json',{'protocol_sha256':digest(PROTOCOL),'code_identity_sha256':cfg['code_identity']['sha256'],'new_Assessment_predictions_generated':False,'protocol_frozen_before_first_assessment_prediction':True})
 print('I_PROTOCOL_FROZEN '+digest(PROTOCOL),flush=True)

if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('--freeze',action='store_true',required=True);parser.parse_args();freeze()
