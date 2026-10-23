"""One frozen CPU experiment; no late data or future target lead as input."""
import argparse,json,platform,subprocess,time,sys,traceback
from datetime import datetime,timezone
import numpy as np
import pandas as pd
import sklearn
from health_common import *
from history_models import *

def utc():return datetime.now(timezone.utc).isoformat()

def freeze():
    path=PHASE/'configs/FROZEN_LONG_HISTORY_PROTOCOL.json'
    if path.exists():raise RuntimeError('NO_PROTOCOL_REFREEZE')
    assert not subprocess.check_output(['git','diff','--name-only'],cwd=REPO,text=True).strip()
    p,authorities,old=load_sources()
    source={k:sha(PROJECT/v) for k,v in SOURCES.items()}
    code={str(f.relative_to(PHASE)).replace('\\','/'):sha(f) for folder in ['scripts','tests'] for f in (PHASE/folder).glob('*.py')}
    cfg={'phase':PHASE.name,'frozen_UTC':utc(),'base_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip(),
        'horizons':{'primary':'H1','secondary':'H3'},'target':'DeltaR=R_target-R_origin; raw CTAE P1S proxy',
        'Vref_V':VREF,'reference_available_after_h':VREF_AVAILABLE,'warmup_excluded':True,'development_boundary_h':BOUNDARY,
        'folds':FOLDS,'histories_h':[96,192],'minimum_observations':{'96':5,'192':10},'minimum_span_fraction':.7,
        'feature_names':FEATURES,'std_ddof':0,'slope_time':'actual reconstructed middle_50 measurement end OpHrs2',
        'models':['M0','M96','M192'],'alpha_grid':ALPHAS,'inner_last_fraction':.15,'inner_count_rule':'ceil(0.15*N)',
        'inner_scaler_scope':'inner fit only','final_scaler_scope':'final outer train only','target_scaled':False,
        'same_train_and_validation_samples_for_all_methods':True,'outer_purge':'train labels end <= first common validation origin; if none, use first original validation origin',
        'inner_purge':'inner fit labels end <= first inner selection origin','purges_apply_to_both_histories':True,
        'fixed_tie_rule':'minimum inner MAE, then smallest alpha','solver':'sklearn Ridge cholesky float64 with intercept',
        'primary_metric':'three-fold equal mean MAE mV equivalent; MISSING FOLD => NOT_EVALUABLE, not imputed',
        'partial_fold_aggregate':'descriptive only, explicitly exclude missing folds from denominator',
        'gain_gate':{'H1_M192_vs_M96_minimum_relative_reduction':.05,'H1_minimum_fold_wins':2,'H1_M192_below_M0':True,'all_three_folds_required':True},
        'no_further_window_or_model':True,'CPU_only':True,'future_lead_feature':False,'new_Test_or_raw_access':False,
        'SOH_bridge':'require >=3 past reference anchors with actual release evidence; current source release UNKNOWN, no auxiliary numeric SOH',
        'source_sha256':source,'execution_code_sha256':code}
    save_json('configs/FROZEN_LONG_HISTORY_PROTOCOL.json',cfg)
    path.with_suffix('.sha256').write_text(sha(path)+'\n',encoding='utf-8',newline='\n')
    save_csv('inputs/CAUSAL_P1S_ACTUAL_MEASUREMENT_IDENTITY.csv',p)
    for horizon,a in authorities.items():save_csv(f'inputs/{horizon}_ORIGINAL_FROZEN_SAMPLE_AUTHORITY.csv',a)
    print('FROZEN '+sha(path))

def verify_protocol():
    path=PHASE/'configs/FROZEN_LONG_HISTORY_PROTOCOL.json';cfg=json.loads(path.read_text(encoding='utf-8'))
    assert sha(path)==path.with_suffix('.sha256').read_text().strip()
    for k,expected in cfg['source_sha256'].items():assert sha(PROJECT/SOURCES[k])==expected,k
    for k,expected in cfg['execution_code_sha256'].items():assert sha(PHASE/k)==expected,k
    return cfg,sha(path)

def run():
    if (PHASE/'outputs/RUN_LEDGER.json').exists():raise RuntimeError('EXPERIMENT_EXISTS_NO_RERUN')
    install_firewall();cfg,digest=verify_protocol();start=time.perf_counter()
    save_json('outputs/RUN_LEDGER.json',{'status':'STARTED','UTC':utc(),'protocol_sha256':digest})
    try:
        p,authorities,old=load_sources();prediction_rows=[];metrics=[];count_rows=[];records=[];parity=[];inner_rows=[];all_details=[];bridges=[];lead_rows=[]
        for horizon in ['H1','H3']:
            f,details=build_features(p,authorities[horizon],horizon);all_details+=details
            save_csv(f'inputs/{horizon}_FEATURES_AND_ELIGIBILITY.csv',f)
            for fold in FOLDS:
                original=f.loc[f.fold==fold];train,val,purged,cut=train_validation(f,fold)
                for role,used in [('train',train),('validation',val)]:
                    g=original.loc[original.role==role]
                    count_rows.append({'horizon':horizon,'fold':fold,'role':role,'original_n':len(g),
                        'L96_history_eligible_n':int(g.L96_eligible.sum()),'L192_history_eligible_n':int(g.L192_eligible.sum()),
                        'common_history_n':int(g.common_history_eligible.sum()),'used_n':len(used),
                        'lost_for_history_n':int(len(g)-g.common_history_eligible.sum()),
                        'outer_parameter_availability_purged_n':len(purged) if role=='train' else 0,
                        'first_validation_origin_cutoff_h':cut})
                save_csv(f'inputs/{horizon}_{fold}_USED_TRAIN.csv',train);save_csv(f'inputs/{horizon}_{fold}_USED_VALIDATION.csv',val)
                save_csv(f'inputs/{horizon}_{fold}_OUTER_LABEL_PURGE.csv',purged)
                # Verify original all-validation Persistence before comparing changed scopes.
                ov=original.loc[original.role=='validation'];original_metric=error_metrics(ov.target_R_P1S_causal,ov.origin_R_P1S_causal,VREF)
                cached=old[horizon].loc[old[horizon].fold==fold]
                cached_metric=error_metrics(cached.true_R_P1S_causal,cached.predicted_R_P1S_causal,VREF)
                assert abs(original_metric['MAE_R']-cached_metric['MAE_R'])<1e-12
                parity.append({'horizon':horizon,'fold':fold,'scope':'FULL_ORIGINAL_VALIDATION_REFERENCE_ONLY',
                    'n':len(ov),'recomputed_MAE_mV':original_metric['MAE_mV_equivalent'],
                    'historical_cached_MAE_mV':cached_metric['MAE_mV_equivalent'],'passed':True})
                outputs={'M0':val.origin_R_P1S_causal.to_numpy(float)}
                for length in [96,192]:
                    delta,rec=ridge_fit(train,val,length);rec.update({'horizon':horizon,'fold':fold,'outer_label_cutoff_h':cut})
                    records.append(rec);outputs[f'M{length}']=val.origin_R_P1S_causal.to_numpy(float)+delta
                    for score in rec['alpha_grid_scores']:inner_rows.append({'horizon':horizon,'fold':fold,'model':f'M{length}',**score,'selected':score['alpha']==rec['alpha'],'inner_fit_n':rec['inner_fit_n'],'inner_selection_n':rec['inner_selection_n'],'inner_purged_n':len(rec['inner_purged_samples'])})
                current=[]
                for method,pred in outputs.items():
                    m=error_metrics(val.target_R_P1S_causal,pred,VREF)
                    metrics.append({'horizon':horizon,'fold':fold,'model':method,'status':'COMPLETED' if len(val) else 'NOT_EVALUABLE_NO_COMMON_VALIDATION',
                        'n_train':len(train),'n_validation':len(val),**m})
                    for row,value in zip(val.itertuples(),pred):
                        error=float(value-row.target_R_P1S_causal)
                        record={'horizon':horizon,'fold':fold,'model':method,'sample_index':int(row.sample_index),
                            'origin_cycle_interval_id':row.origin_cycle_interval_id,'target_cycle_interval_id':row.target_cycle_interval_id,
                            'origin_raw_cycle_id':int(row.origin_raw_cycle_id),'target_raw_cycle_id':int(row.target_raw_cycle_id),
                            'origin_measurement_end_h':float(row.origin_p1s_measurement_end_ophrs2_h),'target_measurement_end_h':float(row.target_p1s_measurement_end_ophrs2_h),
                            'origin_R':float(row.origin_R_P1S_causal),'true_R':float(row.target_R_P1S_causal),'predicted_R':float(value),
                            'predicted_DeltaR':float(value-row.origin_R_P1S_causal),'error_R':error,'absolute_error_R':abs(error),
                            'absolute_error_mV_equivalent':abs(error)*VREF*1000,'common_history_eligible':True,
                            'actual_lead_used_as_model_input':False}
                        current.append(record);prediction_rows.append(record)
                        bridges.append({'horizon':horizon,'fold':fold,'model':method,'sample_index':int(row.sample_index),
                            'predicted_R':float(value),'SOH_bridge_status':'NO_CERTIFIED_ANCHOR_RELEASE_TIME','predicted_SOH_proxy':None,'true_SOH_target':None,'true_SOH_MAE':None})
                save_csv(f'outputs/predictions/{horizon}_{fold}_predictions.csv',pd.DataFrame(current,columns=list(prediction_rows[0]) if prediction_rows else None))
                for row in val.itertuples():lead_rows.append({'horizon':horizon,'fold':fold,'sample_index':int(row.sample_index),
                    'actual_target_end_minus_origin_end_h':float(row.target_p1s_measurement_end_ophrs2_h-row.origin_p1s_measurement_end_ophrs2_h),
                    'role':'EVALUATION_DESCRIPTIVE_ONLY_NOT_INPUT'})
        predframe=pd.DataFrame(prediction_rows);metricframe=pd.DataFrame(metrics);countframe=pd.DataFrame(count_rows)
        save_csv('predictions.csv',predframe);save_csv('metrics_fold.csv',metricframe);save_csv('sample_counts.csv',countframe)
        save_json('outputs/HISTORY_WINDOW_IDENTITIES.json',all_details);save_json('outputs/FITTED_RIDGE_MODELS.json',records)
        save_csv('outputs/INNER_ALPHA_SELECTION.csv',pd.DataFrame(inner_rows));save_csv('outputs/HISTORICAL_PERSISTENCE_PARITY.csv',pd.DataFrame(parity))
        save_csv('outputs/SOH_BRIDGE_STATUS.csv',pd.DataFrame(bridges));save_csv('outputs/ACTUAL_LEAD_DESCRIPTIVE_ONLY.csv',pd.DataFrame(lead_rows))
        summaries=[];decisions={}
        for horizon in ['H1','H3']:
            table=metricframe.loc[metricframe.horizon==horizon]
            for method in ['M0','M96','M192']:
                g=table.loc[table.model==method];good=g.loc[g.status=='COMPLETED'];full=len(good)==3
                pooled=predframe.loc[(predframe.horizon==horizon)&(predframe.model==method)]
                summaries.append({'horizon':horizon,'model':method,'available_folds':len(good),'required_folds':3,
                    'formal_three_fold_equal_MAE_mV':float(good.MAE_mV_equivalent.mean()) if full else None,
                    'descriptive_available_fold_equal_MAE_mV':float(good.MAE_mV_equivalent.mean()),
                    'descriptive_available_fold_equal_RMSE_mV':float(good.RMSE_mV_equivalent.mean()),
                    'pooled_n_validation':len(pooled),'pooled_descriptive_MAE_mV':float(pooled.absolute_error_mV_equivalent.mean()),
                    'pooled_P95_abs_error_mV':float(pooled.absolute_error_mV_equivalent.quantile(.95)),
                    'pooled_max_abs_error_mV':float(pooled.absolute_error_mV_equivalent.max()),
                    'fold3_MAE_mV':float(g.loc[g.fold=='fold_3'].iloc[0].MAE_mV_equivalent),
                    'status':'FULL_THREE_FOLD' if full else 'PARTIAL_FOLD_DIAGNOSTIC_ONLY'})
            pivot=table.pivot(index='fold',columns='model',values='MAE_mV_equivalent').dropna()
            mean=pivot.mean();relative=(mean.M96-mean.M192)/mean.M96*100
            decisions[horizon]={'formal_status':'NOT_EVALUABLE' if len(pivot)<3 else ('CLEAR_LONG_HISTORY_GAIN' if mean.M192<=.95*mean.M96 and int((pivot.M192<pivot.M96).sum())>=2 and mean.M192<mean.M0 else 'NO_CLEAR_LONG_HISTORY_GAIN'),
                'available_folds':len(pivot),'missing_folds':['fold_2'] if len(pivot)<3 else [],
                'diagnostic_M192_vs_M96_relative_improvement_percent':float(relative),
                'diagnostic_fold_wins_M192_vs_M96':int((pivot.M192<pivot.M96).sum()),
                'diagnostic_M192_below_M0':bool(mean.M192<mean.M0),
                'formal_gain_claim':len(pivot)==3 and mean.M192<=.95*mean.M96 and int((pivot.M192<pivot.M96).sum())>=2 and mean.M192<mean.M0}
        save_csv('metrics_summary.csv',pd.DataFrame(summaries));save_json('outputs/RESEARCH_DECISION.json',decisions)
        save_json('outputs/DATA_ACCESS_AUDIT.json',{'historical_open_events':EVENTS,'small_derived_sources_only':True,
            'new_raw_or_Test_files_opened':0,'mixed_N1_table_read':False,'future_lead_used_as_input':False,
            'PHI_posterior_columns_used':False,'PHI_whole_file_sha_for_identity_provenance_only':True,'historical_files_written':False})
        save_json('outputs/RUN_LEDGER.json',{'status':'COMPLETED_WITH_MISSING_FOLD','completed_UTC':utc(),
            'elapsed_seconds':time.perf_counter()-start,'CPU_only':True,'Python':platform.python_version(),'numpy':np.__version__,
            'pandas':pd.__version__,'sklearn':sklearn.__version__,'final_ridge_models':12,'ridge_fits_including_inner_grid':60,
            'validation_prediction_rows':len(predframe),'protocol_sha256':digest,'missing_validation_scopes':['H1_fold_2','H3_fold_2']})
        print(pd.DataFrame(summaries)[['horizon','model','available_folds','descriptive_available_fold_equal_MAE_mV','fold3_MAE_mV']].to_string(index=False))
        print(json.dumps(decisions,ensure_ascii=False))
    except Exception:
        save_json('outputs/EXECUTION_FAILURE.json',{'UTC':utc(),'exception':traceback.format_exc()});raise

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--freeze',action='store_true');parser.add_argument('--run',action='store_true');args=parser.parse_args()
    if args.freeze==args.run:parser.error('Choose --freeze or --run')
    sys.dont_write_bytecode=True
    if args.freeze:freeze()
    else:run()
