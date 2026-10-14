"""Freeze once, predict once, record all three targets, then audit saved artifacts."""
import argparse,hashlib,json,os,platform,subprocess,sys,time,traceback
from datetime import datetime,timezone
import numpy as np
import pandas as pd
from soh_sources import *
from soh_methods import *

def now():return datetime.now(timezone.utc).isoformat()

def freeze():
    cfgpath=PHASE/'configs/FROZEN_SOH_PROTOCOL.json'
    if cfgpath.exists():raise RuntimeError('ALREADY_FROZEN_NO_OVERWRITE')
    if subprocess.check_output(['git','diff','--name-only'],cwd=REPO,text=True).strip():raise RuntimeError('PROTECT_OTHER_CHANGES')
    refs,mapping,p1s,timing,bol,ahc=load_sources()
    source={name:hash_file(PROJECT/path) for name,path in PATHS.items() if name!='bol'}
    code={str(p.relative_to(PHASE)).replace('\\','/'):hash_file(p) for folder in ['scripts','tests'] for p in (PHASE/folder).glob('*.py')}
    cfg={'phase':PHASE.name,'base_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip(),
        'frozen_UTC':now(),'tasks':[{'task_id':task,'origin_nominal_h':o,'target_nominal_h':t,'nominal_horizon_h':t-o} for task,o,t in TASKS],
        'methods':METHODS,'jstar_A_cm2':1.,'BoL_nominal_h':0.,'BoL_reference_voltage_V':float(bol['reference_voltage_V']),
        'P1S_Vref_V':VREF,'Vref_available_after_h':AVAIL,'primary_window':'middle_50',
        'R_trend_window_h':96.,'minimum_complete_cycles':4,'affine':'AHC centered unweighted nonnegative slope',
        'M1_includes_BoL':True,'M2_M3_include_origin_anchor_after_characterization':True,
        'evaluation_scope':'NOMINAL_CHECKPOINT_ORDER_REPLAY',
        'origin_information_time':'After origin characterization, with result immediately available assumed; exact release UNKNOWN',
        'P1S_availability_cutoff':'PAV recorded PAA_gap_cutoff_datetime strictly BEFORE origin characterization',
        'P1S_time_for_trend':'H1 reconstructed actual middle_50 measurement end OpHrs2; conservative availability uses original dwell end DateTime',
        'window_anchor':'latest legal actual P1S measurement end; [latest-96,latest]',
        'target_time_assumption':'nominal target checkpoint numeric h substituted for unobserved actual target OpHrs2; not exact operating-time lead',
        'no_exact_origin_or_target_operating_age':True,'no_result_release_time':True,'strict_online_qualification':False,
        'forecast_scenario':'historical duty-cycle/conditions continue; no new recovery/maintenance; conditional mapping stability',
        'primary_metric':'3-target equal-weight MAE in SOH percentage points','practical_gain_fraction':.05,'minimum_wins_vs_M0':2,
        'strict_CLEAR_GAIN_requires_all_three_time_qualified':True,'threshold':.97,'no_search':True,'CPU_only':True,
        'raw_data_access':False,'late_target_access':False,'mixed_N1_feature_table_access':False,
        'source_sha256':source,'code_sha256':code,
        'BoL_first_record_only':bol,'BoL_logical_record_sha256':hashlib.sha256(json.dumps(bol,sort_keys=True).encode()).hexdigest(),
        'no_full_mixed_anchor_file_hash':True}
    write_json('configs/FROZEN_SOH_PROTOCOL.json',cfg)
    (PHASE/'configs/FROZEN_SOH_PROTOCOL.sha256').write_text(hash_file(cfgpath)+'\n',encoding='utf-8',newline='\n')
    write_csv('inputs/DEVELOPMENT_SOH_REFERENCES.csv',refs)
    write_csv('inputs/FROZEN_P1S_MAPPING.csv',mapping)
    write_csv('inputs/CAUSAL_P1S_IDENTITY.csv',p1s)
    write_csv('inputs/POLARIZATION_TIME_PROVENANCE.csv',timing)
    print('FROZEN '+hash_file(cfgpath))

def verify_protocol():
    path=PHASE/'configs/FROZEN_SOH_PROTOCOL.json';cfg=json.loads(path.read_text(encoding='utf-8'))
    assert hash_file(path)==path.with_suffix('.sha256').read_text().strip()
    for name,expected in cfg['source_sha256'].items():assert hash_file(PROJECT/PATHS[name])==expected,name
    for name,expected in cfg['code_sha256'].items():assert hash_file(PHASE/name)==expected,name
    assert hashlib.sha256(json.dumps(get_bol(),sort_keys=True).encode()).hexdigest()==cfg['BoL_logical_record_sha256']
    return cfg,hash_file(path)

def run():
    if (PHASE/'outputs/RUN_LEDGER.json').exists():raise RuntimeError('RUN_EXISTS_NO_SILENT_RERUN')
    install_firewall();cfg,digest=verify_protocol();start=time.perf_counter()
    write_json('outputs/RUN_LEDGER.json',{'started_UTC':now(),'status':'STARTED','protocol_sha256':digest,'CPU_only':True})
    try:
        refs,mapping,p1s,timing,bol,ahc=load_sources()
        raw=[];details_all=[];cross=[];window_rows=[];curve=[]
        for task,o,t in TASKS:
            inp=origin_inputs(o,refs,mapping,p1s,timing,cfg['R_trend_window_h'])
            pred,details=forecasts(o,t,inp,cfg['minimum_complete_cycles'])
            latest=inp['latest'];stamp=inp['timing'];w=inp['window']
            record={'task_id':task,'origin_nominal_h':o,'target_nominal_h':t,'nominal_horizon_h':t-o,
                **details,'a':inp['a'],'b':inp['b'],'M1_slope_per_nominal_h':inp['soh_age_slope'],'M1_intercept':inp['soh_age_intercept'],
                'origin_polarization_first_datetime':stamp.polarization_first_datetime,'origin_polarization_last_datetime':stamp.polarization_last_datetime,
                'origin_result_release_datetime':None,'origin_actual_operating_h':None,'target_actual_operating_h':None,
                'PAA_pretest_cutoff_datetime':str(inp['cutoff']),'PAA_pretest_recorded_ophrs_h':float(stamp.PAA_gap_cutoff_ophrs2_h),
                'latest_P1S_cycle_id':latest.cycle_interval_id,'latest_raw_cycle_id':int(latest.raw_cycle_id),
                'latest_P1S_measurement_start_actual_h':float(latest.p1s_measurement_start_ophrs2_h),
                'latest_P1S_conservative_available_h':float(latest.recorded_available_end_h),
                'latest_P1S_available_datetime':str(latest.available_datetime),
                'latest_support':latest.condition_support_status,
                'trend_window_start_actual_h':float(latest.actual_measurement_end_h)-96.,
                'trend_window_end_actual_h':float(latest.actual_measurement_end_h),'trend_n_cycles':len(w),
                'trend_cycle_ids':w.cycle_interval_id.tolist(),'trend_raw_cycle_ids':w.raw_cycle_id.astype(int).tolist(),
                'calibration_checkpoints':inp['calibration_checkpoints'],'calibration_R':inp['calibration_R'],'calibration_Y':inp['calibration_Y'],
                'calibration_interval_ids':inp['mapping'].source_interval_id.tolist(),
                'reference_checkpoints':inp['refs'].nominal_h.astype(int).tolist(),
                'reference_Y':inp['refs'].Y.tolist(),'strict_online_time_qualified':False}
            details_all.append(record)
            for method,value in pred.items():
                raw.append({'task_id':task,'method':method,'origin_nominal_h':o,'target_nominal_h':t,
                    'nominal_horizon_h':t-o,'prediction':value,'status':'COMPUTED_WITH_TIME_ASSUMPTIONS' if value is not None else 'INSUFFICIENT_LEGAL_P1S',
                    'evaluation_scope':cfg['evaluation_scope'],'strict_online_time_qualified':False})
                if method=='M0':h0=pred['M0'];s=0.
                elif method=='M1':h0=inp['soh_age_intercept']+inp['soh_age_slope']*o;s=inp['soh_age_slope']
                elif method=='M2':h0=pred['M2'];s=0.
                else:
                    s=None if details['slope_R_per_h'] is None else inp['a']*details['slope_R_per_h']
                    h0=None if s is None else pred['M2']+s*(o-details['latest_R_actual_end_h'])
                crossing_result={'status':'NOT_COMPUTABLE','remaining_h':None} if h0 is None else crossing(h0,s,t-o)
                cross.append({'task_id':task,'method':method,'origin_nominal_h':o,'horizon_nominal_h':t-o,
                    'scenario_origin_health':h0,'slope_health_per_scenario_h':s,**crossing_result,'scope':'EXPLORATORY_COORDINATE_ASSUMPTION_ONLY'})
                if value is not None:
                    for age in np.linspace(o,t,71):
                        y=pred['M0'] if method=='M0' else (inp['soh_age_intercept']+inp['soh_age_slope']*age if method=='M1' else (pred['M2'] if method=='M2' else pred['M2']+s*(age-details['latest_R_actual_end_h'])))
                        curve.append({'task_id':task,'method':method,'nominal_scenario_h':age,'predicted_SOH':y,'is_observed':False})
            wc=w.copy();wc.insert(0,'task_id',task);window_rows.append(wc)
        # Freeze all predictions and input IDs BEFORE evaluator attaches target Y.
        write_json('outputs/PREDICTIONS_BEFORE_EVALUATION.json',raw)
        write_json('outputs/ORIGIN_INPUTS_AND_PARAMETERS.json',details_all)
        predframe=pd.DataFrame(raw)
        predframe['target_SOH']=predframe.target_nominal_h.map(refs.set_index('nominal_h').Y)
        predframe['signed_error_SOH']=predframe.prediction-predframe.target_SOH
        predframe['absolute_error_SOH']=predframe.signed_error_SOH.abs()
        predframe['signed_error_pp']=predframe.signed_error_SOH*100.
        predframe['absolute_error_pp']=predframe.absolute_error_SOH*100.
        write_csv('predictions.csv',predframe)
        metrics=metrics_table(predframe);write_csv('metrics.csv',metrics)
        write_csv('outputs/THRESHOLD97_SCENARIOS.csv',pd.DataFrame(cross))
        write_csv('outputs/USED_P1S_TREND_WINDOWS.csv',pd.concat(window_rows,ignore_index=True))
        write_csv('outputs/FORECAST_CURVE_PLOT_DATA.csv',pd.DataFrame(curve))
        write_json('outputs/DATA_ACCESS_AUDIT.json',{'historical_open_events':EVENTS,'small_derived_inputs_only':True,
            'raw_files_opened':0,'late_target_files_opened':0,'GPU_calls':0,'mixed_feature_table_read':False,
            'BoL_read_contract':'CSV header and FIRST data record only; no mixed anchor file hash',
            'posterior_PHI_columns_used_as_model_input':False,
            'PHI_full_file_hashed_but_only_identity_columns_parsed':True,'history_modified':False})
        write_json('outputs/RUN_LEDGER.json',{'status':'COMPLETED','completed_UTC':now(),'elapsed_s':time.perf_counter()-start,
            'predictions_count':int(predframe.prediction.notna().sum()),'tasks_count':3,'protocol_sha256':digest,
            'Python':platform.python_version(),'numpy':np.__version__,'pandas':pd.__version__,'CPU_only':True})
        print(metrics[['method','MAE_pp','RMSE_pp','wins_vs_M0','research_decision']].to_string(index=False))
    except Exception:
        write_json('outputs/EXECUTION_FAILURE.json',{'UTC':now(),'exception':traceback.format_exc()});raise

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--freeze',action='store_true');parser.add_argument('--run',action='store_true');args=parser.parse_args()
    if args.freeze==args.run:parser.error('Choose --freeze or --run')
    sys.dont_write_bytecode=True
    if args.freeze:freeze()
    else:run()
