"""Read-only consolidation: no sklearn/torch, no raw voltage reads, no experiment reruns."""
import ast,csv,hashlib,json,os,sys,subprocess,re,zipfile
from pathlib import Path
import numpy as np
import pandas as pd
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1];REPO=ROOT.parent
BASE='2afcfced13a40b280e93d01d63e993be6413af58'
N='Phase_P1S_96h_ThreeFold_Comparison';Q='Phase_Polarization_Physics_to_Health_Qualification';I='Phase_Integrated_SOH_Physics_RUL_Research';C='Phase_P1S_ForecastStep_Independent_Contribution';E='Phase_RealHyFC_EIS_HFR_Physics_Foundation'
DATA=REPO/'数据/《耐久性测试期间PEM燃料电池堆的空间解析电化学表征》的基础数据'
S={
 'native_legal':f'{N}/outputs/LEGAL_SAMPLES.csv','native_pred':f'{N}/outputs/PREDICTIONS.csv','native_par':f'{N}/outputs/TRAINED_PARAMETERS.json',
 'native_metric':f'{N}/outputs/FOLD_METRICS.csv','native_protocol':f'{N}/configs/FROZEN_PROTOCOL.json','native_model_code':f'{N}/scripts/models.py','native_verify_code':f'{N}/scripts/verify_results.py',
 'native_inner':f'{N}/outputs/INNER_SPLITS.csv','replay_code':'Phase_P1S_LongTerm_Trajectory_Forecasting/scripts/methods.py',
 'h2_legal':f'{Q}/outputs/H2_USED_SAMPLES.csv','h2_pred':f'{Q}/outputs/H2_PREDICTIONS.csv','h2_metric':f'{Q}/outputs/H2_FOLD_METRICS.csv',
 'h2_code':f'{Q}/scripts/run_h2.py','q_protocol':f'{Q}/configs/FROZEN_PROTOCOL.json','q_verify':f'{Q}/outputs/INDEPENDENT_VERIFICATION.json',
 'q_decision':f'{Q}/outputs/RESEARCH_DECISION.json','q_gates':f'{Q}/outputs/CANDIDATE_EVIDENCE_MATRIX.csv','q_available':f'{Q}/outputs/AVAILABILITY_COUNTS.csv',
 'identity':'Phase_P1S_LongHistory_ShortHorizon_Forecasting/inputs/CAUSAL_P1S_ACTUAL_MEASUREMENT_IDENTITY.csv',
 'measure':'Phase_PAA_P1_StateToken_PreAnchor_Alignment/outputs/P1S_COMPLETE_CYCLE_MEASUREMENTS.csv',
 'catalog':'Phase_RCR_Raw_Resolution_Cycle_Reconstruction_Audit/outputs/RAW_CYCLE_CATALOG.csv',
 'dynamic':f'{I}/outputs/RECOMPUTED_DYNAMIC_METRICS.csv','tft_time':f'{I}/outputs/TFT_PAST_INFORMATION_AND_PARAMETER_AVAILABILITY.csv',
 'links':f'{I}/outputs/P1S_TO_SOH_EXACT_FORECAST_LINKS.csv','calibration':f'{I}/models/PAST_ONLY_AFFINE_PARAMETERS.json',
 'references':'Phase_SOH_LongHorizon_Future_Forecasting/inputs/DEVELOPMENT_SOH_REFERENCES.csv',
 'soh_metrics':f'{C}/outputs/SOH_METRICS.csv','contribution':f'{C}/outputs/SEQUENTIAL_CONTRIBUTION.csv','soh_decision':f'{C}/outputs/RESEARCH_DECISION.json',
 'soh_points':f'{C}/outputs/POINT_PREDICTIONS.csv','soh_errors':f'{C}/outputs/ERROR_DECOMPOSITION.csv',
 'eis_decision':f'{E}/outputs/RESEARCH_DECISION.json','eis_identity':f'{E}/outputs/EIS_STACK_IDENTITY_MATRIX.csv','eis_gates':f'{E}/outputs/CPU_PHYSICS_QUALIFICATION_GATES.csv',
 'eol_old':'Phase_K_Paper_Claims_and_SOH_RUL_Feasibility/reports/K_EOL_AND_CENSORING_FEASIBILITY.md',
 'eol_extraction':'Phase_RUL90_Data_Feasibility/04_OFFICIAL_EOL_AUDIT.md',
 'integrated_decision':f'{I}/outputs/FINAL_RESEARCH_DECISION.json','conditional_slopes':f'{I}/outputs/CONDITIONAL_SLOPE_SENSITIVITY.csv',
 'conditional_code':f'{I}/scripts/supplement_trajectory_uncertainty.py','dynamic_plot_code':f'{I}/scripts/render_results.py',
 'q_plot_code':f'{Q}/scripts/plot_figures.py','soh_plot_code':f'{C}/scripts/render_paper_figures.py',
 'eis_readme':Path(os.path.relpath(DATA/'ReadMe_EIS_measurements.txt',ROOT)).as_posix(),
 'source_paper':Path(os.path.relpath(DATA/'IJHE_2026_Fensterle_Spatially resolved electrochemical characterization of a PEM fuel cell stack during durability testing.pdf',ROOT)).as_posix()}
for fold in ['fold_1','fold_2','fold_3']:
 S['h2_par_'+fold]=f'{Q}/models/H2_{fold}.json'
 for role in ['INNER_FIT','INNER_SELECTION']:
  S[f'h2_{fold}_{role}']=f'{Q}/data_contracts/H2_{fold}_{role}.csv'
FIGS=[('TFT',I,'F1_dynamic_existing_results','dynamic','dynamic_plot_code','Validation seeds; not universal robustness'),
 ('P1S',Q,'05_short_horizon_comparison','h2_metric','q_plot_code','H1/H3 frozen plus newH2; own cohort perH'),
 ('SOH',C,'F1_matched_SOH_three_method_control','soh_points','soh_plot_code','Only3unique retrospective referenceevents'),
 ('SOH_error_cancellation',C,'F3_calibration_forecast_error_cancellation','soh_errors','soh_plot_code','Perfectfuture proxy is diagnostic only'),
 ('polarization',Q,'02_slopes_and_hysteresis',None,'q_plot_code','Descriptive, not HFR/ASR or healthloss'),
 ('physical_identifiability',Q,'04_parameter_identifiability',None,'q_plot_code','No training qualification'),
 ('conditional_lifetime',I,'F6_conditional_calendar_trajectories',None,'dynamic_plot_code','Calendar assumptions, not trueRUL validation')]
for ix,(module,phase,name,_,_,_) in enumerate(FIGS):
 for ext in ['png','pdf']:S[f'fig_{ix}_{ext}']=f'{phase}/figures/{name}.{ext}'
for key,path in {'polar_slope':f'{Q}/outputs/NUMERICAL_SLOPES.csv','polar_hysteresis':f'{Q}/outputs/HYSTERESIS.csv',
 'polar_fit':f'{Q}/outputs/PARAMETER_IDENTIFIABILITY.csv','polar_profile':f'{Q}/outputs/JLIM_PROFILE.csv',
 'trajectory':f'{I}/outputs/CONDITIONAL_CALENDAR_TRAJECTORIES.csv','trajectory_envelope':f'{I}/outputs/CONDITIONAL_TRAJECTORY_SENSITIVITY_ENVELOPE.csv',
 'all123metric':f'{Q}/outputs/H1_H2_H3_FOLD_METRICS.csv'}.items():S[key]=path
FEATURES=['origin_R','mean_R','std_R','slope_R_per_h','origin_minus_median_R'];VREF=.71383023635;ACCESS=[]
META_ZIPS=[DATA/n for n in ['EIS.zip','Loadcycle_data.zip','spatiallyResolvedEIS.zip']]
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def firewall():
 allowed={(ROOT/v).resolve() for v in S.values()}|{p.resolve() for p in META_ZIPS}
 def hook(event,args):
  if event!='open' or isinstance(args[0],int):return
  q=Path(os.fsdecode(args[0])).resolve();mode,flags=args[1:];write=(isinstance(mode,str) and any(c in mode for c in 'wax+')) or (isinstance(flags,int) and flags&(os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC|os.O_APPEND))
  if q.is_relative_to(REPO) and not q.is_relative_to(HERE):
   if write or q not in allowed:raise PermissionError('UNAUTHORIZED_READ_OR_HISTORY_WRITE '+str(q))
   ACCESS.append(q.relative_to(REPO).as_posix())
 sys.addaudithook(hook)
def rd(k):return pd.read_csv(ROOT/S[k],float_precision='round_trip')
def jj(k):return json.loads((ROOT/S[k]).read_text(encoding='utf-8'))
def protected_write(name,data):
 q=HERE/name
 if name.startswith('H123_') and q.exists():
  assert q.read_bytes()==data,'FROZEN_CONTRACT_CHANGED: create a separately versioned audit, do not overwrite '+name
 else:q.write_bytes(data)
def write_json(name,d):protected_write(name,(json.dumps(d,ensure_ascii=False,indent=2,allow_nan=False)+'\n').encode('utf-8'))
def csvout(name,d):protected_write(name,d.to_csv(index=False,float_format='%.17g',lineterminator='\n').encode('utf-8'))
def existing_function(key,name):
 # Execute only this previously saved pure prediction function, never module imports/training.
 tree=ast.parse((ROOT/S[key]).read_text(encoding='utf-8'));node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==name)
 assert not any(isinstance(n,ast.Attribute) and n.attr in ['fit','fit_transform','backward'] for n in ast.walk(node))
 env={'np':np,'FEATURES':FEATURES};exec(compile(ast.Module(body=[node],type_ignores=[]),S[key],'exec'),env);return env[name]
def metric(y,p):
 e=(np.asarray(p)-np.asarray(y))*VREF*1000;a=abs(e)
 return dict(n=len(e),MAE_mVeq=float(a.mean()),RMSE_mVeq=float(np.sqrt((e*e).mean())),P95_mVeq=float(np.quantile(a,.95)),max_mVeq=float(a.max()))
def freeze_h123():
 orig=rd('native_legal');new=rd('h2_legal');ident=rd('identity').set_index('raw_cycle_id');m=rd('measure');m=m[m.stable_window.eq('middle_50')].set_index('cycle_interval_id');oldparams=jj('native_par');params=oldparams+[jj('h2_par_'+f) for f in ['fold_1','fold_2','fold_3']];cat=rd('catalog').set_index('raw_cycle_id');folds=jj('native_protocol')['folds']
 rows=[];fn=['L96_origin_R','L96_history_mean_R','L96_history_std_R','L96_history_slope_R_per_h','L96_origin_minus_median_R']
 for r in orig.to_dict('records'):
  raw=int(r['origin_raw_cycle_id']);target=int(r['target_raw_cycle_id']);o=ident.loc[raw];t=ident.loc[target]
  rows.append(dict(H=int(r['horizon'][1:]),fold=r['fold'],role=r['role'],origin_raw_cycle_id=raw,target_raw_cycle_id=target,origin_cycle_interval_id=r['origin_cycle_interval_id'],target_cycle_interval_id=r['target_cycle_interval_id'],
   origin_actual_end_h=float(o.measurement_end_h),target_actual_start_h=float(t.measurement_start_h),target_actual_end_h=float(t.measurement_end_h),origin_available_h=float(o.conservative_available_end_h),target_available_h=float(t.conservative_available_end_h),
   origin_calendar=m.loc[o.cycle_interval_id,'measurement_end_datetime'],target_calendar=m.loc[t.cycle_interval_id,'measurement_end_datetime'],target_R=r['target_R_P1S_causal'],**dict(zip(FEATURES,[r[k] for k in fn])),
   source_key='native_legal',source_sample_index=int(r['sample_index'])))
 for r in new.to_dict('records'):rows.append({k:r[k] for k in ['H','fold','role','origin_raw_cycle_id','target_raw_cycle_id','origin_cycle_interval_id','target_cycle_interval_id','origin_actual_end_h','target_actual_start_h','target_actual_end_h','origin_available_h','target_available_h','origin_calendar','target_calendar','target_R']+FEATURES}|{'source_key':'h2_legal','source_sample_index':r['sample_id']})
 legal=pd.DataFrame(rows);legal['sample_key']=[f"H{r.H}_{r.fold}_{r.origin_raw_cycle_id}_{r.target_raw_cycle_id}" for r in legal.itertuples()];assert legal.sample_key.is_unique
 canonical_params=[];checks=[];predictions=[];saved=rd('native_pred');h2=rd('h2_pred');replay=existing_function('replay_code','replay');predict=existing_function('h2_code','predict');maxfeature=0
 # Original legal sample identity and features retained; no intersection shrinkage.
 for r in legal.itertuples():
  assert r.target_raw_cycle_id==r.origin_raw_cycle_id+r.H
  assert all(i in cat.index and bool(cat.loc[i,'complete_cycle']) for i in range(r.origin_raw_cycle_id,r.target_raw_cycle_id+1))
  lo,mid,hi=folds[r.fold];assert (lo<=r.target_actual_start_h and r.target_available_h<mid) if r.role=='train' else (mid<=r.target_actual_start_h and r.target_available_h<hi)
  o=ident.loc[r.origin_raw_cycle_id];g=ident[(ident.measurement_end_h>=o.measurement_end_h-96-1e-9)&(ident.measurement_end_h<=o.measurement_end_h+1e-9)&(ident.conservative_available_end_h<=o.conservative_available_end_h+1e-9)].sort_values('measurement_end_h')
  x=g.measurement_end_h.to_numpy();y=g.R.to_numpy();xc=x-x.mean();s=float(xc@(y-y.mean())/(xc@xc));v=[o.R,y.mean(),y.std(),s,o.R-np.median(y)];diff=float(np.max(abs(np.array(v)-[getattr(r,k) for k in FEATURES])));assert diff<1e-12;maxfeature=max(maxfeature,diff);assert len(g)>=5 and np.ptp(x)>=67.2-1e-9
 for (H,fold),g in legal.groupby(['H','fold']):
  train=g[g.role.eq('train')];val=g[g.role.eq('validation')].sort_values('target_actual_end_h');par=next(p for p in params if p.get('horizon')==f'H{H}' and p['fold']==fold) if H!=2 else next(p for p in params if p.get('H')==2 and p['fold']==fold)
  assert train.target_available_h.max()<=val.origin_available_h.min()+1e-9
  assert pd.to_datetime(train.target_calendar).max()<=pd.to_datetime(val.origin_calendar).min()
  if H==2:
   f=rd(f'h2_{fold}_INNER_FIT');sel=rd(f'h2_{fold}_INNER_SELECTION');assert f.target_available_h.max()<=sel.origin_available_h.min()+1e-9
   inner_cut=sel.origin_available_h.min();inner_max=f.target_available_h.max();rp=predict(val,par);mean=par['scaler_mean'];scale=par['scaler_scale'];coef=par['coef'];intercept=par['intercept'];source=f'h2_par_{fold}'
  else:
   sp=rd('native_inner');f=sp[sp.horizon.eq(f'H{H}')&sp.fold.eq(fold)&sp.inner_role.eq('inner_fit')];sel=sp[sp.horizon.eq(f'H{H}')&sp.fold.eq(fold)&sp.inner_role.eq('inner_selection')]
   assert f.target_p1s_measurement_end_ophrs2_h.max()<=sel.origin_p1s_measurement_end_ophrs2_h.min()+1e-9
   inner_max=f.target_p1s_measurement_end_ophrs2_h.max();inner_cut=sel.origin_p1s_measurement_end_ophrs2_h.min();rp=replay(val,par);mean=par['train_scaler_mean'];scale=par['train_scaler_scale'];coef=par['coef_standardized_X'];intercept=par['intercept_deltaR'];source='native_par'
  mu=train[FEATURES].to_numpy().mean(axis=0);sd=train[FEATURES].to_numpy().std(axis=0);sd[sd==0]=1;assert np.max(abs(mu-mean))<1e-12 and np.max(abs(sd-scale))<1e-12
  cp={'H':int(H),'fold':fold,'alpha':par['alpha'],'features':FEATURES,'mean':mean,'scale':scale,'coef':coef,'intercept':intercept,'source_key':source};canonical_params.append(cp)
  maxdiff=0
  for method,result in [('PERSISTENCE',val.origin_R.to_numpy()),('RIDGE96',rp)]:
   old=(h2[h2.fold.eq(fold)&h2.method.eq(method)] if H==2 else saved[saved.horizon.eq(f'H{H}')&saved.fold.eq(fold)&saved.model.eq('RIDGE_96H' if method=='RIDGE96' else method)]).set_index(['origin_raw_cycle_id','target_raw_cycle_id'])
   keys=list(zip(val.origin_raw_cycle_id,val.target_raw_cycle_id));assert set(keys)==set(old.index);matched=old.loc[keys];pp=matched.predicted_R.to_numpy() if H==2 else matched.predicted_R_P1S_causal.to_numpy();truth=matched.target_R.to_numpy() if H==2 else matched.true_R_P1S_causal.to_numpy()
   delta=float(np.max(abs(result-pp)));assert delta<1e-12 and np.max(abs(val.target_R.to_numpy()-truth))<1e-14;maxdiff=max(maxdiff,delta)
   z=val.copy();z['method']=method;z['predicted_R']=result;z['error_mVeq']=(result-z.target_R)*VREF*1000;predictions.append(z)
  checks.append({'H':int(H),'fold':fold,'Train_n':len(train),'Validation_n':len(val),'outer_label_available_max_h':float(train.target_available_h.max()),'first_validation_origin_available_h':float(val.origin_available_h.min()),'inner_label_max_h':float(inner_max),'inner_selection_origin_min_h':float(inner_cut),'max_prediction_difference_R':maxdiff,'prediction_function_source':S['h2_code'] if H==2 else S['replay_code'],'alpha':par['alpha']})
 pred=pd.concat(predictions,ignore_index=True);met=pd.DataFrame([{'H':H,'fold':f,'method':m,**metric(g.target_R,g.predicted_R)} for (H,f,m),g in pred.groupby(['H','fold','method'])]);existing=rd('all123metric')
 for r in met.itertuples():
  q=existing[existing.H.eq(r.H)&existing.fold.eq(r.fold)&existing.method.eq(r.method)].iloc[0];assert q.n==r.n
  for k in ['MAE_mVeq','RMSE_mVeq','P95_mVeq','max_mVeq']:assert abs(q[k]-getattr(r,k))<1e-10
 csvout('H123_LEGAL_SAMPLES.csv',legal);csvout('H123_PREDICTIONS.csv',pred);csvout('H123_FOLD_METRICS.csv',met);csvout('H123_TIME_AND_CODE_CHECKS.csv',pd.DataFrame(checks))
 equal=[]
 for (H,m),g in met.groupby(['H','method']):
  assert len(g)==3 and (g.n>0).all();equal.append({'H':int(H),'method':m,'fold_counts':'|'.join(str(n) for n in g.sort_values('fold').n),'equal_fold_MAE_mVeq':float(g.MAE_mVeq.mean()),'equal_fold_RMSE_mVeq':float(g.RMSE_mVeq.mean()),'equal_fold_P95_mVeq':float(g.P95_mVeq.mean()),'max_across_folds_mVeq':float(g.max_mVeq.max())})
 csvout('H123_EQUAL_FOLD_METRICS.csv',pd.DataFrame(equal))
 write_json('H123_FREEZE.json',{'baseline_git_commit':BASE,'identity':'OriginalH1/H3frozen;H2exploratorypreregisteredCPU;no newfits in closure','Vref_V':VREF,'folds':jj('native_protocol')['folds'],'history_h':96,'minimum_observations':5,'minimum_span_h':67.2,'features':FEATURES,'alpha_grid':[.0001,.01,1.,100.],'legal_rows':len(legal),'prediction_rows':len(pred),'final_parameter_records':canonical_params,'H2_fold1_max_error_worsens':True,'new_model_fits':0,'new_hyperparameter_selection':0,'feature_max_difference':maxfeature,'code_replay':'Only AST-extracted existing pure predict/replay; no fit functions imported or called','original_source_sha256':{k:sha(ROOT/S[k]) for k in ['native_legal','native_pred','native_par','native_model_code','native_verify_code','replay_code','h2_legal','h2_pred','h2_code','h2_par_fold_1','h2_par_fold_2','h2_par_fold_3']}})
 return legal,pred,met
def literature():
 import fitz
 doc=fitz.open(ROOT/S['source_paper']);p4=doc[3].get_text();p8=doc[7].get_text();p1=doc[0].get_text()
 # Select only protocol/limit passages. Do not emit or tabulate late-life result values.
 assert '2000 operating hours' in p4 and '200' in p4 and 'ended' in p4 and '500 mV' in p8 and 'lower voltage limit' in p8
 assert 'linear sweep' in p4 and 'air/air' in p4 and 'cell failure' in p1
 records=[
  {'id':'L1','page':4,'finding':'预定研究终止：超过2000运行小时目标，约200周期后结束并全面表征','source_level':'DIRECT_PRIMARY_PDF_METHOD','exact_short_quote':'the durability test was ended','EOL_implication':'EoT，不证明真实失效时刻','new_to_current_matrix':'本轮直接原文核验；旧报告已有二手摘录'},
  {'id':'L2','page':8,'finding':'60/50kPaG、1.8A/cm2测量触及500mV下限；原文确称durability lower-voltage limit','source_level':'DIRECT_PRIMARY_PDF_LIMIT_PASSAGE_ONLY','exact_short_quote':'lower voltage limit for the durability test','EOL_implication':'有效限值线索；缺触发耐久终止记录、同j*=1参考映射和失效时间','new_to_current_matrix':'不是新发现failure；纠正“原文无任何限值”的过强说法'},
  {'id':'L3','page':1,'finding':'膜缺陷/气体交叉泄漏可涉及安全失效，是机理背景','source_level':'DIRECT_PRIMARY_PDF_INTRODUCTION','exact_short_quote':'','EOL_implication':'介绍失效机制不等于本堆已经发生该失效','new_to_current_matrix':'新增明确区分机制与本数据事件'},
  {'id':'L4','page':4,'finding':'方法提到CV和LSV，但未详述其结果','source_level':'DIRECT_PRIMARY_PDF_METHOD','exact_short_quote':'','EOL_implication':'可能存在未公开诊断；当前不能据此补造失效或删失标签','new_to_current_matrix':'本轮新增缺失资料条目'},
  {'id':'L5','page':4,'finding':'空气/空气启动后额外名义极化用于观察可恢复电压损失；启动也可能引入其他影响','source_level':'DIRECT_PRIMARY_PDF_METHOD','exact_short_quote':'','EOL_implication':'干预/恢复证据方向，非failure事件，也非单纯无混杂恢复','new_to_current_matrix':'对应1500h_afterAirAirStart文件元数据，建议下一项配对实验'},
  {'id':'L6','page':None,'finding':'已审查本地展开目录命名及3个zip中央目录；无LSV/CV/失效/删失记录的明确文件名命中','source_level':'LOCAL_METADATA_ONLY_NOT_CONTENT_PROOF','exact_short_quote':'','EOL_implication':'有限命名范围未发现，不证明未命名文件或未公开资料中不存在','new_to_current_matrix':'新增范围明确的文件/归档目录审查'}]
 csvout('LITERATURE_AND_EOL_AUDIT.csv',pd.DataFrame(records));write_json('LITERATURE_SCOPE.json',{'source_paper_SHA256':sha(ROOT/S['source_paper']),'pages_scoped':[1,4,8],'numeric_results_used':False,'Test_csv_read':False,'new_verified_EOL_event':False,'certified_censoring_record_found':False,'unpublished_CV_LSV_possible_but_not_available':True,'new_recovery_curve_numerics_read':False,'no_external_dataset_or_new_download':True})
def metadata_inventory():
 keywords=['lsv','voltam','leak','cross','failure','eol','censor','cv_','_cv','cv.','cyclic'];records=[]
 names=[p.relative_to(DATA).as_posix() for p in DATA.rglob('*') if p.is_file()]
 records.append({'scope':'expanded_local_names_only','entries':len(names),'keyword_hits':[s for s in names if any(k in Path(s).name.lower() for k in keywords)],'names_SHA256':hashlib.sha256('\n'.join(sorted(names)).encode()).hexdigest(),'numerics_read':False})
 for z in META_ZIPS:
  with zipfile.ZipFile(z) as archive:names=archive.namelist()
  records.append({'scope':z.name+'_central_directory_only','entries':len(names),'keyword_hits':[s for s in names if any(k in Path(s).name.lower() for k in keywords)],'names_SHA256':hashlib.sha256('\n'.join(sorted(names)).encode()).hexdigest(),'numerics_read':False,'members_decompressed':0})
 candidate=[]
 for n in ['Polcurve_Nominal_RHFC_1500h.csv','Polcurve_Nominal_RHFC_1500h_afterAirAirStart.csv']:
  p=DATA/n;candidate.append({'path':p.relative_to(REPO).as_posix(),'exists':p.exists(),'bytes':p.stat().st_size if p.exists() else None,'access':'stat_only_no_header_or_numeric_content','next_experiment_not_executed':True})
 write_json('METADATA_ONLY_INVENTORY.json',{'inventory':records,'next_experiment_candidate_files':candidate,'limitation':'No matching name is not proof of absence in differentlynamed or unpublished files; ZIP centralnamesonly, no member content.'})
def evidence():
 rows=[
 ('TFT','VERIFIED_DYNAMIC_PERFORMANCE','F2三seed Validation query RMSE均值7.245400mV，F0为9.067563mV；seed62尾部307.984904mV，非普遍稳健优势','dynamic;tft_time','动态V均值/DeltaV预测','不能据此声称健康、SOH或RUL增益'),
 ('P1S_H123','VERIFIED_PROXY_PERFORMANCE','H1/H2/H3三折等权MAE相对保持改善10.641466/13.849485/10.420448%；H2fold1最大误差恶化','native_pred;h2_pred;native_par;h2_code','未来真实周期固定电流P1S保持率','单轨迹探索性Development，非确认性证据/真实SOH'),
 ('Reference_SOH','FROZEN_MEASUREMENT_DEFINITION','7个Development极化参考；Vpol(j*=1)/0.6989533305168152','references','稀疏参考健康比例','不是连续真SOH，不是7条寿命'),
 ('M2_NC_SOH_bridge','COMPLETED_RETROSPECTIVE_CONNECTION','H1/H3三事件，起点P1S+校准控制与Ridge+校准已比较；H1增益极小，H3存在误差抵消','links;calibration;soh_metrics;soh_errors','过去锚点仿射映射后的SOH回放','参考排程回顾性、发布时间未知；不能严格在线或泛化增强宣称'),
 ('Polarization','VERIFIED_DESCRIPTIVE_PHYSICS','7真实曲线、方向滞后、综合Rdiff、局部辨识；只有1/5拟合支路通过局部数值筛查','q_decision;polar_slope;polar_fit','跨电流描述与适用域','不是现有固定j预测器的有效物理损失'),
 ('EIS','VERIFIED_DESCRIPTIVE_DIAGNOSTICS','65谱3620点33历史电路拟合；R0模型相关；0认证HFR/ASR，串号及映射未证实','eis_decision;eis_identity;eis_gates','阻抗诊断','不认证膜阻，不接入TFT/P1S，不再拟合'),
 ('Soft_physics_training','NOT_ESTABLISHED','真实健康训练物理损失/梯度/同容量消融均NA；现有输出不承载跨电流导数','q_gates','资格与不可辨识性结果','不能写物理增强训练已完成'),
 ('RUL','NOT_VALIDATED','没有与参考SOH一致的可验证failure-time；研究终止/工况限值不是精确RUL','eol_old;source_paper;integrated_decision','条件外推和失效/删失资格分析','不得报告真实RUL精度、用终点减t、或拟合虚构监督'),
 ('Conditional_lifetime','COMPLETED_SCENARIO_ONLY','历史calendar条件外推对斜率假设敏感，可出现无交点','conditional_slopes;trajectory','阈值/运行情景给定的交点演示','非真实失效时间、非认证寿命置信区间'),
 ('Recovery_intervention','NEXT_EXPERIMENT_METADATA_ONLY','存在1500h空气/空气启动后名义极化文件，本轮不读数值','source_paper','下一项最小可恢复性配对诊断','尚无新恢复量化结果、无物理训练资格')]
 csvout('EVIDENCE_MATRIX.csv',pd.DataFrame(rows,columns=['module','status','actual_evidence','source_keys','output_identity','prohibited_claim']))
 links=[
 ('Development_bins','Frozen_TFT','COMPLETED','PAST_INPUT_VALIDATION_PREDICTION','dynamic','原架构冻结；不是逐起点可早用的健康教师'),
 ('Actual_P1S_history','Frozen_H123_Ridge','COMPLETED','CAUSAL_PROXY_FORECAST','native_pred;h2_pred','固定96h过去信息，目标rawID+H'),
 ('Past_reference_pairs','M2_NC','COMPLETED_CONDITIONAL','PAST_MEASUREMENT_COMPLETION_ONLY','calibration;links','发布时间未认证；映射算子nearestcomplete保持冻结'),
 ('H1_H3_Ridge_P1S','M2_NC_SOH_replay','COMPLETED_CONDITIONAL','RETROSPECTIVE_3EVENTS','links;soh_points','H2无参考SOH验证；H1/H3不是6个独立事件'),
 ('Origin_measured_P1S','M2_NC_SOH_control','COMPLETED_CONDITIONAL','RETROSPECTIVE_MATCHED_CONTROL','soh_points','必要主要对照；大部分收益来自近期测量+校准'),
 ('TFT_voltage','P1S_or_SOH_increment','UNVERIFIED_INTERFACE','PARAMETER_TIME_AND_TARGET_IDENTITY_BLOCKED','tft_time','冻结checkpoint选择截止1460.5h；无postselection已缓存起点/因果事件训练对'),
 ('Polarization_curves','Physical_diagnostics','COMPLETED','DESCRIPTIVE','polar_slope;polar_hysteresis;polar_fit','方向和条件明确，不改变原SOH参考'),
 ('EIS','Physical_diagnostics','COMPLETED','DESCRIPTIVE','eis_decision','模型相关R0，不是认证HFR'),
 ('Physical_diagnostics','H123_training_loss','UNQUALIFIED_INTERFACE','STRUCTURE_PARAMETER_TIME_AND_SAMPLE_BLOCKS','q_gates;eis_gates','损失/健康梯度NA；禁止以平滑或合成梯度替代'),
 ('Physical_features','M2_NC_improvement','UNVERIFIED_INTERFACE','SPARSE_REFERENCE_AND_RELEASE_BLOCKED','q_gates;calibration','只有3事件，无新校准泛化实验'),
 ('Reference_SOH','Conditional_lifetime_scenarios','COMPLETED_CONDITIONAL','ILLUSTRATIVE_EXTRAPOLATION','conditional_slopes;trajectory','calendar情景假设，不是未来真实SOH/EOL验证'),
 ('SOH_or_P1S','Validated_true_RUL','UNQUALIFIED_INTERFACE','EOL_OR_CERTIFIED_CENSORING_MISSING','eol_old;source_paper','不将记录末端当失效；单轨迹无可靠寿命分布识别')]
 csvout('CONNECTION_MATRIX.csv',pd.DataFrame(links,columns=['source_module','target_module','implementation','information_qualification','source_keys','limitation']))
 figrows=[]
 for i,(module,phase,name,data,code,note) in enumerate(FIGS):
  keys=([data] if data else ['polar_slope','polar_hysteresis'] if module=='polarization' else ['polar_fit','polar_profile'] if module=='physical_identifiability' else ['trajectory','trajectory_envelope'])
  if module=='P1S':keys=['all123metric','native_pred','h2_pred']
  for ext in ['png','pdf']:figrows.append({'module':module,'image_path':S[f'fig_{i}_{ext}'],'image_SHA256':sha(ROOT/S[f'fig_{i}_{ext}']),'CSV_paths':';'.join(S[k] for k in keys),'CSV_SHA256':';'.join(sha(ROOT/S[k]) for k in keys),'plot_script':S[code],'plot_script_SHA256':sha(ROOT/S[code]),'baseline_commit':BASE,'scientific_boundary':note})
 csvout('EXISTING_FIGURE_AND_CODE_INDEX.csv',pd.DataFrame(figrows))
def source_index():
 # Repo Git reads occur outside the Python data firewall via Git's read-only object database.
 tree=subprocess.check_output(['git','ls-tree','-rz',BASE],cwd=REPO);blobs={}
 for row in tree.split(b'\0'):
  if not row:continue
  meta,name=row.split(b'\t',1);blobs[name.decode('utf-8')]=meta.decode().split()[2]
 rows=[]
 for k,v in S.items():
  p=(ROOT/v).resolve();rel=p.relative_to(REPO).as_posix();rows.append({'key':k,'path_from_research_root':v,'path_from_repo':rel,'SHA256':sha(p),'bytes':p.stat().st_size,'baseline_commit':BASE,'git_blob_oid':blobs.get(rel,'NOT_TRACKED_AT_BASE'),'access':'content_readonly_or_imagehash'})
 csvout('SOURCE_INDEX.csv',pd.DataFrame(rows));return {r['key']:r['SHA256'] for r in rows}
def main():
 firewall()
 if (HERE/'CLOSURE_MANIFEST.json').exists():
  for item in json.loads((HERE/'CLOSURE_MANIFEST.json').read_text(encoding='utf-8'))['files']:assert sha(HERE/item['path'])==item['SHA256'],'SEALED_LOCAL_ARTIFACT_CHANGED '+item['path']
  for row in pd.read_csv(HERE/'SOURCE_INDEX.csv').to_dict('records'):assert sha((ROOT/row['path_from_research_root']).resolve())==row['SHA256'],'SEALED_SOURCE_CHANGED '+row['key']
 before={k:sha(ROOT/v) for k,v in S.items()};legal,pred,met=freeze_h123();literature();metadata_inventory();evidence();source_index()
 for k,h in before.items():assert sha(ROOT/S[k])==h,k
 write_json('REPRODUCIBILITY.json',{'status':'PASS','baseline':BASE,'legal_sample_rows':len(legal),'Validation_unique_targets_with_method_pairs':len(pred)//2,'prediction_rows':len(pred),'fold_metric_rows':len(met),'existing_prediction_functions_replayed_without_fit':True,'H2_fold1_max_error_worsening_retained':True,'all_source_SHA256_unchanged':True,'new_model_fits':0,'new_physics_losses':0,'Test_numerics_read':False,'H1_H2_H3_experiments_rerun':False,'new_model_search_phase':False,'source_files':len(S),'raw_files_read':0})
 write_json('READ_LEDGER.json',sorted(set(ACCESS)));print('CLOSURE_PASS',len(legal),len(pred),len(met),len(S))
if __name__=='__main__':main()
