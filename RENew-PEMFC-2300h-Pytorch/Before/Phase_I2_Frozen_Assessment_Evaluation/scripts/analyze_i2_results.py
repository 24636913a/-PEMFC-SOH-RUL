"""Scientific comparison only after all nine persisted caches independently pass."""
import math
from i2_common import *

def table(frame):
 def fmt(x): return f'{x:.9g}' if isinstance(x,(float,np.floating)) else str(x)
 return '| '+' | '.join(frame.columns)+' |\n| '+' | '.join(['---']*len(frame.columns))+' |\n'+'\n'.join('| '+' | '.join(fmt(x) for x in row)+' |' for row in frame.itertuples(index=False,name=None))+'\n'

def analyze(cfg):
 audit=read_json(OUT/'metrics/I2_ARTIFACT_INTEGRITY_AUDIT.json')
 if not audit['pass'] or audit['runs']!=9: raise RuntimeError('ALL_NINE_AUDITED_REQUIRED')
 m=read_csv(OUT/'metrics/I2_ASSESSMENT_METRICS.csv'); history=read_csv(H/'outputs/metrics/H_VALIDATION_METRICS.csv')
 groups=[]
 for model,g in m.groupby('model',sort=True):
  row={'model':model,'n_seeds':len(g)}
  for k in ['RMSE_mV','MAE_mV','R2','endpoint_RMSE_mV','P95_mV','P99_mV','max_abs_error_mV','target_time_equal_RMSE_mV']:
   row[k+'_mean']=float(g[k].mean());row[k+'_sample_SD']=float(g[k].std(ddof=1))
  row.update(RMSE_min_mV=float(g.RMSE_mV.min()),RMSE_max_mV=float(g.RMSE_mV.max()),pooled_prediction_point_RMSE_mV_descriptive=float(np.sqrt(np.mean(g.RMSE_mV**2))))
  groups.append(row)
 group=pd.DataFrame(groups);csv_write(OUT/'metrics/I2_MULTISEED_SUMMARY.csv',group)
 pairs=[];pairsummary=[]
 labels=[('D10_F1_minus_F0','F1','F0'),('D21_F2_minus_F1','F2','F1'),('D20_F2_minus_F0','F2','F0')]
 for seed,g in m.groupby('seed',sort=True):
  rm=g.set_index('model').RMSE_mV;row={'seed':int(seed)}
  for label,new,old in labels:
   row[label+'_mV']=float(rm[new]-rm[old]);row[label+'_relative_improvement_percent']=float((rm[old]-rm[new])/rm[old]*100)
  pairs.append(row)
 paired=pd.DataFrame(pairs);csv_write(OUT/'metrics/I2_PAIRED_SEED_DIFFERENCES.csv',paired)
 for label,_,_ in labels:
  d=paired[label+'_mV'];pairsummary.append({'comparison':label,'mean_difference_mV':float(d.mean()),
   'sample_SD_mV':float(d.std(ddof=1)),'improved_seed_count':int((d<0).sum()),'maximum_degradation_mV':max(0.,float(d.max())),
   'mean_relative_improvement_percent':float(paired[label+'_relative_improvement_percent'].mean())})
 ps=pd.DataFrame(pairsummary);csv_write(OUT/'metrics/I2_PAIRED_SUMMARY.csv',ps)
 gaps=[]
 for row in m.to_dict('records'):
  hv=history.loc[history.run_id==row['H_run_id']]
  if len(hv)!=1 or hv.iloc[0].role!='MULTISEED_PRIMARY': raise RuntimeError('H_METRIC_IDENTITY_MISMATCH')
  val=float(hv.iloc[0].RMSE_mV);d=row['RMSE_mV']-val
  gaps.append({'model':row['model'],'seed':row['seed'],'Validation_RMSE_mV':val,'Assessment_RMSE_mV':row['RMSE_mV'],
   'absolute_gap_mV':d,'relative_gap_percent':d/val*100})
 gap=pd.DataFrame(gaps);csv_write(OUT/'metrics/I2_VALIDATION_ASSESSMENT_GAPS.csv',gap)
 rank=[]
 for seed,g in gap.groupby('seed'):
  for _,new,old in labels:
   t=g.set_index('model');dv=t.loc[new,'Validation_RMSE_mV']-t.loc[old,'Validation_RMSE_mV'];da=t.loc[new,'Assessment_RMSE_mV']-t.loc[old,'Assessment_RMSE_mV']
   rank.append({'seed':int(seed),'comparison':new+'-'+old,'Validation_difference_mV':float(dv),'Assessment_difference_mV':float(da),'rank_reversal':bool(dv*da<0)})
 rank=pd.DataFrame(rank);csv_write(OUT/'metrics/I2_RANK_REVERSALS.csv',rank)
 targets=pd.concat([read_csv(OUT/'diagnostics'/(r['run_id']+'_target_SSE.csv')) for r in RUNS],ignore_index=True)
 horizons=pd.concat([read_csv(OUT/'diagnostics'/(r['run_id']+'_horizon.csv')) for r in RUNS],ignore_index=True)
 csv_write(OUT/'diagnostics/I2_ALL_TARGET_SSE.csv',targets);csv_write(OUT/'diagnostics/I2_ALL_HORIZON_METRICS.csv',horizons)
 contributions=[];decomp=[]
 for seed in [42,52,62]:
  maps={g:targets.loc[(targets.seed==seed)&(targets.model==g)].set_index('target_row') for g in ['F0','F1','F2']}
  for _,new,old in labels:
   if not maps[new].index.equals(maps[old].index): raise RuntimeError('TARGET_KEYS_DIFFER')
   t=maps[new][['target_age_h','multiplicity']].copy();t['signed_delta_SSE_mV2']=maps[new].SSE_mV2-maps[old].SSE_mV2
   t['signed_delta_MSE_contribution_mV2']=t.signed_delta_SSE_mV2/9072;t['seed']=seed;t['comparison']=new+'-'+old
   actual=float(t.signed_delta_MSE_contribution_mV2.sum()); rm=m.loc[m.seed==seed].set_index('model').RMSE_mV; expected=float(rm[new]**2-rm[old]**2)
   if not math.isclose(actual,expected,abs_tol=1e-9,rel_tol=1e-12): raise RuntimeError('MSE_DECOMPOSITION_FAILURE')
   decomp.append({'seed':seed,'comparison':new+'-'+old,'summed_delta_MSE_mV2':actual,'expected_delta_MSE_mV2':expected,'difference':abs(actual-expected),'pass':True})
   contributions.append(t.reset_index())
 contributions=pd.concat(contributions,ignore_index=True)
 csv_write(OUT/'diagnostics/I2_PAIRED_TARGET_MSE_CONTRIBUTIONS.csv',contributions)
 csv_write(OUT/'diagnostics/I2_MSE_DECOMPOSITION_AUDIT.csv',pd.DataFrame(decomp))
 top=targets.sort_values(['model','seed','SSE_mV2'],ascending=[True,True,False],kind='stable').groupby(['model','seed'],sort=True).head(5)
 csv_write(OUT/'diagnostics/I2_TOP5_TARGETS_BY_RUN.csv',top)
 dominant=contributions.assign(abs_contribution=lambda t:t.signed_delta_MSE_contribution_mV2.abs()).sort_values(['seed','comparison','abs_contribution'],ascending=[True,True,False],kind='stable').groupby(['seed','comparison']).head(5)
 csv_write(OUT/'diagnostics/I2_LARGEST_PAIRED_TARGET_CONTRIBUTIONS.csv',dominant)
 focus=targets.loc[np.isclose(targets.target_age_h,1793.5,atol=1e-6,rtol=0)|np.isclose(targets.target_age_h,1794.,atol=1e-6,rtol=0)]
 csv_write(OUT/'diagnostics/I2_HISTORICAL_FOCUS_TARGETS.csv',focus)
 primary,extra=scientific_decision(paired)
 summary={'phase':PHASE.name,'base_commit':BASE,'execution_status':'ALL_NINE_INFERRED_AND_INDEPENDENTLY_AUDITED',
  'scientific_status':primary,'additional_descriptive_labels':extra,'protocol_sha256':digest(PROTOCOL),
  'code_identity_sha256':cfg['code_identity']['sha256'],'validation_exact':9,'assessment_runs':9,'records':81648,
  'metrics_per_seed':m.to_dict('records'),'equal_seed_mean_sample_SD':groups,'paired':pairs,'paired_summary':pairsummary,
  'validation_assessment_gaps':gaps,'rank_reversals':rank.to_dict('records'),'history_limits':HISTORY_LIMITS,
  'new_training':False,'checkpoint_modified':False,'new_assessment_inference':True,'formal_test_accessed':False,
  'original_I_ledger_modified':False,'gate_capture':'NOT_COLLECTED','full_N1_hash':False,
  'scaler_refitted':False,'outliers_removed':False,'predictions_ensembled':False,'independent_Test_generalization_proven':False,
  'SOH_RUL_performance_proven':False,'recommended_next_action':'STOP; HUMAN_RESEARCH_DECISION; NO_AUTOMATIC_NEXT_PHASE'}
 json_write(OUT/'I2_SUMMARY.json',summary)
 reload=read_csv(OUT/'validation_reload_audit/I2_VALIDATION_RELOAD_SUMMARY.csv')
 common='\nReused Development Assessment; n=3 frozen seeds. Not independent Test. All 9072 queries per run retained. No new training, no new model selection.\n'
 limits='\n'.join('- '+x for x in HISTORY_LIMITS)
 errorcols=['model','seed','endpoint_RMSE_mV','P95_mV','P99_mV','max_abs_error_mV','target_time_equal_RMSE_mV','top5_target_SSE_share','max_SSE_target_age_h','max_SSE_target_multiplicity','max_SSE_target_share']
 # A target's signed contribution, not a target-deleted alternative score.
 focuscon=contributions.loc[np.isclose(contributions.target_age_h,1793.5,atol=1e-6,rtol=0)|np.isclose(contributions.target_age_h,1794.,atol=1e-6,rtol=0)]
 reports={
 'I2_EXECUTION_CONTRACT.md':'# I2 execution contract\n'+common+'\nProtocol SHA: `'+digest(PROTOCOL)+'`\n\nComplete code identity: `'+cfg['code_identity']['sha256']+'`\n\nCUDA construction -> strict load -> eval -> original e.predict(model,x,32,device) -> inherited torch.no_grad. Original requires_grad=True preserved; it does not imply training. Every parameter/buffer compared bytewise before/after; all gradients None, backward/optimizer disabled during inference. Original float32 inverse scaling retained and independently checked against CSV. Gate NOT_COLLECTED to avoid any additional Assessment path. Only nrows=3007 then Development filter ->3005; no whole-N1 checksum. Saved E scaler consumed, never fit. All source/code fingerprints reverified at startup and after evaluation.\n\n'+limits+'\n',
 'I2_VALIDATION_PARITY.md':'# Fixed Validation exact reload\n'+common+'\n'+table(reload)+'\n9/9 np.array_equal against saved checkpoint 32-origin references before any Assessment inference. Each worker is a fresh process with frozen H environment; detailed JSON and raw NPZ preserve all numeric differences and parameter/buffer hashes. This is not a full 205-origin Validation reevaluation. Original I BLOCKED ledger remains unchanged.\n',
 'I2_ASSESSMENT_RESULTS.md':'# I2 Assessment results\n'+common+'\nForecast-query-weighted RMSE (mV) is primary; MAE/R2 auxiliary. 81648 predictions total. Independent scalar/math.fsum recomputation versus frozen project c.metric: '+str(audit['metric_checks'])+' checks PASS; maximum difference '+str(audit['max_metric_absolute_difference'])+'.\n\n'+table(m)+'\n## Validation versus Assessment\n\n'+table(gap)+'\n## Rank reversals\n\n'+table(rank)+'\nLater-period gaps do not uniquely establish overfitting, distribution shift or a physical mechanism.\n',
 'I2_MULTISEED_ANALYSIS.md':'# Equal-seed and paired analysis\n'+common+'\n'+table(group)+'\nEqual-seed arithmetic mean RMSE is primary aggregation. Pooled prediction-point RMSE is descriptive and distinct; no seed predictions averaged into an ensemble. Sample SD uses ddof=1, not a confidence interval.\n\n'+table(paired)+'\n'+table(ps)+'\nNegative new-minus-old favours new (the first/minuend model). This corrects the contradictory prose in the user instruction without changing its explicit subtraction formulas. No significance inference from 3 seeds.\n',
 'I2_ERROR_DIAGNOSTICS.md':'# Error concentration and tail diagnostics\n'+common+'\n'+table(m[errorcols])+'\n## Top5 target-time SSE per run\n\n'+table(top)+'\n## Largest signed paired contributions (all targets retained)\n\n'+table(dominant)+'\n## Previously documented 1793.5/1794 h targets\n\n'+table(focus)+'\n'+table(focuscon)+'\n## Additive MSE identity\n\n'+table(pd.DataFrame(decomp))+'\nOnly squared-error contributions are additive. Target-time-equal uses mean squared error per target, not prediction averaging. Overlapping forecasts are dependent. No targets deleted, no target-deleted score/ranking, no reweighting to select models. Concentration can explain arithmetic differences, not a confirmed failure mechanism. Gate distributions were not collected.\n',
 'I2_FINAL_SCIENTIFIC_CONCLUSION.md':'# I2 final scientific conclusion\n'+common+'\n`'+primary+'`\n\nAdditional labels: '+(', '.join(extra) or 'NONE')+'\n\n'+table(ps)+'\n## Does H Validation gain transfer?\n\nF2 improves over F0 in '+str(int((paired.D20_F2_minus_F0_mV<0).sum()))+'/3 seeds, and over F1 in '+str(int((paired.D21_F2_minus_F1_mV<0).sum()))+'/3. F1 improves over F0 in '+str(int((paired.D10_F1_minus_F0_mV<0).sum()))+'/3. A stable transfer claim against both comparators requires 3/3 on both comparisons; actual counts, not H Validation ranking, determine the description.\n\nTop5 target SSE shares span '+f'{m.top5_target_SSE_share.min():.2%}–{m.top5_target_SSE_share.max():.2%}'+'. Target-level contributions quantify arithmetic concentration; a unique physical or software failure mechanism is not established. Seed variation, later-time distribution change and error concentration are distinct hypotheses.\n\n'+limits+'\n\nHistorical Development Assessment has already been reused for selection. Even favourable means do not prove independent Test generalization. C1/D0/E0 reproducibility remains unresolved; no new TFT-versus-TimesNet ranking or SOH/RUL claim. Original I blocked status retained; I2 is a newly authorized protocol, not a repaired historical I run.\n\n## Stop\n\nAll nine weights unchanged, training zero, Test zero. No automatic next stage, new seeds, architecture search or Test opening.\n'}
 for name,body in reports.items():text_write(PHASE/'reports'/name,body)
 return summary

if __name__=='__main__': analyze(load_protocol())
