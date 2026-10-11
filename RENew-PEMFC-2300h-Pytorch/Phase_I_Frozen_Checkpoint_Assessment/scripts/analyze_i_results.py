"""Generate all scientific tables and reports from audited persisted outputs."""
import math
from i_common import *
from i_metrics import scientific_decision

def table(frame):
 def fmt(x):return f'{x:.8g}' if isinstance(x,(float,np.floating)) else str(x)
 return '| '+' | '.join(frame.columns)+' |\n| '+' | '.join(['---']*len(frame.columns))+' |\n'+'\n'.join('| '+' | '.join(fmt(x) for x in row)+' |' for row in frame.itertuples(index=False,name=None))+'\n'

def read_csv_local(path):return pd.read_csv(check_read(path),float_precision='round_trip')

def analyze(cfg):
 audit=read_json(OUT/'metrics/I_ARTIFACT_INTEGRITY_AUDIT.json')
 if not audit['pass'] or audit['runs']!=9:raise RuntimeError('ANALYSIS_REQUIRES_ALL_NINE_AUDITED')
 metrics=read_csv_local(OUT/'metrics/I_ASSESSMENT_METRICS.csv');history=read_csv_local(H/'outputs/metrics/H_VALIDATION_METRICS.csv')
 groups=[]
 for model,g in metrics.groupby('model',sort=True):
  row={'model':model,'n_seeds':len(g)}
  for k in ['RMSE_mV','MAE_mV','R2','endpoint_RMSE_mV','P95_mV','P99_mV','target_time_equal_RMSE_mV']:
   row[k+'_mean']=float(g[k].mean());row[k+'_sample_SD']=float(g[k].std(ddof=1))
  row['pooled_prediction_point_RMSE_mV_descriptive']=float(np.sqrt(np.mean(g.RMSE_mV**2)))
  groups.append(row)
 group=pd.DataFrame(groups);csv_write(OUT/'metrics/I_MULTISEED_SUMMARY.csv',group)
 pairs=[];pairsummary=[]
 for seed,g in metrics.groupby('seed',sort=True):
  r=g.set_index('model').RMSE_mV;row={'seed':int(seed)}
  for label,new,old in [('D10_F1_minus_F0','F1','F0'),('D21_F2_minus_F1','F2','F1'),('D20_F2_minus_F0','F2','F0')]:
   row[label+'_mV']=float(r[new]-r[old]);row[label+'_relative_improvement_percent']=float((r[old]-r[new])/r[old]*100)
  pairs.append(row)
 paired=pd.DataFrame(pairs);csv_write(OUT/'metrics/I_PAIRED_SEED_DIFFERENCES.csv',paired)
 for label in ['D10_F1_minus_F0','D21_F2_minus_F1','D20_F2_minus_F0']:
  d=paired[label+'_mV'];pairsummary.append({'comparison':label,'mean_difference_mV':float(d.mean()),'sample_SD_mV':float(d.std(ddof=1)),'improved_seed_count':int((d<0).sum()),'maximum_degradation_mV':max(0.,float(d.max()))})
 pairsummary=pd.DataFrame(pairsummary);csv_write(OUT/'metrics/I_PAIRED_SUMMARY.csv',pairsummary)
 gaps=[]
 for row in metrics.to_dict('records'):
  hv=history.loc[history.run_id==row['H_run_id']]
  if len(hv)!=1 or hv.iloc[0].role!='MULTISEED_PRIMARY':raise RuntimeError('VALIDATION_IDENTITY_MISMATCH')
  val=float(hv.iloc[0].RMSE_mV);gap=row['RMSE_mV']-val
  gaps.append({'model':row['model'],'seed':row['seed'],'Validation_RMSE_mV':val,'Assessment_RMSE_mV':row['RMSE_mV'],'gap_mV':gap,'relative_gap_percent':gap/val*100,'direction':'WORSE' if gap>0 else 'BETTER' if gap<0 else 'TIE'})
 gap=pd.DataFrame(gaps);csv_write(OUT/'metrics/I_VALIDATION_ASSESSMENT_GAPS.csv',gap)
 targets=pd.concat([read_csv_local(OUT/'diagnostic'/f"{r['run_id']}_target_SSE.csv") for r in RUNS],ignore_index=True)
 horizons=pd.concat([read_csv_local(OUT/'diagnostic'/f"{r['run_id']}_horizon.csv") for r in RUNS],ignore_index=True)
 csv_write(OUT/'diagnostic/I_ALL_TARGET_SSE.csv',targets);csv_write(OUT/'diagnostic/I_ALL_HORIZON_METRICS.csv',horizons)
 contributions=[];decomp=[]
 for seed in [42,52,62]:
  maps={model:targets.loc[(targets.seed==seed)&(targets.model==model)].set_index('target_row') for model in ['F0','F1','F2']}
  for label,new,old in [('F1-F0','F1','F0'),('F2-F1','F2','F1'),('F2-F0','F2','F0')]:
   if not maps[new].index.equals(maps[old].index):raise RuntimeError('TARGET_DIAGNOSTIC_KEY_MISMATCH')
   t=maps[new][['target_age_h','multiplicity']].copy();t['signed_delta_SSE_mV2']=maps[new].SSE_mV2-maps[old].SSE_mV2;t['signed_delta_MSE_contribution_mV2']=t.signed_delta_SSE_mV2/9072;t['seed']=seed;t['comparison']=label
   actual=float(t.signed_delta_MSE_contribution_mV2.sum());rm=metrics.loc[metrics.seed==seed].set_index('model').RMSE_mV;expected=float(rm[new]**2-rm[old]**2)
   if not math.isclose(actual,expected,abs_tol=1e-9,rel_tol=1e-12):raise RuntimeError('PAIRED_TARGET_MSE_IDENTITY_FAILURE')
   decomp.append({'seed':seed,'comparison':label,'summed_contribution_mV2':actual,'total_delta_MSE_mV2':expected,'absolute_difference':abs(actual-expected),'pass':True})
   contributions.append(t.reset_index())
 contributions=pd.concat(contributions,ignore_index=True);csv_write(OUT/'diagnostic/I_PAIRED_TARGET_MSE_CONTRIBUTIONS.csv',contributions);csv_write(OUT/'diagnostic/I_MSE_DECOMPOSITION_AUDIT.csv',pd.DataFrame(decomp))
 top=targets.sort_values(['model','seed','SSE_mV2'],ascending=[True,True,False],kind='stable').groupby(['model','seed'],sort=True).head(5)
 csv_write(OUT/'diagnostic/I_TOP5_TARGETS_BY_RUN.csv',top)
 focus=targets.loc[np.isclose(targets.target_age_h,1793.5,atol=1e-6,rtol=0)|np.isclose(targets.target_age_h,1794.,atol=1e-6,rtol=0)]
 csv_write(OUT/'diagnostic/I_HISTORICAL_FOCUS_TARGETS.csv',focus)
 gate_stats=[]
 for spec in RUNS:
  if spec['model']!='F2':continue
  g=read_csv_local(OUT/'diagnostic'/f"{spec['run_id']}_gates.csv").gate
  gate_stats.append({'seed':spec['seed'],'origins':len(g),'mean':float(g.mean()),'sample_SD':float(g.std(ddof=1)),'min':float(g.min()),'P5':float(g.quantile(.05)),'P50':float(g.median()),'P95':float(g.quantile(.95)),'max':float(g.max())})
 gates=pd.DataFrame(gate_stats);csv_write(OUT/'diagnostic/I_F2_GATE_STATISTICS.csv',gates)
 primary,additional=scientific_decision(paired)
 summary={'phase':PHASE.name,'base_commit':BASE,'execution_status':'ALL_NINE_INFERRED_AND_INDEPENDENTLY_AUDITED','scientific_status':primary,'additional_descriptive_labels':additional,'protocol_sha256':digest(PROTOCOL),'code_identity_sha256':cfg['code_identity']['sha256'],'records':81648,'origins_per_run':378,'metrics_per_seed':metrics.to_dict('records'),'seed_equal_mean_sample_SD':group.to_dict('records'),'paired':paired.to_dict('records'),'paired_summary':pairsummary.to_dict('records'),'validation_assessment_gaps':gap.to_dict('records'),'gate_statistics':gate_stats,'history_limits':cfg['history_limits'],'existing_checkpoints_reused':True,'new_training':False,'new_assessment_inference':True,'formal_test_accessed':False,'architecture_modified':False,'hyperparameter_optimization':False,'new_seed_selection':False,'scaler_refitted':False,'anomalous_targets_removed':False,'predictions_ensembled':False,'independent_Test_generalization_proven':False,'SOH_RUL_performance_proven':False,'recommended_next_action':'HUMAN_RESEARCH_DECISION_BEFORE_ANY_INDEPENDENT_TEST; NO_AUTOMATIC_PHASE_J'}
 atomic_json(OUT/'I_SUMMARY.json',summary)
 checkpoint=pd.DataFrame([{k:r[k] for k in ['run_id','H_run_id','model','seed','best_epoch','parameter_count','sha256']} for r in cfg['checkpoints']])
 reload=read_csv_local(OUT/'preparation_audit/I_CHECKPOINT_RELOAD_AND_SMOKE.csv')
 limits='\n'.join('- '+s for s in cfg['history_limits'])
 common='\nReused Development Assessment — n=3 seeds. This is not an independent Test. All anomalous targets are retained in the formal forecast-query-weighted metric. No training or new checkpoint selection.\n'
 reports={
 'I_CHECKPOINT_FREEZE_AUDIT.md':'# Phase I checkpoint freeze audit\n'+common+'\nProtocol SHA256: `'+digest(PROTOCOL)+'`\n\nCode identity SHA256: `'+cfg['code_identity']['sha256']+'`\n\n'+table(checkpoint)+'\n## Fixed Validation reload and real-model synthetic forward checks\n\n'+table(reload)+'\nOnly the first 32 fixed Validation origins were reloaded; this is not a full Validation re-evaluation. No new Assessment prediction existed at protocol freeze. Saved E scaler consumed without refitting. Nine state-dict schemas and complete source manifest are in the protocol.\n',
 'I_ASSESSMENT_RESULTS.md':'# Audited Assessment results\n'+common+'\nPrimary: forecast-query-weighted RMSE, mV. Every run has 378 origins and 9072 complete forecast queries. Independent scalar/math.fsum audit agrees with inherited project metrics within 1e-10.\n\n'+table(metrics)+'\n## Validation versus Assessment\n\n'+table(gap)+'\nPositive gaps describe later-period deterioration, not proof of overfitting: distribution shift and concentrated targets remain alternative explanations.\n',
 'I_MULTISEED_COMPARISON.md':'# Multi-seed comparison\n'+common+'\n## Equal-seed means and sample SD (ddof=1)\n\n'+table(group)+'\nThe separately labelled pooled prediction-point RMSE is descriptive only. It is neither the seed-equal mean RMSE nor a prediction ensemble.\n\n## Within-seed paired differences\n\n'+table(paired)+'\n'+table(pairsummary)+'\nNegative differences favour the first/new model. Maximum degradation is capped below at zero. No significance or independent-target sampling claim is made.\n',
 'I_ERROR_CONCENTRATION_ANALYSIS.md':'# Error concentration and tail diagnostics\n'+common+'\n'+table(metrics[['model','seed','endpoint_RMSE_mV','P95_mV','P99_mV','max_abs_error_mV','target_time_equal_RMSE_mV','top5_target_SSE_share','max_SSE_target_age_h','max_SSE_target_multiplicity','max_SSE_target_share']])+'\n## Five largest SSE targets per run\n\n'+table(top)+'\n## Historical focus (1793.5 / 1794 h), no deletion\n\n'+table(focus)+'\n## Additive MSE decomposition\n\n'+table(pd.DataFrame(decomp))+'\nAll target-time signed contributions are retained in I_PAIRED_TARGET_MSE_CONTRIBUTIONS.csv. Positive contribution worsens the first model. RMSE differences themselves are not additive. Concentration is descriptive: overlapping forecasts are not independent observations, and no outlier-deleted ranking is computed.\n\n## Actual-path F2 gate statistics\n\n'+table(gates)+'\nThe gate is captured by a read-only forward hook during the actual forecast, not recomputed from a separate path. Gate distributions alone cannot identify a physical mechanism.\n',
 'I_FINAL_SCIENTIFIC_CONCLUSION.md':'# Phase I final scientific conclusion\n'+common+'\n## Frozen-rule decision\n\n`'+primary+'`\n\nAdditional descriptive labels: '+(', '.join(additional) or 'NONE')+'\n\n'+table(pairsummary)+'\n## Scientific questions\n\n1. F2 Validation advantage transfers to all seeds against both comparators only if the primary decision is MULTISEED_GAIN_OBSERVED; actual Assessment counts are given above, not replaced by Validation 3/3.\n2. F2 versus native F0: '+str(int((paired.D20_F2_minus_F0_mV<0).sum()))+'/3 seeds improve.\n3. Gate F2 versus fixed F1: '+str(int((paired.D21_F2_minus_F1_mV<0).sum()))+'/3 seeds improve.\n4. Fixed dual path F1 versus F0: '+str(int((paired.D10_F1_minus_F0_mV<0).sum()))+'/3 seeds improve; mixed directions do not establish seed-stable independent benefit.\n5. Extreme-target influence is quantified, not removed: Top5 target SSE shares range '+f'{metrics.top5_target_SSE_share.min():.2%}–{metrics.top5_target_SSE_share.max():.2%}'+'. Per-target additive differences identify contributions to the full primary metric; no alternative ranking or causal mechanism is inferred.\n6. A prespecified future independent-Test candidate is a human research decision, not automatic model selection; this reused Assessment cannot by itself authorize Test opening. If improvement is not stable across all three seeds versus F0, overall architectural superiority is not established.\n7. Historical selection bias and unresolved reproducibility remain:\n\n'+limits+'\n\nAssessment was already used in C/D/E/F. H multi-seed Validation gain remains a Development result; the unique cause of historical E2/F0 differences is unresolved. No checkpoint, architecture, seed or preprocessing is reselected. TFT single-split results cannot be fairly ranked against TimesNet three-fold results. SOH/RUL performance is not established.\n\n## Firewall and stop\n\nExisting checkpoints reused YES; new inference YES; training NO; formal Test NO; architecture changes NO; optimization NO; new seed selection NO. Stop after Phase I; no automatic Phase J, Test, SOH or RUL.\n'}
 for name,body in reports.items():
  path=PHASE/'reports'/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_text(body,encoding='utf-8',newline='\n')
 return summary

if __name__=='__main__':analyze(load_protocol())
