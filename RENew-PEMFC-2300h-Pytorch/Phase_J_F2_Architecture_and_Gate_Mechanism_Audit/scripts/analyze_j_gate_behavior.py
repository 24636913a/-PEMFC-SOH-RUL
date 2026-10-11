"""Descriptive statistics computed only from persistent Validation gate/prediction caches."""
import itertools
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import j_common as j

def analyze():
    cfg=j.load_protocol()
    if not j.read_json(j.OUT/'integrity_audit/J_FINAL_READONLY_AUDIT.json')['pass']:raise RuntimeError('FINAL_INTEGRITY_REQUIRED')
    gates=j.read_csv(j.OUT/'gate_values/J_ALL_F2_GATES.csv')
    metrics=j.read_csv(j.OUT/'fixed_gate_diagnostic/J_VALIDATION_METRICS.csv')
    thresholds=j.read_csv(j.OUT/'gate_diagnostics/J_TRAIN_ONLY_CONDITION_THRESHOLDS.csv')
    stats=[];correlations=[];strata=[];pairs=[];comparisons=[]
    for seed in j.SEEDS:
        f=gates.loc[gates.seed.eq(seed)].copy();g=f.g.to_numpy()
        stats.append({'seed':seed,**j.gate_statistics(g)})
        for name in j.REGIME_FEATURE_NAMES:
            x=f[name+'_raw_mean'].to_numpy();rho=float(spearmanr(x,g).statistic) if np.ptp(x)>0 and np.ptp(g)>0 else None
            correlations.append({'seed':seed,'context':name,'Spearman_rho':rho,'scope':'descriptive mathematical input dependence; overlapping windows; no p-value'})
            t=thresholds.loc[thresholds.feature.eq(name)].iloc[0]
            labels=np.where(x<=t.low_upper,'LOW',np.where(x<=t.high_lower,'MID','HIGH'))
            for label in ['LOW','MID','HIGH']:
                a=g[labels==label]
                strata.append({'seed':seed,'context':name,'group':label,'n':len(a),'g_mean':float(a.mean()) if len(a) else None,
                    'g_median':float(np.median(a)) if len(a) else None,'g_SD':float(a.std(ddof=1)) if len(a)>1 else None,
                    'Train_low_upper':t.low_upper,'Train_high_lower':t.high_lower,'threshold_source':'Train only; frozen before gate inference'})
        learned=metrics.loc[(metrics.seed==seed)&(metrics['mode']=='learned')].iloc[0]
        fixed=metrics.loc[(metrics.seed==seed)&(metrics['mode']=='fixed_0p5')].iloc[0]
        a=j.read_csv(j.OUT/f'fixed_gate_diagnostic/J_F2_Seed{seed}_LEARNED_PREDICTIONS.csv')
        b=j.read_csv(j.OUT/f'fixed_gate_diagnostic/J_F2_Seed{seed}_FIXED0P5_PREDICTIONS.csv')
        d=(a.y_pred_delta_V-b.y_pred_delta_V).to_numpy()*1000
        comparisons.append({'seed':seed,'learned_RMSE_mV':learned.RMSE_mV,'fixed0p5_RMSE_mV':fixed.RMSE_mV,
            'learned_minus_fixed_RMSE_mV':learned.RMSE_mV-fixed.RMSE_mV,
            'prediction_difference_RMSE_mV':float(np.sqrt(np.mean(d*d))),
            'mean_absolute_prediction_difference_mV':float(np.mean(np.abs(d))),
            'P95_absolute_prediction_difference_mV':float(np.quantile(np.abs(d),.95)),
            'max_absolute_prediction_difference_mV':float(np.max(np.abs(d))),
            'different_prediction_fraction':float(np.mean(d!=0))})
    for s1,s2 in itertools.combinations(j.SEEDS,2):
        a=gates.loc[gates.seed.eq(s1)].g.to_numpy();b=gates.loc[gates.seed.eq(s2)].g.to_numpy()
        pairs.append({'seed_a':s1,'seed_b':s2,'mean_absolute_gate_difference':float(np.mean(np.abs(a-b))),
            'gate_difference_RMSE':float(np.sqrt(np.mean((a-b)**2))),'paired_gate_Spearman':float(spearmanr(a,b).statistic)})
    stats=pd.DataFrame(stats);comparisons=pd.DataFrame(comparisons)
    j.csv_write(j.OUT/'gate_diagnostics/J_GATE_STATISTICS.csv',stats)
    j.csv_write(j.OUT/'gate_diagnostics/J_GATE_CONTEXT_SPEARMAN.csv',pd.DataFrame(correlations))
    j.csv_write(j.OUT/'gate_diagnostics/J_TRAIN_DEFINED_STRATA.csv',pd.DataFrame(strata))
    j.csv_write(j.OUT/'gate_diagnostics/J_PAIRED_SEED_GATE_DIFFERENCES.csv',pd.DataFrame(pairs))
    j.csv_write(j.OUT/'fixed_gate_diagnostic/J_FIXED_GATE_COMPARISON.csv',comparisons)
    hist=j.read_csv(j.H/'outputs/metrics/H_VALIDATION_METRICS.csv');hist=hist.loc[hist.role=='MULTISEED_PRIMARY'].copy()
    assess=j.read_csv(j.I2/'outputs/metrics/I2_ASSESSMENT_METRICS.csv')
    historical=[]
    for scope,frame in [('historical Validation',hist),('reused Development Assessment; existing metrics only',assess)]:
        for model in ['F0','F1','F2']:
            f=frame.loc[frame.model.eq(model)].sort_values('seed')
            historical.append({'scope':scope,'model':model,'Seed42_RMSE_mV':float(f.loc[f.seed.eq(42),'RMSE_mV'].iloc[0]),
                'Seed52_RMSE_mV':float(f.loc[f.seed.eq(52),'RMSE_mV'].iloc[0]),'Seed62_RMSE_mV':float(f.loc[f.seed.eq(62),'RMSE_mV'].iloc[0]),
                'equal_seed_mean_RMSE_mV':float(f.RMSE_mV.mean()),'sample_SD_mV':float(f.RMSE_mV.std(ddof=1))})
    j.csv_write(j.OUT/'architecture/J_HISTORICAL_ABLATION_EVIDENCE.csv',pd.DataFrame(historical))
    summary={'phase':j.PHASE.name,'base_commit':j.BASE,'architecture_status':'PHASE_J_F2_ARCHITECTURE_FROZEN',
        'gate_status':j.gate_decision(stats,comparisons),'protocol_sha256':j.digest(j.PROTOCOL),
        'code_identity_sha256':cfg['code_identity']['sha256'],'checkpoint_identity':'9/9 PASS','gate_records':615,
        'gate_statistics':stats.to_dict('records'),'fixed_gate_comparison':comparisons.to_dict('records'),
        'paired_seed_gate_differences':pairs,'historical_evidence':historical,'limitations':j.LIMITS,
        'physical_mechanism_proven':False,'training':False,'architecture_modified':False,'new_search':False,
        'new_Assessment_inference':False,'formal_Test_accessed':False,'seed_selection':False,'checkpoint_modified':False,
        'SOH_RUL_complete':False,'automatic_next_phase':False}
    j.json_write(j.OUT/'J_SUMMARY.json',summary)
    print(summary['architecture_status'],summary['gate_status']);print(comparisons.to_string(index=False))

if __name__=='__main__':analyze()
