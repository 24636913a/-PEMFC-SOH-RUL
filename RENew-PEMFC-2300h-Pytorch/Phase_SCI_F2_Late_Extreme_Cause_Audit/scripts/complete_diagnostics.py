"""Additional descriptive cross-checks, without adjusting any predictions."""
from audit_common import *
def main():
    check_protocol();f=raw();analogs=rd(P/'outputs/HISTORICAL_STOP_MORPHOLOGY_ANALOGS.csv');all_lags=[];comparisons=[];focus=[]
    centers=analogs.loc[(analogs.partition=='LATE_LIFE')&analogs.used_as_forecast_target,'row_id'].astype(int).tolist()
    for model in MODELS:
        for seed in SEEDS:
            z=pred(model,seed);z['se']=((z.y_pred_delta_V-z.y_true_delta_V)*1000)**2
            for center in centers:
                chosen=z.loc[z.target_row.between(center-3,center+2)]
                for o,g in chosen.groupby('origin_row'):
                    if len(g)!=6:continue
                    pt=int(g.loc[g.predicted_voltage_V.idxmin(),'target_row']);at=int(g.loc[g.true_raw_target_voltage_V.idxmin(),'target_row'])
                    all_lags.append(dict(model=model,seed=seed,stop_target_row=center,origin_row=int(o),predicted_min_row=pt,actual_min_row=at,lag_rows=pt-at,core_event=center in [3113,3628]))
            if model=='F2':continue
            f2=pred('F2',seed)
            for k in ['origin_row','target_row','horizon_step','y_true_delta_V']:assert np.array_equal(z[k],f2[k])
            f2se=((f2.y_pred_delta_V-f2.y_true_delta_V)*1000)**2
            for name,mask in [('ALL',np.ones(len(z),bool)),('FOUR_CORE_TARGETS',z.target_row.isin([3112,3113,3627,3628])),('OTHER_TARGETS',~z.target_row.isin([3112,3113,3627,3628]))]:
                comparisons.append(dict(seed=seed,baseline=model,subset=name,queries=int(np.sum(mask)),F2_SSE_mV2=float(f2se[mask].sum()),baseline_SSE_mV2=float(z.loc[mask,'se'].sum()),F2_minus_baseline_SSE_mV2=float(f2se[mask].sum()-z.loc[mask,'se'].sum()),F2_abs_error_smaller_queries=int((f2se[mask]<z.loc[mask,'se']).sum()),F2_abs_error_larger_queries=int((f2se[mask]>z.loc[mask,'se']).sum()),identical_keys=True,scientific_status='descriptive paired frozen-model contrast, no causal isolation'))
    cs('outputs/ALL_STOP_LOCAL_TIMING.csv',all_lags);cs('outputs/MATCHED_F2_BASELINE_DIFFERENCES.csv',comparisons)
    lag=pd.DataFrame(all_lags);cs('outputs/ALL_STOP_LAG_SUMMARY.csv',lag.groupby(['model','seed','core_event','lag_rows']).size().reset_index(name='origins'))
    g=rd(P/'outputs/ORIGINAL_GATE_AND_PATH_OBSERVATIONS.csv')
    for seed in SEEDS:
        train=g.loc[(g.seed==seed)&(g.partition=='TRAIN')]
        for event,(a,b) in EVENTS.items():
            z=g.loc[(g.seed==seed)&(g.partition=='LATE_LIFE')&g.origin_row.between(b-24,a-1)]
            record=dict(seed=seed,event=event,origins=len(z),g_mean=float(z.g.mean()),g_min=float(z.g.min()),g_max=float(z.g.max()),outside_Train_g_range=int(((z.g<train.g.min())|(z.g>train.g.max())).sum()),path_difference_RMS_mean=float(z.path_difference_RMS.mean()),outside_Train_path_difference_range=int(((z.path_difference_RMS<train.path_difference_RMS.min())|(z.path_difference_RMS>train.path_difference_RMS.max())).sum()))
            for c in [c for c in g if c.endswith('_scaled_context')]:record[c+'_outside_Train_range']=int(((z[c]<train[c].min())|(z[c]>train[c].max())).sum())
            focus.append(record)
    cs('outputs/EVENT_GATE_TRAIN_SUPPORT.csv',focus)
    c=rd(P/'outputs/EVENT_50_ROW_CONTEXT.csv');similar=[]
    for scope,mask in [('CORE_6',lambda z:z.relative_row.between(-2,3)),('FULL_50',lambda z:np.ones(len(z),bool))]:
        a=c.loc[(c.event=='E1')&mask(c)].sort_values('relative_row');b=c.loc[(c.event=='E2')&mask(c)].sort_values('relative_row')
        for field in ['cell_voltage_mean','actual_current_A_mean','zero_current_fraction','load_transition_count']:
            x=a[field].to_numpy();y=b[field].to_numpy();similar.append(dict(scope=scope,field=field,n=len(x),correlation=float(np.corrcoef(x,y)[0,1]),RMSE=float(np.sqrt(np.mean((x-y)**2))),maximum_difference=float(abs(x-y).max())))
    cs('outputs/CORE_AND_CONTEXT_SIMILARITY.csv',similar)
    print('DESCRIPTIVE_CROSS_CHECKS_COMPLETE',flush=True)
if __name__=='__main__':main()
