from audit_common import *
def main():
    check_protocol();f=raw();rows=[];contexts=[];similar=[]
    for event,(a,b) in EVENTS.items():
        z=f.iloc[a-24:b+25].copy();z['event']=event;z['relative_row']=z.row_id-a;contexts.append(z)
    cs('outputs/EVENT_50_ROW_CONTEXT.csv',pd.concat(contexts))
    for field in ['cell_voltage_mean','actual_current_A_mean','actual_current_A_std','zero_current_fraction','load_transition_count','raw_sample_count','datetime_span_h']:
        a,b=[z[field].to_numpy(float) for z in contexts];similar.append(dict(field=field,n=50,correlation=float(np.corrcoef(a,b)[0,1]),RMSE=float(np.sqrt(np.mean((a-b)**2))),max_absolute_difference=float(abs(a-b).max())))
    cs('outputs/EVENT_TEMPLATE_SIMILARITY.csv',similar)
    allerrors=[];contrib=[];target=[];horizon=[];origin=[];pair=[];timing=[];nearby=[];checks=[]
    truth=data('LATE_LIFE');source_manifest=rd(FULL/'outputs/audits/ARTIFACT_MANIFEST.csv').set_index('path')
    for model in MODELS:
        for seed in SEEDS:
            file=FULL/f'outputs/predictions/LATE_LIFE_{model}_seed{seed}.csv'
            assert sha(file)==source_manifest.loc[file.relative_to(FULL).as_posix(),'sha256']
            z=pred(model,seed);assert len(z)==13152
            for key,value in [('origin_row',np.repeat(truth['origins'],24)),('target_row',truth['targets'].ravel()),('horizon_step',np.tile(np.arange(1,25),548)),('y_true_delta_V',truth['y'].ravel())]:assert np.array_equal(z[key],value),(model,seed,key)
            z['error_mV']=(z.y_pred_delta_V-z.y_true_delta_V)*1000;z['squared_error_mV2']=z.error_mV**2;z['event']='OTHER'
            for event,ids in EVENTS.items():z.loc[z.target_row.isin(ids),'event']=event
            allerrors.append(z[['model','seed','origin_row','target_row','horizon_step','origin_age_h','target_age_h','error_mV','squared_error_mV2','event']])
            se=z.squared_error_mV2;total=se.sum();bytime=z.groupby('target_row').squared_error_mV2.agg(['sum','mean','size'])
            for event,ids in dict(EVENTS,COMBINED=(3112,3113,3627,3628)).items():
                selected=z.target_row.isin(ids);sse=se[selected].sum();u=bytime.loc[list(ids)]
                contrib.append(dict(model=model,seed=seed,event=event,formal_RMSE_mV=float(np.sqrt(total/len(z))),all_queries=len(z),event_queries=int(selected.sum()),all_unique_targets=len(bytime),event_unique_targets=len(ids),event_SSE_mV2=float(sse),total_SSE_mV2=float(total),query_weighted_SSE_share_pct=float(sse/total*100),target_time_equal_MSE_share_pct=float(u['mean'].sum()/bytime['mean'].sum()*100),formal_target_time_equal_RMSE_mV=float(np.sqrt(bytime['mean'].mean())),POSTHOC_remainder_RMSE_not_new_formal_score=float(np.sqrt((total-sse)/(len(z)-selected.sum())))))
            for t,g in z.groupby('target_row'):
                target.append(dict(model=model,seed=seed,target_row=int(t),age_h=float(g.target_age_h.iloc[0]),n_queries=len(g),SSE_mV2=float(g.squared_error_mV2.sum()),MSE_mV2=float(g.squared_error_mV2.mean()),SSE_share_pct=float(g.squared_error_mV2.sum()/total*100),mean_prediction_V=float(g.predicted_voltage_V.mean()),actual_raw_V=float(g.true_raw_target_voltage_V.iloc[0]),event=g.event.iloc[0]))
            for h,g in z.groupby('horizon_step'):horizon.append(dict(model=model,seed=seed,horizon_step=int(h),SSE_mV2=float(g.squared_error_mV2.sum()),SSE_share_pct=float(g.squared_error_mV2.sum()/total*100),event_SSE_mV2=float(g.loc[g.event!='OTHER','squared_error_mV2'].sum()),queries=len(g)))
            for o,g in z.groupby('origin_row'):origin.append(dict(model=model,seed=seed,origin_row=int(o),origin_age_h=float(g.origin_age_h.iloc[0]),SSE_mV2=float(g.squared_error_mV2.sum()),SSE_share_pct=float(g.squared_error_mV2.sum()/total*100),RMSE_mV=float(np.sqrt(g.squared_error_mV2.mean()))))
            for event,(a,b) in EVENTS.items():
                select=z.loc[z.target_row.between(a-2,b+2)].copy();select['event_context']=event;nearby.append(select)
                pivot=z.loc[z.target_row.isin([a,b])].pivot(index='origin_row',columns='target_row',values='predicted_voltage_V').dropna()
                for o,r in pivot.iterrows():pair.append(dict(model=model,seed=seed,event=event,origin_row=int(o),first_lead_step=int(a-o),second_lead_step=int(b-o),pred_first_V=r[a],pred_second_V=r[b],actual_first_V=f.loc[a,'cell_voltage_mean'],actual_second_V=f.loc[b,'cell_voltage_mean'],pair_inverted=bool(r[a]<r[b])))
                for o,g in select.groupby('origin_row'):
                    if len(g)!=6:continue
                    g=g.sort_values('target_row');pt=int(g.loc[g.predicted_voltage_V.idxmin(),'target_row']);at=int(g.loc[g.true_raw_target_voltage_V.idxmin(),'target_row'])
                    timing.append(dict(model=model,seed=seed,event=event,origin_row=int(o),actual_min_row=at,predicted_min_row=pt,lag_rows=pt-at,lag_operating_h=(pt-at)*.5,first_lead_step=int(a-o),interpretation='post-hoc local minimum identity, not corrected forecasts'))
            checks.append(dict(model=model,seed=seed,keys_and_truth_exact=True,queries=len(z),source_prediction_sha256=sha(file)))
    for name,values in [('POINTWISE_ERRORS',pd.concat(allerrors)),('EVENT_SSE_CONTRIBUTIONS',contrib),('PER_TARGET_ERRORS',target),('PER_HORIZON_ERRORS',horizon),('PER_ORIGIN_ERRORS',origin),('PAIR_STATE_INVERSION',pair),('LOCAL_TIMING_DIAGNOSTIC',timing),('EVENT_NEIGHBOR_PREDICTIONS',pd.concat(nearby)),('INPUT_PREDICTION_IDENTITY',checks)]:cs('outputs/'+name+'.csv',values)
    # Aggregate independent seeds descriptively, never count overlapping queries as events.
    p=pd.DataFrame(pair);t=pd.DataFrame(timing)
    cs('outputs/PAIR_INVERSION_SUMMARY.csv',p.groupby(['model','seed','event']).pair_inverted.agg(['sum','count','mean']).reset_index())
    cs('outputs/LAG_COUNTS.csv',t.groupby(['model','seed','event','lag_rows']).size().reset_index(name='origins'))
    analogs=[];w=rd(FULL/'tables/TIME_PARTITIONS.csv')
    for row in f.itertuples():
        r=int(row.row_id)
        if r<1 or r+1>=len(f) or not row.bin_valid_for_windows:continue
        if row.zero_current_fraction>.5 and f.loc[r-1,'zero_current_fraction']==0 and f.loc[r+1,'zero_current_fraction']==0:
            part=w.loc[(w.start_inclusive<=r)&(r<w.end_exclusive),'partition'].iloc[0]
            if f.loc[r-1,'temporal_segment_id']!=row.temporal_segment_id or f.loc[r+1,'temporal_segment_id']!=row.temporal_segment_id:continue
            d=data(part);eligible=bool(np.isin(r,d['targets']))
            analogs.append(dict(partition=part,row_id=r,age_h=row.bin_end_ophrs2_h,voltage_V=row.cell_voltage_mean,prev_voltage_V=f.loc[r-1,'cell_voltage_mean'],next_voltage_V=f.loc[r+1,'cell_voltage_mean'],zero_current_fraction=row.zero_current_fraction,raw_sample_count=row.raw_sample_count,datetime_span_h=row.datetime_span_h,coverage_ratio=row.coverage_ratio,used_as_forecast_target=eligible))
    cs('outputs/HISTORICAL_STOP_MORPHOLOGY_ANALOGS.csv',analogs)
    nearest=[];curves=[]
    for event,(a,b) in EVENTS.items():
        origin_row=a-12;late=data('LATE_LIFE');i=int(np.flatnonzero(late['origins']==origin_row)[0]);query=late['x'][i].astype(float)
        for part in ['TRAIN','VALIDATION']:
            d=data(part);distance=np.sqrt(np.mean((d['x'].astype(float)-query)**2,axis=(1,2)));selected=np.argsort(distance,kind='stable')[:5]
            for rank,k in enumerate(selected,1):
                o=int(d['origins'][k]);seq=d['target_raw_voltage_V'][k]
                nearest.append(dict(event=event,event_origin_row=origin_row,reference_partition=part,rank=rank,reference_origin_row=o,distance_standardized_RMS=float(distance[k]),reference_minimum_lead=int(np.argmin(seq)+1),late_actual_minimum_lead=int(np.argmin(late['target_raw_voltage_V'][i])+1),reference_V_lead12=float(seq[11]),reference_V_lead13=float(seq[12]),selection='history only; no label distance'))
                for h in range(24):curves.append(dict(event=event,reference_partition=part,rank=rank,reference_origin_row=o,lead_step=h+1,reference_actual_V=float(seq[h]),late_actual_V=float(late['target_raw_voltage_V'][i,h])))
    cs('outputs/NEAREST_HISTORY_ONLY_MATCHES.csv',nearest);cs('outputs/NEAREST_HISTORY_FUTURE_DIAGNOSTIC.csv',curves)
    print('ANALYSIS_COMPLETE; all original queries retained',flush=True)
if __name__=='__main__':main()
