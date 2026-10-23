"""Five-model paired cache comparison; frozen Train-only regime thresholds."""
from baseline_common import *
from audit_run import scalar
MODELS=['LSTM','TCN','F0','F1','F2'];SEEDS=[42,52,62]
CONDITIONS=['current_variability','load_transitions','load_level']

def main():
    c=cfg();assert all(read(P/f'outputs/audits/{m}_seed{s}_AUDIT.json')['status']=='PASS' for m,s in RUNS)
    d=data();authority=rd(P/'outputs/audits/VALIDATION_KEY_TRUTH.csv');rows=[];frames=[]
    for name in MODELS:
        for seed in SEEDS:
            if name in ['LSTM','TCN']:
                r=read(P/f'outputs/metrics/{name}_seed{seed}.json');f=rd(P/r['predictions']);source=P/r['predictions']
            else:
                source_row=rd(H/'outputs/metrics/H_VALIDATION_METRICS.csv').query("role=='MULTISEED_PRIMARY' and model==@name and seed==@seed").iloc[0]
                source=H/source_row.validation_prediction;assert sha(source)==source_row.prediction_file_sha256;f=rd(source)
                for col in ['origin_row','target_row','horizon_step','y_true_delta_V']:assert np.array_equal(f[col],authority[col])
                for col in authority:
                    if col not in f:f[col]=authority[col].to_numpy()
                f['origin_age_h']=np.repeat(d['val_origin_age_h'],24);f['target_age_h']=d['val_target_age_h'].ravel();f['origin_raw_voltage_V']=np.repeat(d['val_origin_raw_voltage_V'],24)
                f['predicted_voltage_V']=f.origin_raw_voltage_V+f.y_pred_delta_V;f['true_label_voltage_V']=f.origin_raw_voltage_V+f.y_true_delta_V
                r=source_row.to_dict()
            assert_keys(f);f['model']=name;f['seed']=seed
            f['true_raw_target_voltage_V']=d['val_target_raw_voltage_V'].ravel()
            frames.append(f[['model','seed','origin_row','target_row','horizon_step','origin_age_h','target_age_h','origin_raw_voltage_V','y_true_delta_V','y_pred_delta_V','predicted_voltage_V','true_label_voltage_V','true_raw_target_voltage_V']])
            rows.append(dict(model=name,seed=seed,**metrics(f),best_epoch=int(r['best_epoch']),stop_epoch=int(r['stop_epoch']),parameter_count=int(r['parameter_count']),training_time_s=float(r['training_time_s']),peak_GPU_memory_bytes=int(r['peak_GPU_memory_bytes']),prediction_source=source.relative_to(REPO).as_posix(),prediction_sha256=sha(source),stage='Development Validation'))
    allf=pd.concat(frames,ignore_index=True);table=pd.DataFrame(rows);cs('tables/FIVE_MODEL_PER_SEED.csv',table);cs('outputs/predictions/ALIGNED_FIVE_MODEL_PREDICTIONS.csv',allf)
    numeric=['RMSE_mV','MAE_mV','R2','endpoint_RMSE_mV','target_time_equal_RMSE_mV','P95_mV','P99_mV','max_abs_error_mV','training_time_s','peak_GPU_memory_bytes']
    summary=[]
    for name,z in table.groupby('model',sort=False):
        r=dict(model=name,seed_n=3,parameter_count=int(z.parameter_count.iloc[0]))
        for metric in numeric:r[metric+'_mean']=float(z[metric].mean());r[metric+'_sample_SD']=float(z[metric].std(ddof=1))
        summary.append(r)
    sm=pd.DataFrame(summary).sort_values('RMSE_mV_mean',kind='stable');sm.insert(0,'RMSE_rank',np.arange(1,6));cs('tables/FIVE_MODEL_SUMMARY.csv',sm)
    paired=[]
    for comparator in ['LSTM','TCN','F0','F1']:
        for seed in SEEDS:
            a=table.query("model=='F2' and seed==@seed").iloc[0];b=table.query('model==@comparator and seed==@seed').iloc[0]
            paired.append(dict(comparator=comparator,seed=seed,**{metric+'_F2_minus_comparator':float(a[metric]-b[metric]) for metric in numeric[:8]},F2_RMSE_win=bool(a.RMSE_mV<b.RMSE_mV)))
    pair=pd.DataFrame(paired);cs('tables/F2_PAIRED_COMPARISONS.csv',pair)
    horizons=[]
    for (name,seed,step),g in allf.groupby(['model','seed','horizon_step']):
        e=(g.y_pred_delta_V-g.y_true_delta_V)*1000;horizons.append(dict(model=name,seed=int(seed),horizon_step=int(step),horizon_h=float(step*.5),n=len(g),RMSE_mV=float(np.sqrt(np.mean(e*e))),MAE_mV=float(np.mean(abs(e)))))
    cs('tables/HORIZON_METRICS.csv',horizons)
    cond=rd(PAPER/'outputs/condition_analysis/ORIGIN_CONDITIONS.csv');thr=rd(PAPER/'outputs/condition_analysis/TRAIN_THRESHOLDS.csv')
    for t in thr.itertuples():
        train=d['train_x_raw'][:,:,t.index].mean(axis=1,dtype='float64');vals=d['val_x_raw'][:,:,t.index].mean(axis=1,dtype='float64')
        assert np.array_equal(np.quantile(train,[1/3,2/3]),[t.q1,t.q2]);assert np.array_equal(vals,cond[t.condition+'_raw'])
        assert np.array_equal(np.where(vals<=t.q1,'LOW',np.where(vals<=t.q2,'MID','HIGH')),cond[t.condition+'_group'])
    cs('outputs/audits/REUSED_TRAIN_THRESHOLDS.csv',thr);cs('outputs/audits/REUSED_ORIGIN_CONDITIONS.csv',cond)
    merged=allf.merge(cond,on=['origin_row','origin_age_h'],how='left',validate='many_to_one');assert len(merged)==73800 and merged.current_variability_group.notna().all()
    groups=[]
    for condition in CONDITIONS:
        for (name,seed,group),g in merged.groupby(['model','seed',condition+'_group']):groups.append(dict(condition=condition,group=group,model=name,seed=int(seed),**metrics(g)))
    gm=pd.DataFrame(groups);cs('tables/CONDITION_METRICS.csv',gm);cp=[]
    for (condition,group),z in gm.groupby(['condition','group']):
        for comparator in ['LSTM','TCN','F0','F1']:
            a=z.query("model=='F2'").set_index('seed');b=z.query('model==@comparator').set_index('seed')
            cp.append(dict(condition=condition,group=group,comparator=comparator,origins=int(a.origins.iloc[0]),F2_RMSE_mean=float(a.RMSE_mV.mean()),comparator_RMSE_mean=float(b.RMSE_mV.mean()),F2_minus_comparator_mean_mV=float((a.RMSE_mV-b.RMSE_mV).mean()),F2_paired_wins=int((a.RMSE_mV<b.RMSE_mV).sum()),seeds=3))
    cs('tables/CONDITION_PAIRED_COMPARISONS.csv',cp)
    # Fully prespecified examples, same single origin and seed across five methods.
    cases=allf[allf.seed.eq(62)&allf.origin_row.isin([2205,2224])];cs('outputs/predictions/FIXED_CASES.csv',cases)
    extreme=allf[allf.target_row.isin([2247,2248])].copy();extreme['absolute_error_mV']=abs(extreme.y_pred_delta_V-extreme.y_true_delta_V)*1000;cs('tables/EXTREME_TARGET_2247_2248.csv',extreme)
    checks=[]
    for scope,frame in [('overall',table),('condition',gm)]:
        for r in frame.itertuples():
            f=allf[(allf.model==r.model)&(allf.seed==r.seed)]
            if scope=='condition':f=f[f.origin_row.isin(cond.loc[cond[r.condition+'_group'].eq(r.group),'origin_row'])]
            for metric,value in scalar(f).items():
                delta=abs(value-getattr(r,metric));assert delta<1e-10
                checks.append(dict(scope=scope,model=r.model,seed=int(r.seed),condition=getattr(r,'condition','ALL'),group=getattr(r,'group','ALL'),metric=metric,stored=getattr(r,metric),independent=value,difference=delta))
    cs('outputs/audits/INDEPENDENT_METRIC_CHECKS.csv',checks)
    n=protect();audit=dict(status='PASS',overall_settings=15,condition_settings=len(gm),independent_metric_checks=len(checks),max_metric_difference=max(r['difference'] for r in checks),all_prediction_queries=len(allf),targets_per_model=4920,origins_per_model=205,distinct_target_times=251,history_files_unchanged=n,Test=False,Assessment=False,all_six_run_audits=True)
    js('outputs/audits/FIVE_MODEL_INDEPENDENT_AUDIT.json',audit)
    js('outputs/SUMMARY.json',dict(status='SCI_BASELINE_6_RUNS_COMPLETED',ranking=sm[['model','RMSE_mV_mean','RMSE_mV_sample_SD']].to_dict('records'),paired_RMSE_wins={name:int(g.F2_RMSE_win.sum()) for name,g in pair.groupby('comparator')},endpoint_best_model=str(sm.sort_values('endpoint_RMSE_mV_mean').iloc[0].model),scientific_identity='Development exploratory; same-data/config budget, unequal capacity and historically TFT-derived learning rate; not independent Test',audit=audit))
    print(sm[['model','RMSE_mV_mean','RMSE_mV_sample_SD','endpoint_RMSE_mV_mean','parameter_count']].to_string(index=False),flush=True)

if __name__=='__main__':main()
