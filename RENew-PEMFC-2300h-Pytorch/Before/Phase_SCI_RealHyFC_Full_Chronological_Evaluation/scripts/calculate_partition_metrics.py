from common import *
def metrics(f):
    y=f.y_true_delta_V.to_numpy(float);e=(f.y_pred_delta_V.to_numpy(float)-y)*1000;a=np.abs(e)
    endpoint=e[f.horizon_step.to_numpy()==24]
    bytime=pd.DataFrame(dict(target=f.target_row,se=e*e)).groupby('target').se.mean()
    ss=np.sum((y-y.mean())**2)
    return dict(RMSE_mV=float(np.sqrt(np.mean(e*e))),MAE_mV=float(a.mean()),R2=float(1-np.sum((e/1000)**2)/ss) if ss>0 else None,endpoint_RMSE_mV=float(np.sqrt(np.mean(endpoint**2))) if len(endpoint) else None,target_time_equal_RMSE_mV=float(np.sqrt(bytime.mean())),P95_mV=float(np.quantile(a,.95)),P99_mV=float(np.quantile(a,.99)),max_abs_error_mV=float(a.max()),queries=len(f),origins=int(f.origin_row.nunique()),unique_targets=int(f.target_row.nunique()))
def calculate():
    rows=[];steps=[];originrows=[]
    for part in PARTS:
        for model in MODELS:
            for seed in SEEDS:
                f=rd(P/f'outputs/predictions/{part}_{model}_seed{seed}.csv')
                rows.append(dict(partition=part,model=model,seed=seed,**metrics(f)))
                for step,z in f.groupby('horizon_step'):
                    e=(z.y_pred_delta_V-z.y_true_delta_V)*1000
                    steps.append(dict(partition=part,model=model,seed=seed,horizon_step=int(step),RMSE_mV=float(np.sqrt(np.mean(e*e))),MAE_mV=float(np.mean(abs(e))),queries=len(z)))
                for origin,z in f.groupby('origin_row'):
                    e=(z.y_pred_delta_V-z.y_true_delta_V)*1000
                    originrows.append(dict(partition=part,model=model,seed=seed,origin_row=int(origin),origin_age_h=float(z.origin_age_h.iloc[0]),RMSE_mV=float(np.sqrt(np.mean(e*e))),max_abs_error_mV=float(abs(e).max())))
    detail=pd.DataFrame(rows);cs('outputs/metrics/PER_MODEL_SEED_PARTITION.csv',detail);cs('outputs/metrics/HORIZON_METRICS.csv',steps);cs('outputs/metrics/PER_ORIGIN_METRICS.csv',originrows)
    fields=['RMSE_mV','MAE_mV','R2','endpoint_RMSE_mV','target_time_equal_RMSE_mV','P95_mV','P99_mV','max_abs_error_mV']
    summary=[]
    for (part,model),z in detail.groupby(['partition','model'],sort=False):
        row=dict(partition=part,model=model,seeds=3,queries_per_seed=int(z.queries.iloc[0]),origins_per_seed=int(z.origins.iloc[0]))
        for key in fields:row[key+'_mean']=float(z[key].mean());row[key+'_sample_sd']=float(z[key].std(ddof=1))
        summary.append(row)
    cs('tables/MODEL_THREE_SEED_SUMMARY.csv',summary)
