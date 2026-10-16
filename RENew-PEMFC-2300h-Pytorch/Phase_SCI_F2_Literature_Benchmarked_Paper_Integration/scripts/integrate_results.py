"""New no-training computations: persistence and Train-defined regime evaluation.
O1 metrics are independently checked and reused. All inputs are saved legal caches.
Execution gated on completed literature matrix AND revised outline (user order).
"""
from common import *
def main():
    c=verified_config();assert c['literature_first_complete']
    for file in ['literature/01_已发表PEMFC论文横向比较矩阵.md','outline/04_正式SCI论文大纲_文献对标修订版.md']:
        assert (P/file).exists()
    d=np.load(R2/'outputs/audits/FROZEN_H_DATA.npz');keys=rd(R2/'outputs/audits/VALIDATION_KEY_TRUTH.csv')
    assert d['val_y'].shape==(205,24) and d['tx'].shape==(1380,48,14)
    thr=rd(J/'outputs/gate_diagnostics/J_TRAIN_ONLY_CONDITION_THRESHOLDS.csv');condition=pd.DataFrame({'origin_row':d['val_origins'],'origin_age_h':d['val_origin_age_h']})
    tr_records=[]
    for name,(feature,idx) in CONDITIONS.items():
        train_x=d['train_x_raw'][:,:,idx].astype('float64').mean(axis=1);x=d['val_x_raw'][:,:,idx].astype('float64').mean(axis=1)
        lo,hi=np.quantile(train_x,[1/3,2/3]);t=thr[thr.feature==feature].iloc[0]
        assert np.array_equal([lo,hi],[t.low_upper,t.high_lower]),('TRAIN_THRESHOLD_IDENTITY',feature)
        condition[name+'_raw']=x;condition[name+'_group']=np.where(x<=lo,'LOW',np.where(x<=hi,'MID','HIGH'))
        tr_records.append(dict(condition=name,feature=feature,index=idx,train_n=1380,q1=lo,q2=hi,rule='LOW<=q1; MID q1<x<=q2; HIGH>q2',source='Phase J frozen Train terciles; exact recheck'))
    cs('outputs/condition_analysis/TRAIN_THRESHOLDS.csv',tr_records);cs('outputs/condition_analysis/ORIGIN_CONDITIONS.csv',condition)
    hm=rd(H/'outputs/metrics/H_VALIDATION_METRICS.csv');hm=hm[hm.role=='MULTISEED_PRIMARY'];assert len(hm)==9
    prior=rd(O1/'outputs/O1_NINE_MODEL_VALIDATION_METRICS.csv');allm=[];allh=[];groups=[];win=[];sse=[];identity=[];frames=[]
    def evaluate(f,model,seed):
        m=metrics(f);allm.append(dict(model=model,seed=seed,**m))
        enriched=f.merge(condition,on='origin_row',how='left',validate='many_to_one')
        for hh,z in f.groupby('horizon_step'):
            e=(z.y_pred_delta_V-z.y_true_delta_V).to_numpy()*1000
            allh.append(dict(model=model,seed=seed,horizon_step=hh,lead_h=hh*.5,n=len(z),RMSE_mV=float(np.sqrt(np.mean(e*e))),MAE_mV=float(np.mean(abs(e)))))
        for name in CONDITIONS:
            for label in ['LOW','MID','HIGH']:
                z=enriched[enriched[name+'_group']==label]
                groups.append(dict(model=model,seed=seed,condition=name,group=label,**(metrics(z) if len(z) else dict(n=0,origins=0,unique_targets=0))))
        for origin,z in f.groupby('origin_row'):
            e=(z.y_pred_delta_V-z.y_true_delta_V).to_numpy()*1000
            win.append(dict(model=model,seed=seed,origin_row=origin,RMSE_mV=float(np.sqrt(np.mean(e*e)))))
        f=f.copy();f['e2_mV2']=((f.y_pred_delta_V-f.y_true_delta_V)*1000)**2
        total=float(f.e2_mV2.sum())
        for target,z in f.groupby('target_row'):
            sse.append(dict(model=model,seed=seed,target_row=target,multiplicity=len(z),SSE_mV2=float(z.e2_mV2.sum()),SSE_share=float(z.e2_mV2.sum()/total),MSE_mV2=float(z.e2_mV2.mean())))
        return m
    for r in hm.itertuples():
        path=H/r.validation_prediction;f=rd(path);assert sha(path)==r.prediction_file_sha256
        for col in ['origin_row','target_row','horizon_step','y_true_delta_V']:assert np.array_equal(f[col],keys[col]),('KEY_OR_LABEL_MISMATCH',r.run_id,col)
        f=f[['origin_row','target_row','horizon_step','y_true_delta_V','y_pred_delta_V']].copy();f['model']=r.model;f['seed']=r.seed
        f['origin_age_h']=np.repeat(d['val_origin_age_h'],24);f['target_age_h']=d['val_target_age_h'].ravel();f['origin_raw_voltage_V']=np.repeat(d['val_origin_raw_voltage_V'],24)
        m=evaluate(f,r.model,r.seed)
        for metric in ['RMSE_mV','MAE_mV','R2']:assert abs(m[metric]-getattr(r,metric))<1e-10
        old=prior[(prior.model==r.model)&(prior.seed==r.seed)].iloc[0]
        for metric in ['RMSE_mV','MAE_mV','endpoint_RMSE_mV','target_time_equal_RMSE_mV','P95_mV','P99_mV','max_abs_error_mV']:
            assert abs(m[metric]-float(old[metric]))<1e-9,('O1_DIFFERENCE',r.run_id,metric)
        identity.append(dict(run_id=r.run_id,keys_exact=True,labels_exact=True,source_sha256=sha(path),O1_metrics_match=True,n=4920))
        frames.append(f)
    f=frames[0].copy();f['model']='Persistence';f['seed']=-1;f['y_pred_delta_V']=0.
    evaluate(f,'Persistence',-1);cs('outputs/baseline/PERSISTENCE_PREDICTIONS.csv',f);frames.append(f)
    combined=pd.concat(frames,ignore_index=True);cs('outputs/experiment_tables/ALIGNED_FORMAL_PREDICTIONS.csv',combined)
    cs('outputs/experiment_tables/FORMAL_VALIDATION_METRICS.csv',allm);cs('outputs/experiment_tables/HORIZON_METRICS.csv',allh)
    cs('outputs/condition_analysis/CONDITION_METRICS.csv',groups);cs('outputs/experiment_tables/WINDOW_METRICS.csv',win);cs('outputs/experiment_tables/TARGET_SSE.csv',sse)
    cs('outputs/audits/FORMAL_CACHE_IDENTITY.csv',identity)
    summary=[]
    for model,z in pd.DataFrame(allm).groupby('model',sort=False):
        for key in ['RMSE_mV','MAE_mV','R2','endpoint_RMSE_mV','target_time_equal_RMSE_mV','P95_mV','P99_mV','max_abs_error_mV']:
            summary.append(dict(model=model,metric=key,n_seeds=len(z) if model!='Persistence' else 0,mean=float(z[key].mean()),sample_SD=float(z[key].std(ddof=1)) if len(z)>1 else None))
    cs('outputs/experiment_tables/EQUAL_SEED_SUMMARY.csv',summary)
    gp=pd.DataFrame(groups);cmp=[]
    for (cond,label),z in gp[gp.model!='Persistence'].groupby(['condition','group']):
        for control in ['F0','F1']:
            p=z[z.model.isin(['F2',control])].pivot(index='seed',columns='model',values='RMSE_mV');delta=p.F2-p[control]
            cmp.append(dict(condition=cond,group=label,control=control,F2_minus_control_mean_RMSE_mV=float(delta.mean()),F2_paired_wins=int((delta<0).sum()),n_seeds=len(delta),origins=int(z.origins.iloc[0])))
    cs('outputs/condition_analysis/PAIRED_REGIME_COMPARISONS.csv',cmp)
    # Reuse I2 saved summary ONLY: no Assessment inference or target values are opened.
    cs('outputs/experiment_tables/REUSED_ASSESSMENT_METRICS.csv',rd(I2/'outputs/metrics/I2_ASSESSMENT_METRICS.csv'))
    cs('outputs/experiment_tables/SIX_SELECTED_INSTANCES.csv',rd(R2/'outputs/F2_SIX_CANDIDATE_SEEDS.csv'))
    cs('outputs/experiment_tables/SIX_GATE_INTERVENTIONS.csv',rd(SIX/'outputs/METRICS.csv'))
    cs('outputs/experiment_tables/SIX_GATE_STATISTICS.csv',rd(SIX/'outputs/GATE_STATISTICS.csv'))
    cs('outputs/experiment_tables/SIX_GATE_CORRELATIONS.csv',rd(SIX/'outputs/GATE_CORRELATIONS.csv'))
    w=pd.DataFrame(win);z=w[w.model=='F2'].groupby('origin_row').RMSE_mV.mean();hard=int(z.sort_values(ascending=False,kind='stable').index[0])
    cs('outputs/experiment_tables/CASE_SELECTION.csv',[dict(case='fixed_typical',origin_row=2205,rule='inherit fixed origin from screening protocol'),dict(case='hardest_formal_F2',origin_row=hard,rule='largest mean window RMSE over all three formal F2 seeds; diagnostic only')])
    out=dict(training=False,model_forward=False,Test=False,new_Assessment_inference=False,Assessment_summary_only=True,formal_runs=9,
        queries_per_run=4920,validation_origins=205,unique_target_times=251,condition_rules='Three pre-existing Train tercile rules; no tuning',
        no_new_seed_selection=True,O1_all_metrics_reproduced=True)
    js('outputs/audits/COMPUTATION_STATUS.json',out);print('CACHE_ONLY_INTEGRATION_COMPLETE',flush=True)
if __name__=='__main__':main()
