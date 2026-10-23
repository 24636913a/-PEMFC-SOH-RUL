"""Fixed-metric analysis, gate perturbation, plots and evidence-based reports.

No training; no data loader for Assessment/late-life. Gate perturbation is a
post-training diagnostic, never a checkpoint-selection criterion.
"""
from experiment import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def rd(s): return pd.read_csv(path(s),float_precision='round_trip')

def metric(frame):
    e=(frame.y_pred_delta_V.to_numpy()-frame.y_true_delta_V.to_numpy())*1000
    y=frame.y_true_delta_V.to_numpy()*1000
    by=pd.DataFrame(dict(target=frame.target_row,SE=e**2)).groupby('target').SE.mean()
    return dict(n_queries=len(e),n_origins=frame.origin_row.nunique(),n_unique_targets=frame.target_row.nunique(),
      RMSE_mV=float(np.sqrt(np.mean(e**2))),MAE_mV=float(np.mean(abs(e))),R2_deltaV=float(1-np.sum(e**2)/np.sum((y-y.mean())**2)),
      endpoint_RMSE_mV=float(np.sqrt(np.mean(e[frame.horizon_step.to_numpy()==24]**2))),
      target_time_equal_RMSE_mV=float(np.sqrt(by.mean())),P95_mV=float(np.quantile(abs(e),.95)),
      P99_mV=float(np.quantile(abs(e),.99)),max_abs_error_mV=float(max(abs(e))))

def strata(z,p):
    raw=z['val_x_raw']; stop=(raw[:,-12:,13]>.5).any(1)
    switch=raw[:,-6:,11].mean(1)>=p['grouping']['frequent_switch_threshold_train_p75']
    labels=np.where(stop,'stop_related',np.where(switch,'frequent_switch','normal'))
    csv('outputs/VALIDATION_HISTORY_STRATA.csv',dict(origin_row=z['val_origins'],stratum=labels,
        zero_recent12_max=raw[:,-12:,13].max(1),load_recent6_mean=raw[:,-6:,11].mean(1)))
    return labels

def gates(z,p):
    device=torch.device('cuda'); old.configure_environment('cuda')
    rows=[]; replay=[]; interventions=[]; equivalence=[]; zero_metrics=[]
    for seed in SEEDS:
        for variant in ['B0','A1','B1']:
            if variant=='B0':
                m=old.initialize_f2(seed,p['hyperparameters'],device)
                ck=torch.load(old.HISTORY/f'outputs/checkpoints/H-F2-Seed{seed}_best.pt',map_location='cpu',weights_only=True)
            else:
                m=initialize(seed,variant,p,device)
                ck=torch.load(path(f'checkpoints/{variant}_seed{seed}_best.pt'),map_location='cpu',weights_only=True)
            m.load_state_dict(ck['state_dict']); m.eval(); before=old.state_sha256(m)
            pred=predict(m,z['vx'],device)
            reference=rd(f'outputs/predictions/{variant}_seed{seed}.csv')
            physical=pred*json.loads((D/'SCALER.json').read_text())['target_scale_V']
            req(np.array_equal(physical.ravel().astype(float),reference.y_pred_delta_V.to_numpy()),'POSTTRAIN_REPLAY')
            if variant!='B0': req(np.array_equal(pred,ck['validation_standardized_prediction'].numpy()),'CKPT_PREDICTION_IDENTITY')
            if variant=='A1':
                native=old.initialize_f2(seed,p['hyperparameters'],device)
                state={k:v for k,v in ck['state_dict'].items() if k in native.state_dict()}
                state['regime_gate.hidden.weight']=state['regime_gate.hidden.weight'][:,:5].contiguous()
                native.load_state_dict(state,strict=True)
                native_pred=predict(native,z['vx'],device)
                difference=float(np.max(abs(native_pred-pred)))
                req(difference<5e-6,'ABLATION_NATIVE_FUNCTION_MISMATCH')
                equivalence.append(dict(seed=seed,max_standardized_difference=difference,
                    max_difference_mV=difference*json.loads((D/'SCALER.json').read_text())['target_scale_V']*1000,
                    comparison='same trained A1 weights sliced to original 5-input gate; not historical B0 weights',
                    numeric_tolerance_standardized=5e-6,identity_keys_exact=True))
                del native
            with torch.no_grad():
                x=torch.from_numpy(z['vx']).to(device)
                g=m.regime_weight(x).cpu().numpy().ravel()
                context=m.regime_context(x)
                if variant=='B1':
                    base_g=g.copy()
                    for j,name in enumerate(FEATURES):
                        altered=context.clone(); altered[:,5+j]=0
                        changed=m.regime_gate(altered).cpu().numpy().ravel()
                        interventions.append(dict(seed=seed,feature=name,gate_change_RMS=float(np.sqrt(np.mean((changed-base_g)**2))),gate_change_max=float(max(abs(changed-base_g)))))
                    m.zero_features_inference=True
                    gp=m.regime_weight(x).cpu().numpy().ravel()
                    altered_prediction=predict(m,z['vx'],device)
                    m.zero_features_inference=False
                    zero_frame=reference.assign(y_pred_delta_V=(altered_prediction*json.loads((D/'SCALER.json').read_text())['target_scale_V']).ravel())
                    csv(f'outputs/predictions/DIAGNOSTIC_B1_new_features_zero_seed{seed}.csv',zero_frame)
                    zero_metrics.append(dict(seed=seed,setting='posthoc same weights: all three features set to Train mean',**metric(zero_frame),
                      prediction_change_RMS_mV=float(np.sqrt(np.mean(((altered_prediction-pred)*json.loads((D/'SCALER.json').read_text())['target_scale_V']*1000)**2)))))
                    interventions.append(dict(seed=seed,feature='ALL_THREE',gate_change_RMS=float(np.sqrt(np.mean((gp-base_g)**2))),gate_change_max=float(max(abs(gp-base_g)))))
            for i,origin in enumerate(z['val_origins']): rows.append(dict(seed=seed,variant=variant,origin_row=int(origin),origin_age_h=float(z['val_origin_age_h'][i]),g=float(g[i])))
            req(before==old.state_sha256(m),'INFERENCE_CHANGED_WEIGHTS')
            replay.append(dict(seed=seed,variant=variant,queries=4920,prediction_exact=True,weights_unchanged=True))
            del m; torch.cuda.empty_cache()
    csv('outputs/GATES.csv',rows); csv('outputs/GATE_FEATURE_INTERVENTIONS.csv',interventions); csv('outputs/FINAL_FORWARD_IDENTITY.csv',replay)
    csv('outputs/A1_NATIVE_FUNCTION_EQUIVALENCE.csv',equivalence)
    stats=pd.DataFrame(rows).groupby(['seed','variant']).g.agg(['mean','std','min','max']).reset_index()
    csv('outputs/GATE_SUMMARY.csv',stats); csv('outputs/POSTHOC_ZERO_FEATURE_METRICS.csv',zero_metrics)

def calculate():
    p=get_protocol(); z,sc=load_data(); labels=strata(z,p)
    budget=[]
    for seed in SEEDS:
        for v in ['A1','B1']:
            ledger=readj(f'outputs/ledger/{v}_seed{seed}.json')
            req(ledger['status']=='COMPLETED','INCOMPLETE_TRAINING')
            budget.append(ledger)
    csv('outputs/TRAINING_BUDGET.csv',budget)
    trajectories=[]
    for seed in SEEDS:
        historic=pd.read_csv(old.HISTORY/f'outputs/training_logs/H-F2-Seed{seed}/epochs.csv',float_precision='round_trip')
        fresh=rd(f'outputs/training_logs/A1_seed{seed}.csv')
        for epoch in range(1,min(len(historic),len(fresh))+1):
            a=historic.iloc[epoch-1]; b=fresh.iloc[epoch-1]
            trajectories.append(dict(seed=seed,epoch=epoch,historical_B0_train_MSE=a.train_standardized_MSE,A1_train_MSE=b.train_standardized_MSE,
              historical_B0_val_RMSE_mV=a.validation_RMSE_mV,A1_val_RMSE_mV=b.validation_RMSE_mV,
              train_MSE_difference=b.train_standardized_MSE-a.train_standardized_MSE,val_RMSE_difference_mV=b.validation_RMSE_mV-a.validation_RMSE_mV))
    csv('outputs/HISTORICAL_VS_FRESH_ABLATION_TRAJECTORY.csv',trajectories)
    states=[]; censor=[]
    for split,key in [('train','tx'),('val','vx')]:
        extra=derive(torch.from_numpy(z[key]),sc['mean'][13],sc['scale'][13]).numpy()
        absent=~(z[split+'_x_raw'][:,:,13]>.5).any(1)
        part=pd.DataFrame(extra,columns=FEATURES).assign(partition=split,origin_row=z[split+'_origins'],censored_no_stop_in48=absent)
        states.append(part)
        censor.append(dict(partition=split,windows=len(part),no_observed_majority_stop=int(absent.sum()),max_observed_distance_bins=int(round(float(np.max(extra[~absent,2]))*48)) if (~absent).any() else None))
    csv('outputs/HISTORY_STATE_FEATURES.csv',pd.concat(states)); csv('outputs/CENSOR_AUDIT.csv',censor)
    all_frames=[]; metrics=[]; horizons=[]; sm=[]
    for seed in SEEDS:
        for v in ['B0','A1','B1']:
            f=rd(f'outputs/predictions/{v}_seed{seed}.csv')
            req(len(f)==4920,'QUERY_COUNT')
            req(np.array_equal(f.origin_row,np.repeat(z['val_origins'],24)) and np.array_equal(f.target_row,z['val_targets'].ravel()) and np.array_equal(f.horizon_step,np.tile(np.arange(1,25),205)),'POINT_IDENTITY')
            req(np.array_equal(f.y_true_delta_V,z['val_y'].ravel().astype(float)),'TRUTH_IDENTITY')
            f=f.assign(seed=seed,variant=v,stratum=np.repeat(labels,24))
            f['origin_age_h']=np.repeat(z['val_origin_age_h'],24)
            f['target_age_h']=z['val_target_age_h'].ravel()
            f['origin_raw_voltage_V']=np.repeat(z['val_origin_raw_voltage_V'],24)
            f['target_raw_voltage_V']=z['val_target_raw_voltage_V'].ravel()
            f['prediction_absolute_voltage_V']=f.y_pred_delta_V+f.origin_raw_voltage_V
            f['error_mV']=(f.y_pred_delta_V-f.y_true_delta_V)*1000
            all_frames.append(f)
            metrics.append(dict(seed=seed,variant=v,**metric(f)))
            for h,part in f.groupby('horizon_step'): horizons.append(dict(seed=seed,variant=v,horizon_step=int(h),RMSE_mV=float(np.sqrt(np.mean(part.error_mV**2)))))
            for group,part in f.groupby('stratum'): sm.append(dict(seed=seed,variant=v,stratum=group,**metric(part)))
    f=pd.concat(all_frames,ignore_index=True)
    csv('outputs/POINTWISE_ERRORS.csv',f)
    target_errors=[]; tail=[]
    for (seed,v),part in f.groupby(['seed','variant']):
        total=float(np.sum(part.error_mV**2))
        for target,group in part.groupby('target_row'):
            target_errors.append(dict(seed=int(seed),variant=v,target_row=int(target),target_age_h=float(group.target_age_h.iloc[0]),
                queries=len(group),SSE_mV2=float(np.sum(group.error_mV**2)),SSE_share_pct=float(np.sum(group.error_mV**2)/total*100)))
        by=pd.DataFrame([a for a in target_errors if a['seed']==seed and a['variant']==v]).sort_values('SSE_mV2',ascending=False)
        for rank,row in enumerate(by.head(3).to_dict('records'),1): tail.append(dict(row,rank=rank))
    csv('outputs/PER_TARGET_SQUARED_ERRORS.csv',target_errors); csv('outputs/TARGET_ERROR_CONCENTRATION.csv',tail)
    csv('outputs/TOP_ABSOLUTE_ERRORS.csv',f.sort_values('error_mV',key=abs,ascending=False)[['seed','variant','origin_row','target_row','horizon_step','origin_age_h','target_age_h','target_raw_voltage_V','prediction_absolute_voltage_V','error_mV']].head(30))
    m=pd.DataFrame(metrics); h=pd.DataFrame(horizons); s=pd.DataFrame(sm)
    csv('outputs/METRICS.csv',m); csv('outputs/HORIZON_METRICS.csv',h); csv('outputs/STRATUM_METRICS.csv',s)
    summary=m.groupby('variant').agg({k:['mean','std'] for k in ['RMSE_mV','MAE_mV','endpoint_RMSE_mV','P95_mV','P99_mV','max_abs_error_mV','target_time_equal_RMSE_mV']})
    summary.columns=['_'.join(c) for c in summary.columns]; csv('outputs/SEED_SUMMARY.csv',summary.reset_index())
    paired=[]; target=[]; per_origin=[]; concentration=[]
    for ref in ['B0','A1']:
        for seed in SEEDS:
            a=f.loc[(f.seed==seed)&(f.variant==ref)].reset_index(drop=True)
            b=f.loc[(f.seed==seed)&(f.variant=='B1')].reset_index(drop=True)
            d=a[['origin_row','target_row','horizon_step','stratum']].copy(); d['seed']=seed; d['reference']=ref
            d['reference_error_mV']=a.error_mV; d['B1_error_mV']=b.error_mV
            d['SSE_gain_mV2']=a.error_mV**2-b.error_mV**2
            d['prediction_change_mV']=(b.y_pred_delta_V-a.y_pred_delta_V)*1000
            paired.append(d)
            for t,part in d.groupby('target_row'): target.append(dict(seed=seed,reference=ref,target_row=int(t),queries=len(part),SSE_gain_mV2=float(part.SSE_gain_mV2.sum())))
            for o,part in d.groupby('origin_row'): per_origin.append(dict(seed=seed,reference=ref,origin_row=int(o),SSE_gain_mV2=float(part.SSE_gain_mV2.sum()),B1_RMSE_mV=float(np.sqrt(np.mean(part.B1_error_mV**2))),reference_RMSE_mV=float(np.sqrt(np.mean(part.reference_error_mV**2)))))
        group=pd.DataFrame(target); group=group.loc[group.reference==ref].groupby('target_row').SSE_gain_mV2.sum()
        n=max(1,math.ceil(.05*len(group))); top=group.sort_values(ascending=False).head(n)
        concentration.append(dict(reference=ref,total_target_times=len(group),removed_top_positive_gain_times=int((top>0).sum()),
            pooled_SSE_gain_mV2=float(group.sum()),gain_after_top5pct_removal_mV2=float(group.sum()-top[top>0].sum()),
            positive_target_fraction=float((group>0).mean()),top5pct_positive_gain_share=float(top[top>0].sum()/group[group>0].sum()) if (group>0).any() else 0))
    csv('outputs/PAIRED_POINTWISE_DELTAS.csv',pd.concat(paired)); csv('outputs/PER_TARGET_GAINS.csv',target); csv('outputs/PER_ORIGIN_GAINS.csv',per_origin); csv('outputs/CONCENTRATION_CHECK.csv',concentration)
    checks=[]; seed_comparisons=[]
    for ref in ['B0','A1']:
        refm=m.loc[m.variant==ref].set_index('seed'); new=m.loc[m.variant=='B1'].set_index('seed')
        for seed in SEEDS:
            seed_comparisons.append(dict(seed=seed,reference=ref,reference_RMSE_mV=float(refm.loc[seed,'RMSE_mV']),B1_RMSE_mV=float(new.loc[seed,'RMSE_mV']),
                RMSE_improvement_pct=float((1-new.loc[seed,'RMSE_mV']/refm.loc[seed,'RMSE_mV'])*100),
                MAE_improvement_pct=float((1-new.loc[seed,'MAE_mV']/refm.loc[seed,'MAE_mV'])*100)))
        gain=1-new.RMSE_mV.mean()/refm.RMSE_mV.mean()
        checks.append(dict(check=ref+'_mean_RMSE_gain_3pct',value=float(gain),passed=bool(gain>=.03)))
        checks.append(dict(check=ref+'_seed_wins_at_least2',value=int((new.RMSE_mV<refm.RMSE_mV).sum()),passed=bool((new.RMSE_mV<refm.RMSE_mV).sum()>=2)))
        ratio=new.RMSE_mV/refm.RMSE_mV-1
        checks.append(dict(check=ref+'_worst_seed_worsening_max5pct',value=float(ratio.max()),passed=bool(ratio.max()<=.05)))
        for field in ['P95_mV','endpoint_RMSE_mV']:
            ratio=float(new[field].mean()/refm[field].mean()-1)
            checks.append(dict(check=ref+'_'+field+'_worsening_max5pct',value=ratio,passed=bool(ratio<=.05)))
        a=s.loc[s.variant==ref].groupby('stratum').RMSE_mV.mean(); b=s.loc[s.variant=='B1'].groupby('stratum').RMSE_mV.mean()
        for group in a.index:
            ratio=float(b[group]/a[group]-1); checks.append(dict(check=ref+'_stratum_'+group+'_worsening_max5pct',value=ratio,passed=bool(ratio<=.05)))
        c=next(c for c in concentration if c['reference']==ref)
        checks.append(dict(check=ref+'_gain_not_only_top5pct_targets',value=c['gain_after_top5pct_removal_mV2'],passed=c['gain_after_top5pct_removal_mV2']>0))
    csv('outputs/FROZEN_SUCCESS_CHECKS.csv',checks)
    csv('outputs/PAIRWISE_SEED_COMPARISONS.csv',seed_comparisons)
    means=m.groupby('variant').RMSE_mV.mean()
    status='DEVELOPMENT_GAIN_CONFIRMED' if all(c['passed'] for c in checks) else 'PROMISING_BUT_UNSTABLE' if means.B1<min(means.B0,means.A1) else 'NO_CONFIRMED_GAIN'
    js('outputs/DECISION.json',dict(status=status,GO_NO_GO='GO' if status=='DEVELOPMENT_GAIN_CONFIRMED' else 'NO-GO',
      mean_RMSE_mV=means.to_dict(),gain_vs_B0_pct=float((1-means.B1/means.B0)*100),gain_vs_A1_pct=float((1-means.B1/means.A1)*100),
      all_frozen_checks_passed=all(c['passed'] for c in checks),check_count=len(checks),new_late_life_evaluation=False,
      independent_generalization_claim=False,new_experiments_allowed_after_completion=False))
    gates(z,p)

def metadata():
    """Explicit bounded prefix, never reads Assessment/LATE_LIFE rows."""
    cols=['bin_end_ophrs2_h','first_datetime','last_datetime','datetime_span_h','zero_current_fraction','bin_valid_for_windows']
    f=pd.read_csv(old.SOURCE,nrows=2404,usecols=cols)
    f.insert(0,'row_id',np.arange(len(f)))
    req(len(f)==2404,'BOUNDED_METADATA')
    # This inspection never supplies features, thresholds or labels to the models.
    train=f.iloc[:2103]; valid=f.iloc[2103:2404]
    rows=[]
    for name,part in [('TRAIN',train),('VALIDATION',valid)]:
        d=part.datetime_span_h
        eligible=part.loc[part.bin_valid_for_windows==1]
        rows.append(dict(partition=name,records=len(part),span_h_min=float(d.min()),span_h_median=float(d.median()),span_h_max_all_rows=float(d.max()),valid_bin_span_h_max=float(eligible.datetime_span_h.max()),span_over_0_75h=int((d>.75).sum()),majority_zero_bins=int((part.zero_current_fraction>.5).sum())))
    csv('outputs/BOUNDED_CLOCK_QUALIFICATION.csv',rows)
    js('outputs/BOUNDED_READ_AUDIT.json',dict(source=str(old.SOURCE.relative_to(R)),read_csv_nrows=2404,usecols=cols,features_use_these_extra_fields=False,full_file_not_loaded=True,first_train_row=0,last_validation_row=2403))

def plots():
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'savefig.dpi':300})
    colors={'B0':'#6B7280','A1':'#D97706','B1':'#2563EB'}
    labels={'B0':'Frozen F2 (B0)','A1':'Feature ablation (A1)','B1':'F2-RegimePlus (B1)'}
    def save(fig,name):
        fig.savefig(path('figures/'+name+'.png'),dpi=300,bbox_inches='tight'); fig.savefig(path('figures/'+name+'.pdf'),bbox_inches='tight'); plt.close(fig)
    m=rd('outputs/METRICS.csv'); h=rd('outputs/HORIZON_METRICS.csv'); g=rd('outputs/GATES.csv')
    fig,ax=plt.subplots(figsize=(7.4,4)); x=np.arange(3)
    for i,v in enumerate(colors):
        a=m.loc[m.variant==v].sort_values('seed'); ax.bar(x+(i-1)*.24,a.RMSE_mV,.23,label=labels[v],color=colors[v])
    ax.set(xticks=x,xticklabels=SEEDS,xlabel='Fixed initialization seed',ylabel='Validation RMSE (mV)',title='Development Validation | 205 origins, 4920 queries per instance'); ax.set_ylim(0,float(m.RMSE_mV.max())*1.22); ax.legend(ncols=3); fig.tight_layout(); save(fig,'01_overall_RMSE')
    fig,axs=plt.subplots(1,3,figsize=(12,3.6),sharey=True)
    for ax,seed in zip(axs,SEEDS):
        for v in colors:
            a=h.loc[(h.seed==seed)&(h.variant==v)]; ax.plot(a.horizon_step,a.RMSE_mV,label=labels[v],color=colors[v])
        ax.set(title=f'Seed {seed}',xlabel='Forecast step (0.5 operating h)'); ax.set_xticks([1,6,12,18,24])
    axs[0].set_ylabel('Validation RMSE (mV)'); axs[-1].legend(); fig.suptitle('Development Validation | all 4920 queries per instance'); fig.tight_layout(); save(fig,'02_horizon_errors')
    fig,axs=plt.subplots(3,1,figsize=(9,7),sharex=True)
    for ax,seed in zip(axs,SEEDS):
        for v in colors:
            a=g.loc[(g.seed==seed)&(g.variant==v)]
            breaks=np.flatnonzero(np.diff(a.origin_row.to_numpy())>1)+1
            age=np.insert(a.origin_age_h.to_numpy(),breaks,np.nan); weight=np.insert(a.g.to_numpy(),breaks,np.nan)
            ax.plot(age,weight,label=labels[v],color=colors[v],lw=1)
        ax.set(ylabel=f'Seed {seed}: g')
    handles,names=axs[0].get_legend_handles_labels(); fig.legend(handles,names,ncols=3,loc='upper center',bbox_to_anchor=(.5,.955),frameon=False,fontsize=9)
    axs[-1].set_xlabel('Observed origin operating hour'); fig.suptitle('Development Validation | 205 histories, g weights the LSTM path'); fig.tight_layout(rect=(0,0,1,.92)); save(fig,'03_gate_behavior')
    s=rd('outputs/STRATUM_METRICS.csv'); summary=s.groupby(['variant','stratum']).RMSE_mV.agg(['mean','std']).reset_index()
    csv('outputs/PLOT_STRATUM_DATA.csv',summary)
    fig,ax=plt.subplots(figsize=(8,4)); groups=['normal','frequent_switch','stop_related']
    for i,v in enumerate(colors):
        a=summary.loc[summary.variant==v].set_index('stratum').reindex(groups); ax.bar(np.arange(3)+(i-1)*.24,a['mean'],.23,yerr=a['std'],capsize=3,label=labels[v],color=colors[v])
    ax.set(xticks=np.arange(3),xticklabels=groups,ylabel='RMSE (mV), seed mean ± sample SD',title='Development Validation | history groups: 43 / 27 / 135 origins'); ax.legend(); fig.tight_layout(); save(fig,'04_history_strata')
    fig,axs=plt.subplots(2,3,figsize=(12,6),sharex=True)
    for col,seed in enumerate(SEEDS):
        for row,v in enumerate(['A1','B1']):
            a=rd(f'outputs/training_logs/{v}_seed{seed}.csv'); ax=axs[row,col]
            ax.plot(a.epoch,np.sqrt(a.train_standardized_MSE),label='Train batch-MSE root',color='#6B7280')
            ax.plot(a.epoch,a.validation_standardized_RMSE,label='Validation RMSE',color=colors[v]); ax.set(title=f'{v}, seed {seed}',ylabel='Standardized error',xlabel='Epoch',yscale='log')
    axs[0,0].legend(fontsize=8); fig.suptitle('Development Train (1380) / Validation (205 windows) | six predeclared fits'); fig.tight_layout(); save(fig,'05_training_curves')
    # Fixed middle origin, plus both favorable and adverse cases by *declared* complete-window SSE comparison.
    po=rd('outputs/PER_ORIGIN_GAINS.csv'); allp=rd('outputs/POINTWISE_ERRORS.csv'); origins=sorted(allp.origin_row.unique())
    pooled=po.loc[po.reference=='A1'].groupby('origin_row').SSE_gain_mV2.sum().sort_values()
    chosen=[('fixed middle',int(origins[len(origins)//2])),('most adverse',int(pooled.index[0])),('most favorable',int(pooled.index[-1]))]
    csv('outputs/PLOT_CASE_SELECTION.csv',[dict(rule=n,origin_row=o,selection='posthoc diagnostic except fixed middle; all seeds shown') for n,o in chosen])
    fig,axs=plt.subplots(3,3,figsize=(12,9))
    z,_=load_data(); anchor=dict(zip(z['val_origins'],z['val_origin_raw_voltage_V']))
    for row,(name,origin) in enumerate(chosen):
        for col,seed in enumerate(SEEDS):
            ax=axs[row,col]
            for v in colors:
                a=allp.loc[(allp.seed==seed)&(allp.variant==v)&(allp.origin_row==origin)]
                ax.plot(a.horizon_step,a.prediction_absolute_voltage_V,label=labels[v],color=colors[v],lw=1)
            ax.plot(a.horizon_step,a.target_raw_voltage_V,label='Observed',color='black',ls='--',lw=1)
            ax.set(title=f'{name}: origin {origin}, seed {seed}',xlabel='Forecast step',ylabel='Absolute voltage (V)')
    axs[0,0].legend(fontsize=7); fig.suptitle('Development Validation | 24 steps per origin, no stitched forecasts'); fig.tight_layout(); save(fig,'06_fixed_favorable_adverse_windows')

def table(frame,cols=None):
    if cols is not None: frame=frame[cols]
    def cell(v):
        if isinstance(v,(float,np.floating)): return f'{v:.6f}'
        return str(v).replace('|','/').replace('\n',' ')
    return '| '+' | '.join(map(str,frame.columns))+' |\n| '+' | '.join(['---']*len(frame.columns))+' |\n'+'\n'.join('| '+' | '.join(cell(v) for v in row)+' |' for row in frame.itertuples(index=False,name=None))

def reports():
    p=get_protocol(); m=rd('outputs/METRICS.csv'); summary=rd('outputs/SEED_SUMMARY.csv'); decision=readj('outputs/DECISION.json'); check=rd('outputs/FROZEN_SUCCESS_CHECKS.csv')
    means=m.groupby('variant').mean(numeric_only=True)
    mae_gain=float((1-means.loc['B1','MAE_mV']/means.loc['B0','MAE_mV'])*100)
    end_worse=float((means.loc['B1','endpoint_RMSE_mV']/means.loc['B0','endpoint_RMSE_mV']-1)*100)
    finding=(f"三seed平均MAE从{means.loc['B0','MAE_mV']:.6f}降至{means.loc['B1','MAE_mV']:.6f}mV（改善{mae_gain:.3f}%），"
      f"但12运行小时终点RMSE从{means.loc['B0','endpoint_RMSE_mV']:.6f}升至{means.loc['B1','endpoint_RMSE_mV']:.6f}mV（恶化{end_worse:.3f}%）。"
      "相对原F2，三个seed整体RMSE均恶化；对A1则42、62改善，52恶化。较小的seed离散度不能抵消三个实例共同的尾部失败。\n\n")
    intro=f"基准`{BASE}`；协议SHA256 `{sha(path('configs/FROZEN_PROTOCOL.json'))}`。本阶段是重复使用Development Validation的探索性研究，未评价新模型的Assessment或后段，不是独立Test。\n\n"
    text('01_现有F2信息与门控缺陷检查.md','# 现有F2信息与门控检查\n\n'+intro+
      '14变量含电压、电流/电流密度统计、转换次数、零电流占比；固定Top10保留0、3、13、11、4、9、6、7、10、8槽，1、2、5、12置零。门控取11、13、9、6、8五项的48步均值，5→8→1，g加权LSTM、1−g加权CNN。\n\n'
      '零电流及负载扰动已经进入旧模型。旧门控均值对历史顺序置换不敏感；TFT/LSTM/CNN仍看到序列，因此不能声称整个F2没有时序信息。卷积分支和融合位置保持不变。新增信息只是同一历史的不同摘要，不是新传感器信息，也不是已证明的物理因果变量。\n\n'
      '旧门控I标准差与电流密度标准差在Train的Spearman相关约0.999997，存在明显冗余，因保护协议本次不删除旧项。新增三项与旧均值的最大绝对Spearman相关分别约0.196、0.303、0.724。最近停机位置仍有相关性，不能称为完全独立信息。详见TRAIN_FEATURE_CORRELATIONS.csv。\n\n'
      '历史诊断表明相似历史存在未来低谷第12/13步两种结果，是假设背景，不用于本次特征挑选或性能评价；本轮没有再次读取那两次后段事件的模型输入/目标。不能据此认定未来状态绝对不可预测。\n')
    clocks=rd('outputs/BOUNDED_CLOCK_QUALIFICATION.csv')
    text('02_时间和启停特征可用性分析.md','# 时间与启停特征\n\n'+intro+
      '运行时钟0.5h分箱与日历时间不同。额外DateTime、桶日历跨度可在测量结束后已知，但它们不是原48×14模型输入；加入即改变历史信息集合。本轮仅有界读取前2404条这些字段用于资格说明，不进入模型、分组阈值或选模。全行统计包含无效桶，最大跨度不可误当成合法训练窗口；valid_bin列单列有效桶。\n\n'+table(clocks)+'\n\n'
      '原零电流占比只说明样本比例，不能恢复准确日历停机持续时间。窗口内最近多数零电流桶位置是运行桶距离，每格0.5运行小时，不是停机后经过的墙钟时间。Train的1380个窗口均有该桶，最远19格；“48格未见”标志恒0，训练前删除。若以后无观测则距离编码1并标记删失，不能当精确时间。\n\n'
      '原24步目标仍是12运行小时，不保证等于12日历小时；本候选不预测实际未来日历目标时刻。状态变量在起点桶完整结束后才可用，不能提前到桶开始时使用。\n\n'
      '分类：同信息=三个历史摘要；扩展历史=DateTime/OpHrs2差、48格之前状态；禁止输入=未来电流、未来零电流、实际未来启停时刻。没有可认证的提前启停计划，本轮不假定其存在，不添加复杂状态辅助网络。\n')
    text('03_F2_RegimePlus结构与设计依据.md','# 唯一候选及训练前冻结\n\n'+intro+
      '三项追加门控特征：①最近6格标准化电流密度均值减48格均值；②最近12格零电流占比均值减此前12格均值；③48格内距最近多数零电流桶的距离/48。它们分别表示近期负载偏离、近期停机比例变化和已观察的历史位置。只用Top10已保留槽8、13，不偷偷恢复被掩码的输入。\n\n'
      '追加特征以1380个Train窗口拟合均值/标准差；旧标准化、EWMA和目标尺度原样复用。门控8→8→1，主体332436→332460（+24，约0.0072%）。Attention、VSN、LSTM/CNN、输出头、融合位置、decoder确定性时间坐标不改。\n\n'
      'B1与A1每seed参数及初始化完全相同，共享旧F2初始化权重；新增隐藏列初始化消耗的随机数不会移动训练dropout流。A1仅将新增标准化槽归零，保留原五项和相同物理参数量；这是特征消融，归零列不产生有效数据自由度，不宣称额外参数均得到相同有效激活。B0为原冻结训练结果，严格训练配对因果比较主要依赖A1。\n\n'
      '六次从头训练：A1/B1×42/52/62；AdamW、MSE、batch32、450epochs/patience40、clip1及历史Fold2配置固定；不重新Optuna。Train1380、Validation205×24。旧优化针对基础TFT，F2继承，不能称F2专属搜索。\n\n'
      '事前判据：相对B0、A1三seed平均RMSE均改善≥3%；各至少2个seed胜出；任一种子、历史工况组、平均P95或12h终点恶化不超过5%；移除最高5%正收益目标的诊断后仍有合并SSE收益。满足全部才确认；均值对两者改善但条件未全满足为不稳定；否则无确认增益。每项均按冻结文件判定，不追改阈值。\n')
    text('04_训练与消融实验结果.md','# 实际训练与消融结果\n\n'+intro+
      table(m,['variant','seed','RMSE_mV','MAE_mV','endpoint_RMSE_mV','P95_mV','P99_mV','max_abs_error_mV'])+'\n\n'+table(summary,['variant','RMSE_mV_mean','RMSE_mV_std','MAE_mV_mean','endpoint_RMSE_mV_mean','endpoint_RMSE_mV_std'])+'\n\n'
      '每模型每seed完整保留4920条目标，不删极端点。R²针对ΔV；mV只逆标准化一次，绝对电压图使用原始未EWMA起点。SEED_SUMMARY的标准差为ddof=1。\n\n'+table(check)+'\n\n'
      +finding+
      'A1为本轮重新训练的特征消融，不是历史B0逐轮训练复制。Seed42首轮指标与旧训练相同、第2轮开始出现数值差异；输入维度改变的运算图即使数学等价，也不能保证原训练轨迹逐位复现。本轮直接把已训练A1的有效五列权重映回原门控，检查同权重前向差异，详见A1_NATIVE_FUNCTION_EQUIVALENCE；不因此重跑或替换任何结果。\n\n'
      '工况组仅由预测起点历史定义，不能称实际未来状态分组；其阈值来自Train，重叠窗口不是独立试验。完整STRATUM_METRICS、HORIZON_METRICS、PAIRED_POINTWISE_DELTAS及PER_TARGET_GAINS可复算。\n\n'
      +table(rd('outputs/PAIRWISE_SEED_COMPARISONS.csv'))+'\n\n'+table(rd('outputs/CONCENTRATION_CHECK.csv'))+'\n\n'
      '目标集中度敏感性移除的是251个唯一目标中最高正收益的前ceil(5%×251)=13个目标（若正收益不足13个则只移除正收益项），仅作代数诊断，未改正式样本和排名。\n\n'
      '另按全部目标分解真实SSE：B1三个seed的主要错误目标均为Validation行2247（1382.5运行小时），单一目标约贡献44%—50%的SSE，见下表。其23/24步查询发生明显状态错判；图06的不利起点2224按冻结SSE规则自然选出，未用于后续改模型。本轮不把这一事后定位当训练前假设，不删除这些目标，不计算替代正式排名。\n\n'+table(rd('outputs/TARGET_ERROR_CONCENTRATION.csv').query("variant == 'B1'"))+'\n\n'
      'GATE_FEATURE_INTERVENTIONS和DIAGNOSTIC_B1_new_features_zero是同权重的事后输入归零诊断；用于检查使用依赖，不替代独立训练A1消融，不用其选择模型。优势/不利窗口按事前固定的完整窗口SSE差规则展示，另有固定中间起点；没有拼接重叠窗口。\n\n'+table(rd('outputs/GATE_SUMMARY.csv'))+'\n\n'+table(rd('outputs/POSTHOC_ZERO_FEATURE_METRICS.csv'),['seed','RMSE_mV','endpoint_RMSE_mV','prediction_change_RMS_mV'])+'\n')
    text('05_SCI论文创新性与后续建议.md','# SCI证据与停止决定\n\n'+intro+
      f"最终状态：**{decision['status']}**；**{decision['GO_NO_GO']}**。B1三seed平均RMSE {decision['mean_RMSE_mV']['B1']:.6f}mV，相对B0改善{decision['gain_vs_B0_pct']:.3f}%，相对A1改善{decision['gain_vs_A1_pct']:.3f}%。正值表示改善。\n\n"
      +finding+
      '这是对旧门控均值摘要丢失近期顺序的最小工程假设检验。模块命名、双时钟现象或门控数值变化本身都不是预测增益或学术原创性的证明。只有三seed、同容量消融、尾部及工况守护均通过，才可把本阶段限定为Development探索性方法证据；仍不证明独立泛化，也未进行文献原创性优先权审查。\n\n'
      '本次只训练三项状态特征的组合，没有逐特征重新训练，不能宣布某一个特征单独提高精度；事后归零可能偏离自然联合分布，只证明已训练实例的计算依赖。新增状态特征并没有新增未来工况信息，也没有直接恢复准确的日历停机时长。\n\n'
      '本轮未确认增益，保留原F2正式结论，停止本候选，不用Validation循环换特征。没有收益不等于已证明未来工况不可观测是唯一原因，可能涉及摘要冗余、训练随机性、门控表达方式等。下一步最小有价值工作是审查预测起点是否有真实可用的提前启停计划及测量时钟语义；这是数据可用性问题，不自动训练。\n\n'
      '新特征确实影响门控，但不能把计算依赖升级成独立精度贡献。本阶段可作为负面消融及方法边界材料，不能作为已证实的精度创新点，不替换论文主模型。完成后停止，不进入SOH/RUL/PINN、新backbone或额外seed搜索。\n')
    text('README.md','# F2-RegimePlus Development探索性实验\n\n'+intro+
      f"**{decision['status']} | {decision['GO_NO_GO']}**\n\n"+
      finding+
      table(m,['variant','seed','RMSE_mV','MAE_mV','endpoint_RMSE_mV'])+'\n\n'
      '实际执行一个三特征门控候选与等容量归零特征消融，固定42/52/62共六次新GPU训练；B0历史预测逐点复现。历史模型及论文结果不替换。完整证据见01—05报告和outputs，冻结判据见configs。\n\n'
      '复现（本目录，既有Python/PyTorch CUDA环境）：\n```powershell\npython -B scripts/experiment.py prepare # 首次专用，已冻结时拒绝重复\npython -B scripts/experiment.py run # 已成功训练跳过；失败现场拒绝静默重跑\npython -B scripts/audit_inputs.py\npython -B scripts/analyze.py\npython -B scripts/verify.py\npython -B scripts/protect_history.py\npython -B scripts/seal.py\n```\n'
      '模型定义只读导入冻结RealHyFC_F2_Main.py，SHA写入协议；新门控及完整训练逻辑位于scripts/experiment.py。只从有界Train/Validation缓存取输入、标签。没有调用旧完整数据读取或旧评价入口。\n')

if __name__=='__main__':
    calculate(); metadata(); plots(); reports(); print('ANALYSIS_COMPLETE',readj('outputs/DECISION.json')['status'])
