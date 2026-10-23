"""Fixed causal target qualification, using only source rows [0,3074).

No forecast models are trained here. OLS is the preregistered statistical
condition adjustment, not a physical polarization model or an SOH calibration.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from scipy.stats import spearmanr
from audit_data import HERE, ROOT, SOURCE, CONFIG, continuity, csv, dump, sha

def trailing(values, valid, edges, size=48):
    result=np.full(len(values),np.nan)
    for i in range(size-1,len(values)):
        a=i-size+1
        if valid[a:i+1].all() and edges[a+1:i+1].all() and np.isfinite(values[a:i+1]).all():
            result[i]=np.mean(values[a:i+1],dtype=np.float64)
    return result

def safe_rho(x,y):
    ok=np.isfinite(x)&np.isfinite(y)
    if ok.sum()<3 or np.std(x[ok])==0 or np.std(y[ok])==0:return None
    return float(spearmanr(x[ok],y[ok]).statistic)

def safe_number(x):
    return float(x) if np.isfinite(x) else None

def run(out):
    if (out/'TARGET_SUMMARY.json').exists(): raise RuntimeError('TARGET_AUDIT_ALREADY_COMPLETE_USE_READ_ONLY_MODE')
    cfg=json.loads(CONFIG.read_text(encoding='utf-8'))
    registration=json.loads((out/'REGISTRATION.json').read_text(encoding='utf-8'))
    assert sha(CONFIG)==registration['protocol_sha256']
    cov=cfg['target_B']['covariates']
    # Crucial boundary: pandas parser cannot read any HOLDOUT measurement values.
    fields=['bin_end_ophrs2_h','first_datetime','last_datetime','temporal_segment_id','bin_valid_for_windows',
        'cell_voltage_mean','load_transition_count','startup_flag','shutdown_flag',*cov]
    frame=pd.read_csv(SOURCE,usecols=list(dict.fromkeys(fields)),nrows=3074)
    assert len(frame)==3074
    age,valid,edges,blocks,_,_=continuity(frame,cfg)
    v=frame.cell_voltage_mean.to_numpy(float); z=frame[cov].to_numpy(float)
    train=(np.arange(len(frame))<2690)&valid&np.isfinite(z).all(1)&np.isfinite(v)
    mean=z[train].mean(0); std=z[train].std(0); std=np.where(std>1e-12,std,1.)
    zn=(z-mean)/std
    design=np.column_stack([np.ones(train.sum()),zn[train]])
    beta,resid,rank,singular=np.linalg.lstsq(design,v[train],rcond=None)
    reference=np.median(z[train],axis=0)
    raw_b=v-(z-reference)@(beta[1:]/std)
    lo=z[train].min(0); hi=z[train].max(0)
    support=np.isfinite(z).all(1)&(z>=lo).all(1)&(z<=hi).all(1)&valid
    A=trailing(v,valid,edges)
    B=trailing(raw_b,support,edges)
    load=frame.actual_current_density_A_cm2_mean.to_numpy(float)
    zero=frame.zero_current_fraction.to_numpy(float)
    smoothload=trailing(load,valid,edges); smoothzero=trailing(zero,valid,edges)
    tree=cKDTree(zn[train]); train_dist=tree.query(zn[train],k=2)[0][:,1]
    cutoff=float(np.quantile(train_dist,.95))
    nn=np.full(len(frame),np.nan)
    finite=np.isfinite(zn).all(1)
    nn[finite]=tree.query(zn[finite],k=1)[0]
    joint=support&(nn<=cutoff)
    outrows=pd.DataFrame(dict(row=np.arange(len(frame)),partition=np.where(np.arange(len(frame))<2690,'TRAIN','VALIDATION'),
        operating_h=age,temporal_segment_id=frame.temporal_segment_id,continuous_block=blocks,valid=valid,
        V_raw=v,current_density=load,zero_current_fraction=zero,load_transitions=frame.load_transition_count,
        startup_flag=frame.startup_flag,shutdown_flag=frame.shutdown_flag,A_causal_mean_V=A,
        B_adjusted_raw_V=np.where(support,raw_b,np.nan),B_causal_adjusted_mean_V=B,
        current_causal_mean=smoothload,zero_causal_mean=smoothzero,B_box_support=support,
        B_joint_neighbor_support=joint,B_neighbor_distance=nn))
    csv(out/'TARGETS_TRAIN_VALIDATION_ONLY.csv',outrows)
    csv(out/'B_OLS_PARAMETERS.csv',pd.DataFrame(dict(covariate=cov,coefficient_V_per_unit=beta[1:]/std,
        train_mean=mean,train_std=std,train_median_reference=reference,train_min=lo,train_max=hi)))
    # Age predictability from conditions warns against attributing corrected residuals to damage.
    agefit=np.linalg.lstsq(design,age[train],rcond=None)[0]
    age_r2=1-np.sum((design@agefit-age[train])**2)/np.sum((age[train]-np.mean(age[train]))**2)
    statistics=[]
    for name,values in [('RAW',v),('A',A),('B',B)]:
        for part,(start,end) in list(cfg['partition_rows'].items())[:2]:
            mask=(np.arange(len(frame))>=start)&(np.arange(len(frame))<end)&valid&np.isfinite(values)
            pairs=mask[1:]&mask[:-1]&edges[1:]
            diff=np.diff(values)[pairs]
            voltagediff=np.diff(v)[pairs]
            slopes=[]
            for block in np.unique(blocks[mask]):
                m=mask&(blocks==block)
                if m.sum()>=48 and np.ptp(age[m])>0:
                    slopes.append(dict(target=name,partition=part,continuous_block=int(block),n=int(m.sum()),
                        first_h=float(age[m][0]),last_h=float(age[m][-1]),slope_mV_per_h=float(np.polyfit(age[m],values[m],1)[0]*1000),
                        end_minus_start_mV=float((values[m][-1]-values[m][0])*1000)))
            if slopes: csv(out/f'{name}_{part}_SEGMENT_SLOPES.csv',pd.DataFrame(slopes))
            rr={f'rho_{n}':safe_rho(values[mask],x[mask]) for n,x in [('current',load),('zero',zero),('trailing_current',smoothload),('trailing_zero',smoothzero)]}
            stat=dict(target=name,partition=part,valid_raw_rows=int(valid[start:end].sum()),target_rows=int(mask.sum()),
                label_fraction=float(mask.sum()/max(1,valid[start:end].sum())),
                p95_minus_p5_mV=float(np.ptp(np.quantile(values[mask],[.05,.95]))*1000) if mask.any() else None,
                median_abs_step_mV=safe_number(np.median(np.abs(diff))*1000) if len(diff) else None,
                step_rms_mV=safe_number(np.sqrt(np.mean(diff**2))*1000) if len(diff) else None,
                raw_step_rms_same_pairs_mV=safe_number(np.sqrt(np.mean(voltagediff**2))*1000) if len(diff) else None,
                fraction_positive_steps=float(np.mean(diff>0)) if len(diff) else None,
                **rr)
            statistics.append(stat)
    csv(out/'TARGET_DIAGNOSTICS.csv',pd.DataFrame(statistics))
    supportstats=[]
    for part,(s,e) in list(cfg['partition_rows'].items())[:2]:
        m=(np.arange(len(frame))>=s)&(np.arange(len(frame))<e)&valid
        supportstats.append(dict(partition=part,valid_rows=int(m.sum()),box_supported=int(support[m].sum()),
            joint_neighbor_supported=int(joint[m].sum()),joint_fraction=float(joint[m].mean()),train_loo_distance_p95=cutoff))
    csv(out/'CONDITION_SUPPORT.csv',pd.DataFrame(supportstats))
    p0=pd.read_csv(out/'P0_QUALIFICATION.csv'); targetwins=[]; gates=[]
    for name,values in [('A',A),('B',B)]:
        for r in p0.itertuples():
            for part in ['TRAIN','VALIDATION']:
                w=pd.read_csv(out/f'L{r.history_steps}_H{r.horizon_steps}_{part}_LEGAL.csv')
                n=0
                for row in w.itertuples():
                    if np.isfinite(values[row.origin_row:row.target_last_row+1]).all(): n+=1
                targetwins.append(dict(target=name,history_steps=r.history_steps,horizon_steps=r.horizon_steps,
                    horizon_h=r.horizon_h,partition=part,metadata_legal_origins=len(w),target_legal_origins=n))
        items=[x for x in statistics if x['target']==name]
        coverage=all(x['label_fraction']>=cfg['target_gates']['min_train_and_validation_continuous_label_fraction'] for x in items)
        vr=[x for x in items if x['partition']=='VALIDATION'][0]
        rhos=[abs(v) for k,v in vr.items() if k.startswith('rho_') and v is not None]
        load_ok=bool(rhos) and max(rhos)<=cfg['target_gates']['max_abs_validation_spearman_with_load_or_zero_fraction']
        eligible=False
        for r in p0[p0.qualified].itertuples():
            tw=[x for x in targetwins if x['target']==name and x['history_steps']==r.history_steps and x['horizon_steps']==r.horizon_steps]
            eligible=eligible or (len(tw)==2 and all(x['target_legal_origins']>=cfg['sufficiency']['min_train_origins' if x['partition']=='TRAIN' else 'min_validation_origins'] for x in tw))
        gates.append(dict(target=name,coverage_gate=coverage,load_correlation_gate=load_ok,
            max_abs_validation_load_rho=max(rhos) if rhos else None,any_P0_qualified_labeled_task=bool(eligible),
            forecast_training_qualified=bool(coverage and load_ok and eligible),validated_SOH=False,irreversible_degradation_proven=False))
    csv(out/'TARGET_LEGAL_WINDOW_COUNTS.csv',pd.DataFrame(targetwins))
    csv(out/'P1_QUALIFICATION.csv',pd.DataFrame(gates))
    summary=dict(rows_read=3074,holdout_outcome_rows_read=0,OLS_training_rows=int(train.sum()),OLS_rank=int(rank),
        OLS_columns=int(design.shape[1]),OLS_condition_number=float(singular[0]/singular[-1]),
        age_R2_from_conditions_train=float(age_r2),target_gates=gates,
        any_target_qualified=any(x['forecast_training_qualified'] for x in gates),neural_training_runs=0,
        warning='Smoothing attenuates transients but does not remove operating-state mix, establish irreversibility or create verified SOH. OLS residuals may remove aging correlated with conditions.')
    dump(out/'TARGET_SUMMARY.json',summary)
    dump(out/'TARGET_READ_LEDGER.json',dict(source=str(SOURCE),usecols=fields,nrows=3074,
        allowed_partitions=['TRAIN','VALIDATION'],holdout_outcomes_read=False,protocol_sha256=sha(CONFIG)))
    print(json.dumps(summary,ensure_ascii=True),flush=True)
    return summary

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True)
    run(p.parse_args().out)
