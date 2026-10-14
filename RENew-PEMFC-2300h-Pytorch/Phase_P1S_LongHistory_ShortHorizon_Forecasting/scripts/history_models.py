"""Five past-only features; matched samples; Ridge inner selection with purging."""
import math
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

FEATURES=['origin_R','history_mean_R','history_std_R','history_slope_R_per_h','origin_minus_median_R']
ALPHAS=[.0001,.01,1.,100.]

def historical_features(p1s,origin_time,origin_R,length):
    p=p1s.loc[(p1s.measurement_end_h<=origin_time+1e-9)&(p1s.measurement_end_h>=origin_time-length-1e-9)].sort_values('measurement_end_h')
    count=len(p);span=float(np.ptp(p.measurement_end_h.to_numpy())) if count else 0.
    required=5 if length==96 else 10
    enough=count>=required and span>=.7*length-1e-9
    reason='OK' if enough else ('COUNT_BELOW_MINIMUM' if count<required else 'SPAN_BELOW_70_PERCENT')
    audit={'length_h':length,'observations':count,'span_h':span,'coverage_fraction':span/length,
        'minimum_observations':required,'eligible':bool(enough),'reason':reason,'cycle_ids':p.cycle_interval_id.tolist(),
        'earliest_measurement_end_h':float(p.iloc[0].measurement_end_h) if count else None,
        'latest_measurement_end_h':float(p.iloc[-1].measurement_end_h) if count else None}
    if not count:return None,audit
    if abs(float(p.iloc[-1].R)-origin_R)>1e-12 or abs(float(p.iloc[-1].measurement_end_h)-origin_time)>1e-9:
        raise ValueError('ORIGIN_LATEST_HISTORY_IDENTITY_CHANGED')
    x=p.measurement_end_h.to_numpy(float);y=p.R.to_numpy(float);xc=x-x.mean()
    slope=0. if count<2 else float(xc@(y-y.mean())/(xc@xc))
    values=np.array([origin_R,y.mean(),y.std(ddof=0),slope,origin_R-np.median(y)],float)
    return values,audit

def build_features(p1s,authority,horizon):
    rows=[];details=[]
    for sample in authority.itertuples():
        row=sample._asdict();row['horizon']=horizon
        for L in [96,192]:
            values,audit=historical_features(p1s,float(sample.origin_p1s_measurement_end_ophrs2_h),float(sample.origin_R_P1S_causal),L)
            for name,value in zip(FEATURES,values if values is not None else [np.nan]*5):row[f'L{L}_{name}']=value
            row[f'L{L}_eligible']=audit['eligible'];row[f'L{L}_observations']=audit['observations'];row[f'L{L}_span_h']=audit['span_h'];row[f'L{L}_reason']=audit['reason']
            details.append({'horizon':horizon,'fold':sample.fold,'role':sample.role,'sample_index':int(sample.sample_index),
                'origin_cycle_interval_id':sample.origin_cycle_interval_id,**audit})
        row['common_history_eligible']=row['L96_eligible'] and row['L192_eligible'];rows.append(row)
    return pd.DataFrame(rows),details

def train_validation(frame,fold):
    original=frame.loc[frame.fold==fold]
    train=original.loc[(original.role=='train')&original.common_history_eligible].sort_values('target_p1s_measurement_end_ophrs2_h').copy()
    validation=original.loc[(original.role=='validation')&original.common_history_eligible].sort_values('target_p1s_measurement_end_ophrs2_h').copy()
    # The final model's training labels must already exist at its first forecast origin.
    candidates=validation if len(validation) else original.loc[original.role=='validation']
    cutoff=float(candidates.origin_p1s_measurement_end_ophrs2_h.min())
    purge=train.target_p1s_measurement_end_ophrs2_h>cutoff+1e-9
    rejected=train.loc[purge].copy();train=train.loc[~purge].copy()
    return train,validation,rejected,cutoff

def inner_split(train):
    n=math.ceil(.15*len(train));selection=train.iloc[-n:].copy();fit=train.iloc[:-n].copy()
    cutoff=float(selection.origin_p1s_measurement_end_ophrs2_h.min())
    purge=fit.target_p1s_measurement_end_ophrs2_h>cutoff+1e-9
    rejected=fit.loc[purge].copy();fit=fit.loc[~purge].copy()
    if len(fit)<2 or len(selection)<1:raise ValueError('INSUFFICIENT_INNER_TRAIN')
    return fit,selection,rejected,cutoff

def ridge_fit(train,validation,length):
    names=[f'L{length}_{f}' for f in FEATURES];fit,selection,purged,cutoff=inner_split(train)
    inner_scaler=StandardScaler().fit(fit[names].to_numpy(float))
    x=inner_scaler.transform(fit[names].to_numpy(float));xval=inner_scaler.transform(selection[names].to_numpy(float))
    y=(fit.target_R_P1S_causal-fit.origin_R_P1S_causal).to_numpy(float)
    yval=(selection.target_R_P1S_causal-selection.origin_R_P1S_causal).to_numpy(float)
    scores=[]
    for alpha in ALPHAS:
        estimator=Ridge(alpha=alpha,fit_intercept=True,solver='cholesky').fit(x,y)
        scores.append({'alpha':alpha,'inner_MAE_R':float(np.mean(np.abs(estimator.predict(xval)-yval)))})
    alpha=min(scores,key=lambda r:(r['inner_MAE_R'],r['alpha']))['alpha']
    scaler=StandardScaler().fit(train[names].to_numpy(float))
    target=(train.target_R_P1S_causal-train.origin_R_P1S_causal).to_numpy(float)
    model=Ridge(alpha=alpha,fit_intercept=True,solver='cholesky').fit(scaler.transform(train[names].to_numpy(float)),target)
    pred=model.predict(scaler.transform(validation[names].to_numpy(float))) if len(validation) else np.array([])
    record={'length_h':length,'alpha':alpha,'features':names,'inner_selection_n':len(selection),'inner_fit_n':len(fit),
        'inner_fit_samples':fit.sample_index.astype(int).tolist(),'inner_selection_samples':selection.sample_index.astype(int).tolist(),
        'inner_purged_samples':purged.sample_index.astype(int).tolist(),'inner_label_availability_cutoff_h':cutoff,
        'inner_scaler_mean':inner_scaler.mean_.tolist(),'inner_scaler_scale':inner_scaler.scale_.tolist(),
        'alpha_grid_scores':scores,'train_samples':train.sample_index.astype(int).tolist(),'validation_samples':validation.sample_index.astype(int).tolist(),
        'train_scaler_mean':scaler.mean_.tolist(),'train_scaler_scale':scaler.scale_.tolist(),
        'coef_standardized_X':model.coef_.tolist(),'intercept_deltaR':float(model.intercept_),
        'target_standardized':False,'solver':'cholesky','ridge_fits':5}
    return pred,record

def error_metrics(truth,pred,VREF):
    e=np.asarray(pred)-np.asarray(truth);ae=np.abs(e)
    if not len(e):return {'n':0,**{name:None for name in ['MAE_R','RMSE_R','MAE_mV_equivalent','RMSE_mV_equivalent','P95_abs_error_R','P95_abs_error_mV_equivalent','max_abs_error_R','max_abs_error_mV_equivalent']}}
    return {'n':len(e),'MAE_R':float(ae.mean()),'RMSE_R':float(np.sqrt(np.mean(e**2))),
        'MAE_mV_equivalent':float(ae.mean()*VREF*1000),'RMSE_mV_equivalent':float(np.sqrt(np.mean(e**2))*VREF*1000),
        'P95_abs_error_R':float(np.quantile(ae,.95)),'P95_abs_error_mV_equivalent':float(np.quantile(ae,.95)*VREF*1000),
        'max_abs_error_R':float(ae.max()),'max_abs_error_mV_equivalent':float(ae.max()*VREF*1000)}

def causal_soh_bridge(predicted_R,anchor_records,origin_datetime,min_anchors=3):
    # No automatic all-six B2; actual release evidence must be supplied upstream.
    if any(r.get('result_release_datetime') is None for r in anchor_records):
        return {'status':'NO_CERTIFIED_ANCHOR_RELEASE_TIME','predicted_SOH_proxy':None}
    eligible=[r for r in anchor_records if pd.Timestamp(r['result_release_datetime'])<=pd.Timestamp(origin_datetime)]
    if len(eligible)<min_anchors:return {'status':'INSUFFICIENT_LEGAL_CALIBRATION','predicted_SOH_proxy':None}
    x=np.array([r['R_anchor'] for r in eligible]);y=np.array([r['SOH_anchor'] for r in eligible]);xc=x-x.mean()
    a=0. if float(xc@xc)==0 else max(0.,float(xc@(y-y.mean())/(xc@xc)));b=float(y.mean()-a*x.mean())
    return {'status':'CONDITIONAL_PROXY_ONLY','predicted_SOH_proxy':float(a*predicted_R+b),'a':a,'b':b,'true_SOH_MAE':None}
