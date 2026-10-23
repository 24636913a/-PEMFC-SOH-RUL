"""Frozen 48-step raw-DeltaV task, horizon-specific causal windows and scaler."""
import numpy as np
import pandas as pd
from common import HERE,SOURCE,CONFIG,read,csv,array_sha,dump,read_csv

FEATURES=['cell_voltage_mean','cell_voltage_std','cell_voltage_range','cell_voltage_min','cell_voltage_max',
    'actual_current_A_mean','actual_current_A_std','actual_current_A_peak_to_peak','actual_current_density_A_cm2_mean',
    'actual_current_density_A_cm2_std','actual_current_density_A_cm2_peak_to_peak','load_transition_count','operation_point_transitions','zero_current_fraction']
TOP10=[0,3,13,11,4,9,6,7,10,8]
PARTS={'TRAIN':(0,2690),'VALIDATION':(2690,3074),'HOLDOUT':(3074,3843)}

def window_indices(frame,horizon,start,end):
    age=frame.bin_end_ophrs2_h.to_numpy(float);valid=frame.bin_valid_for_windows.to_numpy()==1
    segment=frame.temporal_segment_id.to_numpy();values=frame[FEATURES].to_numpy(float)
    first=pd.to_datetime(frame.first_datetime).astype('int64').to_numpy()/1e9
    last=pd.to_datetime(frame.last_datetime).astype('int64').to_numpy()/1e9
    good=[];bad=[]
    for o in range(start,end):
        lo=o-47;hi=o+horizon;reasons=[]
        if lo<0:reasons.append('HISTORY_BEFORE_SOURCE')
        if hi>=end:reasons.append('TARGET_OUTSIDE_PARTITION')
        if lo>=0 and hi<len(frame):
            if not valid[lo:hi+1].all():reasons.append('INVALID_BIN')
            if (segment[lo:hi+1]!=segment[o]).any():reasons.append('SEGMENT_BREAK')
            if not np.isclose(np.diff(age[lo:hi+1]),.5,atol=1e-6,rtol=0).all():reasons.append('OPERATING_CLOCK_BREAK')
            gap=first[lo+1:hi+1]-last[lo:hi]
            if ((gap<0)|(gap>1.01)).any():reasons.append('CALENDAR_CLOCK_BREAK')
            if not np.isfinite(values[lo:hi+1]).all():reasons.append('NONFINITE_INPUT_OR_LABEL')
        if reasons:bad.append(dict(origin_row=o,reasons='|'.join(reasons)))
        else:good.append(dict(origin_row=o,history_first_row=lo,target_first_row=o+1,target_last_row=hi,
            origin_age_h=age[o],last_target_age_h=age[hi],actual_terminal_lead_h=age[hi]-age[o],segment_id=int(segment[o])))
    return pd.DataFrame(good),pd.DataFrame(bad)

def freeze_windows(out):
    frame=pd.read_csv(SOURCE);assert len(frame)==3843
    assert frame.bin_end_ophrs2_h.is_monotonic_increasing and frame.bin_end_ophrs2_h.is_unique
    records=[]
    for h in [6,12,24]:
        for part,(s,e) in PARTS.items():
            w,x=window_indices(frame,h,s,e)
            csv(out/f'windows/H{h}_{part}.csv',w);csv(out/f'windows/H{h}_{part}_EXCLUDED.csv',x)
            targets=w.origin_row.to_numpy(int)[:,None]+np.arange(1,h+1)[None,:]
            records.append(dict(horizon_steps=h,partition=part,records=e-s,legal_origins=len(w),queries=targets.size,
                unique_targets=len(np.unique(targets)),excluded_origins=len(x),origin_sha256=array_sha(w.origin_row.to_numpy(np.int64)),
                target_sha256=array_sha(targets.astype(np.int64))))
    csv(out/'WINDOW_COUNTS.csv',records)
    # Train-only thresholds for descriptive event strata, never loss weights.
    train=frame.iloc[:2690];valid=train.bin_valid_for_windows.to_numpy()==1
    v=train.cell_voltage_mean.to_numpy(float);age=train.bin_end_ophrs2_h.to_numpy(float)
    continuous=valid[1:]&valid[:-1]&(train.temporal_segment_id.to_numpy()[1:]==train.temporal_segment_id.to_numpy()[:-1])&np.isclose(np.diff(age),.5,atol=1e-6,rtol=0)
    thresholds=dict(zero_current_P75=float(train.loc[valid,'zero_current_fraction'].quantile(.75)),
        load_transition_P75=float(train.loc[valid,'load_transition_count'].quantile(.75)),
        raw_voltage_jump_P95_V=float(np.quantile(abs(np.diff(v)[continuous]),.95)),fitted_rows='TRAIN only [0,2690)')
    dump(out/'TRAIN_EVENT_THRESHOLDS.json',thresholds)
    counts=pd.DataFrame(records)
    assert counts[(counts.horizon_steps==24)].legal_origins.tolist()==[1750,218,527]
    return counts

def raw_data(out,horizon,parts=('TRAIN','VALIDATION')):
    nrows=None if 'HOLDOUT' in parts else 3074
    frame=pd.read_csv(SOURCE,nrows=nrows)
    raw=frame[FEATURES].to_numpy(float);voltage=frame.cell_voltage_mean.to_numpy(float)
    age=frame.bin_end_ophrs2_h.to_numpy(float);result={}
    for part in parts:
        w=read_csv(out/f'windows/H{horizon}_{part}.csv');origins=w.origin_row.to_numpy(np.int64)
        targets=origins[:,None]+np.arange(1,horizon+1,dtype=np.int64)[None,:]
        x=np.stack([raw[o-47:o+1] for o in origins]).astype(np.float32)
        y=np.asarray(voltage[targets]-voltage[origins,None],np.float32)
        result[part]=dict(w=w,origins=origins,targets=targets,xraw=x,y=y,origin_voltage=voltage[origins],
            true_voltage=voltage[targets],origin_age_h=age[origins],target_age_h=age[targets])
    return frame,result

def prepare(out,horizon,include_holdout=False,frozen_scaler=None):
    frame,d=raw_data(out,horizon,tuple(PARTS) if include_holdout else ('TRAIN','VALIDATION'))
    if frozen_scaler is None:
        flat=d['TRAIN']['xraw'].astype(float).reshape(-1,14)
        mean=flat.mean(0);scale=flat.std(0);scale=np.where(scale>1e-8,scale,1.)
    else:mean=np.asarray(frozen_scaler['feature_mean']);scale=np.asarray(frozen_scaler['feature_std'])
    for part,item in d.items():
        x=item['xraw'].copy()
        for step in range(1,48):x[:,step,0]=.3*item['xraw'][:,step,0]+.7*x[:,step-1,0]
        item['x']=((x-mean)/scale).astype(np.float32)
        item['x'][:,:,[i for i in range(14) if i not in TOP10]]=0
        item['ynorm']=(item['y']/float(scale[0])).astype(np.float32)
    assert d['TRAIN']['targets'].max()<d['VALIDATION']['origins'].min()
    if include_holdout:assert d['VALIDATION']['targets'].max()<d['HOLDOUT']['origins'].min()
    identity=dict(horizon_steps=horizon,feature_mean=mean.tolist(),feature_std=scale.tolist(),target_scale_V=float(scale[0]),
        partitions={p:dict(origins=len(w['origins']),queries=w['y'].size,inputs_sha256=array_sha(w['x']),labels_sha256=array_sha(w['y']),
            origin_sha256=array_sha(w['origins']),target_sha256=array_sha(w['targets'])) for p,w in d.items()})
    return frame,d,identity

def points(frame,d,delta,horizon,model,partition):
    p=pd.DataFrame(dict(model=model,seed=202,partition=partition,horizon_steps=horizon,
        origin_row=np.repeat(d['origins'],horizon),target_row=d['targets'].ravel(),horizon_step=np.tile(np.arange(1,horizon+1),len(d['origins'])),
        origin_age_h=np.repeat(d['origin_age_h'],horizon),target_age_h=d['target_age_h'].ravel(),
        origin_raw_voltage_V=np.repeat(d['origin_voltage'],horizon),true_voltage_V=d['true_voltage'].ravel(),
        y_true_delta_V=d['y'].ravel(),y_pred_delta_V=delta.ravel()))
    p['predicted_voltage_V']=p.origin_raw_voltage_V+p.y_pred_delta_V
    for name in ['actual_current_density_A_cm2_mean','zero_current_fraction','load_transition_count','startup_flag','shutdown_flag','temporal_segment_id']:
        p['target_'+name]=frame[name].to_numpy()[p.target_row.to_numpy(int)]
    return p

def add_events(frame,p,thresholds):
    p=p.copy();idx=p.target_row.to_numpy(int);v=frame.cell_voltage_mean.to_numpy(float)
    ages=frame.bin_end_ophrs2_h.to_numpy(float);seg=frame.temporal_segment_id.to_numpy()
    continuous=np.r_[False,(seg[1:]==seg[:-1])&np.isclose(np.diff(ages),.5,atol=1e-6,rtol=0)]
    jump=np.r_[np.nan,np.abs(np.diff(v))]
    p['event_high_zero']=frame.zero_current_fraction.to_numpy()[idx]>thresholds['zero_current_P75']
    p['event_high_switching']=frame.load_transition_count.to_numpy()[idx]>thresholds['load_transition_P75']
    p['event_voltage_jump']=continuous[idx]&(jump[idx]>thresholds['raw_voltage_jump_P95_V'])
    p['event_start_stop']=(frame.startup_flag.to_numpy()[idx]==1)|(frame.shutdown_flag.to_numpy()[idx]==1)
    p['event_any']=p[['event_high_zero','event_high_switching','event_voltage_jump','event_start_stop']].any(axis=1)
    return p
