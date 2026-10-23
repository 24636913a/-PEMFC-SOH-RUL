"""Fixed CPU methods. No torch, future P1S, smoothing or parameter selection."""
import numpy as np
import pandas as pd

TASKS = [('SOH-F1',850,1150),('SOH-F2',1150,1500),('SOH-F3',1500,1750)]
METHODS = ['M0','M1','M2','M3']

def ols(x, y):
    x = np.asarray(x,dtype=float); y = np.asarray(y,dtype=float)
    if len(x)<2 or len(x)!=len(y) or not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError('INVALID_OLS_DATA')
    xc=x-x.mean(); d=float(xc@xc)
    if d<=0: raise ValueError('DEGENERATE_OLS_TIME')
    b=float(xc@(y-y.mean())/d)
    return b,float(y.mean()-b*x.mean())

def affine_parameters(x,y):
    # Identical centered, unweighted nonnegative-slope mathematics to frozen AHC.
    x=np.asarray(x,dtype=float); y=np.asarray(y,dtype=float)
    if len(x)<2 or len(x)!=len(y) or not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError('INSUFFICIENT_CALIBRATION')
    xc=x-x.mean(); d=float(xc@xc)
    a=0.0 if d==0 else max(0.0,float(xc@(y-y.mean())/d))
    return a,float(y.mean()-a*x.mean())

def origin_inputs(origin, references, mapping, p1s, timing, window_h=96.):
    # Nominal checkpoint membership is distinct from operating age.
    refs=references.loc[references.nominal_h<=origin].sort_values('nominal_h').copy()
    if not (refs.nominal_h==origin).any(): raise ValueError('ORIGIN_REFERENCE_MISSING')
    stamp=timing.loc[timing.checkpoint_h==origin].iloc[0]
    # Use a conservative recorded pre-test DateTime cutoff, not filename age.
    cutoff=pd.Timestamp(stamp.PAA_gap_cutoff_datetime)
    available=p1s.loc[(p1s.available_datetime<cutoff)&p1s.causal_target_eligible].sort_values('actual_measurement_end_h').copy()
    if available.empty: raise ValueError('NO_ORIGIN_P1S')
    latest=available.iloc[-1]
    window=available.loc[available.actual_measurement_end_h>=float(latest.actual_measurement_end_h)-window_h].copy()
    known=refs.loc[refs.nominal_h>0,'nominal_h'].astype(int).tolist()
    # Preserve all three originally mapped intervals per anchor; no reselection.
    mp=mapping.loc[mapping.checkpoint.isin(known)].copy()
    if any(len(g)!=3 for _,g in mp.groupby('checkpoint')): raise ValueError('MAPPING_CHANGED')
    if not (mp.measurement_end_ophrs2_h<=float(stamp.PAA_gap_cutoff_ophrs2_h)).all():
        raise ValueError('CALIBRATION_P1S_AFTER_RECORDED_ORIGIN_CUTOFF')
    anchor=mp.groupby('checkpoint',sort=True).P1S_operating_voltage_retention.median()
    anchor_y=refs.set_index('nominal_h').loc[anchor.index,'Y']
    a,b=affine_parameters(anchor.to_numpy(),anchor_y.to_numpy())
    sb,si=ols(refs.nominal_h,refs.Y)
    return {'refs':refs,'mapping':mp,'window':window,'latest':latest,'a':a,'b':b,
        'soh_age_slope':sb,'soh_age_intercept':si,'cutoff':cutoff,'timing':stamp,
        'calibration_checkpoints':anchor.index.astype(int).tolist(),
        'calibration_R':anchor.to_numpy().tolist(),'calibration_Y':anchor_y.to_numpy().tolist()}

def forecasts(origin,target,inputs,min_cycles=4):
    latest=inputs['latest']; window=inputs['window']; a,b=inputs['a'],inputs['b']
    r=float(latest.R); t=float(latest.actual_measurement_end_h)
    predictions={'M0':float(inputs['refs'].iloc[-1].Y),
        'M1':inputs['soh_age_intercept']+inputs['soh_age_slope']*target,
        'M2':a*r+b,'M3':None}
    sr=None; ri=None; predicted_r=None
    if len(window)>=min_cycles and window.cycle_interval_id.nunique()==len(window):
        sr,ri=ols(window.actual_measurement_end_h,window.R)
        # Target uses nominal coordinate substitution explicitly registered in config.
        predicted_r=r+sr*(target-t)
        predictions['M3']=a*predicted_r+b
    return predictions,{'R_latest':r,'latest_R_actual_end_h':t,'slope_R_per_h':sr,
        'R_OLS_intercept':ri,'R_pred_target':predicted_r,
        'extrapolation_h_under_coordinate_assumption':target-t,
        'M2_current_proxy':a*r+b}

def crossing(start_health,slope,horizon,threshold=.97):
    if start_health<threshold:return {'status':'ALREADY_BELOW_AT_SCENARIO_ORIGIN','remaining_h':0.}
    if slope is None:return {'status':'NOT_COMPUTABLE','remaining_h':None}
    if slope>=0:return {'status':'NO_FINITE_DOWNCROSSING_NONDECLINING','remaining_h':None}
    root=(threshold-start_health)/slope
    return {'status':'CROSSING_BOUNDARY_WITHIN_HORIZON' if root<=horizon else 'NO_CROSSING_WITHIN_HORIZON',
        'remaining_h':float(root) if root<=horizon else None,
        'unbounded_linear_root_h_descriptive':float(root)}

def metrics_table(predictions):
    rows=[]
    m0=predictions.loc[predictions.method=='M0'].set_index('task_id')
    for method in METHODS:
        p=predictions.loc[predictions.method==method].copy()
        valid=p.loc[p.prediction.notna()]
        count=len(valid)
        mae=float(valid.absolute_error_pp.mean()) if count else None
        rmse=float(np.sqrt(np.mean(valid.signed_error_pp.to_numpy()**2))) if count else None
        wins=sum(float(r.absolute_error_pp)<float(m0.loc[r.task_id].absolute_error_pp) for r in valid.itertuples())
        base=float(m0.absolute_error_pp.mean()); improve=None if mae is None else 100*(base-mae)/base
        numerical_gate=count==3 and mae<=.95*base and wins>=2
        # Exact result-publication/operating-hour mapping is absent for every task.
        status='REFERENCE_BASELINE' if method=='M0' else ('NOT_EVALUABLE' if count<3 else ('LIMITED_SIGNAL' if mae<base else 'NO_GAIN'))
        rows.append({'method':method,'scope':'NOMINAL_CHECKPOINT_ORDER_REPLAY','n_fixed_targets':3,
            'n_computed':count,'n_strict_online_qualified':0,'MAE_pp':mae,'RMSE_pp':rmse,
            'improvement_vs_M0_percent':improve,'wins_vs_M0':wins,
            'numeric_5percent_and_2of3_gate':bool(numerical_gate),'research_decision':status,
            'strict_online_decision':'NOT_EVALUABLE','formal_CLEAR_GAIN_claim':False})
    return pd.DataFrame(rows)
