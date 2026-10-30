"""Causal condition-standardization and sparse-diagnostic Kalman filtering.

This module accepts observations, diagnostic scans and estimation settings only.
It never imports the simulator, reads latent truth, or uses generator parameters.
"""
import numpy as np
import pandas as pd


def _update(x, P, H, y, R):
    """Linear measurement update using the Joseph covariance form."""
    H=np.atleast_2d(H); y=np.atleast_1d(y); R=np.atleast_2d(R)
    innovation=y-H@x
    S=H@P@H.T+R
    K=np.linalg.solve(S,H@P).T
    x=x+K@innovation
    I=np.eye(len(x)); IKH=I-K@H
    P=IKH@P@IKH.T+K@R@K.T
    return x, (P+P.T)/2


def _validate(operation, diagnostics, c):
    opcols=["time_h","current_density_a_cm2","temperature_c","voltage_cell_v"]
    dcols=opcols+["session_id"]
    for label,df,cols in [("operation",operation,opcols),("diagnostics",diagnostics,dcols)]:
        missing=set(cols)-set(df.columns)
        if missing: raise ValueError(f"{label}: missing columns {sorted(missing)}")
        if df.empty: raise ValueError(f"{label}: empty input")
        if not np.isfinite(df[opcols].to_numpy(dtype=float)).all():
            raise ValueError(f"{label}: numeric fields must be finite")
    if not np.all(np.diff(operation.time_h.to_numpy())>0):
        raise ValueError("operation time_h must be strictly increasing; aggregate duplicates first")
    if operation.time_h.iloc[0]!=diagnostics.time_h.min():
        raise ValueError("The first complete diagnostic must coincide with the first operation time")
    for _,g in diagnostics.groupby("session_id"):
        if g.time_h.nunique()!=1: raise ValueError("Each diagnostic session needs one availability time")
        if g.current_density_a_cm2.nunique()<3: raise ValueError("At least three distinct currents per diagnostic")
    # This demo requires acquisition/completion and availability to coincide,
    # and each available scan must align with an operation row. No silent snap.
    observed_times=set(operation.time_h.to_numpy())
    arrived_times=diagnostics.loc[diagnostics.time_h<=operation.time_h.iloc[-1],"time_h"]
    if any(t not in observed_times for t in arrived_times):
        raise ValueError("Diagnostic times must align with operation samples; asynchronous or delayed updates need an explicit time-state model")
    first=diagnostics[diagnostics.time_h==diagnostics.time_h.min()]
    if first.session_id.nunique()!=1: raise ValueError("Exactly one initial diagnostic session is required")
    lo,hi=first.current_density_a_cm2.min(),first.current_density_a_cm2.max()
    if operation.current_density_a_cm2.min()<lo or operation.current_density_a_cm2.max()>hi:
        raise ValueError("Operating currents exceed initial diagnostic support; no silent extrapolation")
    if not lo<=c["reference_current_density_a_cm2"]<=hi:
        raise ValueError("Reference current must be within the initial diagnostic support")


def fit_initial_reference(diagnostics,c):
    initial=diagnostics[diagnostics.time_h==diagnostics.time_h.min()]
    j=initial.current_density_a_cm2.to_numpy()
    v=(initial.voltage_cell_v.to_numpy()
        -c["calibrated_temperature_coefficient_v_per_c"]*(initial.temperature_c.to_numpy()-c["reference_temperature_c"]))
    coeff=np.polyfit(j-c["reference_current_density_a_cm2"],v,c["baseline_polynomial_degree"])
    return coeff


def scan_anchor(scan,coeff,c):
    """Return diagnostic estimates [loss at j_ref, slope increment]."""
    centered=scan.current_density_a_cm2.to_numpy()-c["reference_current_density_a_cm2"]
    voltage=(scan.voltage_cell_v.to_numpy()
         -c["calibrated_temperature_coefficient_v_per_c"]*(scan.temperature_c.to_numpy()-c["reference_temperature_c"]))
    y=np.polyval(coeff,centered)-voltage
    X=np.column_stack([np.ones(len(scan)),centered])
    theta=np.linalg.lstsq(X,y,rcond=None)[0]
    residual=y-X@theta
    sigma2=float(residual@residual/max(len(y)-2,1))
    cov=sigma2*np.linalg.inv(X.T@X)
    # Floors account approximately for reference-scan uncertainty and mismatch.
    cov+=np.diag([c["anchor_loss_noise_floor_v"]**2,c["anchor_slope_noise_floor_ohm_cm2"]**2])
    return theta,cov


def estimate(operation,diagnostics,c,use_post_initial_anchors=True):
    """Estimate the state online in timestamp order, without future diagnostics.

    x=[d_c, degradation_rate, d_r, residual_bias].
    Observation y=V0(j)-[Vop-alpha*(T-Tref)]=d_c+d_r*(j-j_ref)-bias.
    dc and bias cannot be separated from one ordinary measurement alone.
    Diagnostics directly constrain dc and dr; the state model propagates between them.
    """
    operation=operation.reset_index(drop=True).copy()
    diagnostics=diagnostics.copy()
    _validate(operation,diagnostics,c)
    coeff=fit_initial_reference(diagnostics,c)
    vref=float(np.polyval(coeff,0.0))
    if vref<=0: raise ValueError("Non-positive fitted reference voltage")
    tref=float(operation.time_h.iloc[0])
    x=np.array([0.,c["initial_rate_prior_v_per_h"],0.,0.])
    P=np.diag(np.array([c["initial_loss_prior_sd_v"],c["initial_rate_prior_sd_v_per_h"],
                       c["initial_slope_prior_sd_ohm_cm2"],c["initial_bias_prior_sd_v"]])**2)
    # Do not pre-fit future scan values. The sorted list contains availability only.
    sessions=diagnostics[["session_id","time_h"]].drop_duplicates().sort_values(["time_h","session_id"])
    pending=[(r.session_id,float(r.time_h)) for r in sessions.itertuples()
             if r.time_h>tref] if use_post_initial_anchors else []
    scan_index=0
    records=[]; updates=[]; prev=tref
    for row in operation.itertuples():
        time=float(row.time_h);dt=time-prev
        F=np.eye(4);F[0,1]=dt
        Q=np.zeros((4,4)); qa=c["rate_process_sd_v_per_h_per_sqrt_h"]**2
        Q[0,0]=qa*dt**3/3+c["loss_process_sd_v_per_sqrt_h"]**2*dt
        Q[0,1]=Q[1,0]=qa*dt**2/2; Q[1,1]=qa*dt
        Q[2,2]=c["slope_process_sd_ohm_cm2_per_sqrt_h"]**2*dt
        Q[3,3]=c["bias_process_sd_v_per_sqrt_h"]**2*dt
        x=F@x;P=F@P@F.T+Q
        dj=float(row.current_density_a_cm2)-c["reference_current_density_a_cm2"]
        v0j=float(np.polyval(coeff,dj))
        vcond=float(row.voltage_cell_v)-c["calibrated_temperature_coefficient_v_per_c"]*(float(row.temperature_c)-c["reference_temperature_c"])
        normalized_v=vcond+vref-v0j
        H=np.array([[1.,0.,dj,-1.]])
        x,P=_update(x,P,H,np.array([v0j-vcond]),np.array([[c["operation_noise_sd_v"]**2]]))
        # Only diagnostics whose AVAILABLE timestamp has arrived are assimilated.
        count=0
        while scan_index<len(pending) and pending[scan_index][1]<=time:
            sid,available_time=pending[scan_index]
            scan=diagnostics[diagnostics.session_id==sid]
            anchor,R=scan_anchor(scan,coeff,c)
            # Acquisition/completion and result availability coincide in this
            # teaching data. An old measurement must NOT be relabelled with its
            # arrival timestamp and treated as a new current-state measurement.
            if available_time!=time:
                raise ValueError("Align diagnostic availability with an operation sample; delayed states need an out-of-sequence update")
            before=x.copy(); Hdiag=np.array([[1,0,0,0],[0,0,1,0]],dtype=float)
            x,P=_update(x,P,Hdiag,anchor,R)
            updates.append({"session_id":sid,"time_h":time,"diagnostic_dc_v":anchor[0],"diagnostic_dr_ohm_cm2":anchor[1],
                "dc_before_update_v":before[0],"dc_after_update_v":x[0],
                "dr_before_update_ohm_cm2":before[2],"dr_after_update_ohm_cm2":x[2],
                "bias_before_update_v":before[3],"bias_after_update_v":x[3]})
            scan_index+=1;count+=1
        records.append({"time_h":time,"current_density_a_cm2":float(row.current_density_a_cm2),
             "raw_ratio":float(row.voltage_cell_v)/vref,
             "initial_curve_standardized_soh":normalized_v/vref,
             "estimated_soh":(vref-x[0])/vref,"estimated_reference_voltage_v":vref-x[0],
             "estimated_dc_v":x[0],"estimated_rate_v_per_h":x[1],
             "estimated_dr_ohm_cm2":x[2],"estimated_residual_bias_v":x[3],
             "soh_model_sd":np.sqrt(max(P[0,0],0))/vref,
             "anchor_updates_at_this_sample":count})
        prev=time
    metadata={"fitted_initial_reference_voltage_v":vref,"initial_curve_polynomial_coefficients":coeff.tolist(),
        "post_initial_anchor_count_used":scan_index,"state_order":["d_c","rate","d_r","bias"],
        "prior_rate_v_per_h":c["initial_rate_prior_v_per_h"],
        "forecast_note":"Filtering is causal; no future anchors used. This demo estimates present SOH, not RUL."}
    return pd.DataFrame(records),pd.DataFrame(updates),metadata
