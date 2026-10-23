"""Raw hold measurements; statistics are not a quasi-steady certificate."""
import numpy as np
import pandas as pd
from m2_access import check

CELLS=[f'U{i} [V]' for i in [1,2,3,4,5,7,8,9,10,11]]
COND={
 'T_anode_in':'T.Si.A [°C]', 'T_anode_out':'T.So.A [°C]',
 'T_cathode_in':'T.Si.C [°C]', 'T_cathode_out':'T.So.C [°C]',
 'RH_anode':'RH.Si.A  [%]', 'RH_cathode':'RH.Si.C [%]',
 'P_anode_in':'p.Si.A [kPag]', 'P_anode_out':'p.So.A [kPag]',
 'P_cathode_in':'p.Si.C [kPag]', 'P_cathode_out':'p.So.C [kPag]',
 'flow_anode_H2':'F.Si.H.A [nl/min]', 'flow_cathode_air':'F.Si.Air.C [nl/min]',
 'stoic_anode':'Stoic.S.A [-]', 'stoic_cathode':'Stoic.S.C [-]'}
EXTRA=['RH.Si.A.CV [%]','RH.Si.C.CV [%]','DPT.Si.A [°C]','DPT.Si.C [°C]','U.S.AveCell [V]']
RAW_COLUMNS=['OpHrs2 [h]','DateTime','operation_point','I.S.set [A]','I.S.act [A]','j.S.act [A/cm2]']+CELLS+list(COND.values())+EXTRA
STABILITY=['j_std_A_cm2','j_range_A_cm2','voltage_abs_slope_mV_per_min',
 'temperature_max_range_C','RH_max_range_pct','pressure_max_range_kPag','flow_max_CV','stoic_max_CV']

def measured_hold(group,source,bins):
    duration=(group._dt.iloc[-1]-group._dt.iloc[0]).total_seconds()
    if duration<120:
        return None
    start_age=float(group['OpHrs2 [h]'].iloc[0]); end_age=float(group['OpHrs2 [h]'].iloc[-1])
    starts=bins.bin_start_ophrs2_h.to_numpy(float)
    first=int(np.searchsorted(starts,start_age,side='right')-1)
    last=int(np.searchsorted(starts,end_age,side='right')-1)
    check(first>=0 and last<len(bins),'DWELL_BIN_MAPPING_MISSING')
    selected=bins.iloc[first:last+1]
    same=selected.temporal_segment_id.nunique()==1
    source_in_bins=all(source.split('/')[-1] in str(s).split('|') for s in selected.source_file_names)
    contiguous=bool(np.allclose(np.diff(selected.bin_start_ophrs2_h),.5,atol=1e-9,rtol=0))
    time_s=(group._dt-group._dt.iloc[0]).dt.total_seconds().to_numpy(float)
    window=group.loc[(time_s>=duration*.25)&(time_s<=duration*.75)]
    if len(window)<30:
        return None
    voltage=window[CELLS].to_numpy(float).mean(axis=1)
    j=window['j.S.act [A/cm2]'].to_numpy(float)
    active=window['I.S.act [A]'].to_numpy(float)
    finite=bool(np.isfinite(window[CELLS+list(COND.values())].to_numpy(float)).all())
    tt=(window._dt-window._dt.iloc[0]).dt.total_seconds().to_numpy(float)
    denom=float(np.sum((tt-tt.mean())**2))
    slope=float(np.sum((tt-tt.mean())*(voltage-voltage.mean()))/denom*60000) if denom else float('nan')
    rec=dict(source_file=source,raw_record_start_inclusive=int(group._record.iloc[0]),
      raw_record_end_exclusive=int(group._record.iloc[-1])+1,
      hold_start_ophrs2_h=start_age,hold_end_ophrs2_h=end_age,
      hold_start_datetime=group._dt.iloc[0].isoformat(),hold_end_datetime=group._dt.iloc[-1].isoformat(),
      measurement_start_ophrs2_h=float(window['OpHrs2 [h]'].iloc[0]),
      measurement_end_ophrs2_h=float(window['OpHrs2 [h]'].iloc[-1]),
      measurement_raw_record_start=int(window._record.iloc[0]),measurement_raw_record_end_exclusive=int(window._record.iloc[-1])+1,
      measurement_start_datetime=window._dt.iloc[0].isoformat(),measurement_end_datetime=window._dt.iloc[-1].isoformat(),
      train_bin_start_row=first,train_bin_end_row=last,temporal_segment_id=int(selected.temporal_segment_id.iloc[0]) if same else None,
      crosses_bin_boundary=first!=last,same_continuous_train_segment=bool(same and contiguous and source_in_bins),
      operation_point=str(group.operation_point.iloc[0]),set_current_A=float(group['I.S.set [A]'].median()),
      actual_j_median_A_cm2=float(np.median(j)),hold_duration_wall_s=duration,
      hold_duration_age_s=(end_age-start_age)*3600,settling_wait_s=duration*.25,
      raw_hold_samples=len(group),measurement_samples=len(window),measurement_window='MIDDLE_50_PERCENT',
      voltage_median_V=float(np.median(voltage)),voltage_std_V=float(np.std(voltage)),
      voltage_abs_slope_mV_per_min=abs(slope),j_std_A_cm2=float(np.std(j)),j_range_A_cm2=float(np.ptp(j)),
      positive_current_whole_hold=bool((group['I.S.act [A]']>.10).all()),
      finite_voltage_and_conditions=finite,recovery_hysteresis_controlled=False,
      source_domain='calibration' if last<1051 else ('evidence' if first>=1051 else 'DOMAIN_BOUNDARY_EXCLUDED'),
      label_source='median instantaneous active10cell mean; raw cells exclude U6',
      startup_or_stop_in_hold=bool((group['I.S.act [A]'].abs()<=.10).any()),
      dwell_quasi_steady_certified=False)
    ranges={}; cvs={}
    for name,col in COND.items():
        values=window[col].to_numpy(float)
        rec[name+'_median']=float(np.median(values)); ranges[name]=float(np.ptp(values))
        cvs[name]=float(np.std(values)/max(abs(np.mean(values)),1e-12))
    rec.update(temperature_max_range_C=max(ranges[n] for n in ranges if n.startswith('T_')),
       RH_max_range_pct=max(ranges[n] for n in ['RH_anode','RH_cathode']),
       pressure_max_range_kPag=max(ranges[n] for n in ranges if n.startswith('P_')),
       flow_max_CV=max(cvs[n] for n in ['flow_anode_H2','flow_cathode_air']),
       stoic_max_CV=max(cvs[n] for n in ['stoic_anode','stoic_cathode']))
    rec['RH_physical_domain_pending']=bool(((window[[COND['RH_anode'],COND['RH_cathode']]].to_numpy(float)<0)|
                                           (window[[COND['RH_anode'],COND['RH_cathode']]].to_numpy(float)>100)).any())
    rec['flow_or_stoic_domain_pending']=bool((window[[COND['flow_anode_H2'],COND['flow_cathode_air']]].to_numpy(float)<0).any()
                 or (window[[COND['stoic_anode'],COND['stoic_cathode']]].to_numpy(float)<=0).any())
    return rec

def split_complete_runs(frame):
    boundary=(frame['operation_point'].astype(str)!=frame['operation_point'].astype(str).shift())
    boundary |= frame['I.S.set [A]'].round(3).ne(frame['I.S.set [A]'].round(3).shift())
    boundary |= frame._dt.diff().dt.total_seconds().gt(5)
    boundary.iloc[0]=True
    groups=boundary.cumsum()
    last_id=int(groups.iloc[-1])
    complete=[g for k,g in frame.groupby(groups,sort=True) if int(k)!=last_id]
    return complete,frame.loc[groups.eq(last_id)].copy()

def apply_stability_thresholds(dwells):
    """Fixed train-calibration medians, NOT chosen by polarization differences."""
    cal=dwells.loc[(dwells.source_domain=='calibration') & dwells.positive_current_whole_hold &
                   dwells.finite_voltage_and_conditions & dwells.same_continuous_train_segment]
    check(len(cal)>=20,'STABILITY_CALIBRATION_SUPPORT_INSUFFICIENT')
    limits={name:max(float(cal[name].median()),1e-8) for name in STABILITY}
    within=np.ones(len(dwells),dtype=bool)
    for name,limit in limits.items():
        within &= dwells[name].le(limit).to_numpy()
    dwells=dwells.copy()
    dwells['relative_quiet_candidate']=within & dwells.positive_current_whole_hold & dwells.finite_voltage_and_conditions & dwells.same_continuous_train_segment
    dwells['sensor_domain_clear_for_review']=~dwells.RH_physical_domain_pending & ~dwells.flow_or_stoic_domain_pending
    dwells['qualification_status']='DWELL_FOUND_BUT_QUASI_STEADY_NOT_VERIFIED'
    return dwells,dict(algorithm='P50 of each metric on time-isolated calibration holds; floor1e-8; conjunction',
       calibration_hold_count=len(cal),thresholds=limits,physical_tolerance_certified=False,
       no_voltage_order_used=True,thresholds_not_retuned=True)
