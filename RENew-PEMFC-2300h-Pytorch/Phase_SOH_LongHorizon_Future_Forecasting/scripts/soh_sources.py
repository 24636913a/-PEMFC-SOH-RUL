"""Only named, small derived Development artifacts; phase-local writes."""
import csv,hashlib,json,os,sys
from pathlib import Path
import numpy as np
import pandas as pd

PHASE=Path(__file__).resolve().parents[1]; PROJECT=PHASE.parent; REPO=PROJECT.parent
PATHS={
 'targets':'Phase_CHM_Cycle_Aligned_Sparse_Anchor_Health_Model/inputs/CHM_SPARSE_ANCHOR_TARGETS.csv',
 'mapping':'Phase_CHM_Cycle_Aligned_Sparse_Anchor_Health_Model/inputs/CHM_FROZEN_ANCHOR_INTERVAL_MAPPING.csv',
 'ahc':'Phase_AHC_Sparse_Anchor_Affine_Health_Calibration/outputs/B2_CHRONOLOGICAL_PREFIX_EVALUATION.csv',
 'ctae':'Phase_CTAE_P1S_Reference_Availability_Erratum/outputs/P1S_REFERENCE_AVAILABILITY_AUDIT.csv',
 'complete':'Phase_PAA_P1_StateToken_PreAnchor_Alignment/outputs/P1S_COMPLETE_CYCLE_MEASUREMENTS.csv',
 'timing':'Phase_PAV_PAA_Anchor_Validity_Qualification/outputs/POLARIZATION_TIME_PROVENANCE.csv',
 'reconstruction':'Phase_CPRF_H1_Causal_P1S_Retention_Forecasting/outputs/P1S_RAW_MEASUREMENT_INTERVAL_RECONSTRUCTION.csv',
 'identity':'Phase_PHI_P1S_B2_Health_Index_Consolidation/outputs/P1S_B2_FULL_HEALTH_TRAJECTORY.csv',
 'bol':'Phase_N1_Data_Foundation/outputs/soh_anchor_analysis/soh_reference_anchors.csv',
}
VREF=.71383023635; AVAIL=171.82091
EVENTS=[]; BOL_OPEN=False

def hash_file(path):
    if Path(path).resolve()==(PROJECT/PATHS['bol']).resolve():raise PermissionError('NO_FULL_MIXED_ANCHOR_HASH')
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def install_firewall():
    allowed={(PROJECT/name).resolve() for key,name in PATHS.items() if key!='bol'}
    bol=(PROJECT/PATHS['bol']).resolve()
    def hook(event,args):
        if event!='open' or isinstance(args[0],int):return
        raw,mode,flags=args; p=Path(os.fsdecode(raw)).resolve()
        write=bool((isinstance(mode,str) and any(c in mode for c in 'wax+')) or (isinstance(flags,int) and flags&(os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC|os.O_APPEND)))
        if p.is_relative_to(REPO) and not p.is_relative_to(PHASE):
            if write or (p==bol and not BOL_OPEN) or (p!=bol and p not in allowed):
                EVENTS.append({'path':str(p),'writing':write,'allowed':False});raise PermissionError('HISTORICAL_PATH_NOT_ALLOWLISTED')
            EVENTS.append({'path':str(p),'writing':False,'allowed':True})
        elif not p.is_relative_to(PHASE) and (p.name.startswith(('Polcurve_','Loadcycle_')) or '数据' in p.parts):
            raise PermissionError('RAW_OR_LATE_TARGET_DENIED')
    sys.addaudithook(hook)

def get_bol():
    global BOL_OPEN
    BOL_OPEN=True
    try:
        with (PROJECT/PATHS['bol']).open(encoding='utf-8-sig',newline='') as f:
            row=next(csv.DictReader(f))  # exactly one record; never parse late-life anchors
    finally:BOL_OPEN=False
    if row['checkpoint']!='BoL' or float(row['jstar_A_cm2'])!=1. or float(row['ageing_hour'])!=0.:
        raise RuntimeError('BOL_FIRST_RECORD_CONTRACT_CHANGED')
    if float(row['normalized_health_value'])!=1. or float(row['reference_voltage_V'])!=float(row['voltage_anchor_V']):
        raise RuntimeError('BOL_NORMALIZATION_MISMATCH')
    return row

def load_sources():
    data={k:pd.read_csv(PROJECT/PATHS[k]) for k in ['targets','mapping','ahc','ctae','complete','timing','reconstruction']}
    # Identity columns only; no posterior global B2 or smoothing columns consumed.
    identity=pd.read_csv(PROJECT/PATHS['identity'],usecols=['cycle_interval_id','raw_cycle_id','cycle_start_ophrs2_h','cycle_end_ophrs2_h'])
    assert data['targets'].checkpoint.tolist()==[300,600,850,1150,1500,1750]
    assert len(data['mapping'])==18 and len(data['timing'])==6
    bol=get_bol()
    refs=pd.concat([pd.DataFrame([{'nominal_h':0,'Y':float(bol['normalized_health_value'])}]),
        data['targets'][['checkpoint','Y']].rename(columns={'checkpoint':'nominal_h'})],ignore_index=True)
    complete=data['complete'].loc[data['complete'].stable_window.eq('middle_50')].copy()
    assert len(complete)==141 and complete.cycle_interval_id.is_unique
    warm=complete.loc[complete.is_reference_complete_interval]
    assert len(warm)==10 and abs(warm.P1S_voltage_V.median()-VREF)<1e-12
    assert abs(warm.measurement_end_ophrs2_h.max()-AVAIL)<1e-12
    ca=data['ctae'].loc[data['ctae'].causal_target_eligible].copy()
    assert len(ca)==131 and not ca.reference_warmup_record.any()
    base=ca[['cycle_interval_id','R_P1S_causal','ophrs_h','causal_target_eligible','condition_support_status']].rename(columns={'R_P1S_causal':'R','ophrs_h':'recorded_available_end_h'})
    p=base.merge(complete[['cycle_interval_id','measurement_end_datetime','measurement_end_ophrs2_h','P1S_voltage_V']],on='cycle_interval_id',validate='one_to_one')
    r=data['reconstruction']
    assert len(r)==131 and r.voltage_match_pass.all() and r.dwell_count_pass.all() and not r.raw_test_accessed.any()
    assert r.p1s_stable_window.eq('middle_50').all()
    p=p.merge(r[['cycle_interval_id','p1s_measurement_start_ophrs2_h','p1s_measurement_end_ophrs2_h','P1S_voltage_reconstructed_V']],on='cycle_interval_id',validate='one_to_one')
    p=p.merge(identity,on='cycle_interval_id',validate='one_to_one')
    assert np.allclose(p.R,p.P1S_voltage_V/VREF,atol=1e-12,rtol=0)
    assert np.allclose(p.P1S_voltage_V,p.P1S_voltage_reconstructed_V,atol=1e-12,rtol=0)
    assert np.allclose(p.recorded_available_end_h,p.measurement_end_ophrs2_h,atol=1e-12,rtol=0)
    p['actual_measurement_end_h']=p.p1s_measurement_end_ophrs2_h
    p['available_datetime']=pd.to_datetime(p.measurement_end_datetime)
    assert (p.actual_measurement_end_h<=p.recorded_available_end_h).all()
    assert (p.recorded_available_end_h<1819.997738).all() and p.raw_cycle_id.is_unique
    # Cross-check original mapping medians against AHC, without using its future estimates as inputs.
    grouped=data['mapping'].groupby('checkpoint').P1S_operating_voltage_retention.median()
    assert np.allclose(grouped.to_numpy(),data['ahc'].sort_values('checkpoint').R_current.to_numpy(),atol=1e-12,rtol=0)
    return refs,data['mapping'],p.sort_values('actual_measurement_end_h').reset_index(drop=True),data['timing'],bol,data['ahc']

def write_json(name,payload):
    path=(PHASE/name).resolve()
    if not path.is_relative_to(PHASE):raise PermissionError('PHASE_OUTPUT_ONLY')
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w',encoding='utf-8',newline='\n') as f:
        json.dump(payload,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')

def write_csv(name,data):
    p=(PHASE/name).resolve()
    if not p.is_relative_to(PHASE):raise PermissionError('PHASE_OUTPUT_ONLY')
    p.parent.mkdir(parents=True,exist_ok=True)
    data.to_csv(p,index=False,float_format='%.17g',lineterminator='\n')
