"""Small derived Development sources, exact physical identity, phase-local writes."""
from pathlib import Path
import hashlib,json,os,sys
import numpy as np
import pandas as pd

PHASE=Path(__file__).resolve().parents[1];PROJECT=PHASE.parent;REPO=PROJECT.parent
VREF=.71383023635;VREF_AVAILABLE=171.82091;BOUNDARY=1819.997738
FOLDS={'fold_1':(72.788690,946.393214,1164.794345),
 'fold_2':(72.788690,1164.794345,1383.195476),'fold_3':(72.788690,1383.195476,BOUNDARY)}
SOURCES={
 'availability':'Phase_CTAE_P1S_Reference_Availability_Erratum/outputs/P1S_REFERENCE_AVAILABILITY_AUDIT.csv',
 'reconstruction':'Phase_CPRF_H1_Causal_P1S_Retention_Forecasting/outputs/P1S_RAW_MEASUREMENT_INTERVAL_RECONSTRUCTION.csv',
 'identity':'Phase_PHI_P1S_B2_Health_Index_Consolidation/outputs/P1S_B2_FULL_HEALTH_TRAJECTORY.csv',
 'catalog':'Phase_RCR_Raw_Resolution_Cycle_Reconstruction_Audit/outputs/RAW_CYCLE_CATALOG.csv',
 'complete':'Phase_PAA_P1_StateToken_PreAnchor_Alignment/outputs/P1S_COMPLETE_CYCLE_MEASUREMENTS.csv',
 'H1_authority':'Phase_CPRF_H1_Causal_P1S_Retention_Forecasting/inputs/H1_FROZEN_SAMPLE_AUTHORITY.csv',
 'H3_authority':'Phase_CMRF_H3_H6_OriginConditioned_P1S_Forecasting/inputs/H3_FROZEN_SAMPLE_AUTHORITY.csv',
 'H1_persistence':'Phase_CPRF_H1_Causal_P1S_Retention_Forecasting/outputs/H1_BASELINE_PREDICTIONS.csv',
 'H3_persistence':'Phase_CMRF_H3_H6_OriginConditioned_P1S_Forecasting/outputs/H3_PREDICTIONS.csv',
 'H1_protocol':'Phase_CPRF_H1_Causal_P1S_Retention_Forecasting/configs/CPRF_H1_FROZEN_PROTOCOL.yaml',
 'H3_protocol':'Phase_CMRF_H3_H6_OriginConditioned_P1S_Forecasting/configs/CMRF_H3_H6_FROZEN_PROTOCOL.yaml'}
EVENTS=[]

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def frame(path,**kwargs):return pd.read_csv(path,float_precision='round_trip',**kwargs)
def save_json(name,value):
    p=(PHASE/name).resolve()
    if not p.is_relative_to(PHASE):raise PermissionError('NEW_PHASE_ONLY')
    p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('w',encoding='utf-8',newline='\n') as f:json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
def save_csv(name,value):
    p=(PHASE/name).resolve()
    if not p.is_relative_to(PHASE):raise PermissionError('NEW_PHASE_ONLY')
    p.parent.mkdir(parents=True,exist_ok=True)
    value.to_csv(p,index=False,float_format='%.17g',lineterminator='\n')
def install_firewall():
    allowed={(PROJECT/name).resolve() for name in SOURCES.values()}
    def audit(event,args):
        if event!='open' or isinstance(args[0],int):return
        path,mode,flags=args;p=Path(os.fsdecode(path)).resolve()
        writing=bool((isinstance(mode,str) and any(c in mode for c in 'wax+')) or (isinstance(flags,int) and flags&(os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC|os.O_APPEND)))
        if p.is_relative_to(REPO) and not p.is_relative_to(PHASE):
            ok=not writing and p in allowed;EVENTS.append({'path':str(p),'writing':writing,'allowed':ok})
            if not ok:raise PermissionError('HISTORY_WRITE_OR_NONALLOWLISTED_SOURCE')
        elif not p.is_relative_to(PHASE) and (p.name.startswith(('Polcurve_','Loadcycle_')) or '数据' in p.parts):
            raise PermissionError('RAW_TEST_POLARIZATION_DENIED')
    sys.addaudithook(audit)

def load_sources():
    a=frame(PROJECT/SOURCES['availability']);r=frame(PROJECT/SOURCES['reconstruction'])
    identity=frame(PROJECT/SOURCES['identity'],usecols=['cycle_interval_id','raw_cycle_id'])
    cat=frame(PROJECT/SOURCES['catalog'],usecols=['raw_cycle_id','start_ophrs2_h','end_ophrs2_h','complete_cycle'])
    c=frame(PROJECT/SOURCES['complete']);primary=c.loc[c.stable_window=='middle_50']
    warm=primary.loc[primary.is_reference_complete_interval]
    assert len(warm)==10 and abs(warm.P1S_voltage_V.median()-VREF)<1e-12
    assert abs(warm.measurement_end_ophrs2_h.max()-VREF_AVAILABLE)<1e-12
    a=a.loc[a.causal_target_eligible].copy();assert len(a)==131 and not a.reference_warmup_record.any()
    assert len(r)==131 and r.voltage_match_pass.all() and r.dwell_count_pass.all() and not r.raw_test_accessed.any()
    assert (cat.end_ophrs2_h<BOUNDARY).all()
    p=a[['cycle_interval_id','R_P1S_causal','ophrs_h','causal_target_eligible']].merge(
        r[['cycle_interval_id','p1s_measurement_start_ophrs2_h','p1s_measurement_end_ophrs2_h','P1S_voltage_reconstructed_V']],on='cycle_interval_id',validate='one_to_one').merge(identity,on='cycle_interval_id',validate='one_to_one')
    p=p.rename(columns={'R_P1S_causal':'R','ophrs_h':'conservative_available_end_h','p1s_measurement_start_ophrs2_h':'measurement_start_h','p1s_measurement_end_ophrs2_h':'measurement_end_h'})
    assert np.allclose(p.R,p.P1S_voltage_reconstructed_V/VREF,rtol=0,atol=1e-12)
    assert p.raw_cycle_id.is_unique and (p.measurement_end_h>VREF_AVAILABLE).all()
    p=p.sort_values('measurement_end_h').reset_index(drop=True)
    by_id=p.set_index('cycle_interval_id');complete=cat.set_index('raw_cycle_id').complete_cycle
    authorities={};historical={}
    for horizon,step in [('H1',1),('H3',3)]:
        df=frame(PROJECT/SOURCES[horizon+'_authority']);assert not df.duplicated(['fold','role','origin_raw_cycle_id','target_raw_cycle_id']).any()
        for row in df.itertuples():
            assert row.target_raw_cycle_id==row.origin_raw_cycle_id+step==row.required_target_raw_cycle_id
            assert all(bool(complete.loc[k]) for k in range(int(row.origin_raw_cycle_id),int(row.target_raw_cycle_id)+1))
            origin=by_id.loc[row.origin_cycle_interval_id];target=by_id.loc[row.target_cycle_interval_id]
            assert abs(origin.R-row.origin_R_P1S_causal)<1e-12 and abs(target.R-row.target_R_P1S_causal)<1e-12
            assert abs(origin.measurement_end_h-row.origin_p1s_measurement_end_ophrs2_h)<1e-9
            assert abs(target.measurement_start_h-row.target_p1s_measurement_start_ophrs2_h)<1e-9
            assert abs(target.measurement_end_h-row.target_p1s_measurement_end_ophrs2_h)<1e-9
            assert origin.measurement_end_h<target.measurement_start_h
            lo,cut,hi=FOLDS[row.fold];start,end=(lo,cut) if row.role=='train' else (cut,hi)
            assert row.target_p1s_measurement_start_ophrs2_h>=start and row.target_p1s_measurement_end_ophrs2_h<end
        authorities[horizon]=df
        old=frame(PROJECT/SOURCES[horizon+'_persistence']);old=old.loc[old.model=='PERSISTENCE'].copy()
        key=['fold','origin_cycle_interval_id','target_cycle_interval_id']
        joined=df.loc[df.role=='validation'].merge(old,on=key,validate='one_to_one',suffixes=('_authority','_old'))
        assert len(joined)==len(df.loc[df.role=='validation'])
        assert np.allclose(joined.origin_R_P1S_causal if horizon=='H1' else joined.origin_R_P1S_causal_authority,joined.predicted_R_P1S_causal,atol=1e-12,rtol=0)
        assert np.allclose(joined.target_R_P1S_causal,joined.true_R_P1S_causal,atol=1e-12,rtol=0)
        historical[horizon]=old
    return p,authorities,historical
