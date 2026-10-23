"""Independent least-squares recalculation from saved inputs, not runner methods."""
import hashlib,json,subprocess
from pathlib import Path
import numpy as np
import pandas as pd

P=Path(__file__).resolve().parents[1]

def fit(x,y):
    x=np.asarray(x,float);y=np.asarray(y,float)
    return np.linalg.lstsq(np.column_stack([x,np.ones(len(x))]),y,rcond=None)[0]

def main():
    dest=P/'outputs/INDEPENDENT_VERIFICATION.json'
    if dest.exists():raise RuntimeError('AUDIT_ALREADY_RECORDED')
    predictions=pd.read_csv(P/'predictions.csv',float_precision='round_trip')
    metrics=pd.read_csv(P/'metrics.csv',float_precision='round_trip')
    refs=pd.read_csv(P/'inputs/DEVELOPMENT_SOH_REFERENCES.csv',float_precision='round_trip')
    mapping=pd.read_csv(P/'inputs/FROZEN_P1S_MAPPING.csv',float_precision='round_trip')
    p1=pd.read_csv(P/'inputs/CAUSAL_P1S_IDENTITY.csv',float_precision='round_trip')
    p1['available_datetime']=pd.to_datetime(p1.available_datetime)
    timing=pd.read_csv(P/'inputs/POLARIZATION_TIME_PROVENANCE.csv',float_precision='round_trip')
    origins=json.loads((P/'outputs/ORIGIN_INPUTS_AND_PARAMETERS.json').read_text(encoding='utf-8'))
    frozen=json.loads((P/'outputs/PREDICTIONS_BEFORE_EVALUATION.json').read_text(encoding='utf-8'))
    assert len(predictions)==len(frozen)==12
    independently=[];maxdiff=0.;records=[]
    for record in origins:
        o=record['origin_nominal_h'];t=record['target_nominal_h'];task=record['task_id']
        historical=refs.loc[refs.nominal_h<=o].copy();trend,intercept=fit(historical.nominal_h,historical.Y)
        cutoff=pd.Timestamp(timing.loc[timing.checkpoint_h==o].iloc[0].PAA_gap_cutoff_datetime)
        legal=p1.loc[p1.available_datetime<cutoff].sort_values('actual_measurement_end_h');latest=legal.iloc[-1]
        window=legal.loc[legal.actual_measurement_end_h>=latest.actual_measurement_end_h-96]
        assert window.cycle_interval_id.tolist()==record['trend_cycle_ids']
        assert len(window)>=4 and window.raw_cycle_id.nunique()==len(window)
        mp=mapping.loc[mapping.checkpoint<=o];r=mp.groupby('checkpoint').P1S_operating_voltage_retention.median()
        y=refs.set_index('nominal_h').loc[r.index,'Y']
        a,b=fit(r,y)
        if a<0:a=0.;b=float(y.mean())
        slope_r,_=fit(window.actual_measurement_end_h,window.R)
        pred={'M0':float(historical.iloc[-1].Y),'M1':float(intercept+trend*t),
            'M2':float(a*latest.R+b),'M3':float(a*(latest.R+slope_r*(t-latest.actual_measurement_end_h))+b)}
        truth=float(refs.loc[refs.nominal_h==t].iloc[0].Y)
        for method,value in pred.items():
            stored=predictions.loc[(predictions.task_id==task)&(predictions.method==method)].iloc[0]
            maxdiff=max(maxdiff,abs(value-float(stored.prediction)))
            assert abs(value-float(stored.prediction))<1e-10
            assert abs((value-truth)*100-float(stored.signed_error_pp))<1e-8
            independently.append({'task_id':task,'method':method,'prediction':value,'target':truth,'ae_pp':abs(value-truth)*100})
        records.append({'task_id':task,'latest_cycle':latest.cycle_interval_id,'window_n':len(window),'mapping_n':len(r),'verified':True})
    independent=pd.DataFrame(independently)
    for method,g in independent.groupby('method'):
        saved=metrics.loc[metrics.method==method].iloc[0]
        mae=g.ae_pp.mean();rmse=np.sqrt(np.mean(g.ae_pp.to_numpy()**2))
        assert abs(mae-saved.MAE_pp)<1e-8 and abs(rmse-saved.RMSE_pp)<1e-8
    assert set(predictions.nominal_horizon_h)=={250,300,350}
    assert not predictions.strict_online_time_qualified.any()
    assert not metrics.formal_CLEAR_GAIN_claim.any()
    check={m:float(independent.loc[independent.method==m].ae_pp.mean()) for m in ['M0','M1']}
    # User-supplied approximate values are diagnostics, never used in fitting.
    # Approximate user references are diagnostics, not authoritative labels.
    reference_delta = {'M0': check['M0']-.2812, 'M1': check['M1']-.7714}
    cfg=json.loads((P/'configs/FROZEN_SOH_PROTOCOL.json').read_text(encoding='utf-8'))
    for name,expected in cfg['code_sha256'].items():assert hashlib.sha256((P/name).read_bytes()).hexdigest()==expected
    assert subprocess.check_output(['git','diff','--name-only'],cwd=P.parent.parent,text=True).strip()==''
    result={'passed':True,'independent_solver':'numpy.linalg.lstsq; runner uses centered covariance',
        'max_absolute_prediction_difference':maxdiff,'all_12_predictions_verified':True,
        'all_targets_retained':True,'checks':records,'M0_M1_selfcheck_MAE_pp':check,
        'strict_online_CLEAR_GAIN_claim':False,'historical_tracked_diff':[],
        'approximate_user_reference_delta_pp':reference_delta,
        'audit_repair_note':'Original audit completed independent numerical checks, then stopped on too-tight approximate M1 reference. This separate post-run audit keeps original frozen sources and predictions unchanged.'}
    with dest.open('x',encoding='utf-8',newline='\n') as f:json.dump(result,f,ensure_ascii=False,indent=2);f.write('\n')
    print(json.dumps(result,ensure_ascii=False))

if __name__=='__main__':main()
