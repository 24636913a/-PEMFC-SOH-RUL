"""Independent feature/Ridge/metric calculations from saved inputs; no sklearn."""
from pathlib import Path
import hashlib,json,subprocess
import numpy as np
import pandas as pd

P=Path(__file__).resolve().parents[1];VREF=.71383023635
def read(path):return pd.read_csv(P/path,float_precision='round_trip')
def independent_ridge(x,y,v,alpha):
    mean=x.mean(0);scale=x.std(0,ddof=0);scale=np.where(scale==0,1.,scale)
    z=(x-mean)/scale;zmean=z.mean(0);yc=y-y.mean();zc=z-zmean
    coef=np.linalg.solve(zc.T@zc+alpha*np.eye(5),zc.T@yc)
    intercept=float(y.mean()-zmean@coef)
    return (v-mean)/scale@coef+intercept,mean,scale

def main():
    dest=P/'outputs/INDEPENDENT_VERIFICATION.json'
    if dest.exists():raise RuntimeError('INDEPENDENT_AUDIT_ALREADY_RECORDED')
    p1s=read('inputs/CAUSAL_P1S_ACTUAL_MEASUREMENT_IDENTITY.csv')
    model_records=json.loads((P/'outputs/FITTED_RIDGE_MODELS.json').read_text(encoding='utf-8'))
    pred=read('predictions.csv');metrics=read('metrics_fold.csv');maximum=0.;grids_checked=0;features_checked=0
    for h in ['H1','H3']:
        f=read(f'inputs/{h}_FEATURES_AND_ELIGIBILITY.csv')
        for row in f.itertuples():
            for L in [96,192]:
                t=row.origin_p1s_measurement_end_ophrs2_h
                history=p1s.loc[(p1s.measurement_end_h<=t+1e-9)&(p1s.measurement_end_h>=t-L-1e-9)]
                times=history.measurement_end_h.to_numpy();values=history.R.to_numpy()
                span=float(np.ptp(times));minimum=5 if L==96 else 10
                assert bool(getattr(row,f'L{L}_eligible'))==(len(values)>=minimum and span>=.7*L-1e-9)
                slope=0. if len(values)==1 else float(np.linalg.lstsq(np.column_stack([times-times.mean(),np.ones(len(times))]),values,rcond=None)[0][0])
                expected=np.array([row.origin_R_P1S_causal,values.mean(),values.std(ddof=0),slope,row.origin_R_P1S_causal-np.median(values)])
                names=['origin_R','history_mean_R','history_std_R','history_slope_R_per_h','origin_minus_median_R']
                actual=np.array([getattr(row,f'L{L}_{n}') for n in names])
                np.testing.assert_allclose(actual,expected,rtol=0,atol=1e-12);features_checked+=1
        for record in [r for r in model_records if r['horizon']==h]:
            fold=record['fold'];L=record['length_h'];names=record['features'];g=f.loc[f.fold==fold].set_index('sample_index')
            train=g.loc[record['train_samples']];fit=g.loc[record['inner_fit_samples']];selection=g.loc[record['inner_selection_samples']]
            X=train[names].to_numpy();Y=(train.target_R_P1S_causal-train.origin_R_P1S_causal).to_numpy()
            xf=fit[names].to_numpy();yf=(fit.target_R_P1S_causal-fit.origin_R_P1S_causal).to_numpy()
            xs=selection[names].to_numpy();ys=(selection.target_R_P1S_causal-selection.origin_R_P1S_causal).to_numpy()
            for score in record['alpha_grid_scores']:
                delta,_,_=independent_ridge(xf,yf,xs,score['alpha'])
                assert abs(np.mean(np.abs(delta-ys))-score['inner_MAE_R'])<1e-10;grids_checked+=1
            best=min(record['alpha_grid_scores'],key=lambda q:(q['inner_MAE_R'],q['alpha']))['alpha'];assert best==record['alpha']
            assert (train.target_p1s_measurement_end_ophrs2_h<=record['outer_label_cutoff_h']+1e-9).all()
            assert (fit.target_p1s_measurement_end_ophrs2_h<=record['inner_label_availability_cutoff_h']+1e-9).all()
            val=g.loc[record['validation_samples']]
            if len(val):
                delta,mean,scale=independent_ridge(X,Y,val[names].to_numpy(),record['alpha'])
                stored=pred.loc[(pred.horizon==h)&(pred.fold==fold)&(pred.model==f'M{L}')].set_index('sample_index').loc[record['validation_samples']]
                difference=np.abs(delta+val.origin_R_P1S_causal.to_numpy()-stored.predicted_R.to_numpy())
                maximum=max(maximum,float(difference.max()));assert difference.max()<1e-9
                np.testing.assert_allclose(record['train_scaler_mean'],mean,atol=1e-12,rtol=0)
                np.testing.assert_allclose(record['train_scaler_scale'],scale,atol=1e-12,rtol=0)
        for fold in ['fold_1','fold_2','fold_3']:
            scopes=[]
            for method in ['M0','M96','M192']:
                points=pred.loc[(pred.horizon==h)&(pred.fold==fold)&(pred.model==method)]
                scores=metrics.loc[(metrics.horizon==h)&(metrics.fold==fold)&(metrics.model==method)].iloc[0]
                assert len(points)==scores.n_validation
                scopes.append(points[['origin_raw_cycle_id','target_raw_cycle_id']].to_numpy().tolist())
                if len(points):
                    error=points.predicted_R.to_numpy()-points.true_R.to_numpy();ae=np.abs(error)
                    for name,value in [('MAE_R',ae.mean()),('RMSE_R',np.sqrt(np.mean(error**2))),('MAE_mV_equivalent',ae.mean()*VREF*1000),('P95_abs_error_mV_equivalent',np.quantile(ae,.95)*VREF*1000),('max_abs_error_mV_equivalent',ae.max()*VREF*1000)]:
                        assert abs(value-scores[name])<1e-9
                    if method=='M0':np.testing.assert_array_equal(points.predicted_R.to_numpy(),points.origin_R.to_numpy())
                else:assert pd.isna(scores.MAE_R) and scores.status=='NOT_EVALUABLE_NO_COMMON_VALIDATION'
            assert scopes[0]==scopes[1]==scopes[2]
    summary=read('metrics_summary.csv');assert summary.formal_three_fold_equal_MAE_mV.isna().all()
    assert (summary.available_folds==2).all()
    parity=read('outputs/HISTORICAL_PERSISTENCE_PARITY.csv');assert parity.passed.all()
    cfg=json.loads((P/'configs/FROZEN_LONG_HISTORY_PROTOCOL.json').read_text(encoding='utf-8'))
    for name,digest in cfg['execution_code_sha256'].items():assert hashlib.sha256((P/name).read_bytes()).hexdigest()==digest,name
    assert not subprocess.check_output(['git','diff','--name-only'],cwd=P.parent.parent,text=True).strip()
    result={'passed':True,'independent_ridge_solver':'numpy normal equations (no sklearn)',
        'feature_vectors_checked':features_checked,'inner_alpha_scores_checked':grids_checked,
        'max_prediction_difference_R':maximum,'prediction_rows':len(pred),'matched_samples_all_methods':True,
        'empty_fold_not_imputed':True,'historical_persistence_parity':True,'no_three_fold_gain_claim':True}
    with dest.open('x',encoding='utf-8',newline='\n') as out:json.dump(result,out,ensure_ascii=False,indent=2);out.write('\n')
    print(json.dumps(result,ensure_ascii=False))

if __name__=='__main__':main()
