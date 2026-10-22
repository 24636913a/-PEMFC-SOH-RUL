"""Read-only audit of the two existing Seed202 training runs; no training calls."""
from pathlib import Path
import hashlib
import json
import math

import numpy as np
import pandas as pd
import torch

HERE=Path(__file__).resolve().parent
FOLDER=HERE.parent
RUNS=[FOLDER/'F2_Training_Outputs'/name for name in [
    '20261019T052754_499042Z_seed202_compare','20261019T055734_218949Z_seedall_compare']]
OUT=HERE/'outputs'/'saved_evidence'


def read(path):return json.loads(path.read_text(encoding='utf-8'))
def csv(path):return pd.read_csv(path,float_precision='round_trip')
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def save(path,obj):path.write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n',encoding='utf-8',newline='\n')
def first_unequal(a,b):
    bad=np.asarray(a)!=np.asarray(b)
    return int(np.flatnonzero(bad)[0])+1 if bad.any() else None


def early_stop(history,target_scale):
    best=math.inf;best_epoch=0;stale=0
    for row in history.itertuples():
        value=math.sqrt(row.validation_standardized_MSE)
        if value<best-1e-12:best,best_epoch,stale=value,row.epoch,0
        else:stale+=1
        assert row.best_epoch==best_epoch and row.stale_epochs==stale
        assert row.optimizer_steps==row.backward_calls==row.epoch*55
        assert math.isfinite(row.gradient_norm_min) and math.isfinite(row.gradient_norm_max)
    assert stale==40
    return dict(best_epoch=best_epoch,stop_epoch=len(history),stale_epochs=stale,
                best_validation_RMSE_V=best*target_scale,updates=len(history)*55)


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    protocols=[read(run/'PROTOCOL_BEFORE_TRAINING.json') for run in RUNS]
    ledgers=[read(run/'seed202/top10/TRAINING_LEDGER.json') for run in RUNS]
    histories=[csv(run/'seed202/top10/training_history.csv') for run in RUNS]
    a,b=histories;n=min(len(a),len(b))
    identity=[]
    for name in ['PREPROCESSOR.json','TRAIN_ONLY_SCALER.csv','MASK_APPLICATION_AUDIT.csv',
                 'windows/TRAIN.csv','windows/VALIDATION.csv','windows/TEST.csv']:
        hashes=[sha(run/name) for run in RUNS]
        identity.append(dict(artifact=name,run_a_sha256=hashes[0],run_b_sha256=hashes[1],exact_equal=hashes[0]==hashes[1]))
    assert all(row['exact_equal'] for row in identity)
    pd.DataFrame(identity).to_csv(OUT/'IDENTITY_CHECKS.csv',index=False)
    for key in ['data_sha256','code_sha256','hyperparameters','masks','partitions','environment','target_scale_V','max_epochs','patience']:
        assert protocols[0][key]==protocols[1][key],key
    columns=[];delta=[]
    for col in a.columns:
        if col in ['epoch','epoch_seconds']:continue
        aa=a[col].iloc[:n].to_numpy();bb=b[col].iloc[:n].to_numpy()
        columns.append(dict(field=col,first_different_epoch=first_unequal(aa,bb),different_epochs=int(np.count_nonzero(aa!=bb))))
        if a[col].dtype.kind in 'if':
            delta.extend(dict(epoch=i+1,field=col,run_a=float(aa[i]),run_b=float(bb[i]),difference=float(bb[i]-aa[i])) for i in range(n))
    pd.DataFrame(columns).to_csv(OUT/'FIRST_DIVERGENCE.csv',index=False)
    pd.DataFrame(delta).to_csv(OUT/'EPOCH_NUMERICAL_DIFFERENCES.csv',index=False,float_format='%.17g')
    stops=[early_stop(history,protocols[i]['target_scale_V']) for i,history in enumerate(histories)]
    assert all(row['best_epoch']==ledgers[i]['best_epoch'] and row['stop_epoch']==ledgers[i]['stop_epoch'] for i,row in enumerate(stops))
    assert ledgers[0]['initialization_sha256']==ledgers[1]['initialization_sha256']
    assert ledgers[0]['first_update_sha256']==ledgers[1]['first_update_sha256']
    assert a.sampler_order_sha256.iloc[:n].tolist()==b.sampler_order_sha256.iloc[:n].tolist()
    weights=[torch.load(run/'seed202/top10/best_model.pt',map_location='cpu',weights_only=True) for run in RUNS]
    assert weights[0]['state_dict'].keys()==weights[1]['state_dict'].keys()
    weight_rows=[];total_sse=0.;total_norm=0.;total_values=0;unequal_values=0
    for name,left in weights[0]['state_dict'].items():
        right=weights[1]['state_dict'][name]
        assert left.shape==right.shape and left.dtype==right.dtype
        if not left.is_floating_point():continue
        x=left.double().numpy();y=right.double().numpy();d=y-x
        count=int(np.count_nonzero(x!=y));sse=float(np.sum(d*d));norm=float(np.sum(x*x))
        total_sse+=sse;total_norm+=norm;total_values+=x.size;unequal_values+=count
        weight_rows.append(dict(tensor=name,values=x.size,unequal_values=count,
            max_abs_difference=float(np.max(np.abs(d))),mean_abs_difference=float(np.mean(np.abs(d))),L2_difference=math.sqrt(sse)))
    pd.DataFrame(weight_rows).to_csv(OUT/'BEST_CHECKPOINT_TENSOR_DIFFERENCES.csv',index=False,float_format='%.17g')
    metrics=[];points=[]
    for i,run in enumerate(RUNS):
        p=csv(run/'seed202/top10/predictions_TEST.csv');points.append(p)
        truth=p.y_true_delta_V.to_numpy();error=p.y_pred_delta_V.to_numpy()-truth;nq=len(error)
        sse=math.fsum(float(v)**2 for v in error);mean=math.fsum(map(float,truth))/nq
        rmse=math.sqrt(sse/nq);mae=math.fsum(abs(float(v)) for v in error)/nq
        r2=1-sse/math.fsum((float(v)-mean)**2 for v in truth)
        old=csv(run/'HOLDOUT_PER_SEED_METRICS.csv')
        old=old.loc[(old.seed==202)&(old.feature_mode=='top10')].iloc[0]
        assert (rmse,mae,r2)==(old.RMSE_V,old.MAE_V,old.R2)
        metrics.append(dict(run=run.name,RMSE_mV=rmse*1000,MAE_mV=mae*1000,R2=r2,
            best_epoch=stops[i]['best_epoch'],stop_epoch=stops[i]['stop_epoch'],optimizer_updates=stops[i]['updates']))
    for key in ['origin_row','target_row','horizon_step','y_true_delta_V','true_voltage_V']:
        assert np.array_equal(points[0][key],points[1][key]),key
    pd.DataFrame(metrics).to_csv(OUT/'RECOMPUTED_REPEAT_METRICS.csv',index=False,float_format='%.17g')
    values=np.asarray([r['RMSE_mV'] for r in metrics]);maes=np.asarray([r['MAE_mV'] for r in metrics]);r2s=np.asarray([r['R2'] for r in metrics])
    prediction_difference=points[1].y_pred_delta_V.to_numpy()-points[0].y_pred_delta_V.to_numpy()
    summary=dict(status='READ_ONLY_EVIDENCE_VERIFIED',source_program_sha256=protocols[0]['code_sha256'],
        exact_initialization=True,exact_first_update_weights=True,common_epochs=n,exact_common_sampler_orders=True,
        first_loss_and_gradient_min_difference_epoch=first_unequal(a.train_standardized_MSE.iloc[:n],b.train_standardized_MSE.iloc[:n]),
        first_best_epoch_decision_difference=first_unequal(a.best_epoch.iloc[:n],b.best_epoch.iloc[:n]),
        early_stopping_recomputed=stops,metrics=metrics,
        rmse_summary=dict(n=2,mean_mV=float(values.mean()),sample_sd_mV=float(values.std(ddof=1)),
            min_mV=float(values.min()),max_mV=float(values.max()),range_mV=float(np.ptp(values)),
            later_vs_first_percent=float((values[1]/values[0]-1)*100)),
        mae_summary=dict(mean_mV=float(maes.mean()),sample_sd_mV=float(maes.std(ddof=1))),
        R2_summary=dict(mean=float(r2s.mean()),sample_sd=float(r2s.std(ddof=1))),
        repeated_run_prediction_difference=dict(RMS_mV=float(np.sqrt(np.mean(prediction_difference**2))*1000),
            max_abs_mV=float(np.max(np.abs(prediction_difference))*1000)),
        best_weight_difference=dict(unequal_values=unequal_values,total_float_values=total_values,
            L2_difference=math.sqrt(total_sse),relative_L2_to_first=math.sqrt(total_sse/total_norm)),
        historical_rng_and_dropout_masks_recorded=False,historical_per_batch_gradients_recorded=False,
        historical_cublas_workspace_environment_recorded=False,
        important_limit='Two completed repetitions are descriptive; no robust variance estimate or statistical significance.',
        unrelated_protocol_differences=['UTC creation time','requested seed list','~1e-16 descriptive KMeans centres; no model input dependence'])
    save(OUT/'SAVED_EVIDENCE_SUMMARY.json',summary)
    print(json.dumps(summary,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
