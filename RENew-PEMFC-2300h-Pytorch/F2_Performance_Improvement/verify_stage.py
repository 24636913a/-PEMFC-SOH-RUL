"""Independent numerical verification of fresh Train/Validation-only stages."""
import math
import numpy as np
import pandas as pd
import torch
import frozen_f2 as f
import diagnostics as d
import settings as s
from io_utils import js,csv


def independent_metrics(p):
    truth=p.y_true_delta_V.to_numpy(float);prediction=p.y_pred_delta_V.to_numpy(float)
    errors=[float(a)-float(b) for a,b in zip(prediction,truth)];n=len(errors)
    sse=math.fsum(x*x for x in errors);mean=math.fsum(map(float,truth))/n
    sst=math.fsum((float(v)-mean)**2 for v in truth)
    return dict(RMSE_mV=math.sqrt(sse/n)*1000,MAE_mV=math.fsum(abs(x) for x in errors)/n*1000,
        R2=1-sse/sst,P99_mV=float(np.quantile(np.abs(errors)*1000,.99)))


def verify(run,stage):
    environment=f.configure_environment()
    f.model_identity();frame,windows,scaler,prepared,excluded=f.load_selection_data()
    f.require(f.sha(f.SOURCE)==f.read_json(run/'PROTOCOL.json')['data_sha256'],'INDEPENDENT_DATA_SHA_CHANGED')
    frozen=f.read_json(run/'CODE_BEFORE_A.json')
    for name,digest in frozen.items():f.require(f.sha(f.HERE/name)==digest,'AUDIT_FROZEN_SOURCE_CHANGED:'+name)
    folders=sorted((run/'models'/stage).iterdir());rows=[];changes=[];precision=[];kernel_gradients=[];event_rows=[]
    f.require(folders,'NO_MODELS_TO_AUDIT')
    f.require(len(folders)==dict(A=9,B=6,C=6,D=6)[stage],'INDEPENDENT_FULL_STAGE_INSTANCE_COUNT')
    if stage!='A':
        registered=f.read_json(run/f'{stage}_REGISTRATION.json')
        for name,digest in registered['source_sha256'].items():
            f.require(f.sha(f.HERE/name)==digest,'INDEPENDENT_STAGE_SOURCE_CHANGED:'+name)
    report=dict(status='RUNNING',stage=stage,instances=0,holdout_windows_or_metrics_used=False,
        environment=environment,
        source_scope='Full 3843-row contract read; only Train and Validation windows constructed or evaluated')
    path=run/f'audits/{stage}_INDEPENDENT_AUDIT.json';js(path,report)
    try:
        for folder in folders:
            ledger=f.read_json(folder/'TRAINING_LEDGER.json');config=ledger['configuration']
            f.require(ledger['status']=='TRAINED' and ledger['fresh_initialization'] and not ledger['historical_checkpoint_loaded'],'NOT_FRESH')
            f.require(ledger['optimizer_initial_state_empty'],'OPTIMIZER_WARMSTART')
            history=f.read_csv(folder/'training_history.csv');best=math.inf;epoch_best=stale=0
            for row in history.itertuples():
                score=math.sqrt(row.validation_MSE)
                if score<best-1e-12:best,epoch_best,stale=score,row.epoch,0
                else:stale+=1
                f.require(row.best_epoch==epoch_best and row.stale_epochs==stale,'EARLY_STOPPING_TRACE')
                f.require(row.optimizer_steps==row.backward_calls==55*row.epoch,'UPDATE_COUNTS')
                f.require(math.isfinite(row.gradient_norm_max) and math.isfinite(row.gradient_norm_min),'NONFINITE_TRAINING_GRADIENT')
                f.require(0<=row.clipped_fraction<=1,'CLIP_RATE')
            f.require(epoch_best==ledger['best_epoch'] and len(history)==ledger['stop_epoch'],'BEST_OR_STOP_EPOCH')
            f.require(len(history)==450 or stale==40,'STOPPING_POLICY')
            f.require(ledger['first_update_sha256']!=ledger['initialization_sha256'],'NO_FIRST_UPDATE')
            f.require(f.sha(folder/'best_model.pt')==ledger['best_checkpoint_sha256'],'BEST_SHA')
            checkpoint=torch.load(folder/'best_model.pt',map_location='cpu',weights_only=True)
            for key in ['run_id','stage','candidate','seed','configuration','initialization_sha256','best_epoch']:
                f.require(checkpoint[key]==ledger[key],'CHECKPOINT_META:'+key)
            initial=d.new_model(ledger['seed'],config)
            f.require(f.state_sha(initial)==ledger['initialization_sha256'],'INITIALIZATION_RECONSTRUCTION')
            initial_state={name:v.detach().cpu().clone() for name,v in initial.state_dict().items()}
            initial.load_state_dict(checkpoint['state_dict'],strict=True);model=initial
            groups=[('CNN','history_branch.'),('CNN_short','history_branch.short.'),('CNN_medium','history_branch.medium.'),
                ('CNN_pointwise','history_branch.pointwise.'),('CNN_norm','history_branch.norm.'),('LSTM_encoder','lstm_encoder.'),('gate','regime_gate.')]
            for name,prefix in groups:
                values=[(checkpoint['state_dict'][key]-v).double().square().sum().item() for key,v in initial_state.items() if key.startswith(prefix)]
                changes.append(dict(stage=stage,candidate=ledger['candidate'],seed=ledger['seed'],group=name,
                    parameter_change_L2=math.sqrt(math.fsum(values))))
            metric=f.read_csv(folder/'METRICS.csv')
            for part,w in windows.items():
                p=f.read_csv(folder/f'predictions_{part}.csv')
                f.require(np.array_equal(p.origin_row,np.repeat(w.origins,24)) and np.array_equal(p.target_row,w.targets.ravel()),'QUERY_KEYS')
                f.require(np.array_equal(p.horizon_step,np.tile(np.arange(1,25),len(w.origins))),'HORIZON_IDENTITY')
                f.require(np.array_equal(p.y_true_delta_V,w.y.ravel().astype(float)),'LABEL_IDENTITY')
                f.require(np.array_equal(p.origin_raw_voltage_V,np.repeat(w.origin_voltage,24)),'RAW_ORIGIN')
                f.require(np.array_equal(p.true_voltage_V,w.target_voltage.ravel()),'RAW_TARGET')
                f.require(np.array_equal(p.target_age_h,w.target_age_h.ravel()) and np.array_equal(p.origin_age_h,np.repeat(w.origin_age_h,24)),'OPERATING_CLOCKS')
                f.require(np.array_equal(p.predicted_voltage_V,p.origin_raw_voltage_V+p.y_pred_delta_V),'ABSOLUTE_VOLTAGE')
                values=independent_metrics(p);stored=metric.loc[metric.partition==part].iloc[0]
                f.require(all(values[key]==stored[key] for key in ['RMSE_mV','MAE_mV','R2']),'INDEPENDENT_PRIMARY_METRICS')
                # The frozen training implementation uses float32 pandas error
                # columns for tails, then pandas' float64 interpolation. Primary
                # RMSE/MAE/R2 already cast the input arrays to float64. Rebuild
                # that dtype path explicitly, and ALSO expose the independent
                # all-float64 tail value. No tolerance is applied to identities.
                e32=(p.y_pred_delta_V.to_numpy(np.float32)-p.y_true_delta_V.to_numpy(np.float32))*np.float32(1000)
                reconstructed=float(np.quantile(np.abs(e32).astype(float),.99))
                f.require(float(np.quantile(np.abs(e32).astype(float),.95))==float(stored['P95_mV']),'EXACT_P95_RECONSTRUCTION')
                f.require(float(np.abs(e32).max())==float(stored['max_abs_mV']),'EXACT_MAX_ERROR_RECONSTRUCTION')
                difference=values['P99_mV']-float(stored['P99_mV'])
                precision.append(dict(stage=stage,candidate=ledger['candidate'],seed=ledger['seed'],partition=part,
                    P99_all_float64_mV=values['P99_mV'],P99_frozen_float32_error_mV=float(stored['P99_mV']),
                    P99_difference_mV=difference,reconstructed_float32_P99_difference_mV=reconstructed-float(stored['P99_mV'])))
                f.require(reconstructed==float(stored['P99_mV']),'EXACT_FROZEN_TAIL_DTYPE_RECONSTRUCTION')
                # Independent event definition at these legal targets only;
                # target conditions are diagnostics, never predictor inputs.
                target=p.target_row.to_numpy(int)
                at=frame.iloc[target];previous=frame.iloc[target-1]
                zero=at.zero_current_fraction.to_numpy(float)
                jump=abs(at.cell_voltage_mean.to_numpy(float)-previous.cell_voltage_mean.to_numpy(float))
                continuous=(at.temporal_segment_id.to_numpy()==previous.temporal_segment_id.to_numpy()) & (abs(
                    at.bin_end_ophrs2_h.to_numpy(float)-previous.bin_end_ophrs2_h.to_numpy(float)-.5)<1e-6)
                cutoff=f.read_json(run/'PROTOCOL.json')['event_definition']['Train_jump_P95_V']
                event=(at.startup_flag.to_numpy()>0)|(at.shutdown_flag.to_numpy()>0)|((zero>0)&(zero<1))|(continuous&(jump>=cutoff))
                for group,mask in [('event',event),('non_event',~event)]:
                    group_values=independent_metrics(p.loc[mask])
                    for key in ['RMSE_mV','MAE_mV','R2']:
                        f.require(group_values[key]==float(stored[group+'_'+key]),'INDEPENDENT_GROUP_METRIC:'+group+key)
                    f.require(int(mask.sum())==int(stored[group+'_queries']),'GROUP_QUERY_COUNT')
                    f.require(p.loc[mask,'target_row'].nunique()==int(stored[group+'_unique_targets']),'GROUP_TARGET_COUNT')
                    event_rows.append(dict(stage=stage,candidate=ledger['candidate'],seed=ledger['seed'],partition=part,
                        group=group,queries=int(mask.sum()),unique_targets=int(p.loc[mask,'target_row'].nunique()),**group_values))
                if part=='VALIDATION':
                    replay,_=d.predict(model,prepared[part]);replay*=scaler.target_scale
                    f.require(np.array_equal(replay.ravel().astype(float),p.y_pred_delta_V),'EXACT_BEST_CHECKPOINT_REPLAY')
                rows.append(dict(stage=stage,candidate=ledger['candidate'],seed=ledger['seed'],partition=part,**values))
            if stage=='A':
                f.require(config==s.CONFIGS_A[ledger['candidate']],'A_CONFIG')
                f.require(ledger['parameters']==332436,'A_MODEL_SIZE')
            else:
                f.require(config==registered['configurations'][ledger['candidate']],'REGISTERED_STAGE_CONFIG')
                f.require(ledger['parameters']==332436,'STAGE_MODEL_SIZE')
            # Read-only diagnostic of actual kernel connectivity at this best state.
            # cuDNN RNN backward requires train mode; dropout is on, updates remain zero.
            before=f.state_sha(model);model.train();model.zero_grad(set_to_none=True)
            x=torch.from_numpy(prepared['TRAIN'][:32]).cuda()
            y=torch.from_numpy((windows['TRAIN'].y[:32]/scaler.target_scale).astype('float32')).cuda()
            loss=torch.nn.functional.mse_loss(model.predict_delta(x),y);loss.backward()
            for group,prefix in groups:
                parameters=[p for name,p in model.named_parameters() if name.startswith(prefix)]
                norm=d.gradient_norm(parameters)
                f.require(norm is None or math.isfinite(norm),'KERNEL_GRADIENT_NOT_FINITE')
                kernel_gradients.append(dict(stage=stage,candidate=ledger['candidate'],seed=ledger['seed'],group=group,
                    gradient_L2=norm,optimizer_updates=0,mode='Train-mode dropout-on unweighted first Train batch at saved best weights'))
            f.require(f.state_sha(model)==before,'GRADIENT_DIAGNOSTIC_CHANGED_WEIGHTS')
            model.zero_grad(set_to_none=True);d.clear(model)
            del model,initial,checkpoint,initial_state;torch.cuda.empty_cache();report['instances']+=1
        csv(run/f'audits/{stage}_INDEPENDENT_METRICS.csv',rows)
        csv(run/f'audits/{stage}_PATH_PARAMETER_CHANGES.csv',changes)
        csv(run/f'audits/{stage}_QUANTILE_PRECISION.csv',precision)
        csv(run/f'audits/{stage}_KERNEL_GRADIENTS_NO_UPDATE.csv',kernel_gradients)
        csv(run/f'audits/{stage}_INDEPENDENT_EVENT_METRICS.csv',event_rows)
        report.update(status='PASS',fresh_updates_verified=True,unweighted_Validation_early_stopping_verified=True,
            every_primary_Train_Validation_metric_exact=True,P99_max_precision_difference_mV=max(abs(r['P99_difference_mV']) for r in precision),Validation_checkpoint_replays_exact=True,
            frozen_float32_P99_reconstructed_exactly=True,
            P95_and_max_error_dtype_reconstructed_exactly=True,event_non_event_primary_metrics_exact=True,
            tail_precision_note='Full float64 error quantile is independently reported, not substituted into frozen selection',
            original_F2_AST_exact=True,selection_only=True,holdout_windows_or_metrics_used=False)
        js(path,report);print(stage,'INDEPENDENT_AUDIT_PASS',len(folders),'fresh instances',flush=True)
        return report
    except Exception as error:
        report.update(status='FAILURE_PRESERVED',error=repr(error));js(path,report);raise
