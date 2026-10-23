"""Independent numerical and new-checkpoint replay audit; never trains a model."""
from pathlib import Path
import math
import numpy as np
import pandas as pd
import torch
from PIL import Image
import core as c
from io_utils import js,csv


def recompute(p):
    truth=p.y_true_delta_V.to_numpy(float);error=p.y_pred_delta_V.to_numpy(float)-truth;n=len(truth)
    squares=math.fsum(float(v)**2 for v in error);mean=math.fsum(map(float,truth))/n
    return dict(RMSE_mV=math.sqrt(squares/n)*1000,MAE_mV=math.fsum(abs(float(v)) for v in error)/n*1000,
        R2=1-squares/math.fsum((float(v)-mean)**2 for v in truth))


def replay(model,x):
    result=[];model.eval()
    with torch.no_grad():
        for first in range(0,len(x),32):result.append(model.predict_delta(torch.from_numpy(x[first:first+32]).cuda()).cpu().numpy())
    return np.concatenate(result)


def verify(run):
    print('P5: independently verify labels, fresh updates, early stopping, metrics and checkpoints',flush=True)
    out=run/'audits';out.mkdir(exist_ok=True)
    report=dict(status='RUNNING',formal_models=0,pilot_models=0)
    js(run/'P5_AUDIT.json',report)
    try:
        c.model_identity();frame,w,scaler,prepared,excluded=c.load_data()
        protocol=c.read_json(run/'FROZEN_PROTOCOL.json')
        c.require(c.sha(c.SOURCE)==protocol['data_sha256'],'AUDIT_SOURCE_CHANGED')
        c.require(not (run/'INVALIDATED_BEFORE_TRAINING.json').exists(),'INVALIDATED_RUN')
        for entry in protocol['partitions']:
            part=entry['partition'];a=w[part];c.require(c.array_sha(prepared[part])==entry['inputs_sha256'],'AUDIT_INPUTS')
            # Rebuild labels directly from the raw source, with the original float32 target contract.
            voltage=frame.cell_voltage_mean.to_numpy(float)
            raw=(voltage[a.targets]-voltage[a.origins,None]).astype('float32')
            c.require(np.array_equal(raw,a.y),'DIRECT_RAW_LABEL_RECONSTRUCTION')
        saved_metrics=c.read_csv(run/'tables/PARTITION_METRICS.csv');all_metrics=[];training_rows=[];replay_differences=[]
        initializations={}
        for cohort,seeds in [('pilot',[202]),('formal',list(c.SUPPORTED_SEEDS))]:
            for seed in seeds:
                histories={}
                for kind in ['LSTM','F2']:
                    folder=run/'models'/cohort/f'{kind}_seed{seed}';ledger=c.read_json(folder/'TRAINING_LEDGER.json')
                    c.require(ledger['status']=='TRAINED' and ledger['fresh_initialization'] and not ledger['historical_checkpoint_loaded'],'NOT_FRESH_TRAINED')
                    c.require(ledger['optimizer_initial_state_empty'],'NONEMPTY_OPTIMIZER')
                    history=c.read_csv(folder/'training_history.csv');histories[kind]=history
                    best=math.inf;epoch_best=stale=0
                    for row in history.itertuples():
                        score=math.sqrt(row.validation_standardized_MSE)
                        if score<best-1e-12:best,epoch_best,stale=score,row.epoch,0
                        else:stale+=1
                        c.require(row.best_epoch==epoch_best and row.stale_epochs==stale,'EARLY_STOP_SEQUENCE')
                        c.require(row.optimizer_steps==row.backward_calls==55*row.epoch,'REAL_UPDATE_COUNTS')
                        c.require(math.isfinite(row.gradient_norm_min) and math.isfinite(row.gradient_norm_max),'GRADIENT_FINITE')
                    c.require(epoch_best==ledger['best_epoch'] and len(history)==ledger['stop_epoch'],'BEST_EPOCH_METADATA')
                    c.require(len(history)==450 or stale==40,'EARLY_STOP_POLICY')
                    c.require(ledger['initialization_sha256']!=ledger['first_update_sha256'] and ledger['initialization_sha256']!=ledger['final_state_sha256'],'NO_WEIGHT_UPDATES')
                    path=folder/'best_model.pt';c.require(c.sha(path)==ledger['best_checkpoint_sha256'],'CHECKPOINT_SHA')
                    checkpoint=torch.load(path,map_location='cpu',weights_only=True)
                    for key in ['run_id','kind','seed','cohort','best_epoch','prepared_train_sha256']:
                        c.require(checkpoint[key]==ledger[key],'CHECKPOINT_META:'+key)
                    c.require(checkpoint['run_id']==run.name and checkpoint['optimizer_steps']==55*epoch_best,'CHECKPOINT_TRAINING_ID')
                    key=(kind,seed)
                    if key in initializations:c.require(initializations[key]==ledger['initialization_sha256'],'PILOT_FORMAL_INIT_CHANGED')
                    initializations[key]=ledger['initialization_sha256']
                    parts=['VALIDATION'] if cohort=='pilot' else list(w)
                    model=c.new_model(kind,seed);model.load_state_dict(checkpoint['state_dict'],strict=True)
                    for part in parts:
                        p=c.read_csv(folder/f'predictions_{part}.csv');a=w[part]
                        c.require(np.array_equal(p.origin_row,np.repeat(a.origins,24)) and np.array_equal(p.target_row,a.targets.ravel()),'PREDICTION_KEYS')
                        c.require(np.array_equal(p.horizon_step,np.tile(np.arange(1,25),len(a.origins))),'PREDICTION_STEP_IDENTITY')
                        c.require(np.array_equal(p.origin_age_h,np.repeat(a.origin_age_h,24)) and np.array_equal(p.target_age_h,a.target_age_h.ravel()),'PREDICTION_OPERATING_CLOCKS')
                        c.require(np.array_equal(p.true_voltage_V,a.target_voltage.ravel()),'TRUE_RAW_TARGET_VOLTAGE')
                        c.require(np.array_equal(p.y_true_delta_V,a.y.ravel().astype(float)),'PREDICTION_LABELS')
                        c.require(np.array_equal(p.origin_raw_voltage_V,np.repeat(a.origin_voltage,24)),'RAW_ORIGIN_VOLTAGE')
                        c.require(np.array_equal(p.predicted_voltage_V,p.origin_raw_voltage_V+p.y_pred_delta_V),'ABSOLUTE_VOLTAGE_RESTORATION')
                        values=recompute(p)
                        if cohort=='formal':
                            stored=saved_metrics.loc[(saved_metrics.model==kind)&(saved_metrics.seed==seed)&(saved_metrics.partition==part)].iloc[0]
                            c.require(all(values[k]==stored[k] for k in values),'INDEPENDENT_METRIC_RECOMPUTATION')
                            all_metrics.append(dict(model=kind,seed=seed,partition=part,**values))
                        # Validation +holdout replay only; Train labels/metrics are still audited above.
                        if part in ['VALIDATION','TEST']:
                            prediction=replay(model,prepared[part])*scaler.target_scale
                            maximum=float(np.max(np.abs(prediction.ravel().astype(float)-p.y_pred_delta_V.to_numpy(float))))
                            replay_differences.append(dict(cohort=cohort,model=kind,seed=seed,partition=part,max_abs_deltaV_difference=maximum))
                            c.require(np.array_equal(prediction.ravel().astype(float),p.y_pred_delta_V),'NEW_CHECKPOINT_REPLAY_NOT_EXACT')
                    del model,checkpoint;torch.cuda.empty_cache()
                    training_rows.append(dict(cohort=cohort,model=kind,seed=seed,best_epoch=epoch_best,stop_epoch=len(history),
                        optimizer_steps=ledger['optimizer_steps'],training_seconds=ledger['training_seconds'],checkpoint_sha256=ledger['best_checkpoint_sha256']))
                    report[cohort+'_models']+=1
                n=min(len(h) for h in histories.values())
                c.require(np.array_equal(histories['LSTM'].sampler_order_sha256.iloc[:n],histories['F2'].sampler_order_sha256.iloc[:n]),'SAME_SEED_DIFFERENT_SAMPLER')
        repeats=c.read_csv(run/'formal_diagnostics/PILOT_FORMAL_202_VALIDATION_REPEATS.csv')
        for row in repeats.itertuples():
            values=recompute(c.read_csv(run/f'models/{row.cohort}/{row.model}_seed202/predictions_VALIDATION.csv'))
            c.require(all(getattr(row,key)==value for key,value in values.items()),'REPEAT_VALIDATION_METRICS')
        means=c.read_csv(run/'tables/MODEL_MEAN_SD.csv');paired=c.read_csv(run/'tables/PAIRED_DIFFERENCES.csv')
        errors=[]
        for kind in ['LSTM','F2']:
            values=pd.DataFrame([r for r in all_metrics if r['model']==kind and r['partition']=='TEST']);meanrow=means.loc[means.model==kind].iloc[0]
            for key in ['RMSE_mV','MAE_mV','R2']:
                c.require(meanrow[key+'_mean']==math.fsum(values[key])/6,'EQUAL_SEED_MEAN')
                precision_difference=float(meanrow[key+'_sample_sd']-values[key].std(ddof=1));errors.append(dict(model=kind,field=key+'_sample_sd',difference=precision_difference))
                c.require(abs(precision_difference)<1e-10,'SAMPLE_SD_DIFFERENCE')
            c.require(sorted(values.seed.tolist())==sorted(c.SUPPORTED_SEEDS),'SIX_SEED_SET')
        for row in paired.itertuples():
            for key in ['RMSE_mV','MAE_mV','R2']:
                a=next(v for v in all_metrics if v['model']=='LSTM' and v['seed']==row.seed and v['partition']=='TEST')[key]
                b=next(v for v in all_metrics if v['model']=='F2' and v['seed']==row.seed and v['partition']=='TEST')[key]
                c.require(getattr(row,key+'_F2_minus_LSTM')==b-a,'PAIRED_DIFFERENCE')
        figures=[]
        for path in sorted((run/'figures').rglob('*.png')):
            with Image.open(path) as image:image.verify()
            with Image.open(path) as image:c.require(min(image.info['dpi'])>=299,'FIGURE_DPI')
            c.require(path.with_name(path.stem+'_data.csv').exists(),'FIGURE_SOURCE_CSV')
            figures.append(path.relative_to(run).as_posix())
        c.require(len(list((run/'figures/P3').glob('*.png')))==12,'P3_FIGURE_COUNT')
        csv(out/'ALL_FRESH_TRAINING_RECORDS.csv',training_rows);csv(out/'CHECKPOINT_REPLAY_DIFFERENCES.csv',replay_differences)
        csv(out/'NUMERICAL_PRECISION_DIFFERENCES.csv',errors)
        report.update(status='PASS',formal_models=12,pilot_models=2,actual_total_optimizer_updates=sum(r['optimizer_steps'] for r in training_rows),
            exact_model_and_sample_identity=True,all_point_metrics_recomputed=True,all_new_checkpoint_replays_exact=True,
            native_F2_AST_exact=True,all_models_trained_from_scratch=True,no_extra_optimization=True,figures_verified=figures,
            pilot_formal_validation_repeats_recomputed=True,strict_GPU_determinism_claimed=False,holdout_scientific_status='exploratory, historically exposed; not independent blind Test',
            pretraining_unit_error_preserved_and_corrected=True)
        js(run/'P5_AUDIT.json',report)
        print('P5 INDEPENDENT_AUDIT_PASS: 14fresh trainings,12formal models, all new checkpoint replays exact',flush=True)
        return report
    except Exception as error:
        report.update(status='FAILURE_PRESERVED',error=repr(error));js(run/'P5_AUDIT.json',report);raise


def artifact_manifest():
    """Run only after final reports, notebook and history-protection checks finish."""
    run=c.HERE/'outputs'/c.read_json(c.HERE/'ACTIVE_DATA_RUN.json')['run']
    c.require(c.read_json(run/'EXECUTION.json')['status']=='PIPELINE_COMPLETED_HISTORY_UNCHANGED','PIPELINE_NOT_COMPLETE')
    c.require(c.read_json(run/'P5_AUDIT.json')['status']=='PASS','P5_NOT_PASSED')
    rows=[]
    for path in sorted(c.HERE.rglob('*')):
        if not path.is_file() or '__pycache__' in path.parts or path.name in ['ARTIFACT_SHA256.csv','GIT_DELIVERY.json']:continue
        rows.append(dict(path=path.relative_to(c.HERE).as_posix(),bytes=path.stat().st_size,sha256=c.sha(path)))
    csv(c.HERE/'ARTIFACT_SHA256.csv',rows)
    print('ARTIFACT_MANIFEST',len(rows),'files;',sum(r['bytes'] for r in rows),'bytes; SHA256',c.sha(c.HERE/'ARTIFACT_SHA256.csv'),flush=True)
    return rows


if __name__=='__main__':
    run=c.HERE/'outputs'/c.read_json(c.HERE/'ACTIVE_DATA_RUN.json')['run']
    import sys
    if '--manifest' in sys.argv:artifact_manifest()
    else:c.configure_environment();verify(run)
