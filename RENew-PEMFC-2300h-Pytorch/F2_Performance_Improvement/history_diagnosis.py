"""Historical Train/Validation replay and no-update gradient diagnosis only."""
import numpy as np
import pandas as pd
import torch
import frozen_f2 as f
import diagnostics as d
import settings as s
from train import grouped_metrics
from io_utils import js,csv


def analyze(run,frame,windows,scaler,prepared):
    rows=[];paths=[];gradients=[];curves=[];counter=[];identities=[]
    threshold=f.read_json(run/'PROTOCOL.json')['event_definition']['Train_jump_P95_V']
    for seed in s.FINAL_SEEDS:
        for kind in ['LSTM','F2']:
            folder=f.REFERENCE_RUN/'models/formal'/f'{kind}_seed{seed}'
            ledger=f.read_json(folder/'TRAINING_LEDGER.json');history=f.read_csv(folder/'training_history.csv')
            validation=history.validation_standardized_MSE.to_numpy(float)
            curves.append(dict(model=kind,seed=seed,best_epoch=ledger['best_epoch'],stop_epoch=ledger['stop_epoch'],
                validation_spike_fraction=float(np.mean(validation[1:]>2*validation[:-1])),
                maximum_global_gradient_norm=float(history.gradient_norm_max.max()),
                epochs_with_max_gradient_above_clip=int((history.gradient_norm_max>1).sum()),
                fraction_batches_clipped=None,training_seconds=ledger['training_seconds']))
            for part,w in windows.items():
                p=f.read_csv(folder/f'predictions_{part}.csv')
                f.require(np.array_equal(p.origin_row,np.repeat(w.origins,24)) and np.array_equal(p.target_row,w.targets.ravel()),'HISTORICAL_KEYS')
                f.require(np.array_equal(p.y_true_delta_V,w.y.ravel().astype(float)),'HISTORICAL_LABELS')
                rows.append(dict(model=kind,seed=seed,partition=part,**grouped_metrics(p,frame,threshold)))
            if kind=='LSTM':continue
            print('Read-only historical F2 Train/Validation diagnostic Seed',seed,flush=True)
            checkpoint=folder/'best_model.pt';digest=f.sha(checkpoint)
            f.require(digest==ledger['best_checkpoint_sha256'],'HISTORICAL_WEIGHT_SHA')
            model=d.new_model(seed,s.CONFIGS_A['A0'])
            saved=torch.load(checkpoint,map_location='cpu',weights_only=True);model.load_state_dict(saved['state_dict'],strict=True)
            before=f.state_sha(model)
            for part,w in windows.items():
                prediction,observed=d.predict(model,prepared[part],observe=True)
                cached=f.read_csv(folder/f'predictions_{part}.csv').y_pred_delta_V.to_numpy(float)
                f.require(np.array_equal((prediction*scaler.target_scale).ravel().astype(float),cached),'HISTORICAL_REPLAY_IDENTITY')
                p=pd.DataFrame(observed);p['seed']=seed;p['partition']=part;p['origin_row']=w.origins;p['origin_age_h']=w.origin_age_h
                csv(run/f'historical/paths_seed{seed}_{part}.csv',p)
                paths.append(dict(seed=seed,partition=part,**d.summarize_paths(observed)))
            # cuDNN RNN backward requires train mode. No optimizer or update exists.
            model.train();model.zero_grad(set_to_none=True)
            x=torch.from_numpy(prepared['TRAIN'][:32]).cuda();y=torch.from_numpy((windows['TRAIN'].y[:32]/scaler.target_scale).astype('float32')).cuda()
            loss=torch.nn.functional.mse_loss(model.predict_delta(x),y);loss.backward()
            gradients.append(dict(seed=seed,mode='train_mode_dropout_on_first_Train_batch_no_optimizer',optimizer_steps=0,**d.gradient_row(model)))
            model.zero_grad(set_to_none=True);d.clear(model)
            try:
                for mode in ['cnn_off','lstm_only']:
                    model._fusion_mode=mode;prediction,_=d.predict(model,prepared['VALIDATION'])
                    p=f.prediction_frame(frame,windows['VALIDATION'],prediction*scaler.target_scale,seed,'top10','VALIDATION')
                    counter.append(dict(seed=seed,mode=mode,**grouped_metrics(p,frame,threshold)))
            finally:model._fusion_mode='normal'
            f.require(f.state_sha(model)==before and f.sha(checkpoint)==digest,'HISTORICAL_MODEL_CHANGED')
            identities.append(dict(seed=seed,checkpoint_sha256=digest,replay_Train_and_Validation_exact=True,no_optimizer_update=True))
            del model,saved;torch.cuda.empty_cache()
    csv(run/'historical/TRAIN_VALIDATION_METRICS.csv',rows);csv(run/'historical/PATH_SUMMARY.csv',paths)
    csv(run/'historical/EVAL_GRADIENTS_NO_UPDATE.csv',gradients);csv(run/'historical/TRAINING_STABILITY.csv',curves)
    csv(run/'historical/VALIDATION_COUNTERFACTUALS.csv',counter)
    js(run/'historical/IDENTITY_AUDIT.json',dict(status='PASS',models=identities,holdout_predictions_read=False,historical_weights_unchanged=True))

