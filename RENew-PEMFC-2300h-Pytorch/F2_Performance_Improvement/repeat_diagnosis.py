"""Use only the planned A0 and B_control repeats; never adds GPU training."""
import math
import numpy as np
import pandas as pd
import frozen_f2 as f
import settings as s
from io_utils import csv


def compare(run):
    rows=[]
    for seed in s.INITIAL_SEEDS:
        a=run/f'models/A/A0_seed{seed}';b=run/f'models/B/B_control_seed{seed}'
        ma=f.read_json(a/'BEFORE_TRAINING.json');mb=f.read_json(b/'BEFORE_TRAINING.json')
        if ma['configuration']!=mb['configuration']:
            # A future independent run may qualify an A training change. In that
            # case these are NOT same-configuration repeatability replicates.
            rows.append(dict(seed=seed,same_configuration=False,numerical_repeat_comparison='NA'))
            continue
        ha=f.read_csv(a/'training_history.csv');hb=f.read_csv(b/'training_history.csv');n=min(len(ha),len(hb))
        sa=f.read_csv(a/'METRICS.csv').query("partition=='VALIDATION'").iloc[0]
        sb=f.read_csv(b/'METRICS.csv').query("partition=='VALIDATION'").iloc[0]
        state_equal=ha.state_sha256.iloc[:n].to_numpy()==hb.state_sha256.iloc[:n].to_numpy()
        first=np.flatnonzero(~state_equal)
        loss_delta=hb.validation_MSE.iloc[:n].to_numpy(float)-ha.validation_MSE.iloc[:n].to_numpy(float)
        rows.append(dict(seed=seed,same_configuration=True,initialization_equal=ma['initialization_sha256']==mb['initialization_sha256'],
            Train_input_SHA_equal=ma['train_inputs_sha256']==mb['train_inputs_sha256'],
            Validation_input_SHA_equal=ma['validation_inputs_sha256']==mb['validation_inputs_sha256'],
            common_epochs=n,common_epoch_sampler_SHA_equal=bool(np.array_equal(ha.sampler_order_sha256.iloc[:n],hb.sampler_order_sha256.iloc[:n])),
            common_epoch_recorded_CPU_CUDA_RNG_SHA_equal=bool(np.array_equal(ha.rng.iloc[:n],hb.rng.iloc[:n])),
            first_different_epoch_state=int(first[0]+1) if len(first) else None,
            first_epoch_Validation_MSE_difference=float(loss_delta[0]),max_common_epoch_Validation_MSE_difference=float(abs(loss_delta).max()),
            A0_Validation_RMSE_mV=float(sa.RMSE_mV),B_control_Validation_RMSE_mV=float(sb.RMSE_mV),
            RMSE_difference_mV=float(sb.RMSE_mV-sa.RMSE_mV),
            A0_Validation_MAE_mV=float(sa.MAE_mV),B_control_Validation_MAE_mV=float(sb.MAE_mV),
            A0_best_epoch=int(ha.best_epoch.iloc[-1]),B_control_best_epoch=int(hb.best_epoch.iloc[-1]),
            A0_stop_epoch=len(ha),B_control_stop_epoch=len(hb)))
    result=pd.DataFrame(rows)
    csv(run/'tables/B_PLANNED_NATIVE_REPEATABILITY.csv',result)
    return result
