"""Stage A experiment and decision code frozen before its first optimizer update."""
import math
import numpy as np
import pandas as pd
import frozen_f2 as f
import settings as s
from selection import collect,qualify,choose
from train import train_one
from io_utils import js,csv


def run_stage(run,frame,windows,scaler,prepared):
    f.require(f.read_json(run/'P0_ACCEPTANCE.json')['status']=='PASS','P0_REQUIRED')
    f.require(f.read_json(run/'historical/IDENTITY_AUDIT.json')['status']=='PASS','READONLY_DIAGNOSIS_REQUIRED')
    if (run/'A_RESULT.json').exists():return f.read_json(run/'A_RESULT.json')
    done=[]
    for seed in s.INITIAL_SEEDS:
        for candidate,config in s.CONFIGS_A.items():
            folder=run/'models/A'/f'{candidate}_seed{seed}'
            if folder.exists():
                f.require((folder/'EVALUATION_AUDIT.json').exists() and f.read_json(folder/'EVALUATION_AUDIT.json')['status']=='PASS','PARTIAL_INSTANCE_REQUIRES_REVIEW')
                ledger=f.read_json(folder/'TRAINING_LEDGER.json')
                f.require(ledger['configuration']==config and ledger['status']=='TRAINED','RESUME_IDENTITY')
            else:ledger=train_one(run,'A',candidate,seed,config,frame,windows,scaler,prepared)
            done.append(ledger);js(run/'A_PROGRESS.json',dict(completed=len(done),maximum=9,models=done))
        histories=[f.read_csv(run/f'models/A/{candidate}_seed{seed}/training_history.csv') for candidate in s.CONFIGS_A]
        n=min(map(len,histories))
        for h in histories[1:]:f.require(np.array_equal(h.sampler_order_sha256.iloc[:n],histories[0].sampler_order_sha256.iloc[:n]),'A_SAMPLER_MISMATCH')
        initial=[f.read_json(run/f'models/A/{candidate}_seed{seed}/BEFORE_TRAINING.json')['initialization_sha256'] for candidate in s.CONFIGS_A]
        f.require(len(set(initial))==1,'A_INITIALIZATION_MISMATCH')
    table=collect(run,'A');f.require(len(table)==9,'ALL_NINE_REQUIRED')
    control=table.loc[table.candidate=='A0']
    decisions={name:qualify(table.loc[table.candidate==name],control) for name in ['A1','A2']}
    qualified=[name for name,result in decisions.items() if result['qualified']]
    selected=choose(table,qualified) or 'A0'
    paths=[];cnn_nonzero=[];lstm_nonzero=[]
    for seed in s.INITIAL_SEEDS:
        folder=run/f'models/A/A0_seed{seed}'
        p=f.read_csv(folder/'paths_TRAIN.csv');paths.append(p)
        h=f.read_csv(folder/'training_history.csv')
        cnn_nonzero.append(bool(h.cnn_parameter_grad_norm.gt(0).any()))
        lstm_nonzero.append(bool(h.lstm_encoder_parameter_grad_norm.gt(0).any()))
    p=pd.concat(paths,ignore_index=True)
    b_checks=dict(all_A0_CNN_have_nonzero_training_gradient=all(cnn_nonzero),
        all_A0_LSTM_have_nonzero_training_gradient=all(lstm_nonzero),
        median_gate_below_095=bool(p.g.median()<s.STAGE_B_GATE['median_A0_g_max']),
        mean_fusion_to_LSTM_RMS_below_098=bool(p.fusion_to_lstm_rms.mean()<s.STAGE_B_GATE['mean_A0_fused_to_lstm_RMS_max']))
    result=dict(status='A_COMPLETE',fresh_instances=9,qualified_candidates=qualified,decisions=decisions,
        selected_training_configuration=selected,selected_configuration=s.CONFIGS_A[selected],
        B_motivation_qualified=all(b_checks.values()),B_motivation_checks=b_checks,
        A0_median_gate=float(p.g.median()),A0_mean_fusion_to_LSTM_RMS=float(p.fusion_to_lstm_rms.mean()),
        holdout_metrics_used=False,selection_scope='three seeds, reused Validation; exploratory consistency gate, not statistical significance')
    csv(run/'tables/A_VALIDATION_PER_SEED.csv',table)
    means=[]
    for name,g in table.groupby('candidate'):
        means.append(dict(candidate=name,n=3,**{key+'_mean':math.fsum(g[key])/3 for key in ['RMSE_mV','MAE_mV','R2','P99_mV']},
            **{key+'_SD':float(g[key].std(ddof=1)) for key in ['RMSE_mV','MAE_mV','R2','P99_mV']}))
    csv(run/'tables/A_VALIDATION_MEAN_SD.csv',means);js(run/'A_RESULT.json',result)
    print('A_COMPLETE',result,flush=True)
    from report_a import write
    write(run,table,pd.DataFrame(means),result)
    return result
