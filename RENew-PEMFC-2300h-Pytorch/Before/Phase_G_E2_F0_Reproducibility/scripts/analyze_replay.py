"""No inference/training: compare persisted diagnostic summaries and plot."""
from __future__ import annotations
import json
import math
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from g_common import *
from run_controlled_replay import verify,TRACES

def read_lines(path):
    return [json.loads(s) for s in path.read_text(encoding='utf-8').splitlines() if s]

def figure(fig,name):
    fig.tight_layout(pad=1.4)
    for suffix in ['png','pdf']:fig.savefig(PHASE/'figures'/f'{name}.{suffix}',dpi=300)
    plt.close(fig)

def main():
    verify()
    ledger=read_json(OUT/'execution_audit/RUN_LEDGER.json')
    report=PHASE/'reports';report.mkdir(exist_ok=True)
    if ledger['status']!='COMPLETED':
        summary={'status':'PHASE_G_EXECUTION_BLOCKED','ledger':ledger,'no_extra_training':True,
                 'new_assessment_inference':False,'test_accessed':False}
        write_json(OUT/'G_FINAL_REPRODUCIBILITY_SUMMARY.json',summary)
        (report/'G_FINAL_REPRODUCIBILITY_CONCLUSION.md').write_text('# Phase G execution blocked\n\n'
            +json.dumps(summary,indent=2)+'\n\nStrict deterministic execution was not silently relaxed. '
            +'See trajectory TECHNICAL_FAILURE.json; no historical cause can be inferred. No further runs authorized.\n',encoding='utf-8')
        print(summary['status']);return
    folder=OUT/'trajectory_comparison';folder.mkdir(parents=True,exist_ok=True)
    runs={n:OUT/'replay_logs'/n for n in TRACES}
    epochs={n:read_json(p/'epochs.json') for n,p in runs.items()}
    batches={n:read_lines(p/'batches.jsonl') for n,p in runs.items()}
    finals={n:read_json(p/'final.json') for n,p in runs.items()}
    initials={n:read_json(p/'initial.json') for n,p in runs.items()}
    inputs={n:read_json(p/'input_audit.json') for n,p in runs.items()}
    env={n:read_json(p/'environment.json') for n,p in runs.items()}
    if any(len(v)!=12 for v in epochs.values()) or any(len(v)!=220 for v in batches.values()):
        raise RuntimeError('INCOMPLETE_DIAGNOSTIC_RECORDS')
    if not all(v==inputs[TRACES[0]] for v in inputs.values()):raise RuntimeError('INPUT_IDENTITY_MISMATCH')
    if not all(v==env[TRACES[0]] for v in env.values()):raise RuntimeError('CONTROLLED_ENVIRONMENT_MISMATCH')
    if not all(v==initials[TRACES[0]] for v in initials.values()):raise RuntimeError('INITIAL_STATE_OR_RNG_MISMATCH')
    names=[('G-E-A','G-E-B'),('G-F-A','G-F-B'),('G-E-A','G-F-A')]
    comparisons=[]
    fields=['train_standardized_MSE','validation_standardized_RMSE','validation_prediction_sha256',
        'parameters_after_validation','optimizer_after_validation','rng_before_validation','rng_after_validation','rng_after_best_handling']
    for a,b in names:
        cmp=compare_batches(batches[a],batches[b]);cmp['comparison']=a+' vs '+b
        rows=[];first=None
        for x,y in zip(epochs[a],epochs[b]):
            delta={k:(x[k]!=y[k]) for k in fields}
            if first is None and any(delta.values()):first={'epoch':x['epoch'],'fields':[k for k,v in delta.items() if v]}
            rows.append({'epoch':x['epoch'],'left_train_MSE':x['train_standardized_MSE'],
                'right_train_MSE':y['train_standardized_MSE'],
                'train_abs_difference':abs(x['train_standardized_MSE']-y['train_standardized_MSE']),
                'left_validation_RMSE':x['validation_standardized_RMSE'],
                'right_validation_RMSE':y['validation_standardized_RMSE'],
                'validation_abs_difference':abs(x['validation_standardized_RMSE']-y['validation_standardized_RMSE']),
                'all_epoch_evidence_equal':not any(delta.values())})
        pd.DataFrame(rows).to_csv(folder/f'{a}_vs_{b}_epochs.csv',index=False,float_format='%.17g')
        cmp.update(epoch_evidence_identical=first is None,first_epoch_divergence=first,
            final_state_identical=finals[a]['final_state_sha256']==finals[b]['final_state_sha256'],
            best_state_identical=finals[a]['best_state_sha256']==finals[b]['best_state_sha256'])
        cmp['fully_identical']=cmp['identical'] and cmp['epoch_evidence_identical'] and cmp['final_state_identical'] and cmp['best_state_identical']
        comparisons.append(cmp)
    pd.DataFrame([{'comparison':v['comparison'],'fully_identical':v['fully_identical'],
        'first_batch_divergence':json.dumps(v['first_divergence']),
        'first_epoch_divergence':json.dumps(v['first_epoch_divergence']),
        'audited_batches':v['records']} for v in comparisons]).to_csv(folder/'REPLAY_COMPARISON.csv',index=False)
    history=read_json(OUT/'historical_audit/HISTORICAL_SUMMARY.json')
    if all(v['fully_identical'] for v in comparisons):
        status='PHASE_G_CONTROLLED_REPLAY_CONSISTENT_HISTORICAL_CAUSE_UNRESOLVED'
        cause='Current controlled environment reproduces identical paths; the historical root cause remains unresolved.'
    elif any(not v['fully_identical'] for v in comparisons[:2]):
        status='PHASE_G_REPRODUCIBILITY_UNSTABLE'
        cause='Within-path divergence remains under this controlled runtime; no unique low-level cause established.'
    else:
        status='PHASE_G_EXECUTION_DIFFERENCE_IDENTIFIED'
        cause='Between-path execution evidence differs while within-path repeats agree. See first differing field; no untested historical attribution.'
    rows=[]
    # Current controlled trajectories versus historical first12: descriptive, no tuning.
    ht=pd.read_csv(OUT/'historical_audit/HISTORICAL_EPOCH_COMPARISON.csv',float_precision='round_trip')
    for n in TRACES:
        side='E2' if n.startswith('G-E') else 'F0'
        for row in epochs[n]:
            h=ht.loc[ht.epoch==row['epoch']].iloc[0]
            rows.append({'trajectory':n,'epoch':row['epoch'],'train_abs_difference_from_history':abs(row['train_standardized_MSE']-h[f'{side}_train_MSE']),
                'validation_abs_difference_from_history':abs(row['validation_standardized_RMSE']-h[f'{side}_validation_RMSE'])})
    pd.DataFrame(rows).to_csv(folder/'CONTROLLED_VS_HISTORICAL_FIRST12.csv',index=False,float_format='%.17g')
    summary={'status':status,'historical_first_divergence':history['first_divergence'],'comparisons':comparisons,
        'same_inputs':True,'same_initial_parameters_and_RNG':True,'environment':env[TRACES[0]],
        'diagnostic_epochs_completed':48,'per_trajectory_epochs':12,'instrumented_batches_per_trajectory':220,
        'historical_cause':cause,'historical_E2_low_RMSE_stably_reproduced':False,
        'new_assessment_inference':False,'test_accessed':False,'historical_sources_unchanged':True,
        'extra_training_performed':False,'scientific_results':'DIAGNOSTIC_ONLY_NOT_FORMAL_MODEL',
        'limitations':['historical E runtime evidence incomplete','symmetric audit hashing synchronizes CUDA',
            'strict deterministic runtime differs from historical configuration','12 epochs cannot establish best287 stability',
            'single Seed42, historical Assessment reused']}
    write_json(folder/'BATCH_AND_EPOCH_COMPARISONS.json',comparisons)
    write_json(OUT/'G_FINAL_REPRODUCIBILITY_SUMMARY.json',summary)
    table='|Comparison|Batch summaries|12-epoch evidence|Final state|Earliest divergence|\n|---|---|---|---|---|\n'
    table+='\n'.join(f"|{c['comparison']}|{c['identical']}|{c['epoch_evidence_identical']}|{c['final_state_identical']}|{c['first_divergence'] or c['first_epoch_divergence'] or 'NONE'}|" for c in comparisons)
    (report/'G_CONTROLLED_REPLAY_RESULTS.md').write_text('# Controlled replay\n\n'+table+'\n\n'
        +'48 epochs completed: four fresh processes, Seed42, two independently implemented loops. '
        +'No formal performance replacement, no new Assessment inference.\n\nRuntime:\n\n```json\n'
        +json.dumps(env[TRACES[0]],indent=2)+'\n```\n\n'
        +'E retains a two-field TensorDataset and CPU-cloned best state in memory; F retains a '
        +'three-field dataset, finite-gradient check and on-improvement diagnostic checkpoint write. '
        +'Both paths carry symmetric dataset index tracing, pre/post forward/backward/clip/step hashes. '
        +'These copies synchronize CUDA and can alter scheduling/timing; they do not prove historical equivalence.\n',encoding='utf-8')
    (report/'G_EARLIEST_DIVERGENCE_ANALYSIS.md').write_text('# Earliest divergence\n\n'
        +json.dumps({'history':history['first_divergence'],'controlled':comparisons},indent=2)+'\n\n'
        +'Batch indices are zero-based. The 220 audited batches per trajectory cover all 44 batches '
        +'in epochs1–5, including the last partial batch. Input, truth, origins, parameter states, '
        +'forward outputs/loss, pre/post-clipping gradients, AdamW state and CPU/CUDA/loader/Python/NumPy '
        +'RNG are compared exactly. Validation summaries and RNG before/after validation and best-state '
        +'handling cover all12 epochs. If no field differs, a first differing operation cannot be named. '
        +'Historical epoch aggregates do not locate the historic first divergent batch or operation.\n',encoding='utf-8')
    (report/'G_FINAL_REPRODUCIBILITY_CONCLUSION.md').write_text('# Final reproducibility conclusion\n\n'
        +status+'\n\n## A. Historical facts\n\n'
        +'Same native model, parameter count323483, Seed42, initialization digest, Top10, EWMA(.3), '
        +'Train-only scaler and48→24 task. Epoch1–3 match exactly. First difference at epoch4: '
        +json.dumps(history['first_divergence'])+'. Best checkpoints287 and165 differ as expected. '
        +'Saved Assessment cache keys and true labels agree.\n\n## B. Controlled replay\n\n'+table+'\n\n'
        +'All runs use CPUthreads1, strict deterministic algorithms, CUBLAS :4096:8 configured before '
        +'Torch import, cuDNN deterministic with benchmark disabled, TF32 disabled, shared data and seed. '
        +'This is a controlled intervention bundle, not an isolation of each runtime factor.\n\n'
        +'## C. Cause classification\n\n'+cause+'\n\n'
        +'Proven: history diverges; present controlled comparison outcomes are recorded above. '
        +'Supported but not uniquely identified: historic runtime differences or scheduling/numerical sensitivity '
        +'are candidates only. Unresolved: unique historical cause, E2 historic CPUthreads/TF32/RNG, and '
        +'the exact historical first divergent batch. Do not label CUDA nondeterminism, CPUthreads or '
        +'floating point as confirmed causes.\n\n## Research handoff\n\n'
        +'E2 low RMSE lacks independent repeat/multiseed stability evidence; this12-epoch experiment '
        +'does not reproduce its epoch287 optimum or Assessment7.7858mV. F0/F1/F2 remain single-seed '
        +'exploratory ablations with inherited reused-Assessment selection bias; F2 has not beaten F0 '
        +'on Assessment. An independently authorized multiseed reproducibility study may be warranted '
        +'before architecture claims, but is not executed here. Current evidence does not establish '
        +'a robust reason to expand TFT architecture search or overturn TIMESNET_FREEZE / '
        +'BACKBONE_SEARCH_CLOSED. Stop; no extra conditions, seeds, tuning, Test, SOH or RUL.\n',encoding='utf-8')
    (PHASE/'figures').mkdir(exist_ok=True)
    plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Arial','DejaVu Sans'],
        'font.size':11,'axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':1.2,'legend.frameon':False})
    fig,axs=plt.subplots(1,2,figsize=(11,4))
    axs[0].plot(ht.epoch,ht.E2_train_MSE,label='Historical E2',color='#3775BA')
    axs[0].plot(ht.epoch,ht.F0_train_MSE,label='Historical F0',color='#B64342',ls='--')
    axs[0].set(xlabel='Epoch',ylabel='Train standardized MSE')
    axs[1].plot(ht.epoch,ht.E2_validation_RMSE,label='Historical E2',color='#3775BA')
    axs[1].plot(ht.epoch,ht.F0_validation_RMSE,label='Historical F0',color='#B64342',ls='--')
    axs[1].set(xlabel='Epoch',ylabel='Validation standardized RMSE')
    for ax in axs:ax.legend()
    figure(fig,'Fig1_Historical_Training_Trajectories')
    fig,axs=plt.subplots(1,2,figsize=(11,4))
    for n,col,ls in zip(TRACES,['#3775BA','#8BCF8B','#B64342','#767676'],['-','--',':','-.']):
        t=pd.DataFrame(epochs[n]);axs[0].plot(t.epoch,t.train_standardized_MSE,label=n,color=col,ls=ls)
        axs[1].plot(t.epoch,t.validation_standardized_RMSE,label=n,color=col,ls=ls)
    axs[0].set(xlabel='Diagnostic epoch',ylabel='Train standardized MSE')
    axs[1].set(xlabel='Diagnostic epoch',ylabel='Validation standardized RMSE')
    for ax in axs:ax.legend()
    figure(fig,'Fig2_Controlled_Replay_Trajectories')
    print(status,flush=True)
    print(json.dumps(comparisons,indent=2),flush=True)

if __name__=='__main__':main()
