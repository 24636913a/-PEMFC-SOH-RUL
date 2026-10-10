"""Persisted Validation evidence only; never executes model inference."""
from __future__ import annotations
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from h_common import *
from audit_h_predictions import audit

def md_table(frame,cols):
    lines=['|'+'|'.join(cols)+'|','|'+'|'.join(['---']*len(cols))+'|']
    for _,row in frame.iterrows():lines.append('|'+'|'.join(str(row[k]) for k in cols)+'|')
    return '\n'.join(lines)+'\n'

def save(fig,name):
    fig.tight_layout(pad=1.5)
    for suffix in ['png','pdf']:fig.savefig(PHASE/'figures'/f'{name}.{suffix}',dpi=300)
    plt.close(fig)

def main():
    cfg=load_protocol();ledger=read_json(OUT/'audits/H_EXECUTION_LEDGER.json');status=ledger['status']
    rows=audit() if any(v['status']=='COMPLETED' for v in ledger['runs'].values()) else []
    results=pd.DataFrame(rows);reports=PHASE/'reports';reports.mkdir(exist_ok=True);(PHASE/'figures').mkdir(exist_ok=True)
    compatibility={}
    for device in ['cuda','cpu']:
        for kind in ['interpolation','tft']:
            path=OUT/'compatibility'/f'{device}_strict_{kind}/result.json'
            compatibility[device+'_strict_'+kind]=read_json(path) if path.exists() else {'status':'NOT_RUN'}
    (reports/'H_DETERMINISM_COMPATIBILITY.md').write_text('# Determinism compatibility\n\n'+json.dumps(compatibility,indent=2)+'\n\n'
        +'Strict CUDA interpolation failure is a framework compatibility limitation, not proof of historical E2/F0 divergence cause. '
        +'CPU strict success does not imply CUDA bitwise determinism. Normal training uses deterministic_algorithms=False '
        +'and does not claim strict determinism. Crash-safe injection and F0/F1/F2 normal CUDA smoke evidence are separately saved.\n\n'
        +'Count erratum: state_dict_element_count=332955 includes persistent buffers, whereas trainable_parameter_count=323483 counts '
        +'only optimizable F0 parameters. Phase G JSON is retained unchanged.\n',encoding='utf-8')
    gates={}
    for n in [2,3]:
        path=OUT/'repeatability'/f'{n}_run_gate.json'
        if path.exists():gates[str(n)]=read_json(path)
    repeat=results.loc[results.seed.eq(42)&results.model.eq('F0')] if len(results) else results
    repeatcols=['run_id','best_epoch','stop_epoch','RMSE_mV','MAE_mV','R2']
    (reports/'H_SAME_SEED_REPEATABILITY.md').write_text('# Same Seed42 repeatability\n\n'
        +(md_table(repeat,repeatcols) if len(repeat) else 'NOT_RUN\n')+'\n'+json.dumps(gates,indent=2)+'\n\n'
        +'Two-run D=|RA−RB|; T=max(0.5mV,0.10×mean). Three-run D=max−min uses the same mean-based threshold. '
        +'All runs are independent initializations in fresh processes. PASS means no material discrepancy above the registered threshold, '
        +'not bitwise determinism. Repeat-A/B are excluded from multi-seed n=3.\n',encoding='utf-8')
    summary={'status':status,'base_commit':BASE,'protocol_sha256':file_hash(PROTOCOL),'code_identity_sha256':cfg['code_identity']['sha256'],
        'completed_runs':len(rows),'formal_runs_started':ledger['formal_runs_started'],'repeatability':gates,
        'new_assessment_inference':False,'formal_test_accessed':False,'architecture_modification':False,'hyperparameter_search':False,
        'train_accessed':bool(rows),'validation_accessed':bool(rows),'CUDA_STRICT_BITWISE_DETERMINISM_NOT_CLAIMED':True,
        'multi_seed_primary_complete':False,'architecture_signal':'NOT_EVALUABLE','registered_seeds':[42,52,62]}
    primary=results.loc[results.role.eq('MULTISEED_PRIMARY')] if len(results) else results
    stats=[];pairs=[];pair_stats=[]
    if status=='PHASE_H_MULTISEED_VALIDATION_COMPLETED':
        if len(primary)!=9 or any(set(primary.loc[primary.model.eq(g),'seed'])!={42,52,62} for g in COUNTS):raise RuntimeError('MULTISEED_RUN_IDENTITY_INCOMPLETE')
        for g in COUNTS:
            f=primary.loc[primary.model.eq(g)]
            stats.append({'model':g,'n_seeds':3,'RMSE_mean_mV':f.RMSE_mV.mean(),'RMSE_sample_std_mV':f.RMSE_mV.std(ddof=1),
                'RMSE_median_mV':f.RMSE_mV.median(),'RMSE_min_mV':f.RMSE_mV.min(),'RMSE_max_mV':f.RMSE_mV.max(),
                'MAE_mean_mV':f.MAE_mV.mean(),'MAE_sample_std_mV':f.MAE_mV.std(ddof=1),'R2_mean':f.R2.mean(),'R2_sample_std':f.R2.std(ddof=1),
                'best_epochs':f.best_epoch.tolist(),'training_time_mean_s':f.training_time_s.mean(),'total_training_time_s':f.training_time_s.sum()})
        pivot=primary.pivot(index='seed',columns='model',values='RMSE_mV')
        for seed in [42,52,62]:pairs.append({'seed':seed,'D10_F1_minus_F0_mV':pivot.loc[seed,'F1']-pivot.loc[seed,'F0'],
            'D21_F2_minus_F1_mV':pivot.loc[seed,'F2']-pivot.loc[seed,'F1'],'D20_F2_minus_F0_mV':pivot.loc[seed,'F2']-pivot.loc[seed,'F0']})
        pf=pd.DataFrame(pairs)
        for name in list(pf.columns)[1:]:
            values=pf[name];pair_stats.append({'comparison':name,'mean_difference_mV':values.mean(),'sample_std_mV':values.std(ddof=1),
                'improved_seed_count':int((values<0).sum()),'max_degradation_mV':max(0,float(values.max()))})
        if (pf.D21_F2_minus_F1_mV<0).all() and (pf.D20_F2_minus_F0_mV<0).all():signal='F2_VALIDATION_MULTISEED_GAIN_OBSERVED'
        elif (pf.D21_F2_minus_F1_mV<0).sum()>=2:signal='F2_VS_F1_MULTISEED_SIGNAL_ONLY'
        else:signal='F2_STRUCTURAL_GAIN_NOT_STABLE'
        pd.DataFrame(stats).to_csv(OUT/'metrics/H_MULTISEED_STATISTICS.csv',index=False)
        pf.to_csv(OUT/'metrics/H_PAIRED_SEED_DIFFERENCES.csv',index=False)
        pd.DataFrame(pair_stats).to_csv(OUT/'metrics/H_PAIRED_DIFFERENCE_SUMMARY.csv',index=False)
        summary.update(multi_seed_primary_complete=True,architecture_signal=signal,multi_seed_statistics=stats,paired_differences=pairs,paired_statistics=pair_stats)
    else:summary.update(stop_reason=ledger.get('stop_reason'),technical_exception=ledger.get('exception'))
    table=md_table(primary,['model','seed','RMSE_mV','MAE_mV','R2','best_epoch','stop_epoch','training_time_s']) if len(primary) else 'MULTISEED_NOT_RUN\n'
    (reports/'H_MULTISEED_STABILITY_ANALYSIS.md').write_text('# Development Validation multi-seed stability\n\n'+table+'\n'
        +json.dumps(stats,indent=2)+'\n\nOnly designated Seed42/52/62 runs enter n=3; sample standard deviation uses ddof=1. '
        +'No significance tests, no use of 4920 overlapping predictions as independent samples. Not a Test result.\n',encoding='utf-8')
    (reports/'H_ABLATION_ANALYSIS.md').write_text('# Paired Validation ablation\n\n'+json.dumps(pairs,indent=2)+'\n\n'
        +json.dumps(pair_stats,indent=2)+'\n\n'+summary['architecture_signal']+'\n\n'
        +'Differences are F1−F0, F2−F1 and F2−F0; a negative value means F1/F2 (the minuend) is better. '
        +'F2 gains relative to F1 alone do not establish superiority to F0. No model or gate redesign.\n',encoding='utf-8')
    (reports/'H_FINAL_CONCLUSION.md').write_text('# Phase H final conclusion\n\n'+status+'\n\n'+summary['architecture_signal']+'\n\n'
        +'## Same-seed evidence\n\n'+json.dumps(gates,indent=2)+'\n\n## Development Validation results\n\n'+table+'\n\n'
        +'## Paired evidence\n\n'+json.dumps(pair_stats,indent=2)+'\n\n'
        +'## Research boundary and eight answers\n\n'
        +'1. Same-Seed42 material repeatability is determined only by the saved two-/three-run registered gates; no bitwise claim.\n'
        +'2. CUDA strict compatibility is reported by the independent synthetic probes; normal CUDA training is a different mode.\n'
        +'3. Compatibility failures and same-seed numerical outcomes are new evidence, not identification of the unique historical E2/F0 cause.\n'
        +'4. F2 versus F1: see per-seed paired counts above; unavailable if stopped before the primary runs completed.\n'
        +'5. F2 versus F0 must independently be supported by its paired counts, not inferred from F2/F1.\n'
        +'6. Condition-fusion evidence is limited to the observed three-seed Validation signal; no automatic architecture expansion.\n'
        +'7. Three seeds on one reused Development split can support limited descriptive stability statements only, not universal stability or significance.\n'
        +'8. No independent Test promotion is authorized. If fully successful, a separately authorized Test-plan review may be considered, not automatic opening; '
        +'if blocked or unstable, reliability review comes first.\n\n'
        +'## Immutable history\n\nTIMESNET_FREEZE; BACKBONE_SEARCH_CLOSED; PHASE_F_ASSESSMENT_INHERITED_SELECTION_BIAS; '
        +'C1_D0_E0_REPRODUCIBILITY_UNRESOLVED. Historical E/F/G untouched. State-dict332955 is not trainable-parameter323483. '
        +'Train/Validation only; no Assessment inference, formal Test, search, SOH or RUL. Stop, no Phase I.\n',encoding='utf-8')
    atomic_json(OUT/'H_FINAL_SUMMARY.json',summary)
    plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Arial','DejaVu Sans'],'font.size':11,
        'axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':1.2,'legend.frameon':False})
    colors={'F0':'#3775BA','F1':'#8BCF8B','F2':'#B64342'}
    fig,ax=plt.subplots(figsize=(9,4))
    for n in ['H-F0-Repeat-A','H-F0-Repeat-B','H-F0-Seed42']:
        path=OUT/'training_logs'/n/'epochs.csv'
        if path.exists():f=pd.read_csv(path);ax.plot(f.epoch,f.validation_RMSE_mV,label=n)
    ax.set(xlabel='Epoch',ylabel='Validation RMSE (mV)',title='Development Validation — independent F0 Seed42 repeats');ax.legend();save(fig,'Fig1_F0_Seed42_Repeat_Validation_Curves')
    if summary['multi_seed_primary_complete']:
        fig,ax=plt.subplots(figsize=(8,4))
        for g in COUNTS:
            f=primary.loc[primary.model.eq(g)].sort_values('seed');ax.plot([42,52,62],f.RMSE_mV,'o-',label=g,color=colors[g])
        ax.set(xticks=[42,52,62],xlabel='Seed',ylabel='Validation RMSE (mV)',title='Development Validation — n=3 seeds');ax.legend();save(fig,'Fig2_Three_Seed_Validation_RMSE')
        sf=pd.DataFrame(stats);fig,ax=plt.subplots(figsize=(8,4));ax.bar(sf.model,sf.RMSE_mean_mV,yerr=sf.RMSE_sample_std_mV,
            color=[colors[g] for g in sf.model],edgecolor='black',capsize=5)
        ax.set(ylim=(0,None),ylabel='Validation RMSE (mV)',title='Development Validation — mean ± sample SD, n=3 seeds');save(fig,'Fig3_RMSE_Mean_and_Sample_SD')
        pf=pd.DataFrame(pairs);fig,ax=plt.subplots(figsize=(9,4))
        for i,(name,label) in enumerate(zip(list(pf.columns)[1:],['F1−F0','F2−F1','F2−F0'])):
            ax.bar(np.arange(3)+(i-1)*.24,pf[name],width=.24,label=label,edgecolor='black')
        ax.axhline(0,color='black',lw=.8);ax.set(xticks=np.arange(3),xticklabels=['42','52','62'],xlabel='Seed',ylabel='Paired RMSE difference (mV)',title='Development Validation — n=3 seeds');ax.legend();save(fig,'Fig4_Paired_Seed_Differences')
        fig,ax=plt.subplots(figsize=(8,4))
        for i,g in enumerate(COUNTS):ax.scatter(np.full(3,i),primary.loc[primary.model.eq(g),'best_epoch'],label=g,color=colors[g],s=45)
        ax.set(xticks=range(3),xticklabels=list(COUNTS),ylabel='Best epoch',title='Development Validation — n=3 seeds');save(fig,'Fig5_Best_Epoch_Distribution')
        fig,ax=plt.subplots(figsize=(8,4));ax.bar(sf.model,sf.training_time_mean_s/60,color=[colors[g] for g in sf.model],edgecolor='black')
        ax.set(ylim=(0,None),ylabel='Mean training time (min)',title='Development Validation — n=3 seeds; includes crash-safe I/O');save(fig,'Fig6_Training_Cost')
    else:
        # Do not fabricate n=3 plots when the registered stop prevents them.
        atomic_json(OUT/'audits/H_NOT_RUN_MANIFEST.json',{'multi_seed_figures_2_to_6':'NOT_IDENTIFIABLE_REGISTERED_STOP','status':status,
            'not_run_ids':[r['run_id'] for r in cfg['formal_runs'] if r['run_id'] not in ledger['runs']]})
    print(json.dumps(summary,indent=2),flush=True)

if __name__=='__main__':main()
