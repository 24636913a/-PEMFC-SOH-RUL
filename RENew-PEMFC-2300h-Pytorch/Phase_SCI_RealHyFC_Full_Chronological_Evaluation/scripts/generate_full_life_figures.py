"""Paper candidates from audited CSV only. No overlapping windows are stitched."""
from common import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
COLORS=dict(F0='#767676',F1='#E49C3A',F2='#2578AE',LSTM='#39986B',TCN='#A267AC')
LABELS=dict(TRAIN='Train (in-sample)',VALIDATION='Validation (reused)',HISTORICAL_ASSESSMENT='Historical assessment (exploratory)',LATE_LIFE='Late-life (exploratory)')
def style():
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.spines.top':False,'axes.spines.right':False,'axes.labelsize':10,'legend.fontsize':8,'pdf.fonttype':42,'ps.fonttype':42,'savefig.dpi':300})
def save(fig,name,csv):
    cs(f'figures/{name}_plot_data.csv',csv)
    fig.savefig(path(f'figures/{name}.png'),dpi=300,bbox_inches='tight',facecolor='white')
    fig.savefig(path(f'figures/{name}.pdf'),bbox_inches='tight',facecolor='white');plt.close(fig)
def case_frame(origin,seed,part='LATE_LIFE'):
    frames=[]
    for model in MODELS:
        f=rd(P/f'outputs/predictions/{part}_{model}_seed{seed}.csv');frames.append(f.loc[f.origin_row==origin].copy())
    return pd.concat(frames,ignore_index=True)
def voltage_panel(ax,f,title):
    first=f.loc[f.model=='F0'];x=first.horizon_step*.5
    ax.plot(x,first.true_raw_target_voltage_V,color='black',lw=1.8,label='Observed voltage')
    for model,z in f.groupby('model',sort=False):ax.plot(z.horizon_step*.5,z.predicted_voltage_V,color=COLORS[model],lw=1.2,label=model)
    ax.set(xlabel='Lead time (h)',ylabel='Cell voltage (V)',title=title);ax.grid(alpha=.16)
def generate():
    protocol();assert read(P/'outputs/audits/INDEPENDENT_RECALCULATION.json')['pass'];style()
    src=rd(P/'outputs/audits/FULL_ROW_IDENTITY.csv');parts=rd(P/'tables/TIME_PARTITIONS.csv');counts=rd(P/'tables/WINDOW_COUNTS.csv')
    summary=rd(P/'tables/MODEL_THREE_SEED_SUMMARY.csv');detail=rd(P/'outputs/metrics/PER_MODEL_SEED_PARTITION.csv');steps=rd(P/'outputs/metrics/HORIZON_METRICS.csv');origin=rd(P/'outputs/metrics/PER_ORIGIN_METRICS.csv')
    # Timeline uses actual operating hours; points, rather than lines over gaps.
    fig,ax=plt.subplots(figsize=(10,3.6));palette=['#DCEAD9','#DAE8F4','#F7E6C4','#E8DBEB'];plot=[]
    for r,color in zip(parts.itertuples(),palette):
        z=src.iloc[r.start_inclusive:r.end_exclusive].copy();z['partition']=r.partition;plot.append(z[['row_id','partition','bin_end_ophrs2_h','cell_voltage_mean','bin_valid_for_windows']])
        ax.axvspan(z.bin_end_ophrs2_h.min(),z.bin_end_ophrs2_h.max(),color=color,alpha=.65)
        ax.scatter(z.bin_end_ophrs2_h,z.cell_voltage_mean,s=2,color='black',alpha=.5)
        ax.text(z.bin_end_ophrs2_h.median(),1.01,f'{r.partition}\nn={len(z)}',ha='center',va='bottom',transform=ax.get_xaxis_transform(),fontsize=8)
    ax.set(xlabel='Actual operating time at bin end (h)',ylabel='Observed cell voltage (V)',title=f'Full chronological evaluation: N={len(src)} records; late region is exploratory',ylim=(src.cell_voltage_mean.min()-.01,src.cell_voltage_mean.max()+.01));save(fig,'01_full_timeline',pd.concat(plot))
    fig,axes=plt.subplots(2,2,figsize=(10,6.8));
    for ax,part in zip(axes.flat,PARTS):
        z=summary.loc[summary.partition==part].set_index('model').loc[MODELS];x=np.arange(5)
        ax.bar(x,z.RMSE_mV_mean,yerr=z.RMSE_mV_sample_sd,color=[COLORS[m] for m in MODELS],capsize=3)
        ax.set_xticks(x,MODELS);n=counts.loc[counts.partition==part].iloc[0]
        ax.set(title=f'{LABELS[part]}\n{n.legal_origins} origins / {n.prediction_queries} targets per seed',ylabel='RMSE (mV)');ax.grid(axis='y',alpha=.16)
    fig.suptitle('Three frozen seeds per model; bars = mean, whiskers = sample SD');fig.tight_layout();save(fig,'02_partition_model_RMSE',summary)
    binned=[]
    for part in PARTS:
        z=origin.loc[(origin.partition==part)&(origin.model=='F2')].copy();z['bin']=pd.cut(z.origin_age_h,bins=20,labels=False,include_lowest=True)
        for b,g in z.groupby('bin'):
            binned.append(dict(partition=part,bin=int(b),origin_age_h_mean=g.origin_age_h.mean(),RMSE_mV=float(np.sqrt(np.mean(g.RMSE_mV**2))),origins_per_seed=g.origin_row.nunique(),seeds=g.seed.nunique()))
    z=pd.DataFrame(binned);fig,ax=plt.subplots(figsize=(10,3.8))
    for part in PARTS:
        g=z.loc[z.partition==part];ax.scatter(g.origin_age_h_mean,g.RMSE_mV,s=24,label=LABELS[part])
    ax.set(xlabel='Actual forecast-origin operating time (h)',ylabel='F2 pooled query RMSE (mV)',title='F2 local errors: 20 time bins within each partition; all three seeds');ax.legend(ncol=2);ax.grid(alpha=.16);save(fig,'03_F2_life_stage_errors',z)
    fig,axes=plt.subplots(2,2,figsize=(10,6.7));hplot=[]
    for ax,part in zip(axes.flat,PARTS):
        for model in MODELS:
            g=steps.loc[(steps.partition==part)&(steps.model==model)].groupby('horizon_step').RMSE_mV.agg(['mean','std']).reset_index();g['partition']=part;g['model']=model;hplot.append(g)
            x=g.horizon_step.to_numpy()*.5;y=g['mean'].to_numpy();sd=g['std'].to_numpy();ax.plot(x,y,label=model,color=COLORS[model],lw=1.8 if model=='F2' else 1)
            if model=='F2':ax.fill_between(x,y-sd,y+sd,color=COLORS[model],alpha=.15)
        ax.set(title=LABELS[part],xlabel='Forecast lead time (h)',ylabel='RMSE (mV)');ax.grid(alpha=.16)
    axes[0,0].legend(ncol=3);fig.suptitle('Horizon errors: mean of three frozen seeds; F2 band = sample SD');fig.tight_layout();save(fig,'04_horizon_RMSE',pd.concat(hplot))
    legal=data('LATE_LIFE')['origins'];fixed=int(legal[len(legal)//2]);f=case_frame(fixed,42)
    fig,ax=plt.subplots(figsize=(9,4));voltage_panel(ax,f,f'Late-life exploratory: fixed middle origin row {fixed}, Seed42, one 24-step forecast');ax.legend(ncol=3);save(fig,'05_late_fixed_window',f)
    fig,axes=plt.subplots(1,2,figsize=(10,4));z=detail.loc[(detail.partition=='LATE_LIFE')&detail.model.isin(['F2','LSTM'])]
    for ax,key,title in zip(axes,['RMSE_mV','endpoint_RMSE_mV'],['All 24 steps','12-hour endpoint']):
        for seed in SEEDS:
            g=z.loc[z.seed==seed].set_index('model').loc[['F2','LSTM']];ax.plot([0,1],g[key],marker='o',label=f'Seed{seed}')
        ax.set_xticks([0,1],['F2','LSTM']);ax.set(ylabel='RMSE (mV)',title=title);ax.grid(axis='y',alpha=.16)
    axes[0].legend();fig.suptitle(f'Late-life paired comparison: {len(legal)} identical origins per model/seed');fig.tight_layout();save(fig,'06_F2_LSTM_late_comparison',z)
    # Pre-frozen adverse-case rule, with independent absolute maximum case too.
    z=origin.loc[(origin.partition=='LATE_LIFE')&(origin.model=='F2')];g=z.groupby('origin_row',sort=True).RMSE_mV.mean();hard=int(g.idxmax())
    worst=z.sort_values(['max_abs_error_mV','origin_row'],ascending=[False,True]).iloc[0];extreme=int(worst.origin_row);eseed=int(worst.seed)
    cases=[];fig,axes=plt.subplots(1,2,figsize=(12,4.3))
    for ax,o,s,label in [(axes[0],hard,42,'Largest F2 three-seed mean window error'),(axes[1],extreme,eseed,'Largest F2 single-query error window')]:
        f=case_frame(o,s);f['case_rule']=label;cases.append(f);voltage_panel(ax,f,f'{label}\nrow {o}, Seed{s}; late-life exploratory')
    axes[0].legend(ncol=3);fig.tight_layout();save(fig,'07_adverse_windows',pd.concat(cases))
    cs('outputs/audits/FIGURE_CASES.csv',[dict(figure='05',origin_row=fixed,seed=42,selection='fixed middle legal late origin'),dict(figure='07a',origin_row=hard,seed=42,selection='largest F2 three-seed mean per-origin RMSE'),dict(figure='07b',origin_row=extreme,seed=eseed,selection='largest F2 absolute single-query error')])
    js('outputs/audits/FIGURE_PROVENANCE.json',dict(script_sha256=sha(Path(__file__)),protocol_sha256=sha(P/'configs/FROZEN_CHRONOLOGICAL_PROTOCOL.yaml'),source_sha256=sha(SOURCE),metric_csv_sha256=sha(P/'outputs/metrics/PER_MODEL_SEED_PARTITION.csv'),figures=7,dpi=300,PDF='vector paths plus embedded fonts',no_stitched_forecasts=True,units='cell voltage V; prediction error mV; operating time h',scientific_status='in-sample Train, reused Validation, exploratory historical Assessment and late-life'))
