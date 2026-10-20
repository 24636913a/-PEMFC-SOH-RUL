"""Publication-candidate plots exclusively from the saved real audit/pilot CSVs."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from audit_data import HERE,csv

AUDIT=HERE/'outputs/qualification_20261020_v2'
FIG=HERE/'figures'
DATA=HERE/'tables'
COLORS={'LSTM':'#0072B2','F2':'#D55E00','Persistence':'#009E73'}
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,
    'axes.spines.right':False,'pdf.fonttype':42,'ps.fonttype':42,'savefig.dpi':300})

def save(fig,name,data=None):
    fig.tight_layout();fig.savefig(FIG/(name+'.png'),dpi=300,bbox_inches='tight')
    fig.savefig(FIG/(name+'.pdf'),bbox_inches='tight');plt.close(fig)
    if data is not None:csv(DATA/(name+'.csv'),data)

def plot():
    FIG.mkdir(exist_ok=True);DATA.mkdir(exist_ok=True)
    run=Path(json.loads((HERE/'outputs/ACTIVE_PILOT.json').read_text(encoding='utf-8'))['path'])
    meta=pd.read_csv(AUDIT/'CLOCK_METADATA_ALL_ROWS.csv');pieces=pd.read_csv(AUDIT/'CONTINUOUS_VALID_PIECES.csv')
    fig,ax=plt.subplots(figsize=(9,3.6))
    for i,r in pieces.iterrows():ax.plot([r.first_h,r.last_h],[i+1,i+1],lw=4,color='#0072B2')
    for row,label,color in [(2690,'Train / Validation','#D55E00'),(3074,'Validation / holdout','#009E73')]:
        h=meta.bin_end_ophrs2_h.iloc[row];ax.axvline(h,ls='--',color=color,label=f'{label}: {h:g}h (row {row})')
    ax.set(xlabel='Actual operating hours',ylabel='Continuous valid piece',title='3843 bins; clock/validity metadata only, including holdout')
    ax.legend(fontsize=8,loc='upper left');save(fig,'F1_continuous_clock_support',pieces)
    counts=pd.read_csv(AUDIT/'WINDOW_COUNTS.csv');c=counts[counts.history_steps==48]
    fig,axes=plt.subplots(1,2,figsize=(9,3.5))
    for part,color in [('TRAIN','#0072B2'),('VALIDATION','#D55E00'),('HOLDOUT','#009E73')]:
        g=c[c.partition==part]
        axes[0].plot(g.horizon_h,g.legal_origins,'o-',color=color,label=part)
        axes[1].plot(g.horizon_h,g.disjoint_target_trajectories,'o-',color=color,label=part)
    for ax in axes:ax.set(xlabel='Forecast horizon (h)');ax.legend(fontsize=8)
    axes[0].set(ylabel='Legal origins',title='48 historical bins; overlapping origins')
    axes[1].set(ylabel='Disjoint target trajectories',title='96h Validation: 2 origins, 1 trajectory')
    save(fig,'F2_horizon_qualification',c)
    t=pd.read_csv(AUDIT/'TARGETS_TRAIN_VALIDATION_ONLY.csv')
    fig,axes=plt.subplots(3,1,figsize=(9,7),sharex=True)
    cols=[('V_raw','Raw cell voltage (V)'),('A_causal_mean_V','A: causal mean (V)'),('B_causal_adjusted_mean_V','B: adjusted causal mean (V)')]
    for ax,(col,label) in zip(axes,cols):
        for _,g in t.groupby('continuous_block'):ax.plot(g.operating_h,g[col],lw=.7,color='#0072B2')
        ax.set(ylabel=label);ax.axvline(t.operating_h.iloc[2690],color='#D55E00',ls='--')
    axes[0].set_title('TRAIN + VALIDATION only; no bridging breaks; B is a statistical proxy')
    axes[-1].set_xlabel('Actual operating hours');save(fig,'F3_causal_target_diagnostics',t)
    fig,axes=plt.subplots(1,2,figsize=(9,3.5))
    v=t[t.partition=='VALIDATION']
    for ax,col,label in zip(axes,['A_causal_mean_V','B_causal_adjusted_mean_V'],['A','B']):
        ax.scatter(v.zero_causal_mean,v[col],s=10,alpha=.7,color='#0072B2')
        ax.set(xlabel='Causal 48-bin mean zero-current fraction',ylabel='Proxy (V)',title=f'{label}: Validation operating-state sensitivity')
    save(fig,'F4_load_sensitivity',v)
    allmetrics=[];allsteps=[];allpoints=[]
    for d in sorted(run.glob('*_seed202_*h')):
        allmetrics.append(pd.read_csv(d/'METRICS.csv'));allsteps.append(pd.read_csv(d/'STEP_METRICS.csv'))
        allpoints.append(pd.read_csv(d/'VALIDATION_PREDICTIONS.csv',float_precision='round_trip'))
    m=pd.concat(allmetrics).drop_duplicates(['model','seed','horizon_h','partition']);steps=pd.concat(allsteps);points=pd.concat(allpoints)
    csv(DATA/'ALL_PILOT_METRICS.csv',m);csv(DATA/'ALL_STEP_METRICS.csv',steps);csv(DATA/'VALIDATION_POINT_PREDICTIONS.csv',points)
    fig,axes=plt.subplots(1,2,figsize=(9,3.5))
    vm=m[m.partition=='VALIDATION']
    for model,color in COLORS.items():
        g=vm[vm.model==model].sort_values('horizon_h')
        axes[0].plot(g.horizon_h,g.RMSE_mV,'o-',color=color,label=model)
        axes[1].plot(g.horizon_h,g.R2_delta,'o-',color=color,label=model)
    axes[0].set(xlabel='Forecast horizon (h)',ylabel='RMSE (mV of adjusted proxy)',title='Seed202 pilot; all forecast queries')
    axes[1].set(xlabel='Forecast horizon (h)',ylabel='R2 of delta proxy',title='48h pilot failed; no six-seed claim');axes[1].axhline(0,color='.5',ls=':')
    for ax in axes:ax.legend(fontsize=8)
    save(fig,'F5_pilot_model_comparison',vm)
    fig,axes=plt.subplots(1,2,figsize=(9,3.5))
    for ax,h in zip(axes,[24,48]):
        for model,color in list(COLORS.items())[:2]:
            s=steps[(steps.partition=='VALIDATION')&(steps.horizon_h==h)&(steps.model==model)]
            ax.plot(s.lead_h,s.RMSE_mV,color=color,label=model)
        ax.set(xlabel='Actual forecast lead (h)',ylabel='RMSE (mV of proxy)',title=f'{h}h direct trajectory, seed202');ax.legend()
    save(fig,'F6_lead_error',steps[steps.partition=='VALIDATION'])
    # Predefined representative cases: first common origin and worst average squared error.
    # Show both, never choose cases for an F2 advantage.
    cases=[]
    fig,axes=plt.subplots(2,2,figsize=(10,6),sharey='row')
    for row,h in enumerate([24,48]):
        p=points[points.horizon_h==h].copy();p['error2']=(p.predicted_proxy_V-p.true_proxy_V)**2
        first=int(p.origin_row.min());worst=int(p.groupby('origin_row').error2.mean().idxmax())
        for ax,o,label in zip(axes[row],[first,worst],['First common origin','Worst two-model mean SSE origin']):
            g=p[p.origin_row==o]
            a=g[g.model=='LSTM'];ax.plot(a.target_h,a.true_proxy_V,color='black',lw=1.5,label='Observed proxy')
            for model,color in list(COLORS.items())[:2]:
                a=g[g.model==model];ax.plot(a.target_h,a.predicted_proxy_V,color=color,label=model)
            ax.set(xlabel='Actual operating hours',ylabel='Adjusted proxy (V)',title=f'{h}h; {label}; row{o}');ax.legend(fontsize=7)
            g=g.copy();g['case_rule']=label;cases.append(g)
    save(fig,'F7_fixed_origin_favorable_adverse',pd.concat(cases))
    fig,axes=plt.subplots(2,2,figsize=(10,6),sharey='row')
    scatter=[]
    for row,h in enumerate([24,48]):
        for col,model in enumerate(['LSTM','F2']):
            p=points[(points.horizon_h==h)&(points.model==model)].copy();terminal=p[p.horizon_step==int(2*h)].copy()
            terminal['abs_error_mV']=abs(terminal.predicted_proxy_V-terminal.true_proxy_V)*1000
            axes[row,col].scatter(terminal.target_h,terminal.abs_error_mV,s=8,color=COLORS[model])
            axes[row,col].set(xlabel='Target operating hours',ylabel='Terminal absolute error (mV)',title=f'{model}, {h}h terminal only; n={len(terminal)}')
            scatter.append(terminal)
    save(fig,'F8_terminal_error_over_age',pd.concat(scatter))
    fig,axes=plt.subplots(1,2,figsize=(9,3.5))
    for ax,h in zip(axes,[24,48]):
        for model,color in list(COLORS.items())[:2]:
            p=points[(points.horizon_h==h)&(points.model==model)]
            ax.hist((p.predicted_proxy_V-p.true_proxy_V)*1000,bins=40,histtype='step',color=color,label=model)
        ax.set(xlabel='Predicted - observed proxy (mV)',ylabel='Overlapping forecast queries',title=f'{h}h Validation; all tails retained');ax.legend()
    save(fig,'F9_residual_distribution',points)
    gates=[];fig,axes=plt.subplots(1,2,figsize=(9,3.5))
    for ax,h in zip(axes,[24,48]):
        g=pd.read_csv(run/f'F2_seed202_{h}h/VALIDATION_GATE.csv');g['horizon_h']=h;gates.append(g)
        ax.plot(g.origin_h,g.g,'o',ms=2,color=COLORS['F2'])
        ax.set(xlabel='Origin operating hours',ylabel='Historical regime gate g (zoomed)',title=f'F2 {h}h; seed202; n={len(g)}; full g range [0,1]',ylim=(.47,.49))
    save(fig,'F10_f2_gate',pd.concat(gates))
    losses=[];fig,axes=plt.subplots(2,2,figsize=(10,6),sharey=True)
    for ax,d in zip(axes.ravel(),sorted(run.glob('*_seed202_*h'))):
        l=pd.read_csv(d/'epoch_log.csv');l['run']=d.name;losses.append(l)
        ax.semilogy(l.epoch,l.train_MSE,label='Train',color='#0072B2');ax.semilogy(l.epoch,l.validation_MSE,label='Validation',color='#D55E00')
        ax.set(xlabel='Epoch',ylabel='Standardized delta-proxy MSE',title=d.name);ax.legend(fontsize=8)
    save(fig,'F11_actual_training_losses',pd.concat(losses))
    # Descriptive chronological/state diagnostics, not extra model-selection criteria.
    state=points.merge(t[['row','zero_current_fraction','load_transitions','current_density']],left_on='target_row',right_on='row',validate='many_to_one')
    train=t[(t.partition=='TRAIN')&t.valid];zero_cut=float(train.zero_current_fraction.quantile(.75));load_cut=float(train.load_transitions.quantile(.75))
    state['event_group']=np.where((state.zero_current_fraction>zero_cut)|(state.load_transitions>load_cut),'Train-P75 high switching/zero','Other')
    state['chronological_half']=np.where(state.target_row<(2690+3074)//2,'Earlier Validation half','Later Validation half')
    summary=[]
    for keys,g in state.groupby(['model','horizon_h','event_group','chronological_half']):
        err=(g.predicted_proxy_V-g.true_proxy_V).to_numpy(float)*1000
        summary.append(dict(model=keys[0],horizon_h=keys[1],event_group=keys[2],chronological_half=keys[3],queries=len(g),unique_targets=g.target_row.nunique(),RMSE_mV=float(np.sqrt(np.mean(err**2))),MAE_mV=float(np.mean(abs(err)))))
    csv(DATA/'DESCRIPTIVE_STATE_AGE_ERRORS.csv',pd.DataFrame(summary));csv(DATA/'STATE_GROUP_THRESHOLDS.csv',pd.DataFrame([dict(zero_current_train_P75=zero_cut,load_transitions_train_P75=load_cut)]))
    csv(DATA/'DESCRIPTIVE_STATE_PLOT_POINTS.csv',state)
    fig,axes=plt.subplots(1,2,figsize=(9,3.5))
    for ax,h in zip(axes,[24,48]):
        for model,color in list(COLORS.items())[:2]:
            g=state[(state.model==model)&(state.horizon_h==h)&(state.horizon_step==int(2*h))]
            ax.scatter(g.zero_current_fraction,abs(g.predicted_proxy_V-g.true_proxy_V)*1000,s=9,alpha=.65,color=color,label=model)
        ax.set(xlabel='Target zero-current fraction (diagnostic only)',ylabel='Terminal absolute error (mV)',title=f'{h}h proxy, Validation; not causal attribution');ax.legend()
    save(fig,'F12_state_error',state)
    print('Generated 12 actual PNG/PDF figure pairs and source CSVs')

if __name__=='__main__':plot()
