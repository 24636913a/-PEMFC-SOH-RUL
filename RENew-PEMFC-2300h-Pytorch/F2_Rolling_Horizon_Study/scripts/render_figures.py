"""Real CSV -> paper-candidate PNG/PDF, no curve smoothing or sample removal."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from common import HERE,read,read_csv,csv,dump

COL={'F2':'#0072B2','LSTM':'#D55E00','C0':'#0072B2','C1':'#009E73','Historical_F2':'#CC79A7'}
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'ps.fonttype':42,'savefig.dpi':300})

def lines_with_breaks(frame,cols):
    f=frame.sort_values('target_row');rows=[];prev=None
    for r in f.to_dict('records'):
        if prev is not None and (r['target_row']!=prev['target_row']+1 or abs(r['target_age_h']-prev['target_age_h']-.5)>1e-6):rows.append({c:np.nan for c in cols})
        rows.append({c:r[c] for c in cols});prev=r
    return pd.DataFrame(rows)

def run(out,presentation_root=None):
    base=Path(presentation_root) if presentation_root is not None else HERE
    base.mkdir(parents=True,exist_ok=True)
    figdir=base/'figures';tables=base/'tables';figdir.mkdir(exist_ok=True);tables.mkdir(exist_ok=True)
    def save(fig,name,data=None):
        fig.tight_layout();fig.savefig(figdir/(name+'.png'),dpi=300,bbox_inches='tight');fig.savefig(figdir/(name+'.pdf'),bbox_inches='tight');plt.close(fig)
        if data is not None:csv(tables/(name+'.csv'),data)
    jobs=read(HERE/'configs/FROZEN_PROTOCOL.json')['training_jobs']
    hist=read_csv(out/'P1_HISTORICAL_PREFIX_METRICS.csv');histstep=read_csv(out/'P1_HISTORICAL_STEP_METRICS.csv')
    fig,axes=plt.subplots(1,2,figsize=(10,3.6))
    axes[0].plot(hist.prefix_steps*.5,hist.RMSE_mV,'o-',color=COL['Historical_F2']);axes[0].set(xlabel='Included future span (h)',ylabel='RMSE (mV)',title='Historical H24 model truncated; not retrained')
    axes[1].plot(histstep.horizon_step*.5,histstep.RMSE_mV,'o-',ms=3,color=COL['Historical_F2']);axes[1].set(xlabel='Fixed forecast lead (h)',ylabel='RMSE (mV)',title='Selected historical Seed202, same527 origins')
    save(fig,'F1_historical_truncation',histstep)
    metric=read_csv(out/'ALL_PARTITION_METRICS.csv');test=metric[metric.partition=='HOLDOUT']
    final=read_csv(out/'FINAL_RESULT_TABLE.csv');csv(tables/'FINAL_RESULT_TABLE.csv',final)
    fig,axes=plt.subplots(1,2,figsize=(10,3.7),sharey=True)
    for ax,scope,title in zip(axes,['ALL_H_LEGAL','COMMON_H24_ORIGINS'],['All H-specific legal origins','Common original H24 origins']):
        m=test[test.scope==scope]
        f=m[m.model.isin(['F2','C0'])].sort_values('horizon_steps');l=m[m.model=='LSTM'].sort_values('horizon_steps');c=m[m.model=='C1']
        ax.plot(f.horizon_steps*.5,f.RMSE_mV,'o-',color=COL['F2'],label='Native F2 (fresh)')
        ax.plot(l.horizon_steps*.5,l.RMSE_mV,'s-',color=COL['LSTM'],label='Plain LSTM (fresh)')
        ax.scatter(c.horizon_steps*.5,c.RMSE_mV,marker='^',s=70,color=COL['C1'],label='C1 fresh output head')
        ax.scatter([12],[hist.loc[hist.prefix_steps==24,'RMSE_mV'].iloc[0]],marker='D',s=45,color=COL['Historical_F2'],label='Historical selected H24')
        ax.axhline(10,color='.5',ls=':',lw=1,label='10mV criterion');ax.set(xlabel='Forecast horizon (h)',ylabel='Raw DeltaV RMSE (mV)',title=title);ax.set_xticks([3,6,12]);ax.legend(fontsize=7)
    save(fig,'F2_horizon_RMSE_comparison',test)
    points=[];common_steps=[]
    common=set(read_csv(out/'windows/H24_HOLDOUT.csv').origin_row)
    for kind,h in jobs:
        p=read_csv(out/f'models/{kind}_H{h}/predictions_HOLDOUT.csv');points.append(p)
        for step,g in p[p.origin_row.isin(common)].groupby('horizon_step'):
            e=(g.y_pred_delta_V-g.y_true_delta_V).to_numpy(float)*1000
            common_steps.append(dict(model=kind,horizon_steps=h,step=int(step),lead_h=step*.5,RMSE_mV=float(np.sqrt(np.mean(e**2))),MAE_mV=float(np.mean(abs(e))),queries=len(g)))
    allp=pd.concat(points,ignore_index=True);csv(tables/'ALL_HOLDOUT_PLOT_POINTS.csv',allp)
    lead=pd.DataFrame(common_steps);fig,axes=plt.subplots(1,3,figsize=(13,3.5),sharey=True)
    for ax,h in zip(axes,[6,12,24]):
        for kind in (['F2','LSTM'] if h!=24 else ['C0','C1']):
            s=lead[(lead.horizon_steps==h)&(lead.model==kind)];ax.plot(s.lead_h,s.RMSE_mV,'o-',ms=2,color=COL[kind],label=kind)
        if h==24:ax.plot(histstep.lead_h,histstep.RMSE_mV,ls='--',color=COL['Historical_F2'],label='Historical H24')
        ax.set(xlabel='Fixed lead (h)',ylabel='RMSE (mV)',title=f'H={h} ({h*.5:g}h);527 common origins');ax.legend(fontsize=8)
    save(fig,'F3_fixed_lead_RMSE',lead)
    # Selection rule is frozen before new holdout inference: earliest common
    # origin plus window ending at the largest historical target-SSE event.
    case=read(out/'FIGURE_CASE_RULES_BEFORE_NEW_HOLDOUT.json');adverse=int(case['historical_top_target_row'])
    casepoints=[];fig,axes=plt.subplots(2,3,figsize=(13,7),sharey=True)
    for col,h in enumerate([6,12,24]):
        kinds=['F2','LSTM'] if h!=24 else ['C0','C1']
        for row,(o,title) in enumerate([(min(common),'First common origin'),(adverse-h,'Historical dominant-event terminal')]):
            ax=axes[row,col];g=allp[(allp.horizon_steps==h)&(allp.origin_row==o)]
            if g.empty:ax.text(.1,.5,'No legal case for frozen origin',transform=ax.transAxes);continue
            truth=g[g.model==kinds[0]];ax.plot(truth.target_age_h,truth.true_voltage_V,color='black',lw=1.5,label='True')
            for kind in kinds:
                p=g[g.model==kind];ax.plot(p.target_age_h,p.predicted_voltage_V,'o-',ms=3,color=COL[kind],label=kind)
            short='First common origin' if row==0 else 'Historical-event terminal'
            ax.set(xlabel='Target operating hours',ylabel='Cell voltage (V)');ax.set_title(f'{short}\nH{h}; origin row{o}',fontsize=9);ax.legend(fontsize=8)
            cp=g.copy();cp['case_rule']=title;casepoints.append(cp)
    save(fig,'F4_same_origin_direct_trajectories',pd.concat(casepoints))
    fig,axes=plt.subplots(3,2,figsize=(12,9),sharex=True,sharey=True);fixedpoints=[]
    for ax,(kind,h) in zip(axes.ravel(),jobs):
        p=allp[(allp.model==kind)&(allp.horizon_steps==h)&(allp.horizon_step==h)].copy();fixedpoints.append(p)
        ln=lines_with_breaks(p,['target_age_h','true_voltage_V','predicted_voltage_V'])
        ax.plot(ln.target_age_h,ln.true_voltage_V,color='black',lw=.7,label='True')
        ax.plot(ln.target_age_h,ln.predicted_voltage_V,color=COL[kind],ls='--',lw=.7,label='Predicted')
        ax.set(xlabel='Target operating hours',ylabel='Cell voltage (V)',title=f'{kind} H{h}: fixed terminal step, n={len(p)}');ax.legend(fontsize=8)
    save(fig,'F5_chronological_fixed_terminal_voltage',pd.concat(fixedpoints))
    fig,axes=plt.subplots(3,2,figsize=(10,11),sharex=True,sharey=True)
    low=min(allp.true_voltage_V.min(),allp.predicted_voltage_V.min());high=max(allp.true_voltage_V.max(),allp.predicted_voltage_V.max())
    pad=.02*(high-low);low-=pad;high+=pad
    for ax,(kind,h) in zip(axes.ravel(),jobs):
        p=allp[(allp.model==kind)&(allp.horizon_steps==h)]
        ax.scatter(p.true_voltage_V,p.predicted_voltage_V,s=5,alpha=.3,color=COL[kind]);ax.plot([low,high],[low,high],color='.4',ls='--')
        ax.set(xlabel='True cell voltage (V)',ylabel='Predicted cell voltage (V)',title=f'{kind} H{h}, all{len(p)} queries',xlim=(low,high),ylim=(low,high));ax.set_aspect('equal',adjustable='box')
    save(fig,'F6_absolute_voltage_scatter',allp)
    fig,axes=plt.subplots(1,2,figsize=(14,4.5),sharey=True)
    for ax,event,title in zip(axes,['event_any','event_voltage_jump'],['Train-defined event union','Train-P95 voltage jump (posthoc only)']):
        values=[];positions=[];labels=[];colors=[]
        for i,(kind,h) in enumerate(jobs):
            for j,flag in enumerate([False,True]):
                g=allp[(allp.model==kind)&(allp.horizon_steps==h)&(allp[event]==flag)]
                if len(g):
                    values.append(abs(g.y_pred_delta_V-g.y_true_delta_V).to_numpy(float)*1000);positions.append(i*3+j);labels.append(f'{kind}\nH{h}\n{int(flag)}');colors.append(COL[kind])
        b=ax.boxplot(values,positions=positions,patch_artist=True,showfliers=True,flierprops={'markersize':1,'alpha':.25})
        for patch,color in zip(b['boxes'],colors):patch.set_facecolor(color);patch.set_alpha(.55)
        ax.set_xticks(positions,labels,fontsize=8);ax.set_yscale('symlog',linthresh=1)
        ax.set(ylabel='Absolute error (mV); symlog includes zero',title=title)
    save(fig,'F7_operating_state_error_distributions',read_csv(out/'HOLDOUT_EVENT_METRICS.csv'))
    losses=[];fig,axes=plt.subplots(3,2,figsize=(12,9),sharey=True)
    for ax,(kind,h) in zip(axes.ravel(),jobs):
        l=read_csv(out/f'models/{kind}_H{h}/training_history.csv');l['model']=kind;l['horizon_steps']=h;losses.append(l)
        ax.semilogy(l.epoch,l.train_MSE,label='Train',color='#0072B2');ax.semilogy(l.epoch,l.validation_MSE,label='Validation',color='#D55E00')
        ax.set(xlabel='Epoch',ylabel='Standardized raw DeltaV MSE',title=f'{kind} H{h}, best{int(l.best_epoch.iloc[-1])}');ax.legend(fontsize=8)
    save(fig,'F8_actual_training_losses',pd.concat(losses))
    fig,axes=plt.subplots(1,3,figsize=(12,3.5))
    pair=metric[(metric.model.isin(['C0','C1']))&(metric.scope=='ALL_H_LEGAL')]
    for ax,key,title in zip(axes,['RMSE_mV','MAE_mV','P99_mV'],['RMSE','MAE','P99 absolute error']):
        for i,part in enumerate(['TRAIN','VALIDATION','HOLDOUT']):
            for j,kind in enumerate(['C0','C1']):
                row=pair[(pair.model==kind)&(pair.partition==part)].iloc[0]
                ax.bar(i+(j-.5)*.34,row[key],width=.34,color=COL[kind],label=kind if i==0 else None,edgecolor='black',linewidth=.5)
        ax.set_xticks(range(3),['Train','Validation','Holdout']);ax.set(ylabel='mV',title=f'C0/C1 paired Seed202: {title}');ax.legend()
    save(fig,'F9_C0_C1_paired_metrics',pair)
    paths=read_csv(out/'ALL_HOLDOUT_PATH_DIAGNOSTICS.csv');fig,axes=plt.subplots(2,2,figsize=(11,6),sharex=True,sharey=True)
    for ax,(kind,h) in zip(axes.ravel(),[('F2',6),('F2',12),('C0',24),('C1',24)]):
        g=paths[(paths.model==kind)&(paths.horizon_steps==h)]
        ax.scatter(g.origin_age_h,g.g,s=5,color=COL[kind]);ax.set(xlabel='Origin operating hours',ylabel='Historical gate g',ylim=(0,1),title=f'{kind} H{h}; past-history gate only')
    save(fig,'F10_historical_gate_behavior',paths)
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    summary=[]
    for kind,h in [('F2',6),('F2',12),('C0',24),('C1',24)]:
        g=paths[(paths.model==kind)&(paths.horizon_steps==h)]
        summary.append(dict(model=kind,horizon_steps=h,weighted_LSTM_RMS=g.weighted_LSTM_RMS.mean(),weighted_CNN_RMS=g.weighted_CNN_RMS.mean(),g_mean=g.g.mean(),g_std=g.g.std(ddof=0)))
    sm=pd.DataFrame(summary);x=np.arange(len(sm));axes[0].bar(x-.18,sm.weighted_LSTM_RMS,.36,color='#0072B2',label='g*LSTM RMS');axes[0].bar(x+.18,sm.weighted_CNN_RMS,.36,color='#E69F00',label='(1-g)*CNN RMS')
    axes[0].set_xticks(x,[f'{r.model} H{r.horizon_steps}' for r in sm.itertuples()]);axes[0].set(ylabel='Weighted encoder latent RMS (unitless)',title='Latent scale, not voltage gain attribution');axes[0].legend(fontsize=8)
    cp=read_csv(out/'models/C1_H24/HOLDOUT_OUTPUT_COMPONENTS.csv');q=cp.groupby('horizon_step').residual_component_V.agg(['mean','min','max'])
    axes[1].plot(q.index*.5,q['mean']*1000,color=COL['C1'],label='Mean actual head correction');axes[1].fill_between(q.index*.5,q['min']*1000,q['max']*1000,color=COL['C1'],alpha=.2,label='Full min-max')
    axes[1].set(xlabel='Lead (h)',ylabel='Output correction (mV)',title=f'C1 alpha={cp.alpha.iloc[0]:.5f}; real bounded contribution');axes[1].legend(fontsize=8)
    save(fig,'F11_paths_and_output_head',sm);csv(tables/'F11_C1_HEAD_COMPONENTS.csv',cp)
    f=test[test.scope=='ALL_H_LEGAL'];fig,axes=plt.subplots(1,2,figsize=(11,3.7))
    labels=[f'{r.model} H{r.horizon_steps}' for r in f.itertuples()]
    axes[0].bar(range(len(f)),f.top1pct_SSE_fraction*100,color=[COL[x] for x in f.model],edgecolor='black',linewidth=.5)
    axes[0].set_xticks(range(len(f)),labels,rotation=35,ha='right');axes[0].set(ylabel='Total squared error from largest1% queries (%)',ylim=(0,100),title='All extreme points retained')
    axes[1].scatter(f.RMSE_mV,f.target_time_equal_RMSE_mV,c=[COL[x] for x in f.model],s=40)
    for r in f.itertuples():axes[1].annotate(f'{r.model} H{r.horizon_steps}',(r.RMSE_mV,r.target_time_equal_RMSE_mV),fontsize=7)
    lo=min(f.RMSE_mV.min(),f.target_time_equal_RMSE_mV.min());hi=max(f.RMSE_mV.max(),f.target_time_equal_RMSE_mV.max())
    axes[1].plot([lo,hi],[lo,hi],ls='--',color='.5');axes[1].set(xlabel='Query-weighted RMSE (mV)',ylabel='Target-time-equal RMSE (mV)',title='Overlapping queries are not independent events')
    save(fig,'F12_extreme_error_concentration',f)
    dump(out/'FIGURE_INDEX.json',dict(figure_pairs=12,PNG_dpi=300,formats=['PNG','PDF'],all_points_retained=True,
        figure_directory=str(figdir),data_directory=str(tables),fixed_continuous_lead_rule='terminal H; insert NaN at gaps, never concatenate different forecast leads',
        figure_case_rules=case,source='Real CSV only; no prediction smoothing or shifting'))
    print('Generated12 paper-candidate PNG/PDF pairs and real plotting CSVs',flush=True)

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path);a=p.parse_args();run(a.run or Path(read(HERE/'outputs/ACTIVE_RUN.json')['path']))
