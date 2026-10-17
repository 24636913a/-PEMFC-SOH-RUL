"""Real diagnostic CSV only, no synthetic or shifted predictions."""
from audit_common import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
C={'F0':'#777777','F1':'#DF9B39','F2':'#267AAF','LSTM':'#39956A','TCN':'#A169AF'}
def save(fig,name,table):
    cs(f'figures/{name}_plot_data.csv',table)
    for ext in ['png','pdf']:fig.savefig(path(f'figures/{name}.{ext}'),dpi=300,bbox_inches='tight',facecolor='white')
    plt.close(fig)
def main():
    check_protocol();plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
    f=rd(P/'outputs/EVENT_50_ROW_CONTEXT.csv');fig,axes=plt.subplots(4,2,figsize=(11,9),sharex='col')
    fields=[('cell_voltage_mean','Observed cell voltage (V)'),('actual_current_A_mean','Mean current (A)'),('zero_current_fraction','Zero-current fraction'),('datetime_span_h','Calendar span per bin (h)')]
    for j,(event,ids) in enumerate(EVENTS.items()):
        z=f.loc[f.event==event]
        for ax,(field,label) in zip(axes[:,j],fields):
            ax.plot(z.relative_row,z[field],'.-',lw=1,ms=3,color='#267AAF');ax.axvspan(-.2,1.2,color='#E49C3A',alpha=.2);ax.set(ylabel=label);ax.grid(alpha=.15)
        axes[0,j].set_title(f'{event}: target rows {ids[0]}/{ids[1]}, 50 observed bins')
        axes[-1,j].set_xlabel('Row offset from first core target (0.5 operating h / row)')
    fig.suptitle('Exploratory event context: voltage, current, off fraction and unequal calendar duration');fig.tight_layout();save(fig,'01_event_context',f[['event','row_id','relative_row','bin_end_ophrs2_h','cell_voltage_mean','actual_current_A_mean','actual_current_A_std','zero_current_fraction','load_transition_count','datetime_span_h','raw_sample_count','coverage_ratio']])
    minute=rd(P/'outputs/RAW_MINUTE_PLOT_DATA.csv');fig,axes=plt.subplots(3,2,figsize=(11,7),sharex='col')
    for j,event in enumerate(EVENTS):
        z=minute.loc[minute.event==event].copy();dt=pd.to_datetime(z.calendar_time);x=(dt-dt.iloc[0]).dt.total_seconds()/3600
        for ax,y,label in zip(axes[:,j],[z.voltage_V,z.current_A,z.age_min_h-z.age_min_h.iloc[0]],['Cell voltage (V)','Current (A)','Operating-clock advance (h)']):
            ax.plot(x,y,color='#267AAF',lw=1);ax.set(ylabel=label);ax.grid(alpha=.15)
        axes[0,j].set_title(f'{event}: real one-minute averages, 12210 raw samples');axes[-1,j].set_xlabel('Elapsed calendar time (h)')
    fig.suptitle('Raw clock check: the operating clock pauses during the zero-current interval');fig.tight_layout();save(fig,'02_raw_clock_mixture',minute)
    c=rd(P/'outputs/EVENT_SSE_CONTRIBUTIONS.csv');fig,axes=plt.subplots(2,1,figsize=(11,6.5),sharex=True);labels=[f'{m}\n{s}' for m in MODELS for s in SEEDS];x=np.arange(15)
    for ax,column,title in zip(axes,['query_weighted_SSE_share_pct','target_time_equal_MSE_share_pct'],['Original query-weighted squared error','Equal weight per unique target time']):
        first=[];second=[]
        for model in MODELS:
            for seed in SEEDS:
                z=c.loc[(c.model==model)&(c.seed==seed)].set_index('event');first.append(z.loc['E1',column]);second.append(z.loc['E2',column])
        ax.bar(x,first,label='E1: two target times',color='#DF9B39');ax.bar(x,second,bottom=first,label='E2: two target times',color='#A169AF');ax.bar(x,100-np.array(first)-np.array(second),bottom=np.array(first)+np.array(second),label='All other targets',color='#D8DFE3')
        ax.set(ylabel='Squared-error contribution (%)',title=title,ylim=(0,100))
    axes[0].legend(loc='upper center',bbox_to_anchor=(.5,1.27),ncol=3,frameon=False);axes[-1].set_xticks(x,labels);fig.tight_layout();save(fig,'03_event_error_contribution',c)
    near=rd(P/'outputs/EVENT_NEIGHBOR_PREDICTIONS.csv');fig,axes=plt.subplots(3,2,figsize=(11,8.5));cases=[]
    for i,seed in enumerate(SEEDS):
        for j,(event,(a,b)) in enumerate(EVENTS.items()):
            origin=a-12;z=near.loc[(near.seed==seed)&(near.event_context==event)&(near.origin_row==origin)].copy();cases.append(z)
            ax=axes[i,j];true=z.loc[z.model=='F2'];ax.plot(true.horizon_step,true.true_raw_target_voltage_V,'ko-',label='Observed',lw=1.8,ms=3)
            for model in MODELS:
                g=z.loc[z.model==model];ax.plot(g.horizon_step,g.predicted_voltage_V,'.-',label=model,color=C[model],lw=1.2)
            ax.set(title=f'{event}, Seed{seed}, same origin {origin}, n=6 targets',xlabel='Lead step (0.5 operating h)',ylabel='Cell voltage (V)');ax.grid(alpha=.15)
    handles,labels=axes[0,0].get_legend_handles_labels();fig.legend(handles,labels,loc='lower center',bbox_to_anchor=(.5,-.025),ncol=6,frameon=False);fig.suptitle('Exploratory matched predictions: no future conditions supplied, no forecast shifting');fig.tight_layout();save(fig,'04_matched_model_event_windows',pd.concat(cases))
    lags=rd(P/'outputs/ALL_STOP_LOCAL_TIMING.csv');fig,axes=plt.subplots(1,2,figsize=(11,4));lagplot=[]
    for ax,core,title in zip(axes,[True,False],['Two user-specified events','Other eligible stop-dominated bins']):
        z=lags.loc[lags.core_event==core]
        for k,model in enumerate(MODELS):
            g=z.loc[z.model==model];counts=g.lag_rows.value_counts();xx=np.arange(-5,6);freq=np.array([counts.get(n,0)/len(g)*100 for n in xx]);ax.bar(xx+(k-2)*.14,freq,width=.14,label=model,color=C[model])
            for n,v in zip(xx,freq):lagplot.append(dict(model=model,core_events=core,lag_rows=int(n),percent=float(v),overlapping_diagnostic_windows=len(g),physical_stop_bins=int(g.stop_target_row.nunique())))
        lo=min(-2,int(z.lag_rows.min()));hi=max(2,int(z.lag_rows.max()))
        ax.set(xlim=(lo-.6,hi+.6),xlabel='Predicted minimum row minus observed minimum row',ylabel='Diagnostic windows (%)',title=title);ax.set_xticks(range(lo,hi+1));ax.grid(axis='y',alpha=.15)
    axes[0].legend(ncol=3);fig.suptitle('Local timing only: negative = early; overlapping windows are not independent events');fig.tight_layout();save(fig,'05_local_timing',lagplot)
    n=rd(P/'outputs/NEAREST_HISTORY_FUTURE_DIAGNOSTIC.csv');fig,axes=plt.subplots(1,2,figsize=(11,4.2));chosen=[]
    for ax,event in zip(axes,EVENTS):
        z=n.loc[n.event==event];truth=z.loc[(z.reference_partition=='TRAIN')&(z['rank']==1)];ax.plot(truth.lead_step,truth.late_actual_V,color='black',lw=1.8,label='Late observed')
        for part,rank,color in [('TRAIN',1,'#267AAF'),('TRAIN',2,'#A169AF'),('VALIDATION',1,'#DF9B39')]:
            g=z.loc[(z.reference_partition==part)&(z['rank']==rank)];chosen.append(g);ax.plot(g.lead_step,g.reference_actual_V,'--',label=f'{part} neighbor #{rank}',color=color,lw=1)
        ax.set(title=f'{event}: history-only neighbor selection; n=24 future observations',xlabel='Future step (0.5 operating h)',ylabel='Observed cell voltage (V)');ax.grid(alpha=.15)
    handles,labels=axes[0].get_legend_handles_labels();fig.legend(handles,labels,loc='lower center',bbox_to_anchor=(.5,-.07),ncol=4,frameon=False);fig.suptitle('Future observations shown only after neighbors are selected by past inputs');fig.tight_layout();save(fig,'06_historical_neighbor_phase',pd.concat(chosen))
    gate=rd(P/'outputs/ORIGINAL_GATE_AND_PATH_OBSERVATIONS.csv');fig,axes=plt.subplots(2,3,figsize=(11,6.3));gplot=[]
    for j,seed in enumerate(SEEDS):
        for ax,field,label in [(axes[0,j],'g','Gate g (LSTM fraction)'),(axes[1,j],'path_difference_RMS','Hidden path-difference RMS')]:
            groups=[gate.loc[(gate.seed==seed)&(gate.partition==part),field].to_numpy() for part in ['TRAIN','VALIDATION','LATE_LIFE']]
            ax.boxplot(groups,showfliers=False,patch_artist=True);ax.set_xticks([1,2,3],['Train','Val','Late'])
            for k,(event,(a,b)) in enumerate(EVENTS.items()):
                z=gate.loc[(gate.seed==seed)&(gate.partition=='LATE_LIFE')&gate.origin_row.between(b-24,a-1)].copy();ax.scatter(np.full(len(z),3+(k-.5)*.12),z[field],s=7,alpha=.7,color=['#DF9B39','#A169AF'][k],label=event);z['event']=event;gplot.append(z)
            ax.set(ylabel=label,title=f'Seed{seed}');ax.grid(axis='y',alpha=.15)
    gate_plot=gate.copy();gate_plot['event']='BACKGROUND'
    for event,(a,b) in EVENTS.items():gate_plot.loc[(gate_plot.partition=='LATE_LIFE')&gate_plot.origin_row.between(b-24,a-1),'event']=event
    axes[0,0].legend();fig.suptitle('Observe original F2 activations only; no gate replacement, no causal attribution');fig.tight_layout();save(fig,'07_original_gate_and_paths',gate_plot)
    js('outputs/FIGURE_PROVENANCE.json',dict(script_sha256=sha(Path(__file__)),protocol_sha256=sha(P/'configs/DIAGNOSTIC_PROTOCOL.json'),figures=7,PNG_dpi=300,PDF='vector',scientific_status='EXPLORATORY_POSTHOC_DIAGNOSTIC_ONLY',new_model_predictions=False,no_stitched_forecasts=True))
if __name__=='__main__':main()
