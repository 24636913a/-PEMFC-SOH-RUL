"""Six original 300dpi/vector figures, with explicit CSV plot data."""
from baseline_common import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
MODELS=['LSTM','TCN','F0','F1','F2']
COLORS=dict(LSTM='#767676',TCN='#9A4D8E',F0='#B64342',F1='#42949E',F2='#0F4D92')
MARKERS=dict(LSTM='o',TCN='s',F0='^',F1='D',F2='v')
plt.rcParams.update({'font.family':['Arial','DejaVu Sans'],'font.size':13,'axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':1.3,'legend.frameon':False,'pdf.fonttype':42,'savefig.facecolor':'white'})

def main():
    cfg();table=rd(P/'tables/FIVE_MODEL_PER_SEED.csv');sm=rd(P/'tables/FIVE_MODEL_SUMMARY.csv').set_index('model').loc[MODELS];h=rd(P/'tables/HORIZON_METRICS.csv');groups=rd(P/'tables/CONDITION_METRICS.csv');cases=rd(P/'outputs/predictions/FIXED_CASES.csv');index=[]
    def save(fig,stem,source,caption):
        fig.suptitle('RealHyFC Development Validation | 3 fixed seeds | 205 origins x 24 leads',fontsize=12)
        fig.text(.5,.018,caption,ha='center',fontsize=9);fig.tight_layout(rect=[0,.055,1,.93],pad=1.5)
        for ext in ['png','pdf']:fig.savefig(path('figures/'+stem+'.'+ext),dpi=300)
        plt.close(fig);index.append(dict(stem='figures/'+stem,data=source,script='scripts/plot_baselines.py',caption=caption,stage='Development Validation; not independent Test',dpi=300))
    def bars(ax,metric,label):
        for i,name in enumerate(MODELS):
            z=table[table.model.eq(name)][metric];ax.bar(i,z.mean(),yerr=z.std(ddof=1),color=COLORS[name],edgecolor='black',linewidth=.7,capsize=4,alpha=.8)
            ax.scatter(i+np.array([-.11,0,.11]),z,marker=MARKERS[name],color='black',s=24,zorder=3);ax.text(i,max(z.max(),z.mean()+z.std(ddof=1))+max(sm[metric+'_mean'])*.05,f'{z.mean():.3f}',ha='center',fontsize=9)
        ax.set_xticks(range(5),MODELS);ax.set_ylabel(label);ax.set_ylim(bottom=0);ax.margins(y=.22)
    fig,ax=plt.subplots(figsize=(10.5,5.7));bars(ax,'RMSE_mV','Query-weighted RMSE (mV)');save(fig,'F01_five_model_RMSE','tables/FIVE_MODEL_PER_SEED.csv','Bars: equal-seed mean; whiskers: sample SD; points: seeds 42/52/62. No selected F2 seeds.')
    fig,axes=plt.subplots(1,2,figsize=(12,5.5));bars(axes[0],'MAE_mV','MAE (mV)');bars(axes[1],'endpoint_RMSE_mV','12h endpoint RMSE (mV)');save(fig,'F02_MAE_and_endpoint','tables/FIVE_MODEL_PER_SEED.csv','All 4920 queries retained; endpoint: 205 queries per seed. SD is not a confidence interval.')
    fig,ax=plt.subplots(figsize=(10.5,5.8));plot=[]
    for name in MODELS:
        z=h[h.model.eq(name)].groupby('horizon_h').RMSE_mV.agg(['mean','min','max']).reset_index();ax.plot(z.horizon_h,z['mean'],label=name,color=COLORS[name],marker=MARKERS[name],markevery=3,lw=2);ax.fill_between(z.horizon_h,z['min'],z['max'],color=COLORS[name],alpha=.09)
        plot.extend(dict(model=name,horizon_h=r.horizon_h,mean=r.mean,min=r.min,max=r.max) for r in z.itertuples())
    cs('outputs/metrics/F03_plot_data.csv',plot);ax.set(xlabel='Forecast lead (h)',ylabel='RMSE (mV)',xlim=(.5,12),ylim=(0,None));ax.legend(ncol=5);save(fig,'F03_24_step_RMSE','outputs/metrics/F03_plot_data.csv','Lines: mean of three seeds; shading: observed seed range. One fixed 24-step task.')
    fig,grid=plt.subplots(2,2,figsize=(11.8,9));axes=grid.ravel();names={'current_variability':'Historical current variability','load_transitions':'Historical load transitions','load_level':'Historical mean current density'};plot=[]
    for ax,condition in zip(axes[:3],names):
        for j,name in enumerate(MODELS):
            g=groups[(groups.condition==condition)&(groups.model==name)];mu=[];sd=[]
            for level in ['LOW','MID','HIGH']:
                z=g[g.group.eq(level)];mu.append(z.RMSE_mV.mean());sd.append(z.RMSE_mV.std(ddof=1));plot.append(dict(condition=condition,group=level,model=name,mean_RMSE_mV=mu[-1],sample_SD_mV=sd[-1],origins=int(z.origins.iloc[0])))
            ax.errorbar(np.arange(3)+(j-2)*.045,mu,yerr=sd,color=COLORS[name],marker=MARKERS[name],label=name,capsize=2,lw=1.6)
        ns=[int(groups[(groups.condition==condition)&(groups.model=='F2')&(groups.group==lev)].origins.iloc[0]) for lev in ['LOW','MID','HIGH']];ax.set_xticks(range(3),[f'{lev}\nn={n}' for lev,n in zip(['LOW','MID','HIGH'],ns)]);ax.set_title(names[condition],fontsize=11);ax.set_ylim(bottom=0)
    axes[0].set_ylabel('RMSE (mV)');axes[2].set_ylabel('RMSE (mV)');handles,labels=axes[0].get_legend_handles_labels();axes[3].axis('off');axes[3].legend(handles,labels,loc='center',ncol=1,fontsize=14,title='Equal-seed mean +/- sample SD');cs('outputs/metrics/F04_plot_data.csv',plot);save(fig,'F04_history_conditions','outputs/metrics/F04_plot_data.csv','Groups use frozen Train terciles over observed 48-step histories; future conditions are not known.')
    for origin,stem,error in [(2205,'F05_fixed_origin_voltage',False),(2224,'F06_difficult_origin_error',True)]:
        fig,ax=plt.subplots(figsize=(10.5,5.8));z=cases[cases.origin_row.eq(origin)];baseline=z[z.model.eq('F2')].sort_values('horizon_step');x=baseline.horizon_step.to_numpy()*.5
        if not error:ax.plot(x,baseline.true_raw_target_voltage_V,color='black',lw=2.5,label='Observed target voltage')
        else:ax.axhline(0,color='black',lw=1)
        for name in MODELS:
            f=z[z.model.eq(name)].sort_values('horizon_step');y=(f.y_pred_delta_V-f.y_true_delta_V)*1000 if error else f.predicted_voltage_V;ax.plot(x,y,label=name,color=COLORS[name],marker=MARKERS[name],markevery=2,lw=1.8)
        if error:ax.axvspan(11.25,12.05,color='#F6CFCB',alpha=.55,zorder=0)
        ax.set(xlabel='Forecast lead from this origin (h)',ylabel='Signed prediction error (mV)' if error else 'Mean cell voltage (V)',xlim=(.5,12))
        if error:ax.legend(ncol=3)
        else:ax.legend(ncol=3,loc='lower center',bbox_to_anchor=(.5,1.02),fontsize=11)
        save(fig,stem,'outputs/predictions/FIXED_CASES.csv',f'All five models: seed 62, single fixed origin row {origin}; no window stitching. '+('Targets 2247/2248 highlighted.' if error else 'Raw origin voltage plus predicted DeltaV.'))
    cs('outputs/audits/FIGURE_INDEX.csv',index);write('figures/CAPTIONS.md','# Candidate figures\n\n'+'\n\n'.join(f"- {r['stem']}: {r['caption']} Data: `{r['data']}`." for r in index)+'\n')
    print('SIX_FIGURE_PAIRS_CREATED',flush=True)

if __name__=='__main__':main()
