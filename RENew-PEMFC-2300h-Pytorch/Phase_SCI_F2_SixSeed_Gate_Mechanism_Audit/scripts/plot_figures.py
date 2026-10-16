"""Publication candidates, exclusively from saved CSVs; 300dpi PNG and vector PDF."""
from common import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
COL={'B0':'#1764a0','B1':'#c34a45','B2':'#27916f','C':'#73787c'}
LABEL={'B0':'Dynamic (B0)','B1':'Fixed 0.5 (B1)','B2':'Fixed Train mean (B2)','C':'10 permutations'}
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'ps.fonttype':42,'savefig.dpi':300})
def panels(title,ylabel):
    fig,axs=plt.subplots(2,3,figsize=(12,7),layout='constrained',sharex=True)
    fig.suptitle(title+'\nDevelopment Validation | 205 origins/seed | selected F2 instances',fontsize=13)
    for ax,s in zip(axs.flat,SEEDS):ax.set_title(f'Seed {s}');ax.set_ylabel(ylabel);ax.grid(alpha=.18)
    return fig,axs
def save(fig,name):
    fig.savefig(dest('figures/'+name+'.png'),dpi=300);fig.savefig(dest('figures/'+name+'.pdf'));plt.close(fig)
def main():
    cfg();g=rd(P/'outputs/ALL_VALIDATION_GATES.csv');m=rd(P/'outputs/METRICS.csv');h=rd(P/'outputs/HORIZON_METRICS.csv')
    fig,axs=panels('Historical regime-dependent LSTM fusion fraction','g (LSTM fraction)')
    for ax,s in zip(axs.flat,SEEDS):
        f=g[g.seed==s];ax.plot(f.origin_age_h,f.g,color=COL['B0'],lw=1);ax.axhline(.5,color='grey',ls='--',lw=.7);ax.set_xlabel('Origin operating time (h)')
    save(fig,'01_gate_over_time')
    fig,axs=panels('Gate versus historical current-density variability','g (LSTM fraction)')
    for ax,s in zip(axs.flat,SEEDS):
        f=g[g.seed==s];ax.scatter(f.actual_current_density_A_cm2_std_raw_mean,f.g,s=12,alpha=.65,color=COL['B0']);ax.set_xlabel('48-bin mean of bin current-density SD (A/cm²)')
    save(fig,'02_gate_current_variability')
    labels=[str(s) for s in SEEDS];x=np.arange(6);fig,axs=plt.subplots(1,2,figsize=(12,4.5),layout='constrained')
    for k,mode in enumerate(['B0','B1','B2']):
        f=m[m['mode']==mode].set_index('seed').loc[SEEDS]
        axs[0].bar(x+(k-1)*.25,f.RMSE_mV,.24,color=COL[mode],edgecolor='black',lw=.5,hatch=['','//','..'][k],label=LABEL[mode])
        if mode!='B0':axs[1].plot(x,f.delta_RMSE_vs_B0_mV,marker='o',color=COL[mode],label=LABEL[mode])
    for ax in axs:ax.set_xticks(x,labels);ax.set_xlabel('Seed');ax.grid(axis='y',alpha=.2)
    axs[0].set_ylabel('RMSE (mV)');axs[0].legend(fontsize=9);axs[1].set_ylabel('RMSE minus B0 (mV)');axs[1].axhline(0,color='black',lw=.7);axs[1].legend(fontsize=9)
    fig.suptitle('Frozen-weight interventions | Validation | 4920 queries/seed (no target removal)');save(fig,'03_fixed_gate_RMSE')
    fig,axs=plt.subplots(1,2,figsize=(12,4.5),layout='constrained')
    for i,s in enumerate(SEEDS):
        f=m[(m.seed==s)&m['mode'].str.startswith('C_')];b=m[(m.seed==s)&(m['mode']=='B0')].iloc[0]
        axs[0].scatter(np.full(len(f),i),f.RMSE_mV,color=COL['C'],alpha=.5,s=20);axs[0].scatter(i,b.RMSE_mV,color=COL['B0'],marker='D',s=50)
        axs[1].scatter(np.full(len(f),i),f.delta_RMSE_vs_B0_mV,color=COL['C'],alpha=.5,s=20);axs[1].plot([i-.18,i+.18],[f.delta_RMSE_vs_B0_mV.mean()]*2,color=COL['B0'],lw=2)
    for ax in axs:ax.set_xticks(x,labels);ax.set_xlabel('Seed');ax.grid(axis='y',alpha=.2)
    axs[0].set_ylabel('RMSE (mV)');axs[1].set_ylabel('Permutation RMSE minus B0 (mV)');axs[1].axhline(0,color='black',lw=.7)
    axs[0].legend(handles=[Line2D([],[],marker='D',color=COL['B0'],ls='',label='B0'),Line2D([],[],marker='o',color=COL['C'],ls='',label='All 10 permutations')],fontsize=9)
    fig.suptitle('Label-free gate derangements | Validation | all 10 predeclared repeats shown');save(fig,'04_permutation_RMSE')
    fig,axs=panels('Forecast-horizon error under gate interventions','RMSE (mV)')
    for ax,s in zip(axs.flat,SEEDS):
        z=h[h.seed==s]
        for mode in ['B0','B1','B2']:
            f=z[z['mode']==mode];ax.plot(f.lead_h,f.RMSE_mV,color=COL[mode],lw=1.4,label=LABEL[mode])
        f=z[z['mode'].str.startswith('C_')];q=f.groupby('lead_h').RMSE_mV.agg(['mean','min','max'])
        ax.fill_between(q.index,q['min'],q['max'],color=COL['C'],alpha=.15);ax.plot(q.index,q['mean'],color=COL['C'],ls=':',lw=1,label='Permutation mean/range');ax.set_xlabel('Forecast lead (h)')
    handles,ls=axs.flat[0].get_legend_handles_labels();fig.legend(handles,ls,loc='outside lower center',ncol=4,fontsize=9);save(fig,'05_horizon_RMSE')
    selections=rd(P/'outputs/CASE_SELECTION.csv');plotdata=[]
    for case,filename,title in [('fixed_typical','06_fixed_window_predictions','Predeclared common origin 2205'),('most_unfavorable_to_B0_vs_B2','07_unfavorable_windows','Most adverse window for B0 versus B2 (each seed)')]:
        fig,axs=panels(title,'Absolute voltage (V)')
        for ax,s in zip(axs.flat,SEEDS):
            origin=int(selections[(selections.seed==s)&(selections['case']==case)].origin_row.iloc[0]);ax.set_title(f'Seed {s} | origin row {origin}')
            for mode in ['B0','B1','B2']:
                f=rd(P/f'outputs/predictions/seed_{s}_{mode}.csv');f=f[f.origin_row==origin].copy();f['case']=case;f['mode']=mode
                f['V_pred']=f.origin_raw_voltage_V+f.y_pred_delta_V;f['V_true']=f.origin_raw_voltage_V+f.y_true_delta_V;plotdata.append(f)
                ax.plot(f.horizon_step*.5,f.V_pred,color=COL[mode],lw=1.3,label=LABEL[mode])
            ax.plot(f.horizon_step*.5,f.V_true,color='black',lw=1.1,ls='--',label='Measured');ax.set_xlabel('Forecast lead (h)')
        handles,ls=axs.flat[0].get_legend_handles_labels();fig.legend(handles,ls,loc='outside lower center',ncol=4,fontsize=9);save(fig,filename)
    cs('outputs/FIGURE_WINDOW_DATA.csv',pd.concat(plotdata,ignore_index=True))
    print('SEVEN_PNG_AND_PDF_FIGURES_CREATED')
if __name__=='__main__':main()
