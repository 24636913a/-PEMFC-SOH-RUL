"""Presentation-only QA amendment: break missing origins, move legend, show mV errors.
No model inference or statistical rule changes. Run after plot_figures.py.
"""
from plot_figures import *
def main():
    cfg();g=rd(P/'outputs/ALL_VALIDATION_GATES.csv');m=rd(P/'outputs/METRICS.csv')
    fig,axs=panels('Historical gates (gaps have no inferred gate values)','g (LSTM fraction)')
    for ax,s in zip(axs.flat,SEEDS):
        f=g[g.seed==s];splits=np.where(np.diff(f.origin_age_h.to_numpy())>.5+1e-9)[0]+1
        for part in np.split(np.arange(len(f)),splits):
            z=f.iloc[part];ax.plot(z.origin_age_h,z.g,color=COL['B0'],lw=1,marker='.',ms=2)
        ax.axhline(.5,color='grey',ls='--',lw=.7);ax.set_xlabel('Origin operating time (h)')
    save(fig,'01_gate_over_time')
    x=np.arange(6);fig,axs=plt.subplots(1,2,figsize=(12,4.5),layout='constrained')
    for k,mode in enumerate(['B0','B1','B2']):
        f=m[m['mode']==mode].set_index('seed').loc[SEEDS]
        axs[0].bar(x+(k-1)*.25,f.RMSE_mV,.24,color=COL[mode],edgecolor='black',lw=.5,hatch=['','//','..'][k],label=LABEL[mode])
        if mode!='B0':axs[1].plot(x,f.delta_RMSE_vs_B0_mV,marker='o',color=COL[mode],label=LABEL[mode])
    for ax in axs:ax.set_xticks(x,[str(s) for s in SEEDS]);ax.set_xlabel('Seed');ax.grid(axis='y',alpha=.2)
    axs[0].set_ylabel('RMSE (mV)');axs[1].set_ylabel('RMSE minus B0 (mV)');axs[1].axhline(0,color='black',lw=.7)
    handles,ls=axs[0].get_legend_handles_labels();fig.legend(handles,ls,loc='outside lower center',ncol=3,fontsize=9)
    fig.suptitle('Frozen-weight interventions | Validation | all 4920 queries/seed');save(fig,'03_fixed_gate_RMSE')
    cases=rd(P/'outputs/FIGURE_WINDOW_DATA.csv');cases=cases[cases['case']=='most_unfavorable_to_B0_vs_B2']
    fig,axs=panels('Adverse-window errors: B0 vs B2 selection rule fixed before inference','Prediction error (mV)')
    for ax,s in zip(axs.flat,SEEDS):
        for mode in ['B0','B1','B2']:
            f=cases[(cases.seed==s)&(cases['mode']==mode)]
            ax.plot(f.horizon_step*.5,(f.y_pred_delta_V-f.y_true_delta_V)*1000,color=COL[mode],lw=1.3,label=LABEL[mode])
        ax.axhline(0,color='black',lw=.7);ax.set_title(f'Seed {s} | row {int(f.origin_row.iloc[0])}');ax.set_xlabel('Forecast lead (h)')
    handles,ls=axs.flat[0].get_legend_handles_labels();fig.legend(handles,ls,loc='outside lower center',ncol=3,fontsize=9);save(fig,'08_unfavorable_window_errors')
    print('PRESENTATION_QA_AMENDMENT_COMPLETE')
if __name__=='__main__':main()
