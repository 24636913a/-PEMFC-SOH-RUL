"""Original English journal-style candidates. Every output has CSV/spec provenance."""
from common import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch,FancyArrowPatch
from matplotlib.lines import Line2D
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'savefig.dpi':300})
COL={'F0':'#777f87','F1':'#d17d46','F2':'#236fa8','Persistence':'#a74a4a','B0':'#236fa8','B1':'#b94e4b','B2':'#2c8c71'}
INDEX=[]
def save(fig,number,name,source,placement,caption):
    stem=f'F{number:02d}_{name}';fig.savefig(dst('figures/'+stem+'.png'),dpi=300);fig.savefig(dst('figures/'+stem+'.pdf'));plt.close(fig)
    INDEX.append(dict(id=f'F{number:02d}',stem='figures/'+stem,source=source,script='scripts/plot_paper_figures.py',placement=placement,caption=caption,stage='Development; no independent Test'))
def box(ax,x,y,w,h,label,color='#e3eaf1'):
    p=FancyBboxPatch((x,y),w,h,boxstyle='round,pad=.012',fc=color,ec='#364654',lw=1);ax.add_patch(p);ax.text(x+w/2,y+h/2,label,ha='center',va='center',fontsize=10);return p
def arrow(ax,a,b,label=None):
    ax.add_patch(FancyArrowPatch(a,b,arrowstyle='-|>',mutation_scale=12,lw=1,color='#45525c'))
    if label:ax.text((a[0]+b[0])/2,(a[1]+b[1])/2+.025,label,ha='center',fontsize=8)
def main():
    verified_config();m=rd(P/'outputs/experiment_tables/FORMAL_VALIDATION_METRICS.csv');h=rd(P/'outputs/experiment_tables/HORIZON_METRICS.csv');f=rd(P/'outputs/experiment_tables/ALIGNED_FORMAL_PREDICTIONS.csv')
    fig,ax=plt.subplots(figsize=(12,4.8),layout='constrained');ax.set(xlim=(0,1),ylim=(0,1));ax.axis('off')
    box(ax,.02,.63,.23,.23,'Legal past measurements\n48 half-hour bins\n14 raw input slots')
    box(ax,.36,.63,.27,.23,'Frozen Train-derived processing\nTop10 mask + causal EWMA(.3)\nTrain-only scaler')
    box(ax,.74,.63,.23,.23,'Dual-path TFT\nDirect 24-step DeltaV\nFuture actual covariates absent')
    arrow(ax,(.25,.745),(.36,.745));arrow(ax,(.63,.745),(.74,.745))
    box(ax,.05,.16,.25,.22,'Official comparison\nF0 / F1 / F2\nSeeds42 /52 /62','#f0f0ed');box(ax,.37,.16,.27,.22,'Horizon + regime + tail analysis\nTrain-frozen regime thresholds\nAll queries retained','#e8f0eb')
    box(ax,.74,.16,.23,.22,'Separate evidence\nP2 retrained ablations\nSelected-seed diagnostics','#faeee7')
    arrow(ax,(.85,.63),(.5,.38));arrow(ax,(.3,.27),(.37,.27));arrow(ax,(.74,.27),(.64,.27))
    ax.set_title('Forecasting and evaluation workflow | RealHyFC Development only',pad=14)
    save(fig,1,'workflow','outputs/experiment_tables/DIAGRAM_SPEC.json','main','Original workflow; all post-hoc selection and ablations identified separately.')
    fig,ax=plt.subplots(figsize=(13,7),layout='constrained');ax.set(xlim=(0,1),ylim=(0,1));ax.axis('off')
    box(ax,.045,.76,.17,.15,'Past scaled input\n[B,48,14] + time index')
    box(ax,.28,.76,.15,.15,'Native encoder VSN\n[B,48,64]')
    box(ax,.53,.78,.19,.15,'Native LSTM encoder\npost-LSTM residual/norm')
    box(ax,.53,.54,.19,.16,'Added causal CNN path\nk3,d1 and k3,d4\nconcat ->1x1 +residual/norm','#e1edf7')
    box(ax,.80,.65,.17,.19,'Encoder feature fusion\ng LSTM +(1-g) CNN\n[B,48,64]','#dcefdc')
    box(ax,.23,.40,.24,.13,'5 historical regime means\n5->8(ELU)->1(sigmoid)\none scalar g / history','#dcefdc')
    box(ax,.045,.09,.29,.18,'Native LSTM decoder ->[B,24,64]\nExternal input: relative step only\nInitial state: encoder hidden/cell')
    box(ax,.56,.075,.41,.23,'Concat encoder48 + decoder24\nNative enrichment / attention / output head\n24 direct standardized DeltaV predictions')
    arrow(ax,(.215,.835),(.28,.835));arrow(ax,(.43,.84),(.53,.855));arrow(ax,(.43,.79),(.53,.62))
    arrow(ax,(.72,.85),(.80,.75));arrow(ax,(.72,.62),(.80,.68));arrow(ax,(.885,.65),(.885,.305))
    # The decoder receives the ORIGINAL LSTM hidden/cell, not fused encoder states.
    ax.plot([.625,.625,.012,.012,.18],[.93,.975,.975,.315,.315],color='#45525c',lw=1);arrow(ax,(.18,.315),(.18,.27));ax.text(.23,.945,'Original encoder hidden / cell',fontsize=8,ha='center')
    arrow(ax,(.334,.18),(.56,.18),'decoder output')
    ax.plot([.12,.12],[.76,.465],color='#45525c',lw=1);arrow(ax,(.12,.465),(.23,.465))
    ax.plot([.47,.765,.765],[.455,.455,.61],color='#45525c',lw=1);arrow(ax,(.765,.61),(.80,.67));ax.text(.69,.47,'g',fontsize=9)
    ax.set_title('F2 architecture: inherited TFT modules and added history-path fusion',pad=14)
    save(fig,2,'architecture','outputs/experiment_tables/DIAGRAM_SPEC.json','main','Original source-based architecture. g weights encoder features, not final forecast contributions.')
    fig,axs=plt.subplots(1,2,figsize=(12,4),layout='constrained')
    axs[0].broken_barh([(0,2103)],(0,1),facecolors='#dddddd');axs[0].broken_barh([(2103,301)],(0,1),facecolors='#7ca8c8');axs[0].broken_barh([(2404,601)],(0,1),facecolors='#dec8a3')
    axs[0].set(xlim=(0,3005),ylim=(0,1),yticks=[],xlabel='Legal Development row index');axs[0].text(1000,.5,'Train\n1380 windows',ha='center',va='center');axs[0].text(2253,.5,'Val\n205',ha='center',va='center',fontsize=8);axs[0].text(2704,.5,'Reused\nAssessment\n378 windows',ha='center',va='center',fontsize=9)
    axs[1].axvspan(-24,0,color='#dddddd');axs[1].axvspan(0,12,color='#b5d3e8');axs[1].axvline(0,color='black');axs[1].set(xlim=(-24,12),ylim=(0,1),yticks=[],xlabel='Time relative to forecast origin (h)');axs[1].text(-12,.55,'48 historical bins\n24 h observed input',ha='center');axs[1].text(6,.55,'24 direct outputs\n12 h forecast',ha='center');axs[1].text(0,.1,'Origin',ha='center')
    fig.suptitle('Frozen data split and 48-to-24 forecasting task | 0.5 h bins')
    save(fig,3,'data_task','tables/T02_SPLITS.csv;outputs/experiment_tables/DIAGRAM_SPEC.json','main','Schematic only; no raw/Test values plotted. Assessment shown as reused Development.')
    fig,axs=plt.subplots(1,2,figsize=(11,4.5),layout='constrained')
    for i,model in enumerate(['F0','F1','F2']):
        z=m[m.model==model];axs[0].bar(i,z.RMSE_mV.mean(),yerr=z.RMSE_mV.std(ddof=1),color=COL[model],alpha=.65,capsize=4,ec='black',lw=.6)
        axs[0].scatter(i+np.array([-.09,0,.09]),z.RMSE_mV,color=COL[model],edgecolors='black',s=35,zorder=4)
    axs[0].set(xticks=range(3),xticklabels=['F0','F1','F2'],ylabel='Query-weighted RMSE (mV)',xlabel='Formal model | all three predefined seeds')
    axs[1].scatter(range(4),[m[m.model==k].RMSE_mV.mean() for k in ['Persistence','F0','F1','F2']],c=[COL[k] for k in ['Persistence','F0','F1','F2']],s=65);axs[1].set(xticks=range(4),xticklabels=['Persistence','F0','F1','F2'],yscale='log',ylabel='RMSE (mV), logarithmic scale')
    for ax in axs:ax.grid(axis='y',alpha=.2)
    fig.suptitle('Development Validation | 205 origins, 4920 queries/run | bars: seed mean ± sample SD')
    save(fig,4,'formal_model_comparison','outputs/experiment_tables/FORMAL_VALIDATION_METRICS.csv','main','Official H seeds only; persistence is one deterministic baseline; SD is not a confidence interval.')
    fig,axs=plt.subplots(1,2,figsize=(12,4.5),layout='constrained')
    for model in ['F0','F1','F2']:
        z=h[h.model==model].groupby('lead_h').RMSE_mV.agg(['mean','min','max']);axs[0].plot(z.index,z['mean'],color=COL[model],label=model);axs[0].fill_between(z.index,z['min'],z['max'],color=COL[model],alpha=.12)
    for i,model in enumerate(['F0','F1','F2']):
        z=m[m.model==model].sort_values('seed');axs[1].scatter(i+np.array([-.08,0,.08]),z.endpoint_RMSE_mV,color=COL[model],s=45)
        for j,r in enumerate(z.itertuples()):axs[1].annotate(str(r.seed),(i+[-.08,0,.08][j],r.endpoint_RMSE_mV),xytext=(4,3),textcoords='offset points',fontsize=8)
    axs[0].set(xlabel='Forecast lead (h)',ylabel='RMSE (mV)');axs[0].legend();axs[1].set(xticks=range(3),xticklabels=['F0','F1','F2'],ylabel='12 h endpoint RMSE (mV)')
    for ax in axs:ax.grid(alpha=.18)
    fig.suptitle('Fixed 24-step model | Validation | 205 queries per horizon/seed | band: seed min-max')
    save(fig,5,'horizons_endpoint','outputs/experiment_tables/HORIZON_METRICS.csv;outputs/experiment_tables/FORMAL_VALIDATION_METRICS.csv','main','Horizon diagnostics of the same frozen models; not separately trained output-length experiments.')
    gm=rd(P/'outputs/condition_analysis/CONDITION_METRICS.csv');fig,axs=plt.subplots(1,3,figsize=(13,4.5),layout='constrained')
    names=['Historical current variability','Historical transition count','Historical load level']
    for ax,cond,title in zip(axs,CONDITIONS,names):
        z=gm[(gm.condition==cond)&(gm.model!='Persistence')];counts=[]
        for j,lab in enumerate(['LOW','MID','HIGH']):
            counts.append(int(z[z.group==lab].origins.iloc[0]))
            for k,model in enumerate(['F0','F1','F2']):
                zz=z[(z.group==lab)&(z.model==model)];xx=j+(k-1)*.2
                ax.errorbar(xx,zz.RMSE_mV.mean(),yerr=zz.RMSE_mV.std(ddof=1),fmt='o',capsize=3,color=COL[model],label=model if j==0 else None)
        ax.set_title(title,fontsize=11);ax.set(xticks=range(3),xticklabels=[f'{lab}\nn={n}' for lab,n in zip(['LOW','MID','HIGH'],counts)],ylabel='RMSE (mV)',xlabel='Train-defined history group');ax.grid(alpha=.18)
    handles,labels=axs[0].get_legend_handles_labels();fig.legend(handles,labels,loc='outside lower center',ncol=3);fig.suptitle('Validation regime diagnostics | 1380 Train windows define all thresholds | mean ± seed SD')
    save(fig,6,'history_regimes','outputs/condition_analysis/CONDITION_METRICS.csv;outputs/condition_analysis/TRAIN_THRESHOLDS.csv','main','Groups use past inputs only; overlapping windows and relative terciles are not independent operating datasets.')
    p2=rd(P/'tables/T07_INDEPENDENT_TRAINING_ABLATION.csv');gate=rd(SIX/'outputs/METRICS.csv');fig,axs=plt.subplots(1,2,figsize=(12,4.5),layout='constrained')
    for i,model in enumerate(['G0','G1','G2']):
        z=p2[p2.model==model];axs[0].errorbar(i,z.RMSE_mV.mean(),yerr=z.RMSE_mV.std(ddof=1),fmt='o',color='#404d59',capsize=5);axs[0].scatter(i+np.array([-.08,0,.08]),z.RMSE_mV,s=30,color=COL['F2'] if model=='G2' else '#999999')
    axs[0].set(xticks=range(3),xticklabels=['G0 fixed 0.5','G1 global learned','G2 dynamic'],ylabel='RMSE (mV)',title='P2: nine independently trained runs')
    for mode in ['B1','B2']:
        z=gate[gate['mode']==mode].set_index('seed').loc[SELECTED];axs[1].plot(range(6),z.delta_RMSE_vs_B0_mV,marker='o',color=COL[mode],label='Fixed0.5' if mode=='B1' else 'Fixed Train mean')
    axs[1].axhline(0,color='black',lw=.7);axs[1].set(xticks=range(6),xticklabels=SELECTED,ylabel='RMSE minus dynamic B0 (mV)',xlabel='Selected F2 seed',title='Six selected checkpoints: inference only');axs[1].legend(fontsize=9)
    for ax in axs:ax.grid(alpha=.18)
    fig.suptitle('Gate ablations | Development Validation | retraining and inference evidence kept separate')
    save(fig,7,'gate_evidence','tables/T07_INDEPENDENT_TRAINING_ABLATION.csv;outputs/experiment_tables/SIX_GATE_INTERVENTIONS.csv','main','P2 fails predefined independent-gain criteria; B1 is not retrained F1.')
    s=rd(SIX/'outputs/predictions/seed_202_B0.csv');s=s[s.origin_row==2205].copy();s['true_voltage_V']=s.origin_raw_voltage_V+s.y_true_delta_V;s['pred_voltage_V']=s.origin_raw_voltage_V+s.y_pred_delta_V;cs('outputs/experiment_tables/SELECTED_EXAMPLE_DATA.csv',s)
    fig,axs=plt.subplots(2,1,figsize=(10,6),layout='constrained',sharex=True);x=s.horizon_step*.5
    axs[0].plot(x,s.true_voltage_V,'k--',label='Measured');axs[0].plot(x,s.pred_voltage_V,color=COL['F2'],label='F2 seed202');axs[0].set_ylabel('Voltage (V)');axs[0].legend();axs[1].plot(x,(s.y_pred_delta_V-s.y_true_delta_V)*1000,color=COL['F2']);axs[1].axhline(0,color='black',lw=.7);axs[1].set(xlabel='Forecast lead (h)',ylabel='Prediction error (mV)')
    for ax in axs:ax.grid(alpha=.18)
    fig.suptitle('Selected best existing F2 instance | fixed origin2205 | Validation | one24-step forecast')
    save(fig,8,'selected_voltage_example','outputs/experiment_tables/SELECTED_EXAMPLE_DATA.csv','main_optional','Seed202 was selected before this task; example is illustrative, not unbiased performance evidence.')
    pool=rd(P/'outputs/experiment_tables/ALL_COMPLETED_SCREENED_SEEDS.csv');sel=rd(P/'tables/T06_SELECTED_F2_INSTANCES.csv');fig,axs=plt.subplots(1,2,figsize=(12,4.5),layout='constrained')
    rng=np.random.default_rng(20261017)  # display jitter only, unrelated to training seeds
    for i,model in enumerate(['R1','R2']):
        z=pool[pool['round']==model];axs[0].scatter(i+rng.uniform(-.15,.15,len(z)),z.RMSE_mV,s=30,c='#999999',alpha=.7)
    axs[0].set(xticks=[0,1],xticklabels=['Round1:18 completed','Round2:12 completed'],ylabel='Validation RMSE (mV)',title='All30 successful new runs')
    axs[1].scatter(range(6),sel.RMSE_mV,c=COL['F2'],s=50);axs[1].set(xticks=range(6),xticklabels=sel.seed,ylabel='Validation RMSE (mV)',xlabel='Selected instance',title='Six selected instances (42 from historical H)')
    for ax in axs:ax.grid(axis='y',alpha=.2)
    fig.suptitle('Validation-based selection disclosure |32 attempted,30 completed,2 failed | not unbiased seed statistics')
    save(fig,9,'seed_selection','outputs/experiment_tables/ALL_COMPLETED_SCREENED_SEEDS.csv;tables/T06_SELECTED_F2_INSTANCES.csv','supplement','R2 stopped after3 qualified new seeds;38 registered candidates unstarted. Historical42 not part of30 new completions.')
    fig,axs=plt.subplots(1,4,figsize=(13,4),layout='constrained')
    for ax,metric,title in zip(axs,['RMSE_mV','endpoint_RMSE_mV','target_time_equal_RMSE_mV','max_abs_error_mV'],['Query weighted','12 h endpoint','Target-time equal','Maximum absolute error']):
        for s in SEEDS:
            z=m[(m.seed==s)&m.model.isin(['F0','F1','F2'])].set_index('model').loc[['F0','F1','F2']];ax.plot(range(3),z[metric],marker={42:'o',52:'s',62:'^'}[s],lw=1,label=f'Seed{s}')
        ax.set(xticks=range(3),xticklabels=['F0','F1','F2'],ylabel='Error (mV)',title=title);ax.grid(alpha=.18)
    handles,labels=axs[0].get_legend_handles_labels();fig.legend(handles,labels,loc='outside lower center',ncol=3);fig.suptitle('Formal seed-wise robustness | Validation | all4920 targets retained, no outlier removal')
    save(fig,10,'metric_sensitivity','outputs/experiment_tables/FORMAL_VALIDATION_METRICS.csv','main','Seed62 reverses some rankings at endpoint and target-equal metrics; unique targets are not independent batteries.')
    hard=int(rd(P/'outputs/experiment_tables/CASE_SELECTION.csv').query("case=='hardest_formal_F2'").origin_row.iloc[0]);z=f[(f.origin_row==hard)&(f.seed==62)].copy();cs('outputs/experiment_tables/DIFFICULT_WINDOW_DATA.csv',z)
    fig,axs=plt.subplots(1,2,figsize=(12,4.5),layout='constrained')
    for model in ['F0','F1','F2']:
        zz=z[z.model==model];x=zz.horizon_step*.5;axs[0].plot(x,zz.origin_raw_voltage_V+zz.y_pred_delta_V,color=COL[model],label=model);axs[1].plot(x,(zz.y_pred_delta_V-zz.y_true_delta_V)*1000,color=COL[model],label=model)
    axs[0].plot(x,zz.origin_raw_voltage_V+zz.y_true_delta_V,'k--',label='Measured');axs[0].set_ylabel('Voltage (V)');axs[1].set_ylabel('Prediction error (mV)');axs[1].axhline(0,color='black',lw=.7)
    for ax in axs:ax.set_xlabel('Forecast lead (h)');ax.grid(alpha=.18)
    axs[0].legend();fig.suptitle(f'Difficult window | fixed rule: worst mean formal F2 origin={hard} | displayed seed62 |24 queries')
    save(fig,11,'difficult_window','outputs/experiment_tables/DIFFICULT_WINDOW_DATA.csv;outputs/experiment_tables/CASE_SELECTION.csv','main_optional','Selection is objective diagnostic using complete formal pool; not a predictor/model-selection rule.')
    sse=rd(P/'outputs/experiment_tables/TARGET_SSE.csv');fig,axs=plt.subplots(1,2,figsize=(12,4),layout='constrained')
    for model in ['F0','F1','F2']:
        z=sse[(sse.model==model)&(sse.seed==62)].sort_values('SSE_share',ascending=False);axs[0].plot(np.arange(0,len(z)+1),np.r_[0,z.SSE_share.cumsum().to_numpy()],color=COL[model],label=model)
        zz=z.sort_values('target_row');axs[1].scatter(zz.target_row,zz.MSE_mV2**.5,s=12,color=COL[model],alpha=.6,label=model)
    axs[0].set(xlabel='Targets ranked by total SSE contribution',ylabel='Cumulative SSE fraction',ylim=(0,1.02));axs[1].set(xlabel='Unique Validation target row',ylabel='Per-target RMS error (mV)',ylim=(0,None))
    for ax in axs:ax.grid(alpha=.18)
    axs[0].legend();fig.suptitle('Seed62 target concentration |251 unique times | all4920 overlapping forecast queries retained')
    save(fig,12,'target_concentration','outputs/experiment_tables/TARGET_SSE.csv','supplement','All rows plotted; concentrations help explain pooled/target-equal differences without deleting extreme targets.')
    cs('outputs/evidence_index/FIGURE_INDEX.csv',INDEX)
    text('figures/CAPTIONS.md','# Candidate figure captions\n\n'+'\n\n'.join(f"## {r['id']} ({r['placement']})\n\n{r['caption']}\n\nSource: `{r['source']}`; PNG300dpi + vector PDF; all graphics original." for r in INDEX)+'\n')
    print('TWELVE_ORIGINAL_FIGURE_PAIRS_COMPLETE')
if __name__=='__main__':main()
