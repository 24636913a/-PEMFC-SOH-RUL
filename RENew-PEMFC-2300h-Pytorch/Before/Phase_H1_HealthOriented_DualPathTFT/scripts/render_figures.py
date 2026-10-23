"""Actual H1 event predictions only; no illustrative or interpolated predictions."""
from common import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'ps.fonttype':42,'savefig.dpi':300})
COLORS={'H-F0':'#777777','H-F1':'#D97928','H-F2':'#2878B5','Persistence':'#999999','Ridge96_control':'#3B8A65'}
INDEX=[]
def save(fig,name,sources,scope):
 for ext in ['png','pdf']:
  f=P/f'figures/{name}.{ext}';fig.savefig(f,dpi=300,bbox_inches='tight');INDEX.append(dict(figure=f.relative_to(P).as_posix(),CSV_sources=';'.join(sources),code='scripts/render_figures.py',scope=scope))
 plt.close(fig)
def broken_line(ax,g,value,label,color,marker='o',alpha=1.,lw=1.):
 g=g.sort_values('target_raw_cycle_id');blocks=g.target_raw_cycle_id.diff().ne(1).cumsum()
 for i,(_,b) in enumerate(g.groupby(blocks)):
  ax.plot(b.target_last_measurement_h,b[value],marker=marker,ms=3,lw=lw,color=color,alpha=alpha,label=label if i==0 else None)
def main():
 eq=rd(P/'outputs/PRIMARY_EQUAL_FOLD_METRICS.csv');fold=rd(P/'outputs/FOLD_THREE_SEED_MEANS.csv');st=rd(P/'outputs/THREE_SEED_STABILITY.csv').set_index('model');pred=rd(P/'outputs/ALL_POINT_PREDICTIONS.csv')
 order=['H-F0','H-F1','H-F2','Persistence','Ridge96_control'];q=eq.set_index('model').loc[order];fig,axs=plt.subplots(1,2,figsize=(11,4.8),layout='constrained')
 x=np.arange(5);sd=[st.loc[m,'MAE_mV_std'] if m in st.index else 0 for m in order]
 bars=axs[0].bar(x,q.MAE_mV,color=[COLORS[m] for m in order],edgecolor='black',lw=.5,yerr=sd,capsize=3);axs[0].bar_label(bars,fmt='%.3f',padding=5,fontsize=9)
 axs[0].set(xticks=x,xticklabels=['H-F0','H-F1','H-F2','Persist.','Ridge96'],ylabel='Equal-fold MAE (mV)',ylim=(0,1.55),title='Threefold primary; errorbars=3seed SD')
 xs=np.arange(3);width=.15
 for i,m in enumerate(order):
  z=fold.loc[fold.model.eq(m)].sort_values('fold');axs[1].bar(xs+(i-2)*width,z.MAE_mV,width=width,color=COLORS[m],label=m,edgecolor='black',lw=.4)
 axs[1].set(xticks=xs,xticklabels=['Fold1: n=13','Fold2: n=6','Fold3: n=22'],ylabel='Seed-mean MAE (mV)',title='Allfolds shown');axs[1].legend(fontsize=8)
 fig.suptitle('Development expandingfold H1 event forecast | fixed-state voltage | no outerValidation selection')
 save(fig,'F1_H1_ablation_and_baselines',['outputs/PRIMARY_EQUAL_FOLD_METRICS.csv','outputs/FOLD_THREE_SEED_MEANS.csv','outputs/THREE_SEED_STABILITY.csv'],'Threefold equal-weight; baselines not3seed replicated fits')
 fig,axs=plt.subplots(3,1,figsize=(10,8),layout='constrained')
 for ax,ff in zip(axs,['fold_1','fold_2','fold_3']):
  z=pred.loc[pred.fold.eq(ff)&(pred.seed.eq(42)|pred.seed.eq(-1))];truth=z.drop_duplicates('sample_key').copy()
  broken_line(ax,truth,'true_voltage_V','Observed frozen fixed-state V','black',lw=1.2)
  for m in ['H-F0','H-F1','H-F2']:broken_line(ax,z.loc[z.model.eq(m)],'predicted_voltage_V',m+' seed42',COLORS[m],marker='x')
  ax.set(ylabel='Fixed-state voltage (V)',title=f'{ff}, {truth.sample_key.nunique()} targets');ax.grid(alpha=.15)
 axs[0].legend(ncol=2,fontsize=8);axs[-1].set_xlabel('Actual target final measurement operating time (h)')
 fig.suptitle('Development | event target, not12h dynamic forecast | seed42 fixed before results; gaps preserved')
 save(fig,'F2_real_voltage_and_TFT_predictions',['outputs/ALL_POINT_PREDICTIONS.csv'],'Seed42 predeclared; no interpolation across missing cycles')
 fig,axs=plt.subplots(3,1,figsize=(10,8),layout='constrained')
 for ax,ff in zip(axs,['fold_1','fold_2','fold_3']):
  z=pred.loc[pred.fold.eq(ff)&pred.model.eq('H-F2')];truth=z.drop_duplicates('sample_key')
  broken_line(ax,truth,'true_SOH_V_percent','Observed voltage proxy','black',lw=1.2)
  for seed,color in [(42,'#2878B5'),(52,'#D97928'),(62,'#777777')]:broken_line(ax,z.loc[z.seed.eq(seed)],'predicted_SOH_V_percent',f'H-F2 seed{seed}',color,marker='x')
  ax.set(ylabel='Voltage SOH proxy (%)',title=f'{ff}: n={len(truth)}');ax.grid(alpha=.15)
 axs[0].legend(ncol=2,fontsize=8);axs[-1].set_xlabel('Actual operating time of target measurement (h)')
 fig.suptitle('SOH_V=100V/0.71383023635 | fixed-state proxy, NOT certified polarization SOH | no clipping')
 save(fig,'F3_voltage_SOH_proxy',['outputs/ALL_POINT_PREDICTIONS.csv'],'All3seeds; no ensemble, monotonicity or calibration fitting')
 metrics=rd(P/'outputs/FOLD_SEED_METRICS.csv');fig,ax=plt.subplots(figsize=(11,3.8),layout='constrained');mat=np.zeros((3,9))
 for i,m in enumerate(['H-F0','H-F1','H-F2']):
  for j,(ff,ss) in enumerate((f,s) for f in ['fold_1','fold_2','fold_3'] for s in [42,52,62]):mat[i,j]=float(metrics.loc[metrics.model.eq(m)&metrics.fold.eq(ff)&metrics.seed.eq(ss),'MAE_mV'].iloc[0])
 im=ax.imshow(mat,cmap='YlOrRd',aspect='auto')
 for i in range(3):
  for j in range(9):ax.text(j,i,f'{mat[i,j]:.3f}',ha='center',va='center',color='white' if mat[i,j]>1.8 else 'black')
 ax.set(yticks=range(3),yticklabels=['H-F0','H-F1','H-F2'],xticks=range(9),xticklabels=[f'F{f}\nSeed{s}' for f in [1,2,3] for s in [42,52,62]],title='Development H1 | all27 registered runs | MAE (mV)');fig.colorbar(im,ax=ax,label='MAE (mV)')
 save(fig,'F4_seed_and_fold_stability',['outputs/FOLD_SEED_METRICS.csv'],'n=13/6/22 perseed; seed variation is not independent life variability')
 fig,axs=plt.subplots(1,2,figsize=(11,4.8),layout='constrained')
 for ax,m,seed in [(axs[0],'H-F2',42),(axs[1],'Ridge96_control',-1)]:
  z=pred.loc[pred.model.eq(m)&pred.seed.eq(seed)];a=1000*(z.true_voltage_V-z.origin_voltage_V);p=1000*(z.predicted_voltage_V-z.origin_voltage_V)
  for ff,color in [('fold_1','#2878B5'),('fold_2','#D97928'),('fold_3','#777777')]:
   take=z.fold.eq(ff);ax.scatter(a[take],p[take],color=color,s=23,label=ff.replace('_',' '))
  lim=[min(a.min(),p.min())-.1,max(a.max(),p.max())+.1];ax.plot(lim,lim,color='black',lw=.8,ls='--',label='Perfect change prediction');ax.axhline(0,color='#3B8A65',ls=':',lw=1,label='Persistence prediction')
  ax.set(xlabel='Observed next-cycle voltage change (mV)',ylabel='Predicted next-cycle voltage change (mV)',title=m+(' seed42' if seed==42 else '')+f' | n={len(z)}');ax.legend(fontsize=8)
 fig.suptitle('Development | diagnostic of predicted performance changes | 24h dynamic vs96h cycle-history information differs')
 save(fig,'F5_change_tracking_and_controls',['outputs/ALL_POINT_PREDICTIONS.csv'],'Same41targets; direction/lag diagnostic, not a fitted time delay')
 cs('data_contracts/FIGURE_SOURCE_INDEX.csv',pd.DataFrame(INDEX));print('Five PNG/PDF figures generated')
if __name__=='__main__':main()
