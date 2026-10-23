"""Six real-data, nonstitched publication candidate figures, 300dpi PNG + PDF."""
from screen_common import *
from reliable_io import js
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def main():
 c=cfg();rank=rd(P/'outputs/ROUND2_SEED_RANKING.csv') if (P/'outputs/ROUND2_SEED_RANKING.csv').exists() else pd.DataFrame()
 candidates=rd(P/'outputs/F2_SIX_CANDIDATE_SEEDS.csv');horizon=rd(P/'outputs/SIX_CANDIDATE_HORIZON_METRICS.csv')
 new=candidates[candidates.origin_round.eq('ROUND2_SELECTED')];origin=c['fixed_figure_origin_row'];n=len(rank);count=len(candidates)
 plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.labelsize':12,'axes.titlesize':12,'figure.titlesize':13,'pdf.fonttype':42,'ps.fonttype':42,'axes.spines.top':False,'axes.spines.right':False,'savefig.dpi':300})
 colors=plt.get_cmap('tab10').colors;manifest=[]
 def source(name,df):cs('outputs/figure_data/'+name+'.csv',df)
 def save(fig,name,title):
  fig.suptitle(title);fig.savefig(P/'figures'/f'{name}.png',dpi=300,bbox_inches='tight');fig.savefig(P/'figures'/f'{name}.pdf',bbox_inches='tight',metadata={'CreationDate':None,'ModDate':None});plt.close(fig)
  for ext in ['png','pdf']:
   f=P/'figures'/f'{name}.{ext}';manifest.append(dict(path=f.relative_to(P).as_posix(),sha256=sha(f),bytes=f.stat().st_size))
 def empty(ax,text):ax.text(.5,.5,text,ha='center',va='center',transform=ax.transAxes);ax.set_axis_off()
 (P/'outputs/figure_data').mkdir(parents=True,exist_ok=True)
 source('F1_ranking',rank)
 fig,ax=plt.subplots(figsize=(max(8,n*.28),4.5),layout='constrained')
 if n:
  bars=ax.bar(np.arange(n),rank.RMSE_mV,color=['#368765' if q else '#2878B5' for q in rank.qualified],edgecolor='black',linewidth=.6)
  for b,q in zip(bars,rank.qualified):
   if q:b.set_hatch('//')
  ax.axhline(6.3,color='#D46C24',linestyle='--',label='A threshold 6.30 mV');ax.set_xticks(np.arange(n),rank.seed.astype(str),rotation=90 if n>25 else 0);ax.set_ylabel('Validation pooled RMSE (mV)');ax.set_xlabel('Round2 seed, ranked by RMSE');ax.legend()
 else:empty(ax,'No completed new seeds; ranking unavailable')
 save(fig,'F1_round2_all_seed_ranking',f'Development Validation | Round2 screening | completed n={n}; hatch=A+B qualified')
 frames=[]
 for r in candidates.to_dict('records'):
  frame=rd(P/r['predictions']);g=frame[frame.origin_row.eq(origin)].copy();assert len(g)==24
  g['candidate_seed']=r['seed'];g['origin_round']=r['origin_round'];frames.append(g[['candidate_seed','origin_round','origin_row','target_row','horizon_step','target_age_h','true_label_voltage_V','predicted_voltage_V']])
 allcurves=pd.concat(frames,ignore_index=True);source('F2_new_selected_fixed_origin',allcurves[allcurves.candidate_seed.isin(new.seed)])
 fig,axes=plt.subplots(1,max(1,len(new)),figsize=(max(8,5*len(new)),4.3),squeeze=False,layout='constrained')
 if len(new):
  for ax,seed in zip(axes.ravel(),new.seed):
   g=allcurves[allcurves.candidate_seed.eq(seed)];ax.plot(g.target_age_h,g.true_label_voltage_V,'ko-',markersize=3,label='Observed frozen target');ax.plot(g.target_age_h,g.predicted_voltage_V,'x-',color='#2878B5',label=f'Seed{seed}');ax.set_xlabel('Target operating time (h)');ax.set_ylabel('Mean cell voltage (V)');ax.legend(fontsize=9)
 else:empty(axes[0,0],'No new A+B qualified seeds; no prediction curve invented')
 save(fig,'F2_new_qualified_prediction_curves',f'Development Validation | new selected n={len(new)} | fixed origin row{origin}, 24 targets each')
 source('F3_candidate_metrics',candidates)
 fig,axes=plt.subplots(1,2,figsize=(11,4.6),layout='constrained');x=np.arange(count)
 axes[0].bar(x-.18,candidates.RMSE_mV,width=.36,label='Pooled RMSE',color='#2878B5',edgecolor='black');axes[0].bar(x+.18,candidates.endpoint_RMSE_mV,width=.36,label='12h endpoint RMSE',color='#E58A2E',edgecolor='black',hatch='/');axes[0].axhline(6.3,color='gray',ls='--',lw=1);axes[0].set_ylabel('Error (mV)');axes[0].legend(fontsize=9)
 axes[1].bar(x,candidates.max_abs_error_mV,color=['#2878B5' if s=='HISTORICAL_SELECTED' else '#368765' for s in candidates.origin_round],edgecolor='black');axes[1].axhline(50,color='gray',ls='--',label='50mV maximum-error threshold');axes[1].set_ylabel('Maximum absolute error (mV)');axes[1].legend(fontsize=9)
 for ax in axes:ax.set_xticks(x,candidates.seed.astype(str));ax.set_xlabel('Selected candidate seed (historical first)')
 save(fig,'F3_historical_and_new_candidates',f'Development Validation | selected candidates n={count} | not an unbiased model comparison')
 source('F4_candidate_horizon_metrics',horizon)
 fig,ax=plt.subplots(figsize=(9,4.8),layout='constrained')
 for i,seed in enumerate(candidates.seed):
  g=horizon[horizon.seed.eq(seed)];assert len(g)==24;ax.plot(g.horizon_h,g.RMSE_mV,marker=['o','s','^','x','D','v'][i%6],markersize=3,color=colors[i%10],label=f'Seed{seed}')
 ax.axhline(10,color='gray',ls='--',lw=1,label='10mV endpoint criterion');ax.set_xlabel('Forecast horizon (h; 24 half-hour steps)');ax.set_ylabel('RMSE (mV; 205 origins per step)');ax.legend(ncol=2,fontsize=9)
 save(fig,'F4_candidate_24_step_RMSE',f'Development Validation | selected n={count}; threshold applies to step24 only')
 source('F5_all_candidates_fixed_origin',allcurves)
 fig,ax=plt.subplots(figsize=(11,5),layout='constrained');g=allcurves[allcurves.candidate_seed.eq(candidates.iloc[0].seed)];ax.plot(g.target_age_h,g.true_label_voltage_V,'ko-',label='Observed frozen target',markersize=4,lw=2)
 for i,seed in enumerate(candidates.seed):
  g=allcurves[allcurves.candidate_seed.eq(seed)];ax.plot(g.target_age_h,g.predicted_voltage_V,color=colors[i%10],marker=['x','s','^','D','v','+'][i%6],markersize=3,lw=1.1,label=f'Seed{seed}')
 ax.set_xlabel('Actual target operating time (h)');ax.set_ylabel('Mean cell voltage (V)');ax.legend(ncol=2,fontsize=9)
 save(fig,'F5_candidates_same_prediction_origin',f'Development Validation | origin row{origin}, 24 targets | selected n={count}; no stitching')
 distribution=rank[['seed','RMSE_mV','qualified','selection_status']].copy() if n else pd.DataFrame();source('F6_full_and_selected_distribution',distribution)
 fig,ax=plt.subplots(figsize=(8,4.8),layout='constrained')
 if n:
  groups=[rank.RMSE_mV.to_numpy()];labels=[f'All completed\nn={n}']
  selected=rank[rank.qualified]
  if len(selected):groups.append(selected.RMSE_mV.to_numpy());labels.append(f'A+B selected\nn={len(selected)}')
  ax.boxplot(groups,tick_labels=labels,showfliers=False,widths=.35)
  for i,g in enumerate(groups,1):
   jitter=np.linspace(-.12,.12,len(g));ax.scatter(i+jitter,g,color='#2878B5' if i==1 else '#368765',s=30,zorder=3);ax.scatter([i],[np.mean(g)],marker='D',color='#D46C24',s=45,zorder=4)
  ax.axhline(6.3,color='gray',ls='--',label='A threshold; diamonds=group mean');ax.legend(fontsize=9);ax.set_ylabel('Validation pooled RMSE (mV)')
 else:empty(ax,'No completed seeds; distribution unavailable')
 save(fig,'F6_before_after_selection_distribution',f'Development Validation | Round2 n={n} | selection bias shown, not generalization')
 cs('outputs/FIGURE_FILES.csv',pd.DataFrame(manifest));js('outputs/FIGURE_PROVENANCE.json',dict(status='REAL_CSV_DERIVED',base_commit=BASE,protocol_sha256=sha(P/'configs/frozen_seed_screening.json'),plot_script_sha256=sha(Path(__file__)),fixed_origin_row=origin,source_CSV_sha256={f.relative_to(P).as_posix():sha(f) for f in sorted((P/'outputs/figure_data').glob('*.csv'))},figures=manifest,stage='Development Validation; selected candidates',PNG_dpi=300,Test=False,stitched=False))
 print('SIX_FIGURES_GENERATED',flush=True)

if __name__=='__main__':main()
