"""Six real-cache figures; each voltage curve uses exactly one forecast origin."""
from screen_common import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'ps.fonttype':42,'savefig.dpi':300})
INDEX=[]
def save(fig,name,table,scope):
 csv=f'outputs/plot_data/{name}.csv';cs(csv,table);(P/'figures').mkdir(exist_ok=True)
 for ext in ['png','pdf']:
  path=f'figures/{name}.{ext}';fig.savefig(P/path,dpi=300,bbox_inches='tight');INDEX.append(dict(figure=path,data_csv=csv,code='scripts/generate_paper_figures.py',scope=scope))
 plt.close(fig)
def frame(seed):return rd(P/f'outputs/predictions/F2_seed_{seed}.csv')
def main():
 rank=rd(P/'outputs/F2_SEED_RANKING.csv');n=len(rank);top=rank.head(5);best=int(rank.seed.iloc[0]);typical=int(np.sort(data()['val_origins'])[len(data()['val_origins'])//2]);b=frame(best);sample=b.loc[b.origin_row.eq(typical)].sort_values('horizon_step')
 fig,ax=plt.subplots(figsize=(max(7,n*.42),4),layout='constrained');bars=ax.bar(range(n),rank.RMSE_mV,color=['#3B8A65' if i<5 else '#2878B5' for i in range(n)],edgecolor='black',lw=.5)
 for r in bars[:min(5,n)]:r.set_hatch('//')
 ax.set(xticks=range(n),xticklabels=rank.seed.astype(str),xlabel='Seed (ranked by Validation RMSE)',ylabel='RMSE (mV)',title=f'Development Validation | F2 seed screening | n={n}; hatched=Top5 selected')
 ax.bar_label(bars,fmt='%.2f',fontsize=7,padding=2);ax.margins(y=.15);save(fig,'F1_F2_seed_ranking',rank,'All valid seeds, no failed-seed imputation')
 fig,ax=plt.subplots(figsize=(6.8,4),layout='constrained');vals=rank.RMSE_mV.to_numpy();ax.boxplot([vals],positions=[0],widths=.3,showfliers=False)
 ax.scatter(np.linspace(-.09,.09,n),vals,s=30,color='#2878B5',zorder=3,label='Every completed seed')
 if n>1:ax.errorbar(.32,vals.mean(),yerr=vals.std(ddof=1),fmt='s',color='#D97928',capsize=5,label='All-valid mean +/- sample SD')
 historical=json.loads((H/'outputs/H_FINAL_SUMMARY.json').read_text());hist_f2=next(r['RMSE_mean_mV'] for r in historical['multi_seed_statistics'] if r['model']=='F2')
 ax.axhline(hist_f2,color='#777777',ls='--',lw=1,label='Historical F2 mean (n=3; reference only)')
 ax.set(xticks=[0],xticklabels=['F2 complete candidate set'],ylabel='RMSE (mV)',title=f'Development Validation | full distribution n={n}; no Top5-only mean');ax.legend(fontsize=8)
 save(fig,'F2_F2_full_seed_distribution',rank,'SD over initialization seeds, not independent datasets')
 fig,ax=plt.subplots(figsize=(8,4.5),layout='constrained');ax.plot(sample.target_age_h,sample.true_label_voltage_V,'o-',color='black',ms=3,label='Observed frozen target voltage')
 ax.plot(sample.target_age_h,sample.predicted_voltage_V,'x-',color='#2878B5',label=f'Validation-selected best seed{best}')
 ax.set(xlabel='Actual target operating time (h)',ylabel='Mean cell voltage (V)',title=f'Development Validation | one origin row{typical}, 24targets | selected best');ax.legend(fontsize=8)
 save(fig,'F3_best_seed_single_origin',sample,'Predeclared middle origin; one24step forecast, no overlapping-window stitching')
 fig,ax=plt.subplots(figsize=(8,4.5),layout='constrained');ax.plot(sample.target_age_h,sample.true_label_voltage_V,'o-',color='black',ms=3,label='Observed frozen target voltage');tables=[]
 colors=['#2878B5','#D97928','#3B8A65','#9A62A5','#777777']
 for seed,color in zip(top.seed,colors):
  z=frame(int(seed));z=z.loc[z.origin_row.eq(typical)].sort_values('horizon_step');tables.append(z);ax.plot(z.target_age_h,z.predicted_voltage_V,'x-',ms=3,color=color,label=f'Seed{seed}')
 ax.set(xlabel='Actual target operating time (h)',ylabel='Mean cell voltage (V)',title=f'Development Validation | selected Top{len(top)} at same origin row{typical}');ax.legend(fontsize=8,ncol=2)
 save(fig,'F4_top5_same_origin',pd.concat(tables),'Top5 selection disclosure; exact same24 target rows for every curve')
 horizon=rd(P/'outputs/F2_HORIZON_METRICS.csv');g=horizon.groupby('horizon_h').RMSE_mV.agg(['mean','min','max']).reset_index();z=horizon.loc[horizon.seed.eq(best)].sort_values('horizon_h')
 fig,ax=plt.subplots(figsize=(8,4.5),layout='constrained');ax.fill_between(g.horizon_h,g['min'],g['max'],alpha=.15,color='#2878B5',label='All-valid seed min/max, not CI')
 ax.plot(g.horizon_h,g['mean'],'o-',ms=3,color='#2878B5',label='All-valid seeds mean');ax.plot(z.horizon_h,z.RMSE_mV,'x-',color='#3B8A65',label=f'Selected best seed{best}')
 ax.set(xlabel='Forecast horizon (h; step1..24)',ylabel='RMSE (mV)',title=f'Development Validation | 205origins per horizon, {n}valid seeds');ax.set_xticks(np.arange(1,13));ax.legend(fontsize=8)
 save(fig,'F5_all24_horizon_errors',horizon,'All24 horizons retained; seed range is not statistical uncertainty')
 window=b.assign(se=(b.y_pred_delta_V-b.y_true_delta_V)**2).groupby('origin_row').se.mean();worst=int(window.idxmax());z=b.loc[b.origin_row.eq(worst)].sort_values('horizon_step')
 global_seed=int(rank.loc[rank.max_abs_error_mV.idxmax(),'seed']);g=frame(global_seed);point=int((g.y_pred_delta_V-g.y_true_delta_V).abs().idxmax());global_origin=int(g.loc[point,'origin_row']);gz=g.loc[g.origin_row.eq(global_origin)].sort_values('horizon_step')
 cs('outputs/GLOBAL_WORST_POINT.csv',g.loc[[point]]);cs('outputs/GLOBAL_WORST_WINDOW.csv',gz)
 fig,axs=plt.subplots(2,2,figsize=(12,6),layout='constrained');plotparts=[]
 for col,case,t,seed,origin in [(0,'Selected best: worst window RMSE',z,best,worst),(1,'Full pool: maximum absolute error',gz,global_seed,global_origin)]:
  axs[0,col].plot(t.target_age_h,t.true_label_voltage_V,'o-',ms=3,color='black',label='Observed');axs[0,col].plot(t.target_age_h,t.predicted_voltage_V,'x-',color='#2878B5',label=f'Seed{seed}')
  axs[0,col].set(ylabel='Mean cell voltage (V)',title=f'{case}\nSeed{seed}, origin row{origin}');axs[0,col].legend(fontsize=8)
  axs[1,col].bar(t.target_age_h,1000*(t.y_pred_delta_V-t.y_true_delta_V),width=.32,color='#D97928');axs[1,col].axhline(0,color='black',lw=.7);axs[1,col].set(xlabel='Actual target operating time (h)',ylabel='Signed error (mV)');plotparts.append(t.assign(diagnostic_case=case))
 fig.suptitle('Development Validation | adverse windows retained | individual24step forecasts, no stitching')
 save(fig,'F6_worst_window_retained',pd.concat(plotparts),'Original selected-best worst window retained; added full-pool global maximum error window; no target removed')
 cs('outputs/FIGURE_SOURCE_INDEX.csv',pd.DataFrame(INDEX));js('outputs/audits/FIGURE_SELECTION.json',dict(typical_origin=typical,best_seed=best,worst_window_origin=worst,global_max_error_seed=global_seed,global_max_error_origin=global_origin,top5=top.seed.astype(int).tolist(),valid_seeds=n,overlap_stitching=False))
 print('SIX_REAL_CACHE_FIGURES_GENERATED',flush=True)
if __name__=='__main__':main()
