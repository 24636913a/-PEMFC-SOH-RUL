"""Seven print-ready PNG/PDF panels: reused Development, not independent Test."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from i2_common import *

COLORS={'F0':'#3775BA','F1':'#68AE68','F2':'#B64342'}
TITLE='Reused Development Assessment — n=3 seeds'

def plot_all():
 plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Arial','Helvetica','DejaVu Sans'],
  'font.size':11,'axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':1.3,
  'legend.frameon':False,'pdf.fonttype':42,'savefig.dpi':300})
 m=read_csv(OUT/'metrics/I2_ASSESSMENT_METRICS.csv');p=read_csv(OUT/'metrics/I2_PAIRED_SEED_DIFFERENCES.csv')
 g=read_csv(OUT/'metrics/I2_MULTISEED_SUMMARY.csv');gap=read_csv(OUT/'metrics/I2_VALIDATION_ASSESSMENT_GAPS.csv')
 horizons=read_csv(OUT/'diagnostics/I2_ALL_HORIZON_METRICS.csv'); targets=read_csv(OUT/'diagnostics/I2_ALL_TARGET_SSE.csv')
 folder=PHASE/'figures';folder.mkdir(parents=True,exist_ok=True)
 def save(fig,name):
  fig.suptitle(TITLE,fontsize=13);fig.tight_layout(rect=(0,0,1,.91),pad=1.5)
  fig.savefig(folder/(name+'.png'),facecolor='white');fig.savefig(folder/(name+'.pdf'),facecolor='white');plt.close(fig)
 def bars(metric,ax):
  for j,model in enumerate(COLORS):
   v=m.loc[m.model==model].sort_values('seed')[metric].to_numpy()
   ax.bar(np.arange(3)+(j-1)*.24,v,width=.24,label=model,color=COLORS[model],edgecolor='black',linewidth=.6)
  ax.set_xticks(range(3),['42','52','62']);ax.set_xlabel('Frozen training seed');ax.legend()
 fig,ax=plt.subplots(figsize=(7.5,5));bars('RMSE_mV',ax);ax.set_ylabel('Forecast-query-weighted RMSE (mV)');save(fig,'Fig1_Assessment_Seed_RMSE')
 fig,ax=plt.subplots(figsize=(7.5,5));ax.bar(g.model,g.RMSE_mV_mean,yerr=g.RMSE_mV_sample_SD,capsize=5,color=[COLORS[x] for x in g.model],edgecolor='black',linewidth=.6)
 ax.set_ylabel('RMSE (mV)');ax.set_title('Equal-seed mean ± sample SD (ddof=1), not CI',fontsize=11);save(fig,'Fig2_Assessment_Mean_Sample_SD')
 fig,axs=plt.subplots(1,3,figsize=(12,5),sharey=True)
 for ax,model in zip(axs,COLORS):
  q=gap.loc[gap.model==model].sort_values('seed');ax.bar(np.arange(3)-.18,q.Validation_RMSE_mV,width=.36,label='Validation',color=COLORS[model],alpha=.4,edgecolor='black',linewidth=.6)
  ax.bar(np.arange(3)+.18,q.Assessment_RMSE_mV,width=.36,label='Assessment',color=COLORS[model],edgecolor='black',linewidth=.6)
  ax.set_xticks(range(3),['42','52','62']);ax.set_xlabel('Seed');ax.set_title(model)
 axs[0].set_ylabel('RMSE (mV)');axs[-1].legend();save(fig,'Fig3_Validation_Assessment')
 fig,ax=plt.subplots(figsize=(8,5))
 for j,(key,label) in enumerate([('D10_F1_minus_F0_mV','F1−F0'),('D21_F2_minus_F1_mV','F2−F1'),('D20_F2_minus_F0_mV','F2−F0')]):
  ax.bar(np.arange(3)+(j-1)*.24,p[key],width=.24,label=label,color=list(COLORS.values())[j],edgecolor='black',linewidth=.6)
 ax.axhline(0,color='black',lw=.9);ax.set_xticks(range(3),['42','52','62']);ax.set_xlabel('Seed');ax.set_ylabel('Paired RMSE difference (mV)');ax.legend();save(fig,'Fig4_Paired_RMSE_Differences')
 fig,axs=plt.subplots(1,3,figsize=(12,5),sharey=True)
 for ax,seed in zip(axs,[42,52,62]):
  for model in COLORS:
   q=horizons.loc[(horizons.seed==seed)&(horizons.model==model)];ax.plot(q.forecast_hours,q.RMSE_mV,label=model,color=COLORS[model],lw=2)
  ax.set_title('Seed '+str(seed));ax.set_xlabel('Forecast horizon (h)')
 axs[0].set_ylabel('RMSE (mV)');axs[-1].legend();save(fig,'Fig5_Horizon_RMSE')
 fig,axs=plt.subplots(1,3,figsize=(12,5),sharey=True)
 for ax,seed in zip(axs,[42,52,62]):
  for model in COLORS:
   q=targets.loc[(targets.seed==seed)&(targets.model==model)].sort_values('target_age_h')
   ax.plot(q.target_age_h,q.SSE_share*100,label=model,color=COLORS[model],lw=1.3)
  ax.set_title('Seed '+str(seed));ax.set_xlabel('Target operating age (h)')
 axs[0].set_ylabel('Share of forecast-query SSE (%)');axs[-1].legend();save(fig,'Fig6_Target_Time_SSE')
 fig,axs=plt.subplots(1,2,figsize=(11,5))
 for ax,metric,title in zip(axs,['endpoint_RMSE_mV','P99_mV'],['12 h endpoint RMSE','P99 absolute error']):
  bars(metric,ax);ax.set_title(title);ax.set_ylabel('Error (mV)')
 save(fig,'Fig7_Endpoint_and_P99')

if __name__=='__main__':load_protocol();plot_all()
