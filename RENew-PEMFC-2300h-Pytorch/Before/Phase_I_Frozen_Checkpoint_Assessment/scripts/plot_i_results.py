"""Publication-style PNG/PDF panels, with seed SD explicitly not confidence CI."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from i_common import *
from analyze_i_results import read_csv_local

COLORS={'F0':'#4C72B0','F1':'#55A868','F2':'#C44E52'}
TITLE='Reused Development Assessment — n=3 seeds'

def plot_all():
 plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Arial','DejaVu Sans'],'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'savefig.dpi':300,'axes.titleweight':'normal'})
 m=read_csv_local(OUT/'metrics/I_ASSESSMENT_METRICS.csv');p=read_csv_local(OUT/'metrics/I_PAIRED_SEED_DIFFERENCES.csv');g=read_csv_local(OUT/'metrics/I_MULTISEED_SUMMARY.csv');gap=read_csv_local(OUT/'metrics/I_VALIDATION_ASSESSMENT_GAPS.csv');horizon=read_csv_local(OUT/'diagnostic/I_ALL_HORIZON_METRICS.csv');targets=read_csv_local(OUT/'diagnostic/I_ALL_TARGET_SSE.csv')
 folder=PHASE/'figures';folder.mkdir(parents=True,exist_ok=True)
 def save(fig,name):
  fig.suptitle(TITLE,fontsize=12);fig.tight_layout(rect=(0,0,1,.94));fig.savefig(folder/(name+'.png'));fig.savefig(folder/(name+'.pdf'));plt.close(fig)
 def bars(metric,ax):
  for j,model in enumerate(COLORS):
   v=m.loc[m.model==model].sort_values('seed')[metric].to_numpy();ax.bar(np.arange(3)+(j-1)*.24,v,width=.24,label=model,color=COLORS[model])
  ax.set_xticks(range(3),['42','52','62']);ax.set_xlabel('Frozen pretrained seed');ax.legend(frameon=False)
 fig,ax=plt.subplots(figsize=(7,4.6));bars('RMSE_mV',ax);ax.set_ylabel('Forecast-query-weighted RMSE (mV)');save(fig,'Fig1_Assessment_Seed_RMSE')
 fig,ax=plt.subplots(figsize=(6.2,4.6));ax.bar(g.model,g.RMSE_mV_mean,yerr=g.RMSE_mV_sample_SD,capsize=5,color=[COLORS[x] for x in g.model]);ax.set_ylabel('RMSE (mV)');ax.set_title('Equal-seed mean ± sample SD (ddof=1), not CI');save(fig,'Fig2_Assessment_Mean_Sample_SD')
 fig,axs=plt.subplots(1,3,figsize=(11,4.6),sharey=True)
 for ax,model in zip(axs,COLORS):
  t=gap.loc[gap.model==model].sort_values('seed');ax.bar(np.arange(3)-.18,t.Validation_RMSE_mV,width=.36,label='Validation',color=COLORS[model],alpha=.45);ax.bar(np.arange(3)+.18,t.Assessment_RMSE_mV,width=.36,label='Assessment',color=COLORS[model]);ax.set_xticks(range(3),['42','52','62']);ax.set_title(model);ax.set_xlabel('Seed')
 axs[0].set_ylabel('RMSE (mV)');axs[-1].legend(frameon=False);save(fig,'Fig3_Validation_Assessment_RMSE')
 fig,ax=plt.subplots(figsize=(7.4,4.6))
 for j,(key,label) in enumerate([('D10_F1_minus_F0_mV','F1−F0'),('D21_F2_minus_F1_mV','F2−F1'),('D20_F2_minus_F0_mV','F2−F0')]):ax.bar(np.arange(3)+(j-1)*.24,p[key],width=.24,label=label,color=list(COLORS.values())[j])
 ax.axhline(0,color='black',lw=.8);ax.set_xticks(range(3),['42','52','62']);ax.set_xlabel('Seed');ax.set_ylabel('Paired RMSE difference (mV)');ax.legend(frameon=False);save(fig,'Fig4_Paired_RMSE_Differences')
 fig,axs=plt.subplots(1,3,figsize=(11,4.6),sharey=True)
 for ax,seed in zip(axs,[42,52,62]):
  for model in COLORS:
   q=horizon.loc[(horizon.seed==seed)&(horizon.model==model)];ax.plot(q.forecast_hours,q.RMSE_mV,label=model,color=COLORS[model])
  ax.set_title('Seed '+str(seed));ax.set_xlabel('Forecast horizon (h)')
 axs[0].set_ylabel('RMSE (mV)');axs[-1].legend(frameon=False);save(fig,'Fig5_Horizon_RMSE')
 fig,axs=plt.subplots(1,3,figsize=(11,4.6),sharey=True)
 for ax,seed in zip(axs,[42,52,62]):
  for model in COLORS:
   q=targets.loc[(targets.seed==seed)&(targets.model==model)].sort_values('target_age_h');ax.plot(q.target_age_h,q.SSE_share*100,label=model,color=COLORS[model],lw=1)
  ax.set_title('Seed '+str(seed));ax.set_xlabel('Target operating age (h)')
 axs[0].set_ylabel('Share of all forecast-query SSE (%)');axs[-1].legend(frameon=False);save(fig,'Fig6_Target_Time_SSE_Concentration')
 fig,ax=plt.subplots(figsize=(7,4.6));bars('endpoint_RMSE_mV',ax);ax.set_ylabel('12 h endpoint RMSE (mV)');save(fig,'Fig7_Endpoint_RMSE')
 fig,axs=plt.subplots(1,2,figsize=(10,4.6))
 for ax,metric in zip(axs,['P95_mV','P99_mV']):bars(metric,ax);ax.set_ylabel('Absolute error (mV)');ax.set_title(metric.replace('_mV',''))
 save(fig,'Fig8_P95_P99_Absolute_Error')
 fig,ax=plt.subplots(figsize=(7,4.6))
 for seed,color in zip([42,52,62],COLORS.values()):
  q=read_csv_local(OUT/'diagnostic'/f'I_F2_Seed{seed}_gates.csv');ax.hist(q.gate,bins=np.linspace(0,1,31),histtype='step',lw=1.6,label='Seed '+str(seed),color=color)
 ax.set_xlim(0,1);ax.set_xlabel('Actual-path F2 gate weight');ax.set_ylabel('Assessment origins');ax.legend(frameon=False);save(fig,'Fig9_F2_Gate_Distribution')

if __name__=='__main__':plot_all()
