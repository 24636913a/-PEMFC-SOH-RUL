"""Standalone publication-candidate plots from real CSV; portable to verified paper package."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
BASE=Path(__file__).resolve().parents[1];DATA=BASE/'source_csv' if (BASE/'source_csv').exists() else BASE/'outputs';FIG=BASE/'figures';FIG.mkdir(exist_ok=True)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'ps.fonttype':42,'savefig.dpi':300})
METHODS=['PERSISTENCE','LOCAL_TREND96','RIDGE96'];COLORS=['#808080','#B77731','#597FBD'];LABELS=['Persistence','Past local trend','Ridge96']
INDEX=[]
def recovery_panel_data(frame,H,method):
 z=frame[frame.H.eq(H)&frame.method.eq(method)].set_index('state').reindex([False,True])
 assert len(z)==2 and not z['n'].isna().any(),'Both distinct strata required; never broadcast one boolean-selected row'
 return z
def save(fig,name,sources,caption):
 fig.savefig(FIG/f'{name}.png',bbox_inches='tight');fig.savefig(FIG/f'{name}.pdf',bbox_inches='tight');plt.close(fig)
 INDEX.append({'figure':name,'source_csv':'|'.join(sources),'script':'scripts/render_figures.py','caption':caption})
def run():
 t=pd.read_csv(DATA/'REAL_P1S_OBSERVATIONS.csv');fig,ax=plt.subplots(3,1,figsize=(10,7),sharex=True,layout='constrained')
 for role,color,marker,label in [('forecast_trajectory',COLORS[2],'.','131 causal complete-cycle measurements'),('reference_warmup_not_causal_target',COLORS[0],'s','10 reference-construction measurements'),('partial_descriptive_only',COLORS[1],'^','6 partial measurements (descriptive)')]:
  g=t[t.data_role.eq(role)].sort_values('conservative_available_end_h');x=g.actual_stable_end_h.fillna(g.conservative_available_end_h)
  ax[0].scatter(x,g.P1S_operating_voltage_retention*100,s=13,color=color,marker=marker,label=label)
  ax[1].scatter(x,g.anode_inlet_temperature_C,s=12,color=color,marker=marker)
  ax[2].scatter(x,g.anode_inlet_RH_pct,s=12,color=color,marker=marker)
 for a in ax:a.axvline(1819.997738,color='gray',ls='--',lw=.7);a.set_xlim(60,1850)
 ax[0].set_ylabel('P1S retention (%)');ax[0].legend(fontsize=8,loc='upper right');ax[0].set_title('Real discrete P1S observations; no interpolated labels')
 ax[1].set_ylabel('Anode inlet T (deg C)');ax[2].set_ylabel('Anode inlet RH (%)');ax[2].set_xlabel('Actual measurement end / recorded available operating h')
 save(fig,'F1_real_health_measurements_and_conditions',['REAL_P1S_OBSERVATIONS.csv'],'141 complete and6 partial measurements;10 complete warmup values are reference construction only. Exact stable-end hours used when reconstructed, otherwise recorded dwell end;131 causal targets. Five other complete cycles have no cached P1S value, not imputed.')
 e=pd.read_csv(DATA/'THREE_FOLD_EQUAL_MEANS.csv');e=e[e.scope.eq('own_horizon_cohort')];fig,ax=plt.subplots(1,2,figsize=(10.5,3.8),layout='constrained')
 for k,metric in enumerate(['MAE_mVeq','RMSE_mVeq']):
  for j,m in enumerate(METHODS):
   z=e[e.method.eq(m)].set_index('H');xs=np.arange(3)+(j-1)*.25;vals=z.loc[[1,3,6],metric].to_numpy();ax[k].bar(xs,vals,width=.24,color=COLORS[j],edgecolor='black',linewidth=.6,label=LABELS[j])
   for x,v in zip(xs,vals):ax[k].annotate(f'{v:.3f}',(x,v),xytext=(0,4),textcoords='offset points',ha='center',va='bottom',rotation=90,fontsize=7)
  ax[k].set_xticks(range(4),['H1','H3','H6','H12']);ax[k].set_xlim(-.6,3.7);ax[k].text(3,.09,'Not\nevaluable',ha='center',transform=ax[k].get_xaxis_transform(),fontsize=9)
  ax[k].set_ylabel(('MAE' if k==0 else 'RMSE')+' (mV-equivalent)');ax[k].set_title('Threefold equal means; H12 has no mean');ax[k].set_ylim(0,float(e[metric].max())*1.2)
 fig.legend(*ax[0].get_legend_handles_labels(),loc='outside lower center',ncol=3,fontsize=9)
 save(fig,'F2_horizon_primary_equal_fold_metrics',['THREE_FOLD_EQUAL_MEANS.csv'],'H1/H3 native coefficients replayed; H6 new fixed experiment. H6 fold2n1. H12 has emptyfold2 and unqualified Ridgefold1: no fabricated or twofold-as-threefold mean. Cohorts vary across horizons; use matched-origin table for cross-horizon diagnostics.')
 fm=pd.read_csv(DATA/'FOLD_METRICS.csv');own=fm[fm.scope.eq('own_horizon_cohort')];fig,ax=plt.subplots(1,3,figsize=(12,3.6),layout='constrained')
 for i,fold in enumerate(['fold_1','fold_2','fold_3']):
  for j,m in enumerate(METHODS):
   z=own[own.fold.eq(fold)&own.method.eq(m)].set_index('H').loc[[1,3,6,12]];ax[i].plot(range(4),z.MAE_mVeq,color=COLORS[j],marker=['s','^','o'][j],label=LABELS[j])
  p=own[own.fold.eq(fold)&own.method.eq('PERSISTENCE')].set_index('H').loc[[1,3,6,12]]
  ax[i].set_xticks(range(4),['1','3','6','12']);ax[i].set_title(fold+': n='+('/'.join(p.n.astype(int).astype(str))));ax[i].set_xlabel('Future physical cycles');ax[i].set_ylabel('MAE (mV-equivalent)')
  if i==1:ax[i].text(.05,.92,'H6: one observation\nH12: zero observations',transform=ax[i].transAxes,fontsize=8,va='top')
 fig.legend(*ax[0].get_legend_handles_labels(),loc='outside lower center',ncol=3,fontsize=9)
 save(fig,'F3_per_fold_horizon_failure_and_support',['FOLD_METRICS.csv'],'Gaps mean unavailable scores, not zero. H12fold3 Ridge uses15 Train rows,4 inner-fit,3 inner-selection,7 Validation; its large error is retained. H12fold1 baselines are evaluable but Ridge is not.')
 s=pd.read_csv(DATA/'POOLED_STRATUM_DIAGNOSTICS.csv');s=s[s.stratum.eq('future_observed_increase_AUDIT_ONLY')];fig,ax=plt.subplots(1,3,figsize=(11,3.7),layout='constrained')
 for i,H in enumerate([1,3,6]):
  for j,m in enumerate(METHODS):
   z=recovery_panel_data(s,H,m);ax[i].bar(np.arange(2)+(j-1)*.25,z.MAE_mVeq,width=.24,color=COLORS[j],edgecolor='black',linewidth=.5,label=LABELS[j])
  baseline=s[s.H.eq(H)&s.method.eq('PERSISTENCE')].set_index('state');ax[i].set_xticks([0,1],[f'Not increased\nn={int(baseline.loc[False,"n"])}',f'Observed increase\nn={int(baseline.loc[True,"n"])}']);ax[i].set_title('H'+str(H)+' pooled diagnostic');ax[i].set_ylabel('MAE (mV-equivalent)')
 fig.legend(*ax[0].get_legend_handles_labels(),loc='outside lower center',ncol=3,fontsize=9)
 save(fig,'F4_observed_recovery_error_diagnostic',['POOLED_STRATUM_DIAGNOSTICS.csv'],'Future measuredR increase is a post-hoc evaluation stratum, never an input, not certified reversible recovery. Pooled summaries are descriptive and not the equal-fold primary metric. H12 has no observed-increase Validation labels, so no recovery result is drawn.')
 p=pd.read_csv(DATA/'PREDICTIONS.csv');b=p[p.method.eq('PERSISTENCE')];fig,ax=plt.subplots(1,2,figsize=(10,3.6),layout='constrained')
 for i,col in enumerate(['actual_lead_h_AUDIT_ONLY','actual_calendar_lead_h_AUDIT_ONLY']):
  for k,H in enumerate([1,3,6,12]):
   z=b[b.H.eq(H)].sort_values('origin_available_h');ax[i].scatter(k+np.linspace(-.13,.13,len(z)),z[col],s=12,color=COLORS[2],alpha=.65)
  ax[i].set_xticks(range(4),['1','3','6','12']);ax[i].set_xlabel('Future physical cycles');ax[i].set_ylabel('Operating hours' if i==0 else 'Recorded calendar hours');ax[i].set_title('Actual lead, evaluation only')
 save(fig,'F5_actual_operating_and_calendar_leads',['PREDICTIONS.csv'],'Future actual lead is never a model input; trend uses history-only cadence. Calendar intervals use recorded conservative dwell-end clocks, not invented stable-end timestamps. Shutdowns can create hundreds of calendar hours even for H1.')
 fig,ax=plt.subplots(2,2,figsize=(11,6),layout='constrained')
 for i,H in enumerate([1,3,6,12]):
  a=ax.flat[i]
  for j,m in enumerate(METHODS):
   z=p[p.H.eq(H)&p.method.eq(m)].sort_values('target_actual_end_h');group=(z.fold.ne(z.fold.shift())|z.target_raw_cycle_id.diff().ne(1)).cumsum()
   for n,(_,part) in enumerate(z.groupby(group)):a.plot(part.target_actual_end_h,part.predicted_R*100,color=COLORS[j],marker='.',lw=.8,ms=3,label=LABELS[j] if n==0 else None)
  obs=b[b.H.eq(H)].sort_values('target_actual_end_h');a.scatter(obs.target_actual_end_h,obs.target_R*100,s=11,color='black',label='Observed target P1S');a.set_title('H'+str(H)+(' (no complete threefold)' if H==12 else ''));a.set_xlabel('Actual target measurement end (operating h)');a.set_ylabel('P1S retention (%)');a.legend(fontsize=6.8)
 save(fig,'F6_real_pointwise_health_forecasts',['PREDICTIONS.csv'],'All actual predictions retained, broken across folds/missing cycle IDs. Lines guide the eye only; no interpolated true labels. H12 Ridge exists only forfold3. This is P1S retention, not reference SOH or dynamic-voltage RMSE.')
 fig,ax=plt.subplots(1,2,figsize=(9,3.6),layout='constrained');cm=fm[fm.scope.eq('common_origins_all4_horizons')]
 for i,fold in enumerate(['fold_1','fold_3']):
  for j,m in enumerate(METHODS):
   z=cm[cm.fold.eq(fold)&cm.method.eq(m)].set_index('H').loc[[1,3,6,12]];ax[i].plot(range(4),z.MAE_mVeq,color=COLORS[j],marker=['s','^','o'][j],label=LABELS[j])
  nn=int(cm[cm.fold.eq(fold)&cm.method.eq('PERSISTENCE')].n.max());ax[i].set_title(fold+f': exact common origins n={nn}');ax[i].set_xticks(range(4),['1','3','6','12']);ax[i].set_xlabel('Future physical cycles');ax[i].set_ylabel('MAE (mV-equivalent)')
 fig.suptitle('Matched-origin diagnostic; fold2 empty, no threefold mean',fontsize=11);fig.legend(*ax[0].get_legend_handles_labels(),loc='outside lower center',ncol=3,fontsize=9)
 save(fig,'F7_same_origin_cross_horizon_diagnostic',['FOLD_METRICS.csv'],'Exact same fold and raw origin across all4 horizons:5/0/4 origins. Different horizon-specific fits retained; H12fold1 Ridge unavailable. This small descriptive diagnostic cannot be called a formal threefold horizon comparison.')
 pd.DataFrame(INDEX).to_csv(BASE/'FIGURE_SOURCE_INDEX.csv',index=False,lineterminator='\n');print('Rendered7 true-data figure pairs')
if __name__=='__main__':run()
