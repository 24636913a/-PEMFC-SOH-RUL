"""Standalone real-CSV plotting; portable unchanged to the verified paper package."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
BASE=Path(__file__).resolve().parents[1]
DATA=BASE/'source_csv' if (BASE/'source_csv').exists() else BASE/'outputs'
FIG=BASE/'figures';FIG.mkdir(exist_ok=True)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'ps.fonttype':42,'savefig.dpi':300})
BLUE='#597FBD';ORANGE='#B77731';GRAY='#808080';COLORS=[GRAY,ORANGE,BLUE]
METHODS=['M0_REFERENCE_PERSISTENCE','M_ORIGIN_P1S_CALIBRATED','M_RIDGE_P1S_CALIBRATED']
LABELS=['Last reference','Measured origin P1S + fixed calibration','Ridge forecast P1S + same calibration']
INDEX=[]
def save(fig,name,files,caption):
 fig.savefig(FIG/(name+'.png'),bbox_inches='tight');fig.savefig(FIG/(name+'.pdf'),bbox_inches='tight');plt.close(fig)
 INDEX.append({'figure':name,'source_csv':'|'.join(files),'script':'scripts/render_paper_figures.py','caption':caption})
def run():
 met=pd.read_csv(DATA/'SOH_METRICS.csv');fig,ax=plt.subplots(1,2,figsize=(10.5,3.8),layout='constrained')
 for k,metric in enumerate(['MAE_SOH_pp','RMSE_SOH_pp']):
  for j,m in enumerate(METHODS):
   z=met[met.method.eq(m)].set_index('horizon').loc[['H1','H3']];bars=ax[k].bar(np.arange(2)+(j-1)*.25,z[metric],width=.24,color=COLORS[j],edgecolor='black',linewidth=.6,hatch=['','/',''][j],label=LABELS[j])
   for bar,v in zip(bars,z[metric]):ax[k].annotate(f'{v:.6f}',(bar.get_x()+bar.get_width()/2,v),xytext=(0,4),textcoords='offset points',ha='center',fontsize=7)
  ax[k].set_xticks([0,1],['H1 origin','H3 origin']);ax[k].set_ylabel(('MAE' if k==0 else 'RMSE')+' (SOH percentage points)');ax[k].set_ylim(0,float(met[metric].max())*1.23)
  ax[k].set_title('Same three reference events, n = 3')
 handles,labels=ax[0].get_legend_handles_labels();fig.legend(handles,labels,loc='outside lower center',ncol=3,fontsize=8)
 save(fig,'F1_matched_SOH_three_method_control',['SOH_METRICS.csv'],'Three methods at the exact same origins/targets; H1 and H3 reuse the same3 reference events, not6 independent outcomes. Retrospective scheduling; unknown release times.')
 w=pd.read_csv(DATA/'POINTWISE_WINS.csv');w=w[w.candidate.eq(METHODS[2])&w.baseline.eq(METHODS[1])];fig,ax=plt.subplots(1,2,figsize=(9,3.5),layout='constrained')
 for i,h in enumerate(['H1','H3']):
  z=w[w.horizon.eq(h)].sort_values('nominal_h');vals=z.absolute_error_reduction_SOH_pp.to_numpy();ax[i].bar(np.arange(3),vals,color=[BLUE if v>0 else ORANGE for v in vals],edgecolor='black',linewidth=.6)
  ax[i].axhline(0,color='black',lw=.7);lo=min(0,float(vals.min()));hi=max(0,float(vals.max()));span=hi-lo
  ax[i].set_ylim(lo-.22*span,hi+.3*span)
  for x,v in enumerate(vals):ax[i].annotate(f'{v:+.6f}',(x,v),xytext=(0,5 if v>=0 else -12),textcoords='offset points',ha='center',fontsize=9)
  ax[i].set_xticks(range(3),z.nominal_h.astype(str));ax[i].set_xlabel('Nominal reference ID');ax[i].set_ylabel('Origin AE minus Ridge AE (SOH pp)');ax[i].set_title(h+': positive = forecast step improves')
 save(fig,'F2_eventwise_forecast_step_increment',['POINTWISE_WINS.csv'],'Increment relative to measured-origin P1S with identical frozen calibration. Panels use different y scales. No significance or noise-floor claim.')
 d=pd.read_csv(DATA/'ERROR_DECOMPOSITION.csv');fig,ax=plt.subplots(1,2,figsize=(10,3.7),layout='constrained',sharey=True)
 for i,h in enumerate(['H1','H3']):
  z=d[d.horizon.eq(h)].sort_values('nominal_h');x=np.arange(3)
  ax[i].bar(x-.16,z.calibration_at_actual_target_SOH_pp,width=.3,color=GRAY,label='Calibration residual at actual target R')
  ax[i].bar(x+.16,z.ridge_proxy_error_translated_SOH_pp,width=.3,color=BLUE,label='Forecast error translated by fixed slope a')
  ax[i].scatter(x,z.ridge_final_error_SOH_pp,marker='D',color='black',s=28,zorder=3,label='Final SOH error = sum')
  ax[i].axhline(0,color='black',lw=.7);ax[i].set_xticks(x,z.nominal_h.astype(str));ax[i].set_xlabel('Nominal reference ID');ax[i].set_title(h+' signed-error decomposition');ax[i].set_ylabel('Signed error (SOH percentage points)')
 handles,labels=ax[0].get_legend_handles_labels();fig.legend(handles,labels,loc='outside lower center',ncol=1,fontsize=8)
 save(fig,'F3_calibration_forecast_error_cancellation',['ERROR_DECOMPOSITION.csv'],'e_SOH=100(aR_target+b-Y)+100a(R_pred-R_target). Actual future R is used only in retrospective error diagnosis, never a prediction input or deployable competing model.')
 c=pd.read_csv(DATA/'H3_VS_H1_ERROR_CANCELLATION.csv');fig,ax=plt.subplots(figsize=(7.3,4),layout='constrained')
 for j,(_,r) in enumerate(c.iterrows()):
  x1,y1=r.H1_proxy_AE_mV_equivalent,r.H1_SOH_AE_pp;x3,y3=r.H3_proxy_AE_mV_equivalent,r.H3_SOH_AE_pp
  ax.annotate('',xy=(x3,y3),xytext=(x1,y1),arrowprops={'arrowstyle':'->','color':GRAY,'lw':1})
  ax.scatter(x1,y1,color=ORANGE,s=38,marker='o',zorder=3);ax.scatter(x3,y3,color=BLUE,s=45,marker='^',zorder=3)
  ax.annotate(str(int(r.nominal_h)),(x3,y3),xytext=(-2,8),textcoords='offset points',ha='right',fontsize=9)
 ax.set_xlabel('P1S proxy absolute error (mV-equivalent)');ax.set_ylabel('Final SOH absolute error (percentage points)');ax.set_title('H1 to H3: proxy and SOH errors differ')
 ax.set_ylim(-.008,float(c[['H1_SOH_AE_pp','H3_SOH_AE_pp']].max().max())*1.15);ax.margins(x=.08)
 ax.legend(handles=[Line2D([0],[0],marker='o',ls='',color=ORANGE,label='H1 origin'),Line2D([0],[0],marker='^',ls='',color=BLUE,label='H3 origin')],loc='upper right',fontsize=9)
 save(fig,'F4_proxy_error_vs_SOH_error',['H3_VS_H1_ERROR_CANCELLATION.csv'],'Same target and calibration per event; origins differ. All3 H3 proxy errors exceed H1 while SOH errors shrink through residual cancellation. No regression fit or6-event inference.')
 pd.DataFrame(INDEX).to_csv(BASE/'FIGURE_SOURCE_INDEX.csv',index=False,lineterminator='\n')
 print('Rendered4 candidate figures from real CSV')
if __name__=='__main__':run()
