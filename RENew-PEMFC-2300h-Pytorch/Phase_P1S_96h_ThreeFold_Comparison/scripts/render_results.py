"""Export empirical prediction and sample-count figures; never fits a model."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
P=Path(__file__).resolve().parents[1]
COLORS={'PERSISTENCE':'#64748B','RIDGE_96H':'#0072B2','TRUE':'#D55E00'}
plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Arial','DejaVu Sans'],'font.size':10,
 'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'ps.fonttype':42,'savefig.dpi':300})
def save(fig,name):
 fig.savefig(P/f'figures/{name}.png',bbox_inches='tight');fig.savefig(P/f'figures/{name}.pdf',bbox_inches='tight');plt.close(fig)
def main():
 d=pd.read_csv(P/'outputs/PREDICTIONS.csv');m=pd.read_csv(P/'outputs/FOLD_METRICS.csv');c=pd.read_csv(P/'outputs/SAMPLE_COUNTS.csv')
 for h in ['H1','H3']:
  fig,ax=plt.subplots(3,1,figsize=(9,8),layout='constrained')
  for i,f in enumerate(['fold_1','fold_2','fold_3']):
   v=d[(d.horizon==h)&(d.fold==f)];true=v[v.model=='PERSISTENCE'].sort_values('target_raw_cycle_id')
   ax[i].plot(true.target_raw_cycle_id,true.true_R_P1S_causal,marker='o',markersize=3,label='Measured P1S',color=COLORS['TRUE'])
   for method in ['PERSISTENCE','RIDGE_96H']:
    z=v[v.model==method].sort_values('target_raw_cycle_id');ax[i].plot(z.target_raw_cycle_id,z.predicted_R_P1S_causal,marker='.',label=method,color=COLORS[method])
   ax[i].set(title=f'{h} | Fold {i+1} | n={len(true)}',ylabel='P1S retention R');ax[i].grid(axis='y',alpha=.2)
  ax[0].legend(ncol=3,fontsize=8);ax[-1].set_xlabel('Target raw physical cycle ID (only eligible observations)')
  save(fig,f'{h}_prediction_curves')
  fig,ax=plt.subplots(figsize=(7,4),layout='constrained');x=np.arange(3)
  for j,method in enumerate(['PERSISTENCE','RIDGE_96H']):
   v=m[(m.horizon==h)&(m.model==method)].sort_values('fold');bars=ax.bar(x+(j-.5)*.32,v.MAE_mV_equivalent,.32,label=method,color=COLORS[method])
   ax.bar_label(bars,fmt='%.4f',padding=3,fontsize=9)
  ax.set(xticks=x,xticklabels=['Fold 1','Fold 2','Fold 3'],ylabel='MAE (P1S mV-equivalent)',title=f'{h}: same samples within each fold');ax.legend(fontsize=9);ax.set_ylim(0,float(m[m.horizon==h].MAE_mV_equivalent.max())*1.25)
  save(fig,f'{h}_three_fold_MAE')
 fig,axes=plt.subplots(2,2,figsize=(10,6),layout='constrained')
 for i,h in enumerate(['H1','H3']):
  for j,role in enumerate(['train','validation']):
   v=c[(c.horizon==h)&(c.role==role)].sort_values('fold');x=np.arange(3);ax=axes[i,j]
   for k,(col,label,color) in enumerate([('authority_n','Frozen original','#A7AFB9'),('L96_eligible_n','L96 eligible','#56B4E9'),('used_n','Final used','#0072B2')]):
    b=ax.bar(x+(k-1)*.24,v[col],.24,label=label,color=color);ax.bar_label(b,padding=2,fontsize=8)
   ax.set(xticks=x,xticklabels=['Fold 1','Fold 2','Fold 3'],title=f'{h} {role}',ylabel='Samples');ax.set_ylim(0,v.authority_n.max()*1.2)
 axes[0,0].legend(fontsize=8);save(fig,'original_vs_used_counts')
 print('5 figures exported as PNG + PDF')
if __name__=='__main__':main()
