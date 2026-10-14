"""Export real-data scientific figures; no fitting or prediction modification."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
P=Path(__file__).resolve().parents[1]
MODELS=['F2_MATCHED','F2_CAPACITY_CONTROL','F2_P1S96_HEALTH_AWARE']
SHORT=['Original F2','Zero-input control','P1S96 health-aware'];COLORS=['#64748B','#E69F00','#0072B2']
plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Arial','DejaVu Sans'],'font.size':10,
 'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'ps.fonttype':42,'savefig.dpi':300})
def save(fig,name):
 fig.savefig(P/f'figures/{name}.png',bbox_inches='tight');fig.savefig(P/f'figures/{name}.pdf',bbox_inches='tight');plt.close(fig)
def coverage_and_structure():
 d=pd.read_csv(P/'inputs/P1S_TFT_ALIGNMENT.csv');fig,axes=plt.subplots(2,1,figsize=(9,5),layout='constrained')
 for i,role in enumerate(['train','validation']):
  g=d[d.role==role]
  for flag,label,color,y in [(False,'Excluded','#D55E00',0),(True,'Qualified','#0072B2',1)]:
   z=g[g.eligible==flag];axes[i].scatter(z.origin_time_h,np.full(len(z),y),marker='|',s=45,color=color,label=f'{label} n={len(z)}')
  axes[i].set(yticks=[0,1],yticklabels=['Excluded','Qualified'],title=f'{role.title()}: actual bin-end operating hours',ylim=(-.6,1.6));axes[i].legend(loc='upper right',ncol=2,fontsize=8)
 axes[-1].set_xlabel('Forecast origin (actual operating hours)');save(fig,'train_validation_coverage')
 fig,ax=plt.subplots(figsize=(11,5));ax.set_xlim(0,11);ax.set_ylim(0,5);ax.axis('off')
 def box(x,y,w,h,text,color='#E8F2F8'):
  ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.08',linewidth=1,edgecolor='#334155',facecolor=color));ax.text(x+w/2,y+h/2,text,ha='center',va='center',fontsize=9)
 def arrow(a,b):ax.annotate('',xy=b,xytext=a,arrowprops=dict(arrowstyle='->',color='#334155',lw=1.4))
 box(.1,2.2,1.7,1,'Raw F2 history\n[B,48,14]\n24 h input')
 box(2.3,3.1,2.1,.8,'Unchanged native TFT\nVSN + LSTM');box(2.3,1.6,2.1,.8,'Unchanged causal\nmultiscale Conv')
 box(4.95,2.2,1.7,1,'Scalar fusion\ngate * LSTM\n+ (1-gate) * Conv')
 box(7.2,2.2,1.7,1,'Unchanged decoder\n+ attention\n+ output head');box(9.35,2.2,1.45,1,'24-step DeltaV\n12 h forecast')
 box(2.2,.1,2.4,.8,'Past regime context [B,5]');box(5.,.1,2.6,.8,'A: 5 -> 8 -> 1\nB/C: 10 -> 8 -> 1',color='#FCEFD3')
 box(8.2,.1,2.5,.8,'B: five zeros\nC: five past P1S96 features',color='#FCEFD3')
 for a,b in [((1.8,2.9),(2.3,3.5)),((1.8,2.5),(2.3,2.0)),((4.4,3.5),(4.95,2.9)),((4.4,2.0),(4.95,2.5)),((6.65,2.7),(7.2,2.7)),((8.9,2.7),(9.35,2.7)),((4.6,.5),(5.,.5)),((8.2,.5),(7.6,.5)),((5.8,.9),(5.8,2.2))]:arrow(a,b)
 ax.text(5.5,4.6,'Only historical fusion context changes; no future health in the decoder',ha='center',fontsize=12)
 save(fig,'health_fusion_architecture')
def training_figures():
 m=pd.read_csv(P/'outputs/ALL_RUN_METRICS.csv');fig,ax=plt.subplots(figsize=(8,4.5),layout='constrained')
 for i,g in enumerate(MODELS):
  v=m[m.model==g].sort_values('seed');ax.bar(i,v.RMSE_mV.mean(),color=COLORS[i],alpha=.7,width=.6,yerr=v.RMSE_mV.std(ddof=1),capsize=5)
  ax.scatter(i+np.array([-.12,0,.12]),v.RMSE_mV,color='#111827',s=25,zorder=5)
  for x,y,s in zip(i+np.array([-.12,0,.12]),v.RMSE_mV,v.seed):ax.annotate(str(s),(x,y),xytext=(3,4),textcoords='offset points',fontsize=8)
 ax.set(xticks=np.arange(3),xticklabels=SHORT,ylabel='Query-equal RMSE (mV)',title='Matched Development Validation | 3 seeds | mean and sample SD');save(fig,'three_models_RMSE')
 hs=[]
 for g in MODELS:
  for seed in [42,52,62]:
   z=pd.read_csv(P/f'outputs/metrics/{g}_seed{seed}_horizons.csv');z['model']=g;z['seed']=seed;hs.append(z)
 h=pd.concat(hs);h.to_csv(P/'outputs/ALL_HORIZON_METRICS.csv',index=False,float_format='%.17g',lineterminator='\n');fig,ax=plt.subplots(figsize=(8,4.5),layout='constrained')
 for i,g in enumerate(MODELS):
  mean=h[h.model==g].groupby('lead_h').RMSE_mV.mean();ax.plot(mean.index,mean.values,label=SHORT[i],color=COLORS[i])
 ax.set(xlabel='Forecast lead (h)',ylabel='Mean per-seed RMSE (mV)',title='All 24 horizons | same 178 validation origins');ax.legend();ax.grid(axis='y',alpha=.2);save(fig,'all_horizon_errors')
 fig,ax=plt.subplots(figsize=(10,4.5),layout='constrained');truth=None
 for i,g in enumerate(MODELS):
  f=pd.read_csv(P/f'outputs/predictions/{g}_seed42.csv');f=f[f.horizon_step==24].sort_values('origin_time_h');truth=f
  segments=(f.origin_time_h.diff().fillna(.5)>.500001).cumsum()
  for j,(_,part) in enumerate(f.groupby(segments)):
   ax.plot(part.origin_time_h,part.y_pred_delta_V*1000,label=SHORT[i] if j==0 else None,color=COLORS[i],lw=1)
 segments=(truth.origin_time_h.diff().fillna(.5)>.500001).cumsum()
 for j,(_,part) in enumerate(truth.groupby(segments)):
  ax.plot(part.origin_time_h,part.y_true_delta_V*1000,label='Measured 12h DeltaV' if j==0 else None,color='#111827',ls='--',lw=1.2)
 ax.set(xlabel='Origin operating hours',ylabel='12h terminal DeltaV (mV)',title='Seed42 fixed example | all 178 eligible origins');ax.legend(ncol=2,fontsize=8);save(fig,'three_model_predictions_seed42')
 fig,axes=plt.subplots(1,3,figsize=(11,3.5),layout='constrained');gate=[]
 for i,g in enumerate(MODELS):
  for seed in [42,52,62]:
   f=pd.read_csv(P/f'outputs/metrics/{g}_seed{seed}_gate_weights.csv');gate.append(f);axes[i].hist(f.gate_weight,bins=np.linspace(0,1,21),histtype='step',label=f'Seed {seed}',lw=1.3)
  axes[i].set(title=SHORT[i],xlabel='LSTM fusion weight',ylabel='Origins',xlim=(0,1));axes[i].legend(fontsize=8)
 pd.concat(gate).to_csv(P/'outputs/ALL_GATE_WEIGHTS.csv',index=False,float_format='%.17g',lineterminator='\n');save(fig,'gate_weight_distributions')
 fig,axes=plt.subplots(1,3,figsize=(11,3.5),layout='constrained')
 for j,seed in enumerate([42,52,62]):
  for i,g in enumerate(MODELS):
   f=pd.read_csv(P/f'outputs/training_logs/{g}_seed{seed}/epochs.csv');axes[j].plot(f.epoch,f.validation_RMSE_mV,label=SHORT[i],color=COLORS[i],lw=1)
  axes[j].set(title=f'Seed {seed}',xlabel='Epoch',ylabel='Validation RMSE (mV)',yscale='log')
 axes[0].legend(fontsize=7);save(fig,'training_curves')
 print('Seven real-data/architecture figures exported as PNG and PDF')
if __name__=='__main__':
 coverage_and_structure()
 if (P/'outputs/ALL_RUN_METRICS.csv').exists():training_figures()
