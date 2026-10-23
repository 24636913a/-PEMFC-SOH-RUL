from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT=Path(r"E:\沈农信电\ PEMFC-SOH+RUL\RENew-PEMFC-2300h-Pytorch\Phase_TTD_TimeXer_Failure_Diagnosis")
OUT=ROOT/'outputs'; FIG=OUT/'figures'; FIG.mkdir(parents=True,exist_ok=True)
P={'blue':'#0F4D92','green':'#8BCF8B','red':'#B64342','neutral':'#767676','gold':'#FFD700'}
plt.rcParams.update({'font.family':['Arial','Helvetica','DejaVu Sans','sans-serif'],'font.size':12,'axes.linewidth':2,'axes.spines.top':False,'axes.spines.right':False,'legend.frameon':False,'svg.fonttype':'none'})
def save(fig,name):
    fig.tight_layout(pad=1.2); fig.savefig(FIG/f'{name}.png',dpi=300,bbox_inches='tight'); fig.savefig(FIG/f'{name}.pdf',bbox_inches='tight'); plt.close(fig)

base=pd.read_csv(OUT/'reproduction/baseline_reproduction.csv')
fig,ax=plt.subplots(figsize=(7.6,4.5)); ax.bar(base.model,base.mean_rmse_delta_mV,color=[P['neutral'],P['blue'],P['green'],P['gold'],P['red']],edgecolor='black',linewidth=1.2); ax.set_ylabel('Three-fold DeltaV RMSE (mV)'); ax.set_ylim(0,210); ax.grid(axis='y',alpha=.22); save(fig,'baseline_reproduction')

f1=pd.read_csv(OUT/'fold1_12h/fold1_12h_loss_concentration.csv')
fig,ax=plt.subplots(figsize=(7.6,4.5)); ax.plot(f1.rank_top_k,f1.share_positive_D_pct,'o-',lw=2.5,color=P['red'],label='Share of positive D'); ax.axhline(100,color=P['neutral'],lw=1); ax.set_xlabel('Top unique target times'); ax.set_ylabel('Positive D captured (%)'); ax.set_ylim(0,105); ax.grid(alpha=.22); ax.legend(); save(fig,'fold1_h12_loss_concentration')

seg=pd.read_csv(OUT/'fold3_nonzero/fold3_segment_contribution.csv').head(4)
fig,ax=plt.subplots(figsize=(8.3,4.7)); x=np.arange(len(seg)); ax.bar(x-.18,seg.positive_D_sum_mV2/1e6,.36,color=P['red'],edgecolor='black',label='Positive D'); ax.bar(x+.18,seg.negative_D_sum_mV2/1e6,.36,color=P['blue'],edgecolor='black',label='Negative D'); ax.set_xticks(x,[f'Segment {int(v)}' for v in seg.temporal_segment_id]); ax.set_ylabel('D contribution (10$^6$ mV$^2$)'); ax.axhline(0,color='black',lw=1); ax.grid(axis='y',alpha=.22); ax.legend(); save(fig,'fold3_segment_contribution')

sh=pd.read_csv(OUT/'transitions/trajectory_shift_diagnostics.csv')
fig,axes=plt.subplots(1,2,figsize=(10.2,4.2))
for fold,g in sh.groupby('fold'):
    axes[0].plot(g.shift_steps_predicted_relative_to_actual,g.current_MAE_A,'o-',label=fold)
    axes[1].plot(g.shift_steps_predicted_relative_to_actual,100*g.state_mismatch_rate,'o-',label=fold)
axes[0].set(xlabel='Predicted shift relative to actual (steps)',ylabel='Current MAE (A)'); axes[1].set(xlabel='Predicted shift relative to actual (steps)',ylabel='State mismatch (%)')
[a.grid(alpha=.22) for a in axes]; axes[1].legend(); save(fig,'transition_shift_diagnostics')
