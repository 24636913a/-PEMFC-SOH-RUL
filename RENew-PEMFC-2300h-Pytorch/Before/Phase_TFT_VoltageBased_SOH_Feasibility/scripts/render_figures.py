"""Every figure uses saved Development CSVs, no synthetic health curves."""
from common import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'ps.fonttype':42,'savefig.dpi':300})
BLUE='#2878B5';ORANGE='#D97928';GRAY='#737373'
INDEX=[]
def save(fig,name,sources,note):
 fig.savefig(P/f'figures/{name}.png',dpi=300,bbox_inches='tight');fig.savefig(P/f'figures/{name}.pdf',bbox_inches='tight');plt.close(fig)
 for ext in ['png','pdf']:INDEX.append(dict(figure=f'figures/{name}.{ext}',CSV_sources=';'.join(sources),plot_code='scripts/render_figures.py',scope=note))
def main():
 b=read(P/'data_contracts/DEVELOPMENT.csv');pr=read(P/'outputs/RESTORED_F2_PREDICTIONS.csv')
 end=pr.loc[pr.seed.eq(42)&pr.horizon_step.eq(24)].copy();j=end.target_row.map(b.actual_current_density_A_cm2_mean)
 fig,axs=plt.subplots(2,2,figsize=(11,7),layout='constrained')
 for ax,(label,lo,hi) in zip(axs.flat,[('j<0.50',0,.5),('0.50<=j<0.75',.5,.75),('0.75<=j<1.00',.75,1),('j>=1.00',1,2)]):
  d=end.loc[(j>=lo)&(j<hi)]
  ax.scatter(d.target_ophrs_h,d.actual_voltage_V,s=14,color=GRAY,label='Observed raw bin mean')
  ax.scatter(d.target_ophrs_h,d.predicted_voltage_V,s=13,marker='x',color=BLUE,label='Frozen F2 seed42')
  ax.set(title=f'{label} A/cm² | n={len(d)}',xlabel='Actual operating time (h)',ylabel='Cell voltage (V)')
  ax.grid(alpha=.15)
 axs[0,0].legend(fontsize=8,loc='best');fig.suptitle('Development Validation | fixed 12 h endpoint | actual load used only for retrospective grouping')
 save(fig,'F1_dynamic_voltage_by_load',['outputs/RESTORED_F2_PREDICTIONS.csv','data_contracts/DEVELOPMENT.csv'],'205 fixed12h queries; retrospective load grouping')
 c=read(P/'outputs/COVERAGE_BY_LOAD.csv');fig,axs=plt.subplots(1,2,figsize=(11,4.8),layout='constrained')
 z=c.loc[c.load.eq('ALL')].sort_values('caliper_IQR');x=np.arange(len(z));w=.24
 for offset,col,label,color in [(-1,'load_only_any','Load-only match',GRAY),(0,'joint_any','Joint any match',ORANGE),(1,'joint_3cycle_supported','Joint >=3 cycles',BLUE)]:
  bars=axs[0].bar(x+offset*w,z[col],width=w,label=label,color=color,edgecolor='black',linewidth=.5)
  axs[0].bar_label(bars,fontsize=8)
 axs[0].set(xticks=x,xticklabels=z.caliper_IQR.astype(str),xlabel='Train IQR caliper (0.5 frozen primary)',ylabel='Distinct target bins / 251',ylim=(0,275));axs[0].legend(fontsize=8)
 q=c.loc[c.caliper_IQR.eq(.5)&~c.load.eq('ALL')];x=np.arange(4)
 axs[1].bar(x,q.targets,color='#D8D8D8',label='All targets');bars=axs[1].bar(x,q.joint_3cycle_supported,color=BLUE,label='Joint >=3 cycles');axs[1].bar_label(bars)
 axs[1].set(xticks=x,xticklabels=['<0.50','0.50–0.75','0.75–1.00','>=1.00'],xlabel='Actual target mean current density (A/cm²)',ylabel='Distinct target bins',ylim=(0,115));axs[1].legend(fontsize=8)
 fig.suptitle('Retrospective statistical common support | BoL / physically certified / strict online SOH: 0')
 save(fig,'F2_matching_coverage',['outputs/COVERAGE_BY_LOAD.csv'],'251 unique targets, not independent life samples')
 z=read(P/'outputs/FIXED_12H_ENDPOINT_DIAGNOSTIC.csv');z=z.loc[z.seed.eq(42)].sort_values('target_row');fig,axs=plt.subplots(1,2,figsize=(11,4.8),layout='constrained')
 axs[0].vlines(z.target_ophrs_h,z.CR_reference_range_low_percent,z.CR_reference_range_high_percent,color=GRAY,alpha=.35,lw=.7,label='Reference-cycle min/max sensitivity')
 axs[0].scatter(z.target_ophrs_h,z.conditional_voltage_retention_observed_percent,s=14,color=GRAY,label='Observed voltage / early reference')
 axs[0].scatter(z.target_ophrs_h,z.CR_mean_percent,s=15,color=BLUE,marker='x',label='F2 voltage / early reference')
 axs[0].axhline(100,color=ORANGE,lw=.7,ls='--');axs[0].set(xlabel='Actual operating time (h)',ylabel='Conditional voltage retention (%)',title=f'Fixed 12 h, seed42 | n={len(z)}');axs[0].legend(fontsize=7)
 h=read(P/'outputs/CONDITIONAL_RETENTION_HORIZON_ERRORS.csv')
 for seed,color in [(42,BLUE),(52,ORANGE),(62,GRAY)]:
  d=h.loc[h.seed.eq(seed)];axs[1].plot(d.lead_h,d.MAE,label=f'Seed {seed}',color=color)
 axs[1].set(xlabel='Forecast lead (h)',ylabel='Conditional retention MAE (percentage points)',title='Same denominator error propagation');axs[1].legend(fontsize=8)
 fig.suptitle('Exploratory early-Development reference ratio | retrospective; NOT BoL HI or validated SOH')
 save(fig,'F3_conditional_retention_and_error',['outputs/FIXED_12H_ENDPOINT_DIAGNOSTIC.csv','outputs/CONDITIONAL_RETENTION_HORIZON_ERRORS.csv'],'145 supported targets,2871 dependent queries perseed; endpointsubset displayed')
 cycles=read(P/'outputs/FROZEN_CYCLE_PERFORMANCE_PREDICTIONS.csv');fig,axs=plt.subplots(2,1,figsize=(10,7),layout='constrained',sharex=True)
 axs[0].scatter(z.target_ophrs_h,z.CR_mean_percent,s=18,color=BLUE,label=f'F2 conditional ratio, fixed12h, n={len(z)}');axs[0].legend(fontsize=8);axs[0].set(ylabel='Conditional voltage retention (%)',title='Dynamic whole-bin mean | retrospective early-reference diagnostic')
 for Hn,color in [(1,BLUE),(2,ORANGE),(3,GRAY)]:
  d=cycles.loc[cycles.H.eq(Hn)];axs[1].scatter(d.target_actual_end_h,100*d.predicted_R,s=12,label=f'H{Hn}: {len(d)} forecasts',color=color)
 obs=cycles.drop_duplicates('target_raw_cycle_id');axs[1].scatter(obs.target_actual_end_h,100*obs.target_R,s=10,facecolors='none',edgecolors='black',label='Observed fixed-state retention')
 axs[1].set(xlabel='Actual operating time (h)',ylabel='Fixed-state performance retention (%)',title='Frozen Ridge96 | H1/H2/H3 = physical cycles, NOT hours');axs[1].legend(fontsize=8,ncol=2)
 fig.suptitle('Development context only | different denominators / observation operators | qualified comparison pairs: 0')
 save(fig,'F4_dynamic_and_cycle_context',['outputs/FIXED_12H_ENDPOINT_DIAGNOSTIC.csv','outputs/FROZEN_CYCLE_PERFORMANCE_PREDICTIONS.csv'],'Two separate panels; no forced correlation, no shared SOH scale')
 cs('data_contracts/FIGURE_SOURCE_INDEX.csv',pd.DataFrame(INDEX));print('4 PNG + 4 PDF figures generated')
if __name__=='__main__':main()
