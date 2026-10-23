"""Publication candidates drawn exclusively from auditable CSVs; no simulation."""
from common import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'ps.fonttype':42,'savefig.dpi':300})
BLUE='#2274A5';ORANGE='#D97928';GRAY='#686868';COLORS=plt.get_cmap('tab10').colors
def save(fig,name):
 fig.savefig(out('figures/'+name+'.png'),dpi=300,bbox_inches='tight',facecolor='white');fig.savefig(out('figures/'+name+'.pdf'),bbox_inches='tight',facecolor='white');plt.close(fig)
def run():
 dw=pd.read_csv(P/'outputs/POLARIZATION_DWELLS.csv');base=dw[dw.window.eq('middle50')];sl=pd.read_csv(P/'outputs/NUMERICAL_SLOPES.csv');hs=pd.read_csv(P/'outputs/HYSTERESIS.csv')
 fig,axes=plt.subplots(2,4,figsize=(13,6),layout='constrained')
 for ax,k in zip(axes.flat,CHECKPOINTS):
  for b,col,m in [('descending',BLUE,'o'),('ascending',ORANGE,'s')]:
   a=base[(base.checkpoint==k)&base.branch.eq(b)].sort_values('j_A_cm2');ax.plot(a.j_A_cm2,a.V_V,'-',color=col,lw=1);ax.scatter(a.j_A_cm2,a.V_V,s=14,facecolors='none',edgecolors=col,marker=m);z=a[a.stable_domain];ax.scatter(z.j_A_cm2,z.V_V,s=14,c=col,marker=m)
  ax.set(title=k,xlabel=r'$j$ (A cm$^{-2}$)',ylabel='Active10 mean V (V)');ax.grid(alpha=.15)
 axes.flat[-1].axis('off');axes.flat[-1].legend(handles=[Line2D([],[],color=BLUE,marker='o',label='Descending'),Line2D([],[],color=ORANGE,marker='s',label='Ascending'),Line2D([],[],color=GRAY,marker='o',linestyle='',label='Filled: passes domain/gates')],loc='center',frameon=False)
 fig.suptitle('Development polarization measurements: direction and within-dwell screening');save(fig,'01_real_polarization_directions')
 fig,axes=plt.subplots(1,2,figsize=(10,3.8),layout='constrained')
 for i,k in enumerate(CHECKPOINTS):
  for b,ls in [('descending','-'),('ascending','--')]:
   a=sl[(sl.checkpoint==k)&sl.branch.eq(b)&sl.window.eq('middle50')].sort_values('j_mid');axes[0].plot(a.j_mid,a.Rdiff_ohm_cm2,ls,marker='.',color=COLORS[i],label=k if b=='descending' else None,lw=1)
  a=hs[hs.checkpoint.eq(k)].sort_values('j_A_cm2');axes[1].plot(a.j_A_cm2,a.up_minus_down_mV,marker='.',color=COLORS[i],label=k,lw=1)
 axes[0].set(xlabel=r'$j$ midpoint (A cm$^{-2}$)',ylabel=r'$R_{diff}$ ($\Omega$ cm$^2$)',title='Composite local slope; solid down / dashed up');axes[0].legend(ncols=2,fontsize=7,frameon=False)
 axes[1].set(xlabel=r'$j$ (A cm$^{-2}$)',ylabel='Up minus down (mV)',title='Matched-setpoint hysteresis (stable pairs)');axes[1].axhline(0,color=GRAY,lw=.7);save(fig,'02_slopes_and_hysteresis')
 fig,axes=plt.subplots(1,2,figsize=(10,3.8),layout='constrained');sens=pd.read_csv(P/'outputs/SLOPE_WINDOW_SENSITIVITY.csv');z=sens[sens.windows.eq(4)].reset_index(drop=True)
 axes[0].vlines(np.arange(len(z)),z.slope_min,z.slope_max,color=BLUE,lw=1);axes[0].set(xlabel='Stable point pair (all four windows)',ylabel=r'$R_{diff}$ ($\Omega$ cm$^2$)',title='Window sensitivity range; not a confidence interval')
 shapes=pd.read_csv(P/'outputs/J1_CENTERED_SHAPES.csv')
 for i,k in enumerate(CHECKPOINTS):
  a=shapes[shapes.checkpoint.eq(k)&shapes.branch.eq('descending')].sort_values('j_A_cm2');axes[1].plot(a.j_A_cm2,a.V_minus_V1*1000,'.-',color=COLORS[i],label=k,lw=1)
 axes[1].axhline(0,color=GRAY,lw=.7);axes[1].set(xlabel=r'$j$ (A cm$^{-2}$)',ylabel=r'$V(j)-V(1)$ (mV)',title='Voltage-centered shape; diagnostic only');axes[1].legend(ncols=2,fontsize=7,frameon=False);save(fig,'03_sensitivity_and_centered_shapes')
 fig,axes=plt.subplots(1,2,figsize=(10,3.8),layout='constrained');prof=pd.read_csv(P/'outputs/JLIM_PROFILE.csv');fit=pd.read_csv(P/'outputs/PARAMETER_IDENTIFIABILITY.csv');z=fit[fit.status.eq('FITTED_DESCRIPTIVE')]
 for i,((k,b),g) in enumerate(prof.groupby(['checkpoint','branch'])):axes[0].plot(g.fixed_jlim,g.RMSE_mV,'o-',lw=1,label=k+' '+b,color=COLORS[i])
 axes[0].set(xlabel=r'Fixed $j_{lim}$ (A cm$^{-2}$)',ylabel='Profile fit RMSE (mV)',title='Predeclared limiting-current profiles');axes[0].legend(fontsize=7,frameon=False)
 x=np.arange(len(z));axes[1].bar(x,z.jacobian_normalized_condition,color=[BLUE if v else GRAY for v in z.identifiable],edgecolor='black',lw=.5);axes[1].set_xticks(x,[r.checkpoint+'\n'+r.branch for r in z.itertuples()],rotation=25,ha='right');axes[1].set(ylabel='Column-normalized Jacobian condition',title='Local numerical screen; blue passes only this screen');save(fig,'04_parameter_identifiability')
 metrics=pd.read_csv(P/'outputs/H1_H2_H3_FOLD_METRICS.csv');eq=pd.read_csv(P/'outputs/H1_H2_H3_EQUAL_MEANS.csv');fig,axes=plt.subplots(1,3,figsize=(11.5,3.5),layout='constrained')
 for ax,H in zip(axes,[1,2,3]):
  for method,off,col,hatch in [('PERSISTENCE',-.18,GRAY,''),('RIDGE96',.18,BLUE,'//')]:
   z=metrics[(metrics.H==H)&metrics.method.eq(method)].sort_values('fold');y=list(z.MAE_mVeq)+[float(eq[(eq.H==H)&eq.method.eq(method)].MAE_mVeq.iloc[0])];ax.bar(np.arange(4)+off,y,.35,label=method,color=col,edgecolor='black',linewidth=.5,hatch=hatch)
  ax.set_xticks(range(4),['Fold1','Fold2','Fold3','Equal\nmean']);ax.set(ylabel='MAE (mV-equivalent)',title=f'H{H} '+('new exploratory CPU' if H==2 else 'frozen replay'));ax.set_ylim(bottom=0)
 axes[0].legend(fontsize=7,frameon=False);fig.suptitle('P1S retention prediction: own legal cohort per horizon');save(fig,'05_short_horizon_comparison')
 pred=pd.read_csv(P/'outputs/H2_PREDICTIONS.csv');fig,axes=plt.subplots(3,1,figsize=(10,7),layout='constrained')
 for ax,(fold,g) in zip(axes,pred.groupby('fold')):
  a=g[g.method.eq('PERSISTENCE')].sort_values('target_actual_end_h');b=g[g.method.eq('RIDGE96')].sort_values('target_actual_end_h');ax.plot(a.target_actual_end_h,a.target_R*100,'k.-',label='Measured target');ax.plot(a.target_actual_end_h,a.predicted_R*100,'--',color=GRAY,label='Persistence');ax.plot(b.target_actual_end_h,b.predicted_R*100,'o-',ms=3,color=BLUE,label='Ridge96');ax.set(title=fold+f' (n={len(a)})',xlabel='Actual target operating hour',ylabel='P1S retention (%)');ax.grid(alpha=.15)
 axes[0].legend(ncols=3,fontsize=8,frameon=False);fig.suptitle('New H2 predictions on genuine complete-cycle targets');save(fig,'06_H2_predictions')
 av=pd.read_csv(P/'outputs/AVAILABILITY_COUNTS.csv');av=av[av.role.eq('validation')];fig,ax=plt.subplots(figsize=(9,3.5),layout='constrained');x=np.arange(len(av));ax.bar(x,av.completion_min,color=BLUE,label='Completed by origin: minimum');ax.errorbar(x,av.completion_min,yerr=[np.zeros(len(av)),av.completion_max-av.completion_min],fmt='none',ecolor=ORANGE,capsize=5,label='Within-fold completion range');ax.scatter(x,np.zeros(len(av)),marker='x',c='black',label='Certified publication records: 0');ax.set_xticks(x,[r.horizon+' '+r.fold for r in av.itertuples()]);ax.set(ylabel='Available polarization events',title='Completion-only replay is conditional; publication time is unknown',ylim=(-.4,7));ax.legend(fontsize=8,ncols=1,frameon=False);save(fig,'07_availability_boundary')
 cs('data_contracts/FIGURE_SOURCE_MAP.csv',pd.DataFrame([{'figure':n,'CSV_sources':s,'script':'scripts/plot_figures.py','identity':'EXPLORATORY_DEVELOPMENT_NOT_PHYSICS_ENHANCED_HEALTH_TRAINING'} for n,s in [('01_real_polarization_directions','POLARIZATION_DWELLS.csv'),('02_slopes_and_hysteresis','NUMERICAL_SLOPES.csv;HYSTERESIS.csv'),('03_sensitivity_and_centered_shapes','SLOPE_WINDOW_SENSITIVITY.csv;J1_CENTERED_SHAPES.csv'),('04_parameter_identifiability','JLIM_PROFILE.csv;PARAMETER_IDENTIFIABILITY.csv'),('05_short_horizon_comparison','H1_H2_H3_FOLD_METRICS.csv;H1_H2_H3_EQUAL_MEANS.csv'),('06_H2_predictions','H2_PREDICTIONS.csv'),('07_availability_boundary','AVAILABILITY_COUNTS.csv')]]));print('FIGURES_COMPLETED_7_PNG_PDF')
if __name__=='__main__':firewall();run()
