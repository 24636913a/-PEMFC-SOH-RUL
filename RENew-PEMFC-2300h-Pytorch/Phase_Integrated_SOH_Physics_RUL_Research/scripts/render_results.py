"""Real CSV inputs; publication exports, no invented experiment series."""
import sys
from common import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'ps.fonttype':42,'savefig.dpi':300})
COLORS=['#597FBD','#BF5B62','#808080','#6D9F71']
INDEX=[]
def save(fig,name,source,meaning):
 fig.savefig(P/f'figures/{name}.png',bbox_inches='tight');fig.savefig(P/f'figures/{name}.pdf',bbox_inches='tight');plt.close(fig)
 INDEX.append({'figure':name,'source_csv':source,'script':'scripts/render_results.py','meaning':meaning})
def render_a():
 d=pd.read_csv(P/'outputs/RECOMPUTED_DYNAMIC_METRICS.csv');fig,ax=plt.subplots(1,2,figsize=(9,3.4),layout='constrained')
 for k,metric in enumerate(['RMSE_mV','MAE_mV']):
  for i,model in enumerate(['F0','F1','F2']):
   z=d[d.model.eq(model)];ax[k].bar(i,z[metric].mean(),color=[COLORS[2],COLORS[1],COLORS[0]][i],edgecolor='black',linewidth=.6)
   ax[k].scatter(i+np.linspace(-.12,.12,3),z[metric],color='black',s=14,zorder=3)
  ax[k].set_xticks(range(3),['F0 native','F1 fixed fusion','F2 gated']);ax[k].set_ylabel(metric.replace('_',' '));ax[k].set_title('Development Validation; 3 seeds')
 save(fig,'F1_dynamic_existing_results','outputs/RECOMPUTED_DYNAMIC_METRICS.csv','Formal query RMSE and MAE, points are seeds; metric dependence retained')
 p=source('preds');cs('paper_materials/P1S_FIGURE_SOURCE.csv',p)
 fig,axes=plt.subplots(2,2,figsize=(10,6),layout='constrained')
 for i,h in enumerate(['H1','H3']):
  for j,m in enumerate(['PERSISTENCE','RIDGE_96H']):
   z=p[p.horizon.eq(h)&p.model.eq(m)].sort_values('target_p1s_measurement_end_ophrs2_h')
   groups=(z.fold.ne(z.fold.shift())|z.target_raw_cycle_id.diff().ne(1)).cumsum()
   for k,(_,part) in enumerate(z.groupby(groups)):
    axes[i,0].plot(part.target_p1s_measurement_end_ophrs2_h,part.predicted_R_P1S_causal,color=COLORS[j],marker='.',ms=3,lw=.8,label=m if k==0 else None)
   if j==0:axes[i,0].scatter(z.target_p1s_measurement_end_ophrs2_h,z.true_R_P1S_causal,color='black',s=8,label='Observed P1S')
   axes[i,1].scatter(z.target_p1s_measurement_end_ophrs2_h,z.absolute_error_mV_equivalent,color=COLORS[j],s=12,label=m)
  axes[i,0].set_title(h+' physical-cycle forecast');axes[i,0].set_ylabel('P1S retention');axes[i,1].set_ylabel('Absolute error (mV-equivalent)')
  for a in axes[i]:a.set_xlabel('Target actual measurement end (operating h)');a.legend(fontsize=7)
 save(fig,'F2_P1S_actual_forecasts','paper_materials/P1S_FIGURE_SOURCE.csv','Historical legal Validation predictions; disconnected fold sections shown as measurements')
 e=pd.read_csv(P/'outputs/SOH_EVENT_PREDICTIONS.csv');fig,ax=plt.subplots(1,2,figsize=(10,3.7),layout='constrained')
 for k,h in enumerate(['H1','H3']):
  for j,m in enumerate(['M0_REFERENCE_PERSISTENCE','M1_RIDGE96_M2_NC']):
   z=e[e.horizon.eq(h)&e.model.eq(m)].sort_values('nominal_h');x=np.arange(len(z));ax[k].plot(x,z.predicted_SOH*100,marker=['s','o'][j],color=COLORS[j],label=['M0 last reference','M1 forecast + past affine'][j])
  ax[k].plot(x,z.true_SOH*100,'kD',label='Reference observed');ax[k].set_xticks(x,z.nominal_h.astype(str));ax[k].set_xlabel('Nominal reference identity (not exact operating h)');ax[k].set_ylabel('Reference retention (%)');ax[k].set_title(h+': retrospective calendar replay, n=3');ax[k].legend(fontsize=7)
 save(fig,'F3_SOH_event_predictions','outputs/SOH_EVENT_PREDICTIONS.csv','Actual sparse references and past-only forecasts; unknown release prohibits online claim')
 fig,ax=plt.subplots(1,2,figsize=(9,3.5),layout='constrained')
 for k,h in enumerate(['H1','H3']):
  for j,m in enumerate(['M0_REFERENCE_PERSISTENCE','M1_RIDGE96_M2_NC']):
   z=e[e.horizon.eq(h)&e.model.eq(m)].sort_values('nominal_h');ax[k].bar(np.arange(len(z))+(j-.5)*.3,z.abs_error_SOH_pp,width=.3,color=COLORS[j],edgecolor='black',linewidth=.5,label=['M0','M1'][j])
  ax[k].set_xticks(range(len(z)),z.nominal_h.astype(str));ax[k].set_ylabel('Absolute SOH error (percentage points)');ax[k].set_title(h+' same-event comparison');ax[k].legend()
 save(fig,'F4_SOH_event_errors','outputs/SOH_EVENT_PREDICTIONS.csv','No comparison of different event populations')
 cs('paper_materials/FIGURE_SOURCE_INDEX_A.csv',pd.DataFrame(INDEX))
def render_bc():
 d=pd.read_csv(P/'outputs/PHYSICS_GRADIENT_DIAGNOSTIC.csv');fig,ax=plt.subplots(1,2,figsize=(9,3.4),layout='constrained')
 ax[0].plot(d.node,d.slow_prior_gradient,color=COLORS[0],label='Slow-state prior');ax[0].scatter(d.node,d.ordinary_smoothing_gradient,color=COLORS[1],s=10,marker='x',label='Matched ordinary smoothing')
 ax[0].set_xlabel('Train sequence node');ax[0].set_ylabel('Gradient w.r.t. latent node (diagnostic)');ax[0].legend(fontsize=8)
 q=pd.read_csv(P/'outputs/PHYSICS_STATE_DECOMPOSITION.csv');ax[1].plot(q.available_end_h,q.observed_R,'k.',ms=3,label='Observed Train P1S')
 for j,name in enumerate(['D_flat','D_declining']):ax[1].plot(q.available_end_h,q[name],color=COLORS[j],label=name)
 ax[1].set_xlabel('Available operating h');ax[1].set_ylabel('Retention / latent state');ax[1].legend(fontsize=8);ax[1].set_title('Two decompositions, identical observed values')
 save(fig,'F5_physics_identifiability_and_gradients','outputs/PHYSICS_GRADIENT_DIAGNOSTIC.csv;outputs/PHYSICS_STATE_DECOMPOSITION.csv','Non-identifiability and exact smoothing equivalence, not trained physics improvement')
 z=pd.read_csv(P/'outputs/CONDITIONAL_CALENDAR_TRAJECTORIES.csv');fig,ax=plt.subplots(figsize=(7.5,4),layout='constrained')
 band=pd.read_csv(P/'outputs/CONDITIONAL_TRAJECTORY_SENSITIVITY_ENVELOPE.csv')
 ax.fill_between(band.elapsed_calendar_h/24,band.low_SOH*100,band.high_SOH*100,color='#D3D3D3',alpha=.6,label='Slope sensitivity envelope (not CI)')
 for j,(m,d) in enumerate(z.groupby('model',sort=False)):ax.plot(d.elapsed_calendar_h/24,d.projected_SOH*100,color=COLORS[j],label=m)
 for y in config()['scenario_thresholds']:ax.hlines(y*100,0,365,ls='--',color='gray',lw=.7);ax.text(367,y*100,f'{y*100:g}% assumption',fontsize=8,va='center')
 ax.set_xlim(0,420);ax.set_xlabel('Calendar days after latest Development reference');ax.set_ylabel('Projected reference retention (%)');ax.set_title('Conditional extrapolation; no failure validation');ax.legend(fontsize=7,framealpha=1,facecolor='white')
 save(fig,'F6_conditional_calendar_trajectories','outputs/CONDITIONAL_CALENDAR_TRAJECTORIES.csv;outputs/CONDITIONAL_TRAJECTORY_SENSITIVITY_ENVELOPE.csv','Calendar-time assumptions; no operating-hour RUL or confidence interval')
 fig,ax=plt.subplots(figsize=(11,5));ax.set_axis_off()
 nodes=[(.03,.62,'48 steps / 24 h history\nFrozen F2\n12 h dynamic voltage'),(.03,.20,'96 h P1S history\nFrozen Ridge\nH1 / H3 physical cycles'),(.36,.20,'Past reference calibration\nM2_NC affine\nSparse future SOH replay'),(.36,.62,'TFT health increment\nParameter-time gate failed\nNo matched gain evidence'),(.70,.20,'Past observed references\nCalendar lifetime scenarios\nNo validated true RUL'),(.70,.62,'Soft physics qualification\nP1 unidentifiable\nP2 uncertified')]
 for x,y,s in nodes:
  ax.add_patch(FancyBboxPatch((x,y),.25,.22,boxstyle='round,pad=.01',transform=ax.transAxes,facecolor='#eef2f8',edgecolor=COLORS[0]));ax.text(x+.125,y+.11,s,transform=ax.transAxes,ha='center',va='center',fontsize=8)
 ax.annotate('',xy=(.36,.31),xytext=(.28,.31),xycoords='axes fraction',arrowprops={'arrowstyle':'->','color':COLORS[0]})
 ax.annotate('',xy=(.70,.31),xytext=(.61,.31),xycoords='axes fraction',arrowprops={'arrowstyle':'->','linestyle':'--','color':'gray'})
 ax.text(.655,.10,'Unvalidated\ninterface only',transform=ax.transAxes,ha='center',fontsize=7,color='gray')
 ax.text(.5,.94,'Actual information flow and blocked interfaces',transform=ax.transAxes,ha='center',fontsize=12)
 save(fig,'F7_integrated_evidence_flow','outputs/EVIDENCE_MATRIX.csv;configs/FROZEN_PROTOCOL.json','Only supported mathematical/data links shown; not an end-to-end network')
 d=pd.read_csv(P/'outputs/RECOMPUTED_DYNAMIC_METRICS.csv');fig,ax=plt.subplots(1,2,figsize=(9.5,3.6),layout='constrained')
 for k,metric in enumerate(['target_time_equal_RMSE_mV','max_abs_error_mV']):
  for j,model in enumerate(['F0','F1','F2']):
   z=d[d.model.eq(model)].sort_values('seed');ax[k].plot(z.seed,z[metric],color=[COLORS[2],COLORS[1],COLORS[0]][j],marker=['s','^','o'][j],label=model)
  ax[k].set_xticks([42,52,62]);ax[k].set_xlabel('Frozen seed');ax[k].set_ylabel('RMSE (mV)' if k==0 else 'Max absolute error (mV)');ax[k].set_title('Equal target-time weighting' if k==0 else 'Tail error retained');ax[k].legend()
 save(fig,'F8_dynamic_robustness_limits','outputs/RECOMPUTED_DYNAMIC_METRICS.csv','O1 metric dependence and Seed62 extreme retained; no universal robustness claim')
 old=pd.read_csv(P/'paper_materials/FIGURE_SOURCE_INDEX_A.csv');cs('paper_materials/FIGURE_SOURCE_INDEX.csv',pd.concat([old,pd.DataFrame(INDEX)],ignore_index=True))
if __name__=='__main__':
 firewall();{'a':render_a,'bc':render_bc}[sys.argv[1]]();print('Rendered real results')
