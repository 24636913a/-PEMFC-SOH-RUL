"""Real CSV-only publication figures, separate reconstructed and future-event tasks."""
from common import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch,FancyArrowPatch
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'ps.fonttype':42,'savefig.dpi':300})
BLUE='#2274A5';ORANGE='#CE7B28';GRAY='#737373'
def save(fig,name):
 for ext in ['png','pdf']:fig.savefig(out('figures/'+name+'.'+ext),bbox_inches='tight',dpi=300,facecolor='white')
 plt.close(fig)
def run():
 cfg=config();data=pd.read_csv(P/'data_contracts/ALL_LEGAL_SAMPLES.csv');base=pd.read_csv(P/'data_contracts/ALL_154_PLATFORM_QUALIFICATION.csv');pred=pd.read_csv(P/'outputs/ALL_POINT_PREDICTIONS.csv');dense=pd.read_csv(P/'outputs/ALL_EVALUATION_SLOPES.csv');ev=pd.read_csv(P/'outputs/EVENT_SEED_METRICS.csv');decision=jl('outputs/RESEARCH_DECISION.json')
 fig,axes=plt.subplots(2,4,figsize=(13,6),layout='constrained')
 for ax,event in zip(axes.flat,EVENTS):
  for branch,c,mark in [('ascending',ORANGE,'s'),('descending',BLUE,'o')]:
   g=base[base.checkpoint.eq(event)&base.branch.eq(branch)].sort_values('set_j_A_cm2');ax.plot(g.set_j_A_cm2,g.V_V,'-',c=c,lw=.8);ax.scatter(g.set_j_A_cm2,g.V_V,s=17,marker=mark,facecolors='none',edgecolors=c);z=g[g.stable_domain];ax.scatter(z.set_j_A_cm2,z.V_V,s=17,marker=mark,color=c)
  ax.set(title=event,xlabel=r'Commanded $j$ (A cm$^{-2}$)',ylabel='Mean cell voltage (V)');ax.grid(alpha=.12)
 axes.flat[-1].axis('off');axes.flat[-1].legend(handles=[Line2D([],[],c=BLUE,marker='o',label='Descending'),Line2D([],[],c=ORANGE,marker='s',label='Ascending'),Line2D([],[],c=GRAY,marker='o',linestyle='',label='Filled: passes frozen gates')],loc='center',frameon=False);fig.suptitle('154 candidate platforms; 105 pass frozen near-steady screening');save(fig,'F1_real_polarization_support')
 fig,axes=plt.subplots(2,4,figsize=(13,6),layout='constrained');aa=data[data.task.eq('A')];dd=dense[dense.task.eq('A')]
 for ax,event in zip(axes.flat,EVENTS):
  for branch,style in [('descending','-'),('ascending','--')]:
   for arm,col in [('A_DATA_ONLY',BLUE),('B_PHYSICS',ORANGE)]:
    z=dd[dd.checkpoint.eq(event)&dd.branch.eq(branch)&dd.arm.eq(arm)].groupby('j_design_A_cm2').predicted_V.mean();ax.plot(z.index,z.values,style,color=col,lw=1.2,label=arm+' '+branch)
  tr=aa[aa.checkpoint.eq(event)&aa.role.eq('train')];va=aa[aa.checkpoint.eq(event)&aa.role.eq('validation')];ax.scatter(tr.j_design_A_cm2,tr.true_V,c=GRAY,s=12);ax.scatter(va.j_design_A_cm2,va.true_V,facecolors='none',edgecolors='black',marker='s',s=25);ax.set(title=event,xlabel=r'$j$ (A cm$^{-2}$)',ylabel='V (V)')
 axes.flat[-1].axis('off');axes.flat[-1].legend(handles=[Line2D([],[],color=BLUE,label='Data only, seed mean'),Line2D([],[],color=ORANGE,label='Physics, seed mean'),Line2D([],[],color='black',linestyle='-',label='Descending'),Line2D([],[],color='black',linestyle='--',label='Ascending'),Line2D([],[],color='black',marker='s',markerfacecolor='none',linestyle='',label='Held-out platform')],loc='center',frameon=False,fontsize=8);fig.suptitle('Task A: measured-checkpoint reconstruction; A/B curves nearly overlap');save(fig,'F2_AB_reconstructed_curves')
 fig,axes=plt.subplots(1,2,figsize=(10,4),layout='constrained');labels=EVENTS;x=np.arange(len(labels))
 for arm,col,off in [('A_DATA_ONLY',BLUE,-.17),('B_PHYSICS',ORANGE,.17)]:
  g=ev[ev.task.eq('A')&ev.arm.eq(arm)].groupby('checkpoint').MAE_mV.agg(['mean','std']).reindex(labels);axes[0].bar(x+off,g['mean'],.32,color=col,label=arm,edgecolor='black',linewidth=.5);axes[0].errorbar(x+off,g['mean'],yerr=g['std'],fmt='none',ecolor='black',capsize=2)
 axes[0].set_xticks(x,labels,rotation=25);axes[0].set(ylabel='Held-out MAE (mV)',title='Task A: mean +/- sample SD across 3 seeds');axes[0].legend(fontsize=8,frameon=False)
 p=pd.read_csv(P/'outputs/PAIRED_SEED_AB_COMPARISON.csv');seeds=cfg['seeds']
 for task,c,off in [('A',BLUE,-.14),('B',ORANGE,.14)]:
  z=p[p.task.eq(task)].set_index('seed').loc[seeds];axes[1].bar(np.arange(3)+off,(z.MAE_data_mV-z.MAE_physics_mV)*1000,.27,color=c,label='Task '+task,edgecolor='black',linewidth=.5)
 axes[1].axhline(0,c=GRAY,lw=.8);axes[1].set_xticks(range(3),seeds);axes[1].set(xlabel='Matched random seed',ylabel='MAE reduction (microvolt)',title='Tiny paired differences; positive favors physics');axes[1].legend(frameon=False);save(fig,'F3_holdout_errors_and_tiny_AB_differences')
 logs=[]
 for q in sorted((P/'outputs/logs').glob('*.csv')):
  g=pd.read_csv(q);g['task']=q.name[0];g['arm']='B_PHYSICS' if '_B_PHYSICS' in q.name else 'A_DATA_ONLY';logs.append(g)
 log=pd.concat(logs);summary=log.groupby(['task','arm','epoch'])[['physics_loss_normalized','violation_rate','weighted_physics_grad_ratio']].mean().reset_index();cs('outputs/FIGURE_TRAINING_ACTIVITY.csv',summary)
 fig,axes=plt.subplots(1,3,figsize=(12,3.8),layout='constrained')
 for task,style in [('A','-'),('B','--')]:
  for arm,col in [('A_DATA_ONLY',BLUE),('B_PHYSICS',ORANGE)]:
   g=summary[summary.task.eq(task)&summary.arm.eq(arm)];z=g[g.epoch<=25];axes[0].plot(z.epoch,z.violation_rate,style,c=col,label=f'{task}: {arm}');axes[1].plot(g.epoch,g.physics_loss_normalized,style,c=col)
  g=summary[summary.task.eq(task)&summary.arm.eq('B_PHYSICS')];axes[2].plot(g.epoch,g.weighted_physics_grad_ratio,style,label='Task '+task)
 axes[0].set(xlabel='Epoch (first 25)',ylabel='Train probe violation fraction',title='Early transient activation');axes[0].legend(fontsize=6,frameon=False)
 axes[1].set(xlabel='Epoch',ylabel='Actual normalized physics loss',title='Later loss is exactly zero');axes[1].set_yscale('symlog',linthresh=1e-10)
 axes[2].set(xlabel='Epoch',ylabel='Weighted physics / data gradient L2',title='Real parameter-gradient contribution');axes[2].set_yscale('symlog',linthresh=1e-8);axes[2].legend(frameon=False);save(fig,'F4_real_physics_activity_and_gradients')
 fig,axes=plt.subplots(1,2,figsize=(10,4),layout='constrained');events=['1150h','1500h','1750h']
 for arm,col,off in [('A_DATA_ONLY',BLUE,-.17),('B_PHYSICS',ORANGE,.17)]:
  g=ev[ev.task.eq('B')&ev.arm.eq(arm)].groupby('checkpoint').MAE_mV.agg(['mean','std']).reindex(events);axes[0].bar(np.arange(3)+off,g['mean'],.32,color=col,label=arm,edgecolor='black',lw=.5);axes[0].errorbar(np.arange(3)+off,g['mean'],yerr=g['std'],fmt='none',c='black',capsize=3)
  z=ev[ev.arm.eq(arm)].groupby('task').violation_rate.mean().reindex(['A','B']);axes[1].scatter(np.arange(2)+off,z.values,c=col,marker='o' if arm=='A_DATA_ONLY' else 'x',s=60,label=arm)
 axes[0].set_xticks(range(3),events);axes[0].set(ylabel='Voltage MAE (mV)',title='Task B: next scheduled event, conditional protocol');axes[0].legend(frameon=False,fontsize=8)
 axes[1].set_xticks(range(2),['Reconstruction','Future-event replay']);axes[1].set(ylabel='Final evaluation slope violation rate',ylim=(-.01,.1),title='Both arms: 0 on the evaluated grids');axes[1].legend(frameon=False,fontsize=8);save(fig,'F5_future_event_errors_and_final_violations')
 architecture=pd.DataFrame([
 {'module':'Dual-pathTFT','inputs':'48x14 frozen history','internal':'VSN -> nativeLSTM + causal multiscaleConv -> historicalregimegate -> staticenhancement/attention','output':'24stepDeltaV','status':'FROZEN_EXISTING','connection':'No newweightupdate','source':SOURCES['tft_arch']},
 {'module':'Ridge96','inputs':'96h causalP1S','internal':'FrozenH1/H2/H3','output':'P1Sretention','status':'FROZEN_EXISTING','connection':'M2_NC conditional3eventreplay unchanged','source':SOURCES['h_contract']},
 {'module':'IndependentMLP','inputs':'jdesign, direction, legalcontext','internal':'32SiLU ->32SiLU ->1,1217parameters','output':'polarizationV; autodiffdV/dj','status':'COMPLETED_AB_NO_INCREMENT','connection':'Independent; noactualTFT/SOHfusion','source':'outputs/RESEARCH_DECISION.json'},
 {'module':'P1_TFT_auxiliary','inputs':'Pastonlycrossfitphysicalteacher','internal':'Sameinfo/capacitycontrolrequired','output':'TFTauxiliaryloss','status':'UNVERIFIED_NOT_STARTED','connection':'No validatedincrement, do notstart','source':'outputs/RESEARCH_DECISION.json'},
 {'module':'P2_referenceSOH','inputs':'Causallyknownstate/referenceconditions','internal':'Vpol(j1)/0.6989533305168152','output':'ReferenceSOH','status':'UNVERIFIED_NOT_STARTED','connection':'Not same label twice; frozenM2_NC control','source':'outputs/RESEARCH_DECISION.json'}]);cs('outputs/ARCHITECTURE_STATUS.csv',architecture)
 fig,ax=plt.subplots(figsize=(12,6));ax.set(xlim=(0,12),ylim=(0,6));ax.axis('off')
 def box(x,y,w,h,text,c):ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=.08',facecolor=c,edgecolor=GRAY,lw=.8));ax.text(x+w/2,y+h/2,text,ha='center',va='center',fontsize=9)
 def arrow(a,b,c=BLUE,style='-'):ax.add_patch(FancyArrowPatch(a,b,arrowstyle='-|>',mutation_scale=12,color=c,linestyle=style,lw=1.4))
 box(.2,4.25,2.0,1.0,'48 x 14 history\nFrozen TFT VSN','#EAF3F7');box(2.8,4.8,2.25,.7,'Native LSTM path','#EAF3F7');box(2.8,3.7,2.25,.7,'Causal multiscale Conv','#EAF3F7');box(5.7,4.25,2.2,1.0,'Historical regime gate\nDual-path fusion','#EAF3F7');box(8.65,4.1,3,1.3,'Frozen TFT static enhancement\nattention + output\n24-step dynamic DeltaV','#EAF3F7');arrow((2.2,4.8),(2.8,5.1));arrow((2.2,4.6),(2.8,4));arrow((5.05,5.1),(5.7,4.9));arrow((5.05,4),(5.7,4.5));arrow((7.9,4.75),(8.65,4.75))
 node_table=pd.DataFrame([
 {'module':'independent','x':.2,'y':1.7,'w':3.3,'h':1.1,'label':'Independent MLP: j, direction, context\nV(j); actual autodiff weak loss\nA/B complete: no clear increment','color':'#FFF4DF'},
 {'module':'ridge','x':4.2,'y':1.7,'w':3.1,'h':1.1,'label':'Frozen Ridge96 H1/H2/H3\nPast-only M2_NC replay retained\nNo weight or label changes','color':'#EAF3F7'},
 {'module':'interfaces','x':8.2,'y':1.7,'w':3.4,'h':1.1,'label':'P1: TFT auxiliary teacher\nP2: reference SOH readout\nUnverified; no integration training','color':'#FAEEEE'}]);cs('outputs/FIGURE_ARCHITECTURE_NODES.csv',node_table)
 for r in pd.read_csv(P/'outputs/FIGURE_ARCHITECTURE_NODES.csv').itertuples():box(r.x,r.y,r.w,r.h,r.label,r.color)
 ax.add_patch(FancyArrowPatch((3.5,1.9),(8.2,1.9),arrowstyle='-|>',mutation_scale=12,color='#B34B4B',linestyle='--',lw=1.3,connectionstyle='arc3,rad=.3'))
 ax.text(5.9,3.15,'Separate branches: no demonstrated TFT or SOH improvement from the weak-physics model',ha='center',fontsize=10)
 ax.text(.2,.75,'Solid blue: existing frozen TFT internal connections.  Dashed red: proposed interfaces only.\nOne RealHyFC trajectory; no Test values, new EOL labels or RUL training.',fontsize=10);fig.subplots_adjust(left=.02,right=.98,top=.98,bottom=.02);save(fig,'F6_frozen_dual_path_TFT_and_independent_branch')
 mapping=[('F1_real_polarization_support','data_contracts/ALL_154_PLATFORM_QUALIFICATION.csv'),('F2_AB_reconstructed_curves','outputs/ALL_EVALUATION_SLOPES.csv;data_contracts/ALL_LEGAL_SAMPLES.csv'),('F3_holdout_errors_and_tiny_AB_differences','outputs/EVENT_SEED_METRICS.csv;outputs/PAIRED_SEED_AB_COMPARISON.csv'),('F4_real_physics_activity_and_gradients','outputs/FIGURE_TRAINING_ACTIVITY.csv'),('F5_future_event_errors_and_final_violations','outputs/EVENT_SEED_METRICS.csv'),('F6_frozen_dual_path_TFT_and_independent_branch','outputs/ARCHITECTURE_STATUS.csv;outputs/FIGURE_ARCHITECTURE_NODES.csv')]
 cs('data_contracts/FIGURE_SOURCE_INDEX.csv',pd.DataFrame([{'figure':n,'CSV_sources':paths,'plot_script':'scripts/render_figures.py','script_SHA256':sha(P/'scripts/render_figures.py'),'protocol_SHA256':sha(P/'configs/FROZEN_PROTOCOL.json'),'base_commit':cfg['base_commit']} for n,paths in mapping]));print('SIX_FIGURE_PAIRS_COMPLETED')
if __name__=='__main__':firewall();run()
