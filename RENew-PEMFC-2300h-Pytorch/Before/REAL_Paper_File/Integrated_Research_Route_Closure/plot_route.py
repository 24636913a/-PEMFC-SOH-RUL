"""One integration figure; every edge comes from CONNECTION_MATRIX.csv."""
from pathlib import Path
import csv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch,FancyArrowPatch
P=Path(__file__).resolve().parent
def main():
 with (P/'CONNECTION_MATRIX.csv').open(encoding='utf-8') as f:rows=list(csv.DictReader(f))
 plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'svg.fonttype':'none'})
 fig,ax=plt.subplots(figsize=(13,8));ax.set(xlim=(0,13),ylim=(0,8));ax.axis('off')
 boxes={
 'Development_bins':((.2,6.35),'Development dynamic bins\nPast inputs + time contract'),
 'Frozen_TFT':((3.45,6.35),'Frozen TFT F2\nDynamic voltage evidence'),
 'P1S_or_SOH_increment':((6.9,6.35),'TFT -> health increment\nNot yet causally validated'),
 'Actual_P1S_history':((.2,4.35),'Real P1S observations\n96 h causal history'),
 'Frozen_H123_Ridge':((3.45,4.35),'Frozen H1/H2/H3 Ridge\nProxy forecast evidence'),
 'M2_NC_SOH_replay':((6.9,4.35),'Past-only M2_NC -> SOH\nCompleted 3-event replay'),
 'Validated_true_RUL':((10.15,4.35),'True RUL validation\nNo verified EOL labels'),
 'Physical_diagnostics':((.2,1.55),'Polarization / EIS diagnostics\nMeasured, descriptive'),
 'H123_training_loss':((3.45,1.55),'Health physics loss\nNot qualified; loss/gradient NA'),
 'Reference_SOH':((6.9,1.55),'Measured reference SOH\n7 events, past-only calibration'),
 'Conditional_lifetime_scenarios':((10.15,1.55),'Conditional lifetime scenarios\nCompleted, assumptions only')}
 width=2.65;height=1.05
 for k,(xy,label) in boxes.items():
  fail=k in ['P1S_or_SOH_increment','Validated_true_RUL','H123_training_loss'];color='#FAEEEE' if fail else '#FFF4DF' if k in ['M2_NC_SOH_replay','Conditional_lifetime_scenarios'] else '#EAF3F7'
  ax.add_patch(FancyBboxPatch(xy,width,height,boxstyle='round,pad=0.08',facecolor=color,edgecolor='#4F5B66',linewidth=.8));ax.text(xy[0]+width/2,xy[1]+height/2,label,ha='center',va='center')
 edges=[('Development_bins','Frozen_TFT','Development_bins','Frozen_TFT','dynamic'),
 ('Frozen_TFT','P1S_or_SOH_increment','TFT_voltage','P1S_or_SOH_increment','target / parameter-time gap'),
 ('Actual_P1S_history','Frozen_H123_Ridge','Actual_P1S_history','Frozen_H123_Ridge','H1 primary; H3 auxiliary; H2'),
 ('Frozen_H123_Ridge','M2_NC_SOH_replay','H1_H3_Ridge_P1S','M2_NC_SOH_replay','H1/H3 only; release unknown'),
 ('M2_NC_SOH_replay','Validated_true_RUL','SOH_or_P1S','Validated_true_RUL','EOL not established'),
 ('Physical_diagnostics','H123_training_loss','Physical_diagnostics','H123_training_loss','structure + parameters + time'),
 ('Reference_SOH','M2_NC_SOH_replay','Past_reference_pairs','M2_NC','past pairs; release unknown'),
 ('Reference_SOH','Conditional_lifetime_scenarios','Reference_SOH','Conditional_lifetime_scenarios','reference scenario only')]
 for a,b,source,target,label in edges:
  r=next(r for r in rows if r['source_module']==source and r['target_module']==target);x,y=boxes[a][0];xx,yy=boxes[b][0];same=abs(y-yy)<.01
  start=(x+width,y+height/2) if same else (x+width/2,y+height if yy>y else y);end=(xx,yy+height/2) if same else (xx+width/2,yy if yy>y else yy+height)
  conditional=r['implementation']=='COMPLETED_CONDITIONAL';unqualified=r['implementation'] in ['UNQUALIFIED_INTERFACE','UNVERIFIED_INTERFACE'];color='#BA4545' if unqualified else '#AF7B24' if conditional else '#2274A5';style='--' if unqualified else '-'
  ax.add_patch(FancyArrowPatch(start,end,arrowstyle='-|>',mutation_scale=13,linestyle=style,color=color,linewidth=1.4,connectionstyle='arc3,rad=0' if same else 'arc3,rad=.04'))
  ax.text((start[0]+end[0])/2,(start[1]+end[1])/2+(.22 if same else -.4),label,ha='center',va='bottom',fontsize=7,color=color,bbox={'facecolor':'white','edgecolor':'none','alpha':.9,'pad':1})
 ax.text(6.5,7.8,'One RealHyFC paper: completed evidence and unresolved interfaces',ha='center',fontsize=15)
 ax.text(3.35,3.45,'Required SOH control:\nmeasured-origin P1S + frozen M2_NC\nH2 has no reference-SOH validation',ha='center',fontsize=9)
 ax.text(.2,.65,'Solid blue: completed module connection     Solid amber: completed conditional replay / scenario\nDashed red: unverified or unqualified interface     No new training or Test numerical access in this closure',fontsize=10,color='#3B4650')
 fig.subplots_adjust(left=.02,right=.98,bottom=.02,top=.97)
 for ext in ['png','svg']:fig.savefig(P/f'integrated_route.{ext}',dpi=300,bbox_inches='tight',facecolor='white')
 plt.close(fig);print('ROUTE_FIGURE_COMPLETED')
if __name__=='__main__':main()
