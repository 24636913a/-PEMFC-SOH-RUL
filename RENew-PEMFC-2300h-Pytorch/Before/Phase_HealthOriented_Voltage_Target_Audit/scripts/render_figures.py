"""Publication candidate plots exclusively from actual Development CSVs."""
from common import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch,FancyArrowPatch
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'ps.fonttype':42,'savefig.dpi':300})
BLUE='#2878B5';ORANGE='#D97928';GRAY='#777777';records=[]
def save(fig,name,csvs,scope,svg=False):
 for ext in ['png','pdf']+(['svg'] if svg else []):
  path=P/f'figures/{name}.{ext}';fig.savefig(path,dpi=300,bbox_inches='tight')
  if ext=='svg':path.write_bytes(('\n'.join(line.rstrip() for line in path.read_text(encoding='utf-8').splitlines())+'\n').encode())
  records.append(dict(figure=path.relative_to(P).as_posix(),CSV_sources=';'.join(csvs),code='scripts/render_figures.py',scope=scope))
 plt.close(fig)
def main():
 d=rd(P/'outputs/STATE_CYCLE_OBSERVATIONS.csv');fig,axs=plt.subplots(3,1,figsize=(10,8),sharex=True,layout='constrained')
 for ax,token in zip(axs,['156||156.800','221||221.200','313||313.600']):
  q=d.loc[d.state_token.eq(token)];ax.scatter(q.last_measurement_h,q.V_cycle_median,s=15,color=BLUE,label=f'All-dwell cycle median, n={len(q)}')
  ax.scatter(q.last_measurement_h,q.V_QC_cycle_median,s=12,marker='x',color=ORANGE,label='QC-subset median: diagnostic only')
  ax.set(ylabel='Fixed-state voltage (V)',title=f'{token}: actual j={q.current_density.median():.6f} A/cm²');ax.legend(fontsize=8,ncol=2);ax.grid(alpha=.15)
 axs[-1].set_xlabel('Actual operating time of final selected measurement (h)')
 fig.suptitle('Development | 146 complete-cycle blocks per state | one durability trajectory; no interpolated labels')
 save(fig,'F1_real_fixed_state_voltage',['outputs/STATE_CYCLE_OBSERVATIONS.csv'],'Nonmonotonic true measurements; different current states remain separate')
 p=rd(P/'outputs/FIXED_PRIMARY_CYCLE_OBSERVATIONS.csv');fig,axs=plt.subplots(4,1,figsize=(10,9),sharex=True,layout='constrained')
 old=p.raw_cycle_id.le(150)
 for ax,col,label in zip(axs,['V_cycle_median','anode_inlet_temperature_C','anode_inlet_RH_pct','anode_inlet_pressure_kPag'],['Fixed-state voltage (V)','Anode temperature (°C)','Anode relative humidity (%)','Anode inlet pressure (kPa gauge)']):
  ax.scatter(p.loc[old,'last_measurement_h'],p.loc[old,col],s=15,color=BLUE,label='141 frozen-cycle reconstruction')
  ax.scatter(p.loc[~old,'last_measurement_h'],p.loc[~old,col],s=25,marker='x',color=ORANGE,label='5 additional Development cycles, audit only')
  ax.set_ylabel(label);ax.grid(alpha=.15)
 axs[0].legend(fontsize=8,ncol=2);axs[-1].set_xlabel('Actual operating time (h)');fig.suptitle('Fixed0.78923 A/cm² does not guarantee constant environment | n=146 cycle blocks')
 save(fig,'F2_voltage_and_condition_context',['outputs/FIXED_PRIMARY_CYCLE_OBSERVATIONS.csv'],'Development; cold/humid operation and voltage recovery retained')
 c=rd(P/'outputs/T0_T1_SPLIT_COUNTS.csv');w=rd(P/'outputs/T0_T1_WINDOW_QUALIFICATION.csv');fig,axs=plt.subplots(1,2,figsize=(11,4.8),layout='constrained');x=np.arange(2)
 bars=axs[0].bar(x-.17,c.T0_legal_origins,width=.34,label='T0 legal / T1 at least1label',color=BLUE);axs[0].bar_label(bars)
 bars=axs[0].bar(x+.17,c.T1_dense24_origins,width=.34,label='T1 fully observed24step target',color=ORANGE);axs[0].bar_label(bars)
 axs[0].set(xticks=x,xticklabels=c.role,ylabel='Prediction origins',ylim=(0,1600),title='Same48x14 historical input');axs[0].legend(fontsize=8)
 for role,color in [('Train',BLUE),('Validation',ORANGE)]:
  z=w.loc[w.role.eq(role),'QC_fixed_bin_labels'];axs[1].hist(z,bins=np.arange(-.5,25.5),histtype='step',lw=1.7,label=f'{role}: {len(z)} windows',color=color)
 axs[1].axvline(24,color=GRAY,ls='--',lw=1);axs[1].set(xlabel='Observed fixed-state labels in24 future bins',ylabel='Overlapping window count',title='Missing labels stay missing');axs[1].legend(fontsize=8)
 fig.suptitle('Development target audit | Train:982 labels/82cycles; Validation:151labels/14cycles | dense24targets:0')
 save(fig,'F3_native_target_coverage',['outputs/T0_T1_SPLIT_COUNTS.csv','outputs/T0_T1_WINDOW_QUALIFICATION.csv'],'Observed labels only, not independent forecast samples')
 fig,ax=plt.subplots(figsize=(12,6));ax.set(xlim=(0,12),ylim=(0,6));ax.axis('off')
 def box(x,y,width,height,text,color):
  ax.add_patch(FancyBboxPatch((x,y),width,height,boxstyle='round,pad=0.08',facecolor=color,edgecolor=GRAY,lw=.8));ax.text(x+width/2,y+height/2,text,ha='center',va='center',fontsize=10)
 def arrow(a,b,dash=False):ax.add_patch(FancyArrowPatch(a,b,arrowstyle='-|>',mutation_scale=14,lw=1.2,color=GRAY,linestyle='--' if dash else '-'))
 box(.2,4.5,2.2,.9,'Development only\nRaw preprocessing','#ECECEC');box(3.1,4.5,2.3,.9,'Spearman Top10\nEWMA voltage alpha0.3','#DDEBF5');box(6.1,4.5,2.4,.9,'Frozen dual-path TFT\n48x14 ->24 DeltaV','#DDEBF5');box(9.2,4.5,2.4,.9,'Dynamic voltage forecast\nCore paper evidence','#DDEBF5')
 arrow((2.4,4.95),(3,4.95));arrow((5.4,4.95),(6,4.95));arrow((8.5,4.95),(9.1,4.95))
 box(.2,2.5,2.2,1.,'Frozen141 cycle\nobservations\n(+5: audit only)','#FBE8D6');box(3.1,2.5,2.3,1.,'Frozen Ridge96\nH1/H2/H3 cycles\nAuxiliary reference','#FBE8D6');box(6.1,2.5,2.4,1.,'Voltage retention proxy\nReference0.713830V\nNot standard SOH','#FBE8D6');box(9.2,2.5,2.4,1.,'7 polarization events\nSparse SOH reference\nPast-only M2_NC context','#ECECEC')
 arrow((2.4,3),(3,3));arrow((5.4,3),(6,3));arrow((8.6,3),(9.1,3),True)
 box(.7,.4,10.6,1.2,'Recommendation C: retain dynamic TFT; no target replacement or new training now.\n2703 QC dwells belong to146 cycle blocks from one life trajectory.\nA future health TFT requires a new target contract, causal qualification and fresh paired ablation.','#F4F4F4')
 ax.set_title('Recommended unified route | no new training in this audit',fontsize=14)
 save(fig,'F4_recommended_route',['outputs/FINAL_DECISION.json','outputs/SAMPLE_UNIT_LEDGER.csv'],'Dashed connection is calibration context, not established TFT-to-SOH bridge',True)
 cs('data_contracts/FIGURE_SOURCE_INDEX.csv',pd.DataFrame(records));print('4 PNG/PDF figure sets plus route SVG')
if __name__=='__main__':main()
