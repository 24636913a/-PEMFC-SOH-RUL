"""Publication exports from persistent caches; no model or data-loader calls."""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import j_common as j

COLORS=['#0F4D92','#42949E','#B64342']
plt.rcParams.update({'font.family':['Arial','Helvetica','DejaVu Sans','sans-serif'],'font.size':12,
    'axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':1.5,'legend.frameon':False,
    'svg.fonttype':'none','pdf.fonttype':42})

def export(fig,path):
    for extension in ['png','pdf','svg']:
        fig.savefig(j.write_target(path.with_suffix('.'+extension)),dpi=300,bbox_inches='tight',pad_inches=.2)
    plt.close(fig)

def architecture():
    root=j.PHASE/'figures/architecture'
    fig,ax=plt.subplots(figsize=(17,13));ax.set_xlim(0,18);ax.set_ylim(0,17);ax.axis('off')
    def box(x,y,w,h,text,color='#DDF3DE',fontsize=11):
        ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.09',fc=color,ec='#4D4D4D',lw=1.4))
        ax.text(x+w/2,y+h/2,text,ha='center',va='center',fontsize=fontsize)
    def arrow(a,b,label=None,rad=0):
        ax.add_patch(FancyArrowPatch(a,b,arrowstyle='-|>',mutation_scale=14,color='#4D4D4D',lw=1.5,connectionstyle=f'arc3,rad={rad}'))
        if label:ax.text((a[0]+b[0])/2,(a[1]+b[1])/2,label,ha='center',va='center',fontsize=10,bbox={'facecolor':'white','edgecolor':'none','pad':2})
    box(4.5,15.3,7,1,'Observed Raw F2 history: 48 bins x 14 variables\n0.5 h / bin; origin voltage known', '#CFCECE')
    box(4.5,13.5,7,1,'Frozen Train-only Spearman Top10\nCausal per-window voltage EWMA (alpha=0.3)\nSaved Train scaler -> standardized-space zero mask','#DCE8F3')
    arrow((8,15.3),(8,14.5))
    box(4.5,11.9,7,.9,'TFT continuous prescalers + encoder VSN\n14 masked slots + relative historical coordinate -> [B,48,64]','#DCE8F3')
    arrow((8,13.5),(8,12.8))
    box(.5,9.5,5.1,1.25,'Native LSTM history encoder\nLSTM -> GLU + AddNorm\nH_LSTM: [B,48,64]')
    box(6.5,9.3,5.1,1.55,'Causal convolution on encoder VSN\nDepthwise k3,d1 (RF3), k3,d4 (RF9)\nConcat -> pointwise 128->64\nDropout + residual + LayerNorm')
    box(12.8,11.9,4.3,1.1,'5 historical operating contexts\nMean of scaled/masked 48 bins\nNot VSN; not future observations','#F6CFCB')
    box(12.8,9.5,4.3,1.1,'Linear 5->8 -> ELU\nLinear 8->1 -> Sigmoid\ng: [B,1,1], 57 parameters','#F6CFCB')
    arrow((4.5,12.2),(3,10.75));arrow((8,11.9),(9,10.85));arrow((11.5,13.8),(15,13),rad=-.15)
    arrow((15,11.9),(15,10.6))
    box(5.5,7.15,6.8,1.1,'Historical representation fusion only\ng H_LSTM + (1-g) H_Conv\n[B,48,64]; one gate per origin','#AADCA9')
    arrow((3.05,9.5),(6.7,8.25));arrow((9,9.3),(9,8.25));arrow((15,9.5),(11.4,8.25))
    box(.5,5.2,4.6,1.4,'Untouched native LSTM decoder\nOnly relative steps 1..24 / 24\nNo future physical covariates\nGLU + AddNorm','#CFCECE')
    arrow((1.4,9.5),(1.4,6.6),'Original LSTM h/c',rad=.06)
    box(6.3,5.4,6.1,1,'Concat fused history + native decoder\n[B,72,64] -> static enrichment GRN','#DCE8F3')
    arrow((9,7.15),(9,6.4));arrow((5.1,5.9),(6.3,5.9))
    box(6.3,3.45,6.1,1.2,'Native interpretable attention\nQ: decoder; K/V: history + decoder\nCausal decoder attention mask','#DCE8F3')
    arrow((9.3,5.4),(9.3,4.65))
    box(6.3,1.7,6.1,.95,'Post-attention GateNorm -> position-wise GRN\nPre-output GateNorm (native decoder skip)\nLinear -> standardized 24-step DeltaV','#DCE8F3')
    arrow((9.3,3.45),(9.3,2.65));arrow((4.6,5.2),(6.3,2.3),'Native decoder skip',rad=.2)
    box(6.3,.1,6.1,.8,'Inverse target scale -> DeltaV [B,24] in V\nOptional voltage reconstruction: V_origin + DeltaV','#CFCECE')
    arrow((9.3,1.7),(9.3,.9))
    ax.text(.3,.6,'F2 frozen architecture\nNo SOH / RUL head\nNo physical equation constraint',fontsize=11,va='center')
    ax.set_title('F2: regime-aware dual-path strict-history TFT',fontsize=19,pad=16)
    export(fig,root/'J_F2_Full_Architecture')
    mermaid='''flowchart TD
 Raw["48 x 14 observed Raw F2 history"] --> Prep["Train-only frozen Top10; voltage causal EWMA alpha .3; saved scaler; mask"]
 Prep --> VSN["Native TFT prescalers + encoder VSN; relative historical coordinate"]
 VSN --> LSTM["LSTM encoder -> GLU + AddNorm: H_LSTM"]
 VSN --> Conv["Depthwise k3 d1 / d4 -> concat -> pointwise -> dropout + VSN residual -> LayerNorm: H_Conv"]
 Prep --> Context["48-step mean of five retained standardized operating variables"]
 Context --> Gate["Linear 5-8; ELU; Linear 8-1; sigmoid; one g per origin"]
 LSTM --> Fuse["g H_LSTM + (1-g) H_Conv: historical representation only"]
 Conv --> Fuse
 Gate --> Fuse
 LSTM -. "Original encoder h/c, before fusion" .-> Decoder["Native LSTM decoder; deterministic relative steps only -> GLU + AddNorm"]
 Relative["Known relative forecast coordinate; no future measured conditions"] --> Decoder
 Fuse --> Cat["Concat fused encoder and native decoder; static enrichment"]
 Decoder --> Cat
 Cat --> Attention["Native causal interpretable attention; decoder queries; history+decoder K/V"]
 Attention --> Head["GateNorm; position-wise GRN; pre-output GateNorm; scalar projection at each of 24 steps"]
 Decoder -. "native decoder residual" .-> Head
 Head --> Delta["DeltaV standardized -> frozen voltage-scale inverse transform"]
'''
    j.text_write(root/'J_F2_Full_Architecture.mmd',mermaid)
    fig,axes=plt.subplots(1,3,figsize=(17,7))
    bodies=[('F0: native TFT','Encoder VSN -> native LSTM\nGLU + AddNorm -> native TFT tail\nNo convolution branch; no regime gate',323483),
        ('F1: fixed dual path','Encoder VSN -> LSTM and causal Conv\n0.5 H_LSTM + 0.5 H_Conv\nNative decoder / attention / output tail',332379),
        ('F2: regime-aware dual path','F1 branches + historical context z\n5 -> 8 -> 1 sigmoid gate g\ng H_LSTM + (1-g) H_Conv\nNative decoder / attention / output tail',332436)]
    for ax,(title,body,n),color in zip(axes,bodies,['#CFCECE','#DDF3DE','#DCE8F3']):
        ax.axis('off');ax.add_patch(FancyBboxPatch((.06,.23),.88,.6,boxstyle='round,pad=.025',fc=color,ec='#4D4D4D',transform=ax.transAxes))
        ax.text(.5,.91,title,ha='center',fontsize=17,transform=ax.transAxes)
        ax.text(.5,.55,body,ha='center',va='center',fontsize=12,linespacing=1.9,transform=ax.transAxes)
        ax.text(.5,.14,f'{n:,} parameters',ha='center',fontsize=17,transform=ax.transAxes)
    fig.suptitle('Matched implementation contrast; fusion is not output averaging',fontsize=18)
    fig.text(.5,.035,'F1 adds 8,896 parameters; F2 adds 57. Original LSTM encoder h/c still seeds native decoder.',ha='center',fontsize=12)
    export(fig,root/'J_F0_F1_F2_Comparison')
    j.text_write(root/'J_F0_F1_F2_Comparison.mmd','''flowchart LR
 F0["F0: native encoder representation; 323483 params"] --> F1["F1: causal multiscale branch; fixed .5 encoder fusion; +8896 params"]
 F1 --> F2["F2: replace fixed fusion with historical 5-8-1 gate; +57 params; 332436 total"]
''')

def gates():
    root=j.PHASE/'figures/gate_analysis';f=j.read_csv(j.OUT/'gate_values/J_ALL_F2_GATES.csv')
    fig,axes=plt.subplots(1,3,figsize=(16,4.5))
    for ax,seed,color in zip(axes,j.SEEDS,COLORS):
        g=f.loc[f.seed.eq(seed)].g;ax.hist(g,bins=np.linspace(0,1,21),color=color,edgecolor='white')
        ax.axvline(.5,color='#767676',linestyle='--');ax.set(xlim=(0,1),xlabel='Gate g (LSTM weight)',ylabel='Validation origins',title=f'Seed{seed}, n=205')
    fig.tight_layout(pad=2);export(fig,root/'J_Gate_Distributions')
    fig,ax=plt.subplots(figsize=(13,5))
    for seed,color in zip(j.SEEDS,COLORS):
        a=f.loc[f.seed.eq(seed)];ax.plot(a.origin_age_h,a.g,label=f'Seed{seed}',color=color,lw=1.7)
    ax.axhline(.5,color='#767676',ls='--');ax.set(xlabel='Operating age at Validation origin (h)',ylabel='Gate g',ylim=(0,1),title='Learned historical fusion weights; Validation only');ax.legend(ncol=3)
    fig.tight_layout(pad=2);export(fig,root/'J_Gates_vs_Validation_Age')
    labels=['Load transitions / bin','Zero-current fraction','Current-density SD (A/cm²)','Current SD (A)','Mean current density (A/cm²)']
    fig,axes=plt.subplots(3,5,figsize=(22,11))
    corr=j.read_csv(j.OUT/'gate_diagnostics/J_GATE_CONTEXT_SPEARMAN.csv')
    for row,(seed,color) in enumerate(zip(j.SEEDS,COLORS)):
        a=f.loc[f.seed.eq(seed)]
        for col,(name,label) in enumerate(zip(j.REGIME_FEATURE_NAMES,labels)):
            ax=axes[row,col];ax.scatter(a[name+'_raw_mean'],a.g,s=13,alpha=.6,color=color)
            rho=corr.loc[(corr.seed==seed)&(corr.context==name),'Spearman_rho'].iloc[0]
            ax.set(xlabel=label,ylabel=f'Seed{seed}: g',ylim=(0,1),title=f'Spearman rho={rho:.3f}')
    fig.suptitle('Mathematical input dependence, not physical causal evidence; no p-values',fontsize=17)
    fig.tight_layout(pad=2);export(fig,root/'J_Gate_Condition_Scatter')
    cmp=j.read_csv(j.OUT/'fixed_gate_diagnostic/J_FIXED_GATE_COMPARISON.csv')
    fig,ax=plt.subplots(figsize=(10,5));x=np.arange(3)
    for offset,column,color,label,hatch in [(-.19,'learned_RMSE_mV','#0F4D92','Learned gate',''),(.19,'fixed0p5_RMSE_mV','#E9A6A1','Temporary fixed g=0.5','//')]:
        bars=ax.bar(x+offset,cmp[column],width=.36,color=color,label=label,edgecolor='#272727',hatch=hatch)
        for bar in bars:ax.text(bar.get_x()+bar.get_width()/2,bar.get_height(),f'{bar.get_height():.3f}',ha='center',va='bottom',fontsize=11)
    ax.set(xticks=x,xticklabels=[f'Seed{s}' for s in j.SEEDS],ylabel='Validation DeltaV RMSE (mV)',title='Same F2 checkpoint: inference-time gate dependence')
    ax.set_ylim(0,max(cmp.learned_RMSE_mV.max(),cmp.fixed0p5_RMSE_mV.max())*1.18);ax.legend()
    fig.tight_layout(pad=2);export(fig,root/'J_Learned_vs_Fixed_Gate')

if __name__=='__main__':
    j.load_protocol();architecture();gates();print('J scientific figures PNG/PDF/SVG + editable Mermaid complete')
