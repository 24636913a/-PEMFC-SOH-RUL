"""Post-freeze layout-only QA repair. Frozen experiment and plotting CSV unchanged.

Run AFTER figures and reports. Fix timeline title/labels overlap and move the
adverse-case legend outside the data. Does not edit original frozen source.
"""
from common import *
from generate_full_life_figures import plt,style,voltage_panel
def generate():
    protocol();style();root=P/'figures';before={f.name:sha(f) for f in root.glob('*_plot_data.csv')}
    src=rd(root/'01_full_timeline_plot_data.csv');counts=rd(P/'tables/WINDOW_COUNTS.csv')
    fig,ax=plt.subplots(figsize=(10,4.2));palette=['#DCEAD9','#DAE8F4','#F7E6C4','#E8DBEB']
    short=['Train','Validation','Historical assessment','Late-life']
    for part,color,label in zip(PARTS,palette,short):
        z=src.loc[src.partition==part];n=counts.loc[counts.partition==part].iloc[0]
        ax.axvspan(z.bin_end_ophrs2_h.min(),z.bin_end_ophrs2_h.max(),color=color,label=f'{label}: {len(z)} records / {n.legal_origins} origins')
        ax.scatter(z.bin_end_ophrs2_h,z.cell_voltage_mean,s=2,color='black',alpha=.5)
    ax.set(xlabel='Actual operating time at bin end (h)',ylabel='Observed cell voltage (V)',title='Full chronological evaluation: N=3843; assessment and late-life are exploratory')
    ax.legend(loc='upper center',bbox_to_anchor=(.5,-.20),ncol=2,frameon=False);fig.tight_layout()
    for ext in ['png','pdf']:fig.savefig(path(f'figures/01_full_timeline.{ext}'),dpi=300,bbox_inches='tight',facecolor='white')
    plt.close(fig)
    f=rd(root/'07_adverse_windows_plot_data.csv');fig,axes=plt.subplots(1,2,figsize=(12,4.5))
    for ax,(_,z) in zip(axes,f.groupby('case_rule',sort=False)):
        label=z.case_rule.iloc[0];voltage_panel(ax,z,f'{label}\nrow {int(z.origin_row.iloc[0])}, Seed{int(z.seed.iloc[0])}; late-life exploratory')
    handles,labels=axes[0].get_legend_handles_labels();fig.legend(handles,labels,loc='lower center',bbox_to_anchor=(.5,-.05),ncol=6,frameon=False);fig.tight_layout()
    for ext in ['png','pdf']:fig.savefig(path(f'figures/07_adverse_windows.{ext}'),dpi=300,bbox_inches='tight',facecolor='white')
    plt.close(fig)
    assert before=={f.name:sha(f) for f in root.glob('*_plot_data.csv')}
    provenance=read(P/'outputs/audits/FIGURE_PROVENANCE.json');provenance['layout_only_postfreeze_repair']=dict(script='scripts/polish_figure_layout.py',sha256=sha(Path(__file__)),figures=['01_full_timeline','07_adverse_windows'],plot_CSV_unchanged=True,scientific_protocol_unchanged=True)
    js('outputs/audits/FIGURE_PROVENANCE.json',provenance)
    name='README.md';s=(P/name).read_text(encoding='utf-8');command='python -B "RENew-PEMFC-2300h-Pytorch/Phase_SCI_RealHyFC_Full_Chronological_Evaluation/scripts/polish_figure_layout.py"'
    if command not in s:s=s.replace('python -B $pipeline finalize',command+'\npython -B $pipeline finalize')
    note='\n图形目视QA发现原图01标注重叠、图07图例遮挡低电压，独立polish_figure_layout.py仅调整排版。原冻结源码及协议未改，绘图CSV SHA完全不变。重绘时按上述顺序在reports后执行排版脚本；事后极端工况诊断可单独运行diagnose_extreme_errors.py。\n'
    if note not in s:s+=note
    write(name,s)
if __name__=='__main__':generate()
