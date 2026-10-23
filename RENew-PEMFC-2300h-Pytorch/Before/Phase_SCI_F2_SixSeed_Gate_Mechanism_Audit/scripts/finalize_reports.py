"""Dependency-free rendering and explicit evidence interpretation, after numeric audit.
Original frozen report generator is preserved byte-for-byte; only its table renderer
is substituted in memory because optional pandas dependency tabulate is unavailable.
This amendment performs no model forward, fitting, metric change or seed selection.
"""
from common import *
import write_reports
def table(df):
    def fmt(x):
        if pd.isna(x):return 'NA'
        if isinstance(x,(float,np.floating)):return f'{x:.8g}'
        return str(x).replace('|','\\|').replace('\n',' ')
    return '|'+ '|'.join(df.columns)+'|\n|'+'|'.join(['---']*len(df.columns))+'|\n'+'\n'.join('|'+ '|'.join(fmt(v) for v in row)+'|' for row in df.itertuples(index=False,name=None))
def append(name,text):
    q=P/'reports'/name;txt('reports/'+name,q.read_text(encoding='utf-8')+'\n'+text+'\n')
def main():
    cfg();assert read(P/'outputs/audits/INDEPENDENT_VERIFICATION.json')['pass_']
    write_reports.table=table;write_reports.main()
    g=rd(P/'outputs/GATE_STATISTICS.csv');cor=rd(P/'outputs/GATE_CORRELATIONS.csv');m=rd(P/'outputs/METRICS.csv');p=rd(P/'outputs/PERMUTATION_SUMMARY.csv')
    jg=rd(J/'outputs/gate_values/J_ALL_F2_GATES.csv');jg=jg[jg.seed==42];now=rd(P/'outputs/validation_gates_seed_42.csv');sc=read(R2/'outputs/audits/SCALER.json');precision=[]
    for name,idx in zip(FEATURES,INDICES):
        x=now[name+'_raw_mean'].to_numpy();old=jg[name+'_raw_mean'].to_numpy()
        assert np.array_equal(x,old),('J_RAW_FEATURE_IDENTITY',name)
        recovered=now[name+'_standardized'].to_numpy()*sc['scale'][idx]+sc['mean'][idx]
        precision.append(dict(feature=name,n=205,J_raw_mean_exact=True,max_raw_difference_vs_J=float(abs(x-old).max()),max_scaled_inverse_roundoff_raw_units=float(abs(recovered-x).max())))
    q=rd(P/'outputs/audits/TRAIN_THRESHOLD_PRECISION.csv');assert (q.max_absolute_difference==0).all()
    cs('outputs/audits/CONTEXT_NUMERIC_PRECISION.csv',precision)
    append('01_六种子门控权重行为.md',f"## 实际行为解释\n\n六个模型均无g<0.1或g>0.9饱和；g整体范围{g['min'].min():.6f}～{g['max'].max():.6f}。Seed202和212变化较小（SD分别{g[g.seed==202].SD.iloc[0]:.6f}、{g[g.seed==212].SD.iloc[0]:.6f}），即使达到J的描述性dynamic定义，也不能称为强动态融合。Seed103和210的平均值接近0.5，但SD分别{g[g.seed==103].SD.iloc[0]:.6f}、{g[g.seed==210].SD.iloc[0]:.6f}，均值接近0.5不等于逐窗口恒定0.5。42/193/202平均偏CNN，212平均偏LSTM。权重大小不能直接解释为某路径对最终预测的贡献百分比，后续非线性层、归一化及两路径相关性都参与输出。")
    append('02_门控与历史工况关系.md',"## 规律不完全一致\n\nSeed42/103/193/202/212对历史电流密度波动均为负相关；Seed210为正相关（ρ="+f"{cor[(cor.seed==210)&(cor.feature=='actual_current_density_A_cm2_std')].Spearman_rho.iloc[0]:.6f}"+"）。因而不能从J三个旧Seed的同向关系推广成所有优秀实例共有的机制。不同初始化的两路径内部表示和下游层不同，g不具有跨模型可辨识的物理意义；这里仅提出这一结构性限制，不把它当作经过因果实验验证的解释。\n\nTrain阈值重新以float64聚合后与J原值全部严格相同，五项Validation原始历史均值也与J的205行严格相同。首次冻结前的失败来自float32直接求均值的精度差异；没有模型运行、没有数据身份错位。精度表给出标准化逆变换的实际差值，不改变用于前向的原张量或键。协议前错误及纠正说明均保留。")
    append('03_固定门控反事实结果.md',"## 逐Seed方向与效应规模\n\n固定0.5变差的是103、193、202、212，改善的是42、210；固定Train均值变差的是103、202、212，改善的是42、193、210。前者相对RMSE变化范围−0.1944%～+1.2143%，后者−0.1395%～+0.5865%。最接近“仅去掉逐窗口动态性”的B2并未呈现一致优势，不能把B1的平均差异全部归因于动态调整。\n\nB1替换使预测本身的RMS变化为0.2646～0.5597mV，个别点最大变化9.2861mV，尽管整体RMSE变化小，局部输出确实依赖门控。两者量级和含义不同。")
    append('04_门控置换实验结果.md',"## 主要来源与反例\n\nSeed103十次均变差，平均+0.091037mV，是平均效应的主要来源；Seed193平均反而改善0.001143mV。Seed202平均仅+0.000111mV，十次中5次变差5次改善。不能将“5/6平均变差”解释为所有实例都有稳定、实质性的精度增益，也不应把10个置换排名用于重新挑Seed。")
    tail=[]
    for mode in ['B1','B2']:
        b=m[m['mode']=='B0'].set_index('seed');z=m[m['mode']==mode].set_index('seed')
        for metric in ['endpoint_RMSE_mV','target_time_equal_RMSE_mV','P95_mV','P99_mV','max_abs_error_mV']:
            delta=z[metric]-b[metric];tail.append(dict(mode=mode,metric=metric,seeds_worse=int((delta>0).sum()),mean_delta_mV=float(delta.mean()),min_delta_mV=float(delta.min()),max_delta_mV=float(delta.max())))
    cs('outputs/TAIL_TRADEOFFS.csv',tail)
    append('05_六种子完整指标对比.md',"## 尾部与终点不能被平均掩盖\n\n"+table(pd.DataFrame(tail))+"\n\nB1虽然使4/6整体RMSE变差，但12h终点只在3/6变差；B2整体RMSE只3/6变差，最大误差却5/6变差。没有一个汇总方向能代替逐Seed的尾部结果。预定共同窗口与客观选出的不利窗口均公开，图08以mV误差尺度补充图07中不易看清的细节。")
    append('06_动态门控独立贡献证据判断.md',"## 通俗结论\n\n门控真正进入了计算，也确实会改变预测；六个模型没有统一的工况响应方向，关闭动态调整也不会一致降低精度。特别是保持各自Train平均融合比例时，六个模型正好三好三坏，平均差异仅0.00357mV。当前可以保留工况感知融合设计和机制诊断，不能把门控独立提高精度写成SCI论文已证实的核心创新。P2已完成的独立训练对照仍是负面/不一致证据，不能被筛选后的六实例诊断覆盖。\n\n当前不必为凑正面结论重复训练；若坚持独立增益主张，合格独立训练对照是必要条件，但已有P2并未满足它。最节省实验的选择是降低该项主张，保留动态电压预测的实际证据。")
    # Preserve and clarify both engineering incidents without touching frozen sources.
    js('outputs/audits/REPORTING_AMENDMENT.json',dict(scope='report rendering and figure presentation only, after inference and independent numerical audit',
        original_report_generator_sha256=sha(P/'scripts/write_reports.py'),failure='optional tabulate dependency missing; original report generation stopped before writing reports',
        remedy='new finalize_reports.py supplies a local Markdown renderer in memory; no dependency installation or frozen-source modification',
        prefreeze_precision_clarification='initial float32 mean reduction differed; float64 reduction reproduces all five J Train threshold pairs EXACTLY and all205 J raw Validation context means EXACTLY; original prefreeze note overstated serialization as cause',
        tests='9 passed; synthetic constant-truth R2 warning is preserved in test log; real validation targets have nonzero variance',
        inference_repeated_for_reporting=False,metrics_modified=False,protocol_modified=False))
    index=rd(P/'outputs/FIGURE_INDEX.csv');index.loc[index.figure.str.contains('01_gate|03_fixed'),'script']='scripts/plot_figures.py + scripts/finalize_figures.py'
    extra=pd.DataFrame([dict(figure='figures/08_unfavorable_window_errors',csv='outputs/FIGURE_WINDOW_DATA.csv',script='scripts/finalize_figures.py',scientific_scope='24 queries/window, unfavorable example for each selected instance')]);cs('outputs/FIGURE_INDEX.csv',pd.concat([index,extra],ignore_index=True))
    text=(P/'README.md').read_text(encoding='utf-8').replace('python -B scripts/write_reports.py','python -B scripts/finalize_figures.py  # presentation-only QA\npython -B scripts/finalize_reports.py # dependency-free reports + evidence interpretation')
    text+='\n初始冻结源码保持不变。后加的两个finalize脚本仅修复报告可选依赖缺失及绘图排版/缺口连线，均不运行模型或改变计算规则；身份由最终交付SHA清单锁定。完整错误说明见outputs/audits/REPORTING_AMENDMENT.json。图01不跨缺失起点连线；图片仅显示合法观测。图06/07/08每个panel为单一起点24步，标题中的205为每Seed总体起点数。\n'
    txt('README.md',text)
    print('REPORTS_FINALIZED_WITHOUT_TABULATE')
if __name__=='__main__':main()
