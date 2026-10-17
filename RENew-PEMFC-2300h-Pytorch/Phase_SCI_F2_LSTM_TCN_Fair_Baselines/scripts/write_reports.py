"""Evidence-driven Chinese handoff; no rewriting any existing manuscript/table."""
from baseline_common import *

def md(f,digits=6):
    def s(x):
        if isinstance(x,(float,np.floating)):return f'{x:.{digits}f}'
        return str(x).replace('|','\\|').replace('\n',' ')
    return '|'+ '|'.join(f.columns)+'|\n|'+'|'.join(['---']*len(f.columns))+'|\n'+'\n'.join('|'+ '|'.join(s(x) for x in row)+'|' for row in f.itertuples(index=False,name=None))

def main():
    c=cfg();summary=read(P/'outputs/SUMMARY.json');r=rd(P/'tables/FIVE_MODEL_PER_SEED.csv');sm=rd(P/'tables/FIVE_MODEL_SUMMARY.csv');pa=rd(P/'tables/F2_PAIRED_COMPARISONS.csv');cp=rd(P/'tables/CONDITION_PAIRED_COMPARISONS.csv');gm=rd(P/'tables/CONDITION_METRICS.csv');ext=rd(P/'tables/EXTREME_TARGET_2247_2248.csv')
    def report(name,body):write('reports/'+name,body.rstrip()+'\n')
    common='数据身份：RealHyFC Development Validation，原205预测起点×24步=4920查询/模型，251个不同目标时间；Seed42/52/62，不是三个时间折或三个独立电堆。所有结果为ΔV物理单位反缩放一次；R²在ΔV目标上计算。除授权的3007行前缀读取及筛选外，不构造或使用Test/Assessment窗口、目标或评价结果，不作显著性检验。'
    report('01_训练协议与公平性.md',f'''# 训练协议与公平性

{common}

前3007记录读取，再按`bin_end_ophrs2_h < 1819.997738`筛得3005行；没有无界读取或哈希混合Raw CSV。Train[0,2103)、Validation[2103,2404)，1380/205窗口。输入[48,14]，Top10其余四槽置零，逐窗口因果EWMA(.3)仅用于历史均值电压，公共scaler仅拟合未滤波合法Train历史；ΔV除以公共电压尺度{read(P/'outputs/audits/SCALER.json')['target_scale_V']:.17g}V。重建结果与Phase H指纹及R2完整数组逐位一致。

访问边界按科学使用定义：严格继承被授权前缀后筛选，未构造Assessment/Test目标或窗口、未用其指标训练/选模/调整。历史保护只对既有文件计算字节SHA（可能含旧Assessment产物），没有解析其数值来开展新的评价；该完整性操作不能被写成新的Assessment实验。

LSTM/TCN类从N2逐类精确复制并用AST核验；没有调用N2训练器或其per-horizon scaler。LSTM为2层64hidden、LayerNorm+Linear24，TCN为3个48通道残差模块、每模块2卷积、kernel3、dilation1/2/4、BatchNorm和全局均值池化。TCN保留对称padding，在48步历史内部不是严格causal convolution；其整个输入均截至起点，没有未来真实输入。BatchNorm统计仅训练批更新。

AdamW、lr0.0009605449687795866、weight_decay0.00000695942232671908、batch32、max450、patience40、MSE、clip1、shuffleTrue、workers0。早停严格`standardized_RMSE < best - 1e-12`。每Run从头初始化；model/DataLoader同数值seed，各自独立进程。不启用严格确定性算法，继承H普通CUDA运行；不声称逐位CUDA训练可重复。初始rng/参数SHA与采样顺序保留。

学习率来自历史TFT冻结配置，未针对两种基线优化。相同数据/预算不等于相同容量，也不等于每个模型均充分调优。这里只评价固定、合理、透明的两种标准结构；不能将其不理想结果解释为LSTM/TCN所有实现的上限。

执行次序LSTM42→TCN42→两Run全量重载/独立审计通过→LSTM52→LSTM62→TCN52→TCN62。没有重试已完成Run。冻结协议`configs/FROZEN_BASELINE_PROTOCOL.yaml`及SHA固定全部6组、来源与训练代码。只有非训练分析/绘图/报告代码随后补充；没有根据结果更改科学配置。''')
    columns=['model','seed','RMSE_mV','MAE_mV','R2','best_epoch','stop_epoch','training_time_s','parameter_count']
    report('02_六组训练结果.md',f'''# 六组正式训练结果

6/6训练完成并通过单Run审计；前两组是完整正式训练，不是缩小窗口/epoch的smoke。实际提前终止只来自冻结patience40或达到450。合成前反向检查没有optimizer更新，不计正式训练。

{md(r[r.model.isin(['LSTM','TCN'])][columns])}

每Run的最佳checkpoint、逐epoch日志、全量预测CSV、固定输入重载和独立指标审计均独立保存。checkpoint内保存完整205起点标准化预测；重新载入后逐位一致，反尺度后的CSV也逐位一致。日志按1e-12规则重建最佳epoch和stale，未使用最后一轮替代最佳模型。全部六个状态在`outputs/execution_ledger/RUN_LEDGER.json`。''')
    wins='\n'.join(f"- F2 vs {name}：RMSE胜出{count}/3个配对Seed。" for name,count in summary['paired_RMSE_wins'].items())
    report('03_五模型整体性能比较.md',f'''# 五模型整体性能比较

{common}

{md(sm[['RMSE_rank','model','RMSE_mV_mean','RMSE_mV_sample_SD','MAE_mV_mean','R2_mean','endpoint_RMSE_mV_mean','target_time_equal_RMSE_mV_mean']])}

排名按三个预定Seed等权平均query-RMSE。标准差为样本ddof1，不是置信区间。六个精选F2实例不进入本表。

{wins}

同Seed逐项差值定义F2−对照，负数表示F2较好，完整RMSE/MAE/R²/终点/尾部差值见`tables/F2_PAIRED_COMPARISONS.csv`。本任务未要求F2获胜，不通过改指标、换seed或删除异常点修正结果。三Seed及重叠目标不足以作夸大的显著性结论。

无训练Persistence已在上一阶段计算，可作背景，本阶段主表为五个模型，不能用Persistence巨大误差代替强时序基线。''')
    endpoint=sm.sort_values('endpoint_RMSE_mV_mean')
    report('04_不同预测步长误差分析.md',f'''# 不同步长与12小时终点

同一48→24模型在每个步长各有205查询。24步曲线见F03，逐模型逐Seed原始值360行见`tables/HORIZON_METRICS.csv`，没有训练24种不同跨度模型。

按三Seed平均，12h终点最小的是**{summary['endpoint_best_model']}**。

{md(endpoint[['model','endpoint_RMSE_mV_mean','endpoint_RMSE_mV_sample_SD','RMSE_mV_mean','target_time_equal_RMSE_mV_mean']])}

Query-RMSE、最后一步和target-time-equal可能产生不同排序，必须同时保留。target-time-equal先在每个实际目标时间内平均平方误差，再给251个不同目标时间等权，最后开方；不是先平均预测再算误差。图中的Seed范围是实际3个初始化观测范围，不是统计置信带。''')
    grouped=gm.groupby(['condition','group','model'],sort=False).agg(RMSE_mV_mean=('RMSE_mV','mean'),MAE_mV_mean=('MAE_mV','mean'),endpoint_RMSE_mV_mean=('endpoint_RMSE_mV','mean'),target_time_equal_RMSE_mV_mean=('target_time_equal_RMSE_mV','mean'),origins=('origins','first')).reset_index()
    cs('tables/CONDITION_EQUAL_SEED_SUMMARY.csv',grouped)
    winners=grouped.sort_values('RMSE_mV_mean').groupby(['condition','group'],sort=False).head(1).sort_values(['condition','group'])
    cs('tables/CONDITION_MEAN_WINNERS.csv',winners)
    high=cp[cp['group'].eq('HIGH')&cp.comparator.isin(['LSTM','TCN'])]
    report('05_不同历史工况模型比较.md',f'''# 不同历史工况的公平比较

完全复用上一阶段Train1/3、2/3阈值；本阶段再次由Train raw历史float64均值核对阈值和所有205组身份，无Validation误差调阈值。三个条件每次均划分同一205起点，并不是九个独立数据集。

九组RMSE均值最优模型（仅描述性）：

{md(winners[['condition','group','model','origins','RMSE_mV_mean']])}

重点HIGH电流波动/HIGH负载转换中F2对LSTM/TCN：

{md(high[['condition','comparator','origins','F2_RMSE_mean','comparator_RMSE_mean','F2_minus_comparator_mean_mV','F2_paired_wins']])}

全部135个组×模型×Seed指标和36个配对汇总见`CONDITION_METRICS.csv`与`CONDITION_PAIRED_COMPARISONS.csv`。组内target-time-equal仅对该组查询实际覆盖的目标时间等权，样本覆盖数随组不同，不能把不同组的差当作相同目标的受控实验。所有方法在同一组中具有完全相同的origin/target键。

这些是过去48步历史统计，不是未来24步真实工况、温湿压控制组或物理因果证据。哪组均值获胜并不自动意味着3/3配对Seed获胜。报告保留所有反例。''')
    extremes=ext.groupby(['model','seed'],sort=False).agg(n_queries=('absolute_error_mV','size'),MAE_mV=('absolute_error_mV','mean'),max_abs_error_mV=('absolute_error_mV','max')).reset_index();cs('tables/EXTREME_TARGET_SUMMARY.csv',extremes)
    report('06_极端误差与稳定性.md',f'''# 极端误差、后期误差与敏感性

目标行2247/2248、固定案例origin2205、困难案例origin2224均在训练前冻结，不参与最优epoch选择以外的任何模型/超参数筛选。F05/F06统一使用未经筛选的Seed62和同一个起点，直接画单次24步，未拼接窗口。完整两个目标的全部历史查询保留。

{md(r[['model','seed','RMSE_mV','endpoint_RMSE_mV','target_time_equal_RMSE_mV','P95_mV','P99_mV','max_abs_error_mV']])}

预指定两目标诊断：

{md(extremes)}

所有模型的全点最大误差都已列出，不能把平均RMSE较低解释成没有极端风险。历史F2 Seed62的终点/目标时间等权较F0/F1差，max介于二者之间的事实没有改变。新基线是否更稳，应同时查看本表和逐步曲线；不能只选整体或尾部中有利的一项。

本阶段没有平滑预测、删掉异常点或重新定位容易预测的区段。这里的高误差可指示任务/转折困难，但没有未经证据将其归因为传感器坏点或电化学损伤。''')
    cost=r[['model','seed','parameter_count','best_epoch','stop_epoch','training_time_s','peak_GPU_memory_bytes']].copy();cost['training_minutes']=cost.training_time_s/60;cost['peak_allocated_MiB']=cost.peak_GPU_memory_bytes/1024**2;cs('tables/CAPACITY_AND_COST.csv',cost)
    report('07_模型容量与计算成本.md',f'''# 模型容量与计算成本

{md(cost[['model','seed','parameter_count','best_epoch','stop_epoch','training_minutes','peak_allocated_MiB']])}

LSTM/TCN分别{c['parameter_counts']['LSTM']:,}/{c['parameter_counts']['TCN']:,}参数，F2为332,436；F0/F1参数量直接来自冻结H记录。没有为了凑容量添加无意义层，当前比较不是参数量匹配实验。

新训练全部RTX3070 Laptop8GB、PyTorch2.6.0+cu124、CPUthreads1、普通CUDA模式，具体版本和TF32/cuDNN开关保存在PREFLIGHT及每Run initial.json。H历史训练时间与当前运行不是控制其他程序/功率/温度后的标准计时试验；H逐batch故障记录更重、主机实时负载不同。GPU峰值为PyTorch allocated峰值，不等于整机显存占用或能源。因此只能报告实际观察成本，不能据此宣称严格吞吐/能效优势。新计时含训练/验证/保存最佳权重/日志，不含数据预检查和后续重载审计。

AdamW学习率从TFT冻结方案继承，未单独优化LSTM/TCN；早停轮数差异属于原规则结果，而非人为减少某模型预算。容量与配置局限必须伴随准确率比较共同披露。''')
    f2mean=float(sm.loc[sm.model.eq('F2'),'RMSE_mV_mean'].iloc[0]);relative=[]
    for baseline in ['LSTM','TCN']:
        baseline_mean=float(sm.loc[sm.model.eq(baseline),'RMSE_mV_mean'].iloc[0]);relative.append(f"F2对{baseline}平均RMSE差{f2mean-baseline_mean:+.6f}mV，按对照均值计变化{100*(f2mean/baseline_mean-1):+.4f}%，配对胜数{summary['paired_RMSE_wins'][baseline]}/3")
    verdict='本轮F2在五模型平均RMSE中居首，但优势仍限于固定配置Development评价。' if sm.iloc[0]['model']=='F2' else f"本轮五模型平均RMSE首位为{sm.iloc[0]['model']}，因此不能再用本任务证据主张F2在标准时序模型中整体最优。"
    report('08_对SCI论文的影响及下一步建议.md',f'''# 对SCI论文的影响及下一步

本轮补齐了同H数据与评价协议下的标准LSTM/TCN对照，五模型表现在可以作为**Development探索性正式模型比较**。完整排名和配对计数见报告03，不把F2必须最优作为成功条件。当前首位为{sm.iloc[0]['model']}，F2对LSTM胜出{summary['paired_RMSE_wins']['LSTM']}/3个配对Seed，对TCN胜出{summary['paired_RMSE_wins']['TCN']}/3个配对Seed。

{verdict}

{'；'.join(relative)}。正差表示F2更差，不取绝对值掩盖方向。具体工况判断见报告05逐组均值及配对胜数。论文必须依据这些结果解释适用范围；不能推广为所有LSTM/TCN实现的优劣。历史门控独立增益未证明、P2负面证据不受本次五模型排名改变。

除独立评价问题外，投稿仍应明确参数容量不同、学习率非基线优化、仅3初始化、重叠窗口与极端/终点风险，完善架构说明和模型选择披露。若要主张跨堆或任意未来工况泛化，需要新的相应证据；当前不主张即可，不需为写作扩大模型搜索。P3/P5全文和作者/基金/利益冲突等稿件资料仍待负责人补。

最有价值的下一步是负责人审核五模型及分组结果后统一改论文，再判断是否有合法未暴露的时间外评价数据。没有该条件时，不将已复用Assessment重新命名Test。本轮不构造或使用Assessment/Test目标与指标，不训练更多Seed，不调模型，不追加门控实验。

没有改写上一阶段论文正文、表格或任何历史模型。新成果先独立交付，由研究负责人决定正式稿更新。''')
    for name in ['FIVE_MODEL_PER_SEED','FIVE_MODEL_SUMMARY','F2_PAIRED_COMPARISONS','CAPACITY_AND_COST']:write('tables/'+name+'.md','# '+name+'\n\n'+md(rd(P/('tables/'+name+'.csv')))+'\n')
    write('README.md',f'''# SCI：LSTM与TCN最小公平基线

状态：`SCI_BASELINE_6_RUNS_COMPLETED`。本阶段6/6正式GPU训练和完整重载审计完成，无重跑、无额外Seed。除授权3007行前缀读取后筛选外，不构造/使用Test或Assessment窗口、目标或评价。历史论文正文/权重只读。

入口：[五模型比较](reports/03_五模型整体性能比较.md)、[工况分析](reports/05_不同历史工况模型比较.md)、[论文影响](reports/08_对SCI论文的影响及下一步建议.md)。表格与真实预测在`tables/`和`outputs/predictions/`；6组英文300dpi PNG/矢量PDF在`figures/`。

RealHyFC数据归属Fensterle等，数据记录[10.5281/zenodo.20715542](https://doi.org/10.5281/zenodo.20715542)，官方已保存元数据为CC BY4.0；原数据论文[10.1016/j.ijhydene.2026.156060](https://doi.org/10.1016/j.ijhydene.2026.156060)。本目录的数据仅为已授权Development缓存/派生预测，不重新发布第三方全文或新的晚期目标。

冻结源：`configs/FROZEN_BASELINE_PROTOCOL.yaml`及.sha256，`outputs/audits/PREFLIGHT_REPORT.json`；模型类逐字继承N2前三个相关class，AST一致。TCN对称padding，不冒称历史内严格因果。两基线参数较少，学习率来自TFT历史配置，没有为其调优。

## 运行与复算

Python3.11.9、torch2.6.0+cu124、pytorch_forecasting1.8.0，RTX3070 Laptop。设置PYTHONUTF8=1/PYTHONDONTWRITEBYTECODE=1并用`python -B`。

- `scripts/preflight.py`：已执行一次性严格前缀读取/身份核验/协议冻结，禁止重新冻结。
- `scripts/run_baselines.py`：已完成6次；只允许跳过已完成Run，失败/中断不自动重训；不得重新运行训练来取得不同结果。
- `scripts/audit_run.py --model LSTM --seed 42`：仅重载最佳checkpoint和现有合法Validation输入、独立复算，不训练。其余5组同理。
- `scripts/compare_models.py`：15设置/135工况设置的公平表、1200标量独立指标核对。
- `scripts/audit_execution.py`：正式6次预算、每轮44个batch和同Seed采样顺序与H的一致性核验。
- `scripts/plot_baselines.py`、`scripts/write_reports.py`：从真实CSV生成图与8份报告。
- `python -B -m unittest discover -s tests -v`：工程和交付检查。
- `scripts/finalize_baselines.py --verify-seal`：最终文件SHA核验；无参数终检仅用于本阶段提交前，勿覆盖原审计。

早停使用Validation属于原Development规则，并未获得独立Test资格。门控/优秀Seed历史结论不因本轮结果自动修改。Git发布身份以提交和被忽略的`outputs/GIT_RECEIPT.local.json`回执为准。没有删除历史数据或上传第三方文献PDF。
''')
    print('EIGHT_REPORTS_AND_README_CREATED',flush=True)

if __name__=='__main__':main()
