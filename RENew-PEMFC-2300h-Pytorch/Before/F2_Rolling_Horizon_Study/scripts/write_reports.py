"""Six requested Chinese reports, generated only from verified real outputs."""
from pathlib import Path
import numpy as np
import pandas as pd
from common import HERE,CONFIG,read,read_csv,csv,dump,sha
from historical_audit import mdtable

LABELS={'F2':'原生F2','LSTM':'普通LSTM','C0':'新训练原生F2 C0','C1':'输出改进F2 C1','Historical_F2_202_selected':'历史F2 Seed202（已筛选）'}

def run(out):
    assert read(out/'INDEPENDENT_VERIFICATION.json')['PASS']
    assert read(out/'HOLDOUT_EVALUATION_COMPLETED.json')['completed_models']==6
    reports=HERE/'reports';reports.mkdir(exist_ok=True)
    final=read_csv(out/'FINAL_RESULT_TABLE.csv');allm=read_csv(out/'ALL_PARTITION_METRICS.csv');step=read_csv(out/'ALL_STEP_METRICS.csv')
    counts=read_csv(out/'WINDOW_COUNTS.csv');decision=read(out/'C1_VALIDATION_DECISION.json')
    records=[read(out/f'models/{kind}_H{h}/COMPLETED.json') for kind,h in read(CONFIG)['training_jobs']]
    table=final[['model','source','history_steps','horizon_steps','RMSE_mV','MAE_mV','R2','origins','queries']].copy();table['model']=table.model.map(LABELS)
    csv(HERE/'tables/论文主结果表.csv',table)
    jobtable=pd.DataFrame([dict(model=r['model'],H=r['horizon_steps'],parameters=r['parameters'],best_epoch=r['best_epoch'],stop_epoch=r['stop_epoch'],optimizer_updates=r['optimizer_steps'],training_seconds=r['training_seconds']) for r in records])
    native=final[(final.model=='F2')|((final.model=='C0')&(final.horizon_steps==24))]
    best_native=native.loc[native.RMSE_mV.idxmin()]
    compare=[]
    for h in [6,12]:
        f=final[(final.model=='F2')&(final.horizon_steps==h)].iloc[0];l=final[(final.model=='LSTM')&(final.horizon_steps==h)].iloc[0]
        compare.append(dict(H=h,hours=h*.5,F2_RMSE_mV=f.RMSE_mV,LSTM_RMSE_mV=l.RMSE_mV,
            F2_RMSE_change_vs_LSTM_pct=(f.RMSE_mV/l.RMSE_mV-1)*100,F2_better=bool(f.RMSE_mV<l.RMSE_mV)))
    cmp=pd.DataFrame(compare);csv(out/'F2_LSTM_PAIRED_HORIZON_COMPARISON.csv',cmp)
    common=allm[(allm.partition=='HOLDOUT')&(allm.scope=='COMMON_H24_ORIGINS')]
    oldprefix=read_csv(out/'P1_HISTORICAL_PREFIX_METRICS.csv').set_index('prefix_steps')
    oldpoints=read_csv(out/'P1_HISTORICAL_POINTS.csv')
    matched=[]
    for r in common[common.horizon_steps.isin([6,12])].itertuples():
        h=int(r.horizon_steps)
        newpoints=read_csv(out/f'models/{r.model}_H{h}/predictions_HOLDOUT.csv')
        newpoints=newpoints[newpoints.origin_row.isin(oldpoints.origin_row.unique())]
        merged=newpoints.merge(oldpoints[oldpoints.horizon_step<=h],on=['origin_row','target_row','horizon_step'],suffixes=('_new','_old'),validate='one_to_one')
        assert len(merged)==527*h==len(newpoints)
        np.testing.assert_array_equal(merged.y_true_delta_V_new,merged.y_true_delta_V_old)
        np.testing.assert_array_equal(merged.target_age_h_new,merged.target_age_h_old)
        base=float(oldprefix.loc[h,'RMSE_mV'])
        matched.append(dict(model=r.model,H=h,matched_queries=len(merged),historical_H24_prefix_RMSE_mV=base,
            new_independent_H_RMSE_mV=r.RMSE_mV,descriptive_RMSE_reduction_pct=(1-r.RMSE_mV/base)*100))
    matched=pd.DataFrame(matched);csv(out/'HISTORICAL_PREFIX_VS_NEW_EXACT_COMMON_QUERIES.csv',matched)
    gate=read_csv(out/'ALL_HOLDOUT_PATH_DIAGNOSTICS.csv').groupby(['model','horizon_steps']).g.agg(['mean','std','min','max']).reset_index()
    tail=[]
    for kind,h in read(CONFIG)['training_jobs']:
        p=read_csv(out/f'models/{kind}_H{h}/predictions_HOLDOUT.csv');error=(p.y_pred_delta_V-p.y_true_delta_V).to_numpy(float)
        event=p.event_any.to_numpy(bool)
        tail.append(dict(model=kind,H=h,event_queries=int(event.sum()),event_unique_targets=int(p.loc[event,'target_row'].nunique()),
            event_SSE_fraction=float(np.sum(error[event]**2)/np.sum(error**2)),all_unique_targets=int(p.target_row.nunique())))
    tail=pd.DataFrame(tail);csv(out/'EVENT_UNION_SSE_CONTRIBUTION.csv',tail)
    threshold=final[['model','horizon_steps','RMSE_mV','R2']].copy();threshold['RMSE_below10mV']=threshold.RMSE_mV<10;threshold['R2_above098']=threshold.R2>.98
    threshold['both_pass']=threshold.RMSE_below10mV&threshold.R2_above098;csv(out/'SUCCESS_THRESHOLD_TABLE.csv',threshold)
    hist=read(out/'P1_HISTORICAL_AUDIT.json');core=read(out/'REGISTRATION.json')
    text=f'''# 00 协议与Seed202确认

本轮回到原始动态电压多步预测，不研究长期退化、SOH/RUL。仅使用Seed202，选择依据严格限定为20261019T055734_218949Z_seedall_compare批次Top10的历史后留RMSE33.837465mV，不能混入另一批对照的Seed212或挑选新种子。历史主结果已逐点核验并复算为{hist['full24']['RMSE_mV']:.9f}mV。该种子既然按后留成绩被选中，本轮所有结果属于探索性单种子研究，非独立盲测、非六种子总体均值、非跨电池泛化。

固定3843行、Train[0,2690)、Validation[2690,3074)、后留[3074,3843)；48步历史，H6/12/24分别3/6/12h。原14槽位、固定Top10索引[0,3,13,11,4,9,6,7,10,8]，仅历史电压因果EWMA(.3)。未平滑目标，ΔV=V(t+h)−V(t)。每个跨度以其合法Train历史拟合scaler，原始电压Train标准差用于目标尺度。两模型在相同H下共享输入、标签、Scaler和窗口身份；不同H的Train窗口和Scaler允许按原规则变化，因此跨H差异不能全部归因于输出长度。

{mdtable(counts[['horizon_steps','partition','legal_origins','queries','unique_targets']])}
三个H分别重新检查有效标志、原始分段、真实运行钟0.5h、日历连续性和有限值，标签不跨分区。Train/Val历史可使用起点之前已发生的前一区间观测，不借用未来标签。所有H的全部合法起点以及与原H24共同起点的结果均保留。

滚动方式为每个起点取真实最近48步历史，直接输出H步，下个起点更新已经观察到的历史，不将预测递归反馈。未来decoder只有确定的相对时间坐标，14个原始未来通道为零占位且不进入decoder VSN。绝对电压由原始、未经EWMA的起点电压加预测ΔV恢复。

AdamW lr=.0009605449687795866、weight_decay=6.95942232671908e−6、batch32、MAX450、PATIENCE40、clip1、MSE；F2 dropout=.1。Validation标准化RMSE最低保存best（改进阈值1e−12）。普通单层LSTM hidden64无内部Dropout，与F2正则化/容量不等，不宣称控制了全部结构差异。

完整配置见configs/FROZEN_PROTOCOL.json，其SHA为`{sha(CONFIG)}`；数据SHA为`{core['data_sha256']}`。训练前代码版本及输入/标签/窗口SHA见COHORT_BEFORE_TRAINING.json及各STARTED.json。6个模型均从头初始化且optimizer初始为空，实际训练共{sum(r['stop_epoch'] for r in records)}epoch、{sum(r['optimizer_steps'] for r in records)}次反向/更新。P2四个模型及P3两个模型全部best冻结后，才做一次统一新后留评价；后续重加载回放仅用于验收，不选模或重训。

普通CUDA存在已知非逐位训练重复性；本轮保留原计算设置，未替换算子强行确定性。C0/C1初始主干、CPU/CUDA RNG状态相同，随机批次顺序核验一致，但这不保证两条非线性训练轨迹只差新头这一项数值效应。所有历史源码、模型和结果保留。
'''
    (reports/'00_协议与Seed202确认.md').write_text(text,encoding='utf-8')
    subset=final[final.horizon_steps.isin([6,12])]
    trainval=allm[(allm.horizon_steps.isin([6,12]))&(allm.scope=='ALL_H_LEGAL')]
    text=f'''# 02 48到6与48到12训练结果

已按原生F2/普通LSTM分别真实训练H6和H12，共4个新实例。不是截断历史H24缓存，不加载旧权重热启动。模型容量分别为F2 332436、LSTM H6 20870/H12 21260；不同容量是结论限制。

## 全部H合法后留起点

{mdtable(subset[['model','horizon_steps','RMSE_mV','MAE_mV','R2','P99_mV','max_abs_mV','origins','queries']])}

## 相同原H24起点的辅助比较

{mdtable(common[common.horizon_steps.isin([6,12])][['model','horizon_steps','RMSE_mV','MAE_mV','R2','origins','queries']])}

每个模型该H全部合法起点是正式主口径；共同起点是辅助口径。不能混用两套数量，或将同一H24模型截断成绩冒充独立训练。相同H的F2/LSTM严格使用相同目标和身份。跨H训练支持及Scaler不同，共同起点比较也不是纯输出长度的因果消融。

## 与指定历史H24模型前缀的精确同查询比较

{mdtable(matched)}
按origin_row、target_row、horizon_step一对一匹配，真ΔV及实际目标小时严格相同。这说明独立训练短跨度在这批共同查询上确有描述性改善，但不是12h原任务成绩，也不将收益全部归因于预测头长度。两套训练支持、Scaler及随机数值轨迹不同，仍只有一个已筛选种子。

## Train/Validation与真实训练记录

{mdtable(trainval[['model','horizon_steps','partition','RMSE_mV','MAE_mV','R2']])}

{mdtable(jobtable[jobtable.H.isin([6,12])])}

{mdtable(cmp)}
最后20%已被历史研究查看，Seed202也来自历史后留选择。本轮只报告该固定实例的结果，不切换种子，不给出不存在的均值/标准差或显著性检验。H24没有本轮新训练LSTM，故不能填写该跨度的同轮LSTM胜负。
'''
    (reports/'02_48到6与48到12训练结果.md').write_text(text,encoding='utf-8')
    pair=allm[(allm.model.isin(['C0','C1']))&(allm.scope=='ALL_H_LEGAL')]
    cp=read_csv(out/'models/C1_H24/HOLDOUT_OUTPUT_COMPONENTS.csv');alpha=float(cp.alpha.iloc[0])
    ab=abs(cp.residual_component_V.to_numpy(float))*1000
    off=read_csv(out/'C1_HEAD_OFF_METRICS.csv')
    c0hold=pair[(pair.model=='C0')&(pair.partition=='HOLDOUT')].iloc[0]
    native_repeat_gain=(1-c0hold.RMSE_mV/hist['full24']['RMSE_mV'])*100
    head_off_rmse=float(off.loc[off.model=='C1_head_off_inference_only','RMSE_mV'].iloc[0])
    full_c1_rmse=float(off.loc[off.model=='C1_full_same_weights','RMSE_mV'].iloc[0])
    claim='达到预先规定的Validation初步改善条件，但未证明稳定提升' if decision['preliminary_validation_improvement'] else '没有达到预先规定的Validation升级条件'
    text=f'''# 03 F2输出结构改进结果

C0与C1均为本轮Seed202从头训练的48→24实例，历史较好权重没有替代C0。C1 {claim}。

## 训练前冻结的结构

原生TFT、LSTM历史路径、因果多尺度CNN与工况凸融合全部保留。令z=mean_time(H_fused_history)∈R^64，仅取48个编码历史状态；s=h/24。

r(z,s)=tanh(W₂ ELU(W₁[z,s]+b₁)+b₂)，共享65→16→1；α=.25 sigmoid(a)，a初始logit(.1)，α初始.025。

ΔV_new,std(h)=ΔV_native,std(h)+α r(z,s)。新增1074参数，总333510；原生C0为332436。最终线性层零初始化，初始输出与C0逐点相等；head在fork_rng内初始化，保存了相同主干初始化及训练前CPU/CUDA RNG。不是过去隐藏状态固定.25残差融合公式，没有修改原生融合，也没有额外未来工况输入。

## 真实Train/Validation/后留结果

{mdtable(pair[['model','partition','RMSE_mV','MAE_mV','R2','P99_mV','max_abs_mV']])}

Validation RMSE改善比例为{decision['validation_RMSE_gain_fraction']*100:.3f}%，MAE变化{decision['validation_MAE_change_fraction']*100:.3f}%，P99变化{decision['validation_P99_change_fraction']*100:.3f}%（变化正数为恶化）。预先条件为RMSE至少降低5%，MAE/P99均不恶化，未事后改门槛。即使后留更好，也不能替代Validation选择依据，更不能凭单种子宣称稳定创新收益。

{mdtable(jobtable[jobtable.H==24])}

## 真正参与与只读贡献诊断

best checkpoint的α={alpha:.8f}，实际输出补充分量绝对值RMS={np.sqrt(np.mean(ab**2)):.6f}mV、P99={np.quantile(ab,.99):.6f}mV、最大={ab.max():.6f}mV。物理尺度的补充上界是α×Train电压标准差，而不是任意大小的电压修正。没有为了消除极端点放大α上界。

{mdtable(off[['model','partition','RMSE_mV','MAE_mV','R2','P99_mV','max_abs_mV']])}
head-off使用C1同一套已训练权重的原生输出分量，是推理反事实，不是重新训练C0、F1或LSTM；不能把它当独立训练消融。CNN、LSTM编码/解码、门控和新头的梯度及参数更新已经逐项核验，初始零头导致α首批梯度为零是预期行为，后续训练有真实非零更新。普通CUDA数值波动与仅一次配对仍限制归因。

在C1同一权重内，打开输出头仅让后留RMSE从{head_off_rmse:.6f}降至{full_c1_rmse:.6f}mV，差值{head_off_rmse-full_c1_rmse:.6f}mV；它没有修复严重状态错误。C1自身主干在联合训练后的误差已经明显大于C0，结合Train/Validation差距与Loss波动，说明本次候选没有形成有效学习收益，不能只把失败归因于输出端某个固定偏置或门控。新C0比指定历史H24的RMSE低{native_repeat_gain:.3f}%，但其结构和超参数没有改变，这个重训差异不能当作输出头创新收益。

逐提前量误差见ALL_STEP_METRICS.csv，训练曲线见F8，配对指标见F9，固定共同起点轨迹见F4，真实门控和路径幅值见F10/F11。没有增加另一轮搜索、追加种子或重跑较好结果。
'''
    (reports/'03_F2输出结构改进结果.md').write_text(text,encoding='utf-8')
    mhold=allm[(allm.partition=='HOLDOUT')&(allm.scope=='ALL_H_LEGAL')]
    leadrows=[]
    for kind,h in read(CONFIG)['training_jobs']:
        s=step[(step.model==kind)&(step.horizon_steps==h)&(step.partition=='HOLDOUT')]
        best=s.loc[s.RMSE_mV.idxmin()];last=s.loc[s.horizon_step==h].iloc[0]
        leadrows.append(dict(model=kind,H=h,lowest_error_step=int(best.horizon_step),lowest_step_RMSE_mV=best.RMSE_mV,
            step1_RMSE_mV=s.loc[s.horizon_step==1,'RMSE_mV'].iloc[0],terminal_RMSE_mV=last.RMSE_mV))
    csv(out/'FIXED_LEAD_SUMMARY.csv',leadrows)
    text=f'''# 04 误差与图表分析

所有指标均来自原始动态ΔV；绝对电压图使用原始起点电压恢复，图的电压散点R²没有替换正式ΔV R²。未删除异常点、平滑预测或移动曲线。

## 查询加权、唯一时间与尾部

{mdtable(mhold[['model','horizon_steps','RMSE_mV','target_time_equal_RMSE_mV','P95_mV','P99_mV','max_abs_mV','top1pct_SSE_fraction','queries','unique_targets']])}
H6/12/24全部合法后留集合各覆盖619个唯一目标时间，但每个目标对应起点/提前量数不同；查询分别3594/6900/12648。按唯一目标等权时，先平均该目标所有查询的平方误差，再对目标平均开根号，未对预测值做集成。重叠查询不视为独立重复实验，不能据此计算虚假的统计显著性。

{mdtable(tail)}
事件阈值只由Train决定。事件联合组包含高零电流比例、高负载转换、真实目标电压跳变和启停标志；未来实际状态仅用于事后诊断。分组可相互重叠，逐组SSE不能简单相加。联合组贡献为一次计数，不把高误差相关性称为因果归因。

{mdtable(pd.DataFrame(leadrows))}
上述最好提前量只是描述，不作为更换任务/选取新H的依据。误差不必随步长单调增加，因为每个提前量对应的实际目标状态与少数极端事件不同。历史H24前6步RMSE38.296011mV大于其全24步33.837465mV，就是不能只根据“短预测一定更准”推断结果的例子。

## 门控与双路径

{mdtable(gate)}
F10显示实际g，F11显示g×LSTM和(1−g)×CNN的潜变量RMS。它们是编码空间的幅值，不能称作对电压精度的百分比贡献。TFT还有LSTM解码、Attention和残差连接，g也不等于整个网络中的LSTM占比。真实梯度和参数更新证明参与训练，不能单凭此证明独立增益。

## 图像索引与固定规则

|图|实际用途|
|---|---|
|F1|历史H24前缀截断与逐提前量误差，明确未重新训练|
|F2|H特定全部起点与共同H24起点两套跨度比较|
|F3|同527共同起点的逐提前量RMSE|
|F4|首个共同起点，以及以历史主导事件目标为终点的固定规则完整轨迹|
|F5|每个模型固定末步的滚动真实/预测电压曲线，断点插NA，不混拼提前量|
|F6|全部真实电压/预测电压散点，同坐标，不剔除异常|
|F7|事后真实工况的误差分布，symlog包含零和全部极端点|
|F8|本轮6个真实训练Loss曲线|
|F9|本轮C0/C1 Train、Validation和后留配对指标|
|F10|历史门控真实权重|
|F11|潜变量路径幅值和输出残差头实际幅度，不做因果宣传|
|F12|最大1%查询对总SSE的贡献，以及唯一时间等权差异|

所有图为300dpi PNG和PDF，同一绘图对象导出；真实绘图CSV位于tables，代码scripts/render_figures.py。典型窗口规则在新后留推理前保存为FIGURE_CASE_RULES_BEFORE_NEW_HOLDOUT.json，不根据新模型哪一段最好来挑选。
'''
    (reports/'04_误差与图表分析.md').write_text(text,encoding='utf-8')
    passes=threshold[(threshold.RMSE_below10mV)&(threshold.model!='Historical_F2_202_selected')]
    passing='；'.join(f'{LABELS.get(r.model,r.model)} {r.horizon_steps*.5:g}小时' for r in passes.itertuples()) or '没有新训练实例'
    comparison='；'.join(f'{r.hours:g}h：'+('F2更低' if r.F2_better else 'LSTM更低')+f'（F2相对LSTM RMSE变化{r.F2_RMSE_change_vs_LSTM_pct:.2f}%）' for r in cmp.itertuples())
    text=f'''# 05 最终通俗总结

本轮已完成全部6次新GPU训练：F2/LSTM各自48→6、48→12，以及原生F2 C0与输出改进F2 C1各一次48→24。全部Seed202，从头初始化，没有用旧权重或缓存冒充。历史H24只读复算，不计入新训练。

## 真实后留结果

单位mV，R²为原始ΔV。这里后留已被历史研究使用，Seed202也曾据此选出，因此只能作为探索性单种子结果。

{mdtable(table)}

1. **未来3/6/12小时怎样？** 上表分别对应H6/H12/H24，不能把3h成绩称为12h成绩；历史33.837465mV单独标记，不替代本轮C0。共同原H24起点的另一套辅助指标也完整保存。
2. **原生F2哪个跨度最准确？** 按本轮全部合法起点，该固定实例最低RMSE出现在{best_native.horizon_steps*.5:g}h，为{best_native.RMSE_mV:.6f}mV。不同H训练样本和Scaler不同，不能把这一差异全部解释为跨度本身的因果作用。
   在原527起点的精确同查询比较中，新F2的3h为{float(common[(common.model=='F2')&(common.horizon_steps==6)].RMSE_mV.iloc[0]):.6f}mV，旧H24模型截取3h为{float(oldprefix.loc[6,'RMSE_mV']):.6f}mV；新F2的6h为{float(common[(common.model=='F2')&(common.horizon_steps==12)].RMSE_mV.iloc[0]):.6f}mV，旧截取6h为{float(oldprefix.loc[12,'RMSE_mV']):.6f}mV。这是短跨度独立训练的描述性改善，仍未达到10mV，不冒称12h性能升级。
3. **F2与LSTM谁更好？** {comparison}。12h本轮没有新训练LSTM对照，不能借别批结果补齐，也没有把LSTM成绩改名为F2。
4. **是否低于10mV？** {passing}达到严格RMSE<10mV。R²>.98是另一项条件，完整判定如下；短跨度通过不替长跨度通过。

{mdtable(threshold)}

5. **新输出结构有实质提升吗？** C1{claim}；Validation RMSE相对新C0变化为{-decision['validation_RMSE_gain_fraction']*100:.3f}%。它增加1074参数并保留整个F2。无论单次后留是否更好，都不能称为稳定改进，普通CUDA训练不完全逐位重复，没有追加重跑寻找正面结果。
6. **主要误差在哪里？** 完整尾部和事件贡献见04报告、HOLDOUT_TOP_TARGETS.csv及EVENT_UNION_SSE_CONTRIBUTION.csv。历史最大1%误差贡献94.69%总SSE，少数启停/负载混合/电压状态切换是重要诊断对象。新模型也保留全部大误差及其真实目标时间，不能仅凭低平均MAE忽略极端点。未来工况未知、窗口重叠及状态切换可能共同影响预测，当前统计不能单独证明某条网络路径是原因。
7. **下一步最值得做什么？** 先在Train/Validation内，对固定起点的第1–6步电压状态切换方向和时间误判做最小可预测性核验，确定现有历史信息能否提前识别这些转折；再决定是否值得设计一次事件监督改进。不得借后留反复调整事件阈值或继续筛种子，本轮不自动开展该实验。

## 验收及保存

6个best checkpoint独立重加载、18份Train/Validation/后留逐点预测严格一致；9组窗口独立复算，Train-only scaler、全部随机批次顺序、Early Stopping、梯度和路径参数更新均核对。测试、失败记录、运行身份和SHA保留。训练合计{sum(r['stop_epoch'] for r in records)}epoch、{sum(r['optimizer_steps'] for r in records)}次真实反向传播与优化器更新。

VS Code入口F2_Rolling_Horizon_Interactive.py具有show/analyze/train三种模式；默认只展示，analyze只读重算，train才在新目录执行6次从头训练及真实Epoch/图像显示。交互图像显示通道单独测试，不将IPython自动化验收冒称为人工VS Code GUI验收。所有旧代码、权重和结果保留。

没有执行SOH/RUL、长期趋势任务，也没有超过6次新训练或开启下一轮搜索。任务完成状态不以必须低于10mV为条件。
'''
    (reports/'05_最终通俗总结.md').write_text(text,encoding='utf-8')
    dump(HERE/'outputs/DELIVERED_RUN.json',dict(path=str(out.resolve())))
    dump(out/'FINAL_RESEARCH_STATUS.json',dict(status='F2_ROLLING_HORIZON_STUDY_COMPLETED',new_training_runs=6,historical_model_training_runs=0,
        total_epochs=sum(r['stop_epoch'] for r in records),optimizer_updates=sum(r['optimizer_steps'] for r in records),
        seed=202,new_holdout_evaluation_rounds=1,independent_replay_passed=True,source='exploratory selected single seed',
        C1_preliminary_validation_improvement=decision['preliminary_validation_improvement'],stable_gain_proven=False,
        new_models_below10mV=[dict(model=r.model,H=int(r.horizon_steps)) for r in passes.itertuples()],next_phase_started=False))
    print('Generated six reports (historical P1 retained) and final actual-result tables',flush=True)

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path);a=p.parse_args();run(a.run or Path(read(HERE/'outputs/ACTIVE_RUN.json')['path']))
