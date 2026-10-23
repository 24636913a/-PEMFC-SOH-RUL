"""Close a negative/mixed finite study truthfully. No model training here."""
from pathlib import Path
import math
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import frozen_f2 as f
from io_utils import csv,js,text
from report_a import save


def write_stopped(run):
    decision=f.read_json(run/'FINAL_DECISION.json')
    f.require(decision['status']=='NO_QUALIFIED_UPGRADE','NEGATIVE_CLOSURE_ONLY')
    a=f.read_csv(run/'tables/A_VALIDATION_MEAN_SD.csv');b=f.read_csv(run/'tables/B_VALIDATION_MEAN_SD.csv')
    summary=pd.concat([a.assign(stage='A'),b.assign(stage='B')],ignore_index=True)
    summary['partition']='VALIDATION';summary['queries_per_model']=5232
    summary['new_training']=True
    csv(run/'tables/FINAL_VALIDATION_STUDY_SUMMARY.csv',summary)
    # Historical holdout background is read only AFTER the immutable decision
    # to stop. It cannot select or revise any A/B/C/D model configuration.
    prior=f.REFERENCE_RUN/'tables/MODEL_MEAN_SD.csv'
    source=f.read_csv(prior)
    f.require(set(source.model)=={'F2','LSTM'} and source.n.eq(6).all(),'HISTORICAL_SIX_SEED_TABLE_IDENTITY')
    expected={'F2':44.220508,'LSTM':24.885444}
    for model,value in expected.items():
        observed=float(source.loc[source.model==model,'RMSE_mV_mean'].iloc[0])
        f.require(round(observed,6)==value,'REQUEST_BASELINE_DISAGREEMENT:'+model)
    source['evidence']='Historical completed six-seed exploratory holdout; not new training'
    source['source']=prior.relative_to(f.ROOT).as_posix();source['source_SHA256']=f.sha(prior)
    csv(run/'tables/HISTORICAL_HOLDOUT_BACKGROUND.csv',source)
    js(run/'audits/HISTORICAL_BACKGROUND_READ_SCOPE.json',dict(status='PASS',
        final_decision_already_frozen_SHA256=f.sha(run/'FINAL_DECISION.json'),
        source_SHA256=f.sha(prior),new_holdout_inference=False,
        use='Final background comparison only; no configuration choice remains'))
    unavailable=[dict(model='New qualified F2',seeds_trained=0,RMSE_mV_mean=None,MAE_mV_mean=None,R2_mean=None,
        status='NA_NO_QUALIFIED_UPGRADE_NO_NEW_HOLDOUT_EVALUATION')]
    csv(run/'tables/NEW_SIX_SEED_HOLDOUT_NOT_EXECUTED.csv',unavailable)
    repeats=f.read_csv(run/'tables/B_PLANNED_NATIVE_REPEATABILITY.csv')
    old=f.read_csv(run/'historical/TRAIN_VALIDATION_METRICS.csv').groupby(['model','partition'])[['RMSE_mV','MAE_mV','R2']].mean()
    all_ledgers=[f.read_json(path) for path in sorted((run/'models').glob('*/*/TRAINING_LEDGER.json'))]
    f.require(len(all_ledgers)==15 and all(value['status']=='TRAINED' for value in all_ledgers),'FIFTEEN_REAL_TRAININGS_REQUIRED')
    failures=list((run/'models').glob('*/*/FAILURE.txt'))
    totals=dict(fresh_models=len(all_ledgers),epochs=sum(v['stop_epoch'] for v in all_ledgers),
        optimizer_updates=sum(v['optimizer_steps'] for v in all_ledgers),
        real_backward_calls=sum(v['backward_calls'] for v in all_ledgers),
        training_wall_seconds=math.fsum(v['training_seconds'] for v in all_ledgers),
        formal_training_failures=len(failures),source_base_git_sha=f.read_json(run/'PROTOCOL.json')['base_git_sha'])
    js(run/'TRAINING_TOTALS.json',totals)
    fig,axes=plt.subplots(1,2,figsize=(12,4.5));x=np.arange(len(summary))
    colors=['#0F4D92','#42949E','#B56C36','#547899','#6BAF9D']
    for ax,key in zip(axes,['RMSE_mV','MAE_mV']):
        ax.bar(x,summary[key+'_mean'],yerr=summary[key+'_SD'],capsize=3,color=colors,edgecolor='black',lw=.6)
        ax.set_xticks(x,summary.candidate);ax.set(ylabel=key.replace('_mV',' (mV)'),
            title='Validation: 3 fresh seeds per method, mean ± sample SD')
        ax.set_xlabel('Predeclared methods; B_control is a planned repeat, not an upgrade')
    save(run,'FINAL/01_Validation_study_summary',fig,summary)
    rows=''.join(f'| {r.candidate} | {int(r.n)} | {r.RMSE_mV_mean:.6f} ± {r.RMSE_mV_SD:.6f} | {r.MAE_mV_mean:.6f} ± {r.MAE_mV_SD:.6f} | {r.R2_mean:.9f} |\n' for r in summary.itertuples())
    history=''.join(f'| {r.model}（旧结果） | {int(r.n)} | {r.RMSE_mV_mean:.6f} | {r.MAE_mV_mean:.6f} | {r.R2_mean:.9f} |\n' for r in source.itertuples())
    rA=f.read_json(run/'A_RESULT.json');rB=f.read_json(run/'B_RESULT.json')
    delta=100*(1-rA['decisions']['A1']['mean_RMSE_ratio'])
    paired_A1=rA['decisions']['A1']['paired_wins']
    a_points=f.read_csv(run/'tables/A_VALIDATION_PER_SEED.csv')
    base=a_points.loc[a_points.candidate=='A0'].set_index('seed')
    lower=a_points.loc[a_points.candidate=='A1'].set_index('seed').loc[base.index]
    unsafe=[int(seed) for seed in base.index if lower.loc[seed,'MAE_mV']>1.1*base.loc[seed,'MAE_mV']
        or lower.loc[seed,'P99_mV']>1.1*base.loc[seed,'P99_mV']]
    spread='较低' if float(a.loc[a.candidate=='A1','RMSE_mV_SD'].iloc[0])<float(a.loc[a.candidate=='A0','RMSE_mV_SD'].iloc[0]) else '较高'
    mean_change=('下降' if delta>=0 else '上升')+f'{abs(delta):.2f}%'
    variation=float(repeats.RMSE_difference_mV.abs().max())
    text(run/'04_最终研究总结.md',f'''# 本轮结论：完成实验，但没有合格的新F2升级

本轮完成了15次从头开始的真实GPU训练：A九次、B六次。没有候选通过预先冻结的验证一致性及尾部守门条件。因此不启动C事件加权和D六种子评价，也不计算新后留成绩。这是有限实验的真实负面/混合结论，不是把未完成训练伪装为成功。

## 1. F2为什么比普通LSTM差？

已证实：已有同协议结果中，F2在Train和Validation的拟合也落后于普通LSTM；这不能只解释成“最后20%太难”。旧六种子只读复算均值如下：

```text
{old.to_string()}
```

本轮原生F2模型定义与正式版AST严格一致，参数332436；输入、标签、样本与时间边界核验通过，没有发现样本错位、未来工况输入或scaler泄漏。CNN各核、LSTM与门控有梯度和权重变化，不能说CNN整条没有训练。Early Stopping轨迹、best checkpoint及预测均已独立核对。

已证实：计划内同配置、同Seed重复，初始化/输入/共同采样顺序相同，训练轨迹和最终成绩仍可明显分歧；所见最大配对RMSE差为{variation:.6f} mV。记录的随机状态与首个权重分歧Epoch详见下面的表。

较可信的解释：F2较复杂的优化过程、非严格确定性数值轨迹放大及较大模型容量共同构成风险。普通LSTM与完整TFT/F2的容量、Dropout位置和结构不同，不能把差距全部归因于CNN或门控。尚无法确认：某个单一结构或算子是最终精度差距的唯一原因。当前结果不是对所有F2超参数或所有残差公式的否定。

## 2. 首先改善训练过程，还是网络融合？

实际顺序为A训练配置→独立验收→B唯一残差融合。A1仅降低学习率，A2仅降低Dropout；没有先叠加多个改动。两者没有取得稳定升级资格，所以B保持A0训练参数。B只把凸组合换成有界残差，TFT主体、原生LSTM/CNN、历史门控与24步头均保留，参数量不变。

原式的隐向量RMS较小本身不是信息丢失证据。后续归一化和Attention可能补偿尺度。只有从头训练的配对比较才检验本轮具体假设；即使推理关闭CNN后误差变差，也不是独立重训的因果模块消融。

## 3. 哪项真正改善Validation？

| 方法 | 新训练种子数 | Validation RMSE均值±SD(mV) | MAE均值±SD(mV) | R²均值（原始ΔV） |
|---|---:|---:|---:|---:|
{rows}

A1的三种子平均RMSE{mean_change}，种子间SD{spread}，但只有{paired_A1}/3种子改善，MAE或P99个别种子守门失败名单为{unsafe}。应明确区分平均变化与逐种子一致性，不能写成稳定升级成功。A2未达条件。

B_v2与新原融合对照相比，RMSE配对改善{rB['matched_control_decision']['paired_wins']}/3；相对阶段A原始配置改善{rB['A0_reference_decision']['paired_wins']}/3；最终初步资格为{rB['qualified']}。完整逐种子/尾部/事件结果均保留。B_control只是计划内原配置重复，不允许选它或A0中较差的一次作为唯一有利基线。

CNN关闭对照、真实补充幅度、各路径训练梯度及权重变化证明分支是否参与，不能自动证明整体精度升级。所有最大误差、P99、不利种子和5232个Validation查询均保留。

## 4. 改进F2在探索性后留区间表现如何？

NA。本轮没有符合固定条件的唯一方案，因此D六种子训练数为0，新后留推理数为0。没有拿旧预测、最好种子或Validation成绩冒充新的后留成绩。以下仅为原任务已完成的六种子历史背景，读取发生在停止决定冻结之后：

| 历史方法 | 种子数 | 后留RMSE(mV) | 后留MAE(mV) | 后留R²（原始ΔV） |
|---|---:|---:|---:|---:|
{history}
| 本轮合格新F2 | 0 | NA | NA | NA |

这些区间已被研究使用，均不称独立盲测；不可把上表与本轮Validation误差当作同区间排名。没有声称超过24.885444 mV的普通LSTM历史均值。

## 5. 是否真正优于原始F2和LSTM？

尚无证据支持这样的升级结论。个别种子和平均Validation有改善，但未满足本轮预定一致性条件，且没有新的六种子后留证据。原生F2继续保留为主模型，普通LSTM仍仅作历史对照，没有更名、替换或改写任何旧成绩。

## 6. 哪些稳定，哪些受到波动影响？

稳定的是协议/样本/输入/初始化身份，以及冻结checkpoint到对应CSV的逐点重放；不稳定的是普通CUDA训练的数值轨迹、最佳Epoch和最终误差。每个Seed只有计划内少量重复，不能从中估计一般成绩分布或宣称统计显著性。

```text
{repeats.to_string(index=False)}
```

不通过更换权重、重跑差Seed或改严格确定性算子来消除本轮不利结果。下一步首先应判断候选收益是否大于已经量化的重复波动；在新的独立预注册之前，不扩大模型、不继续搜索残差系数或损失权重。

## 执行及可追溯证据

真实训练{totals['fresh_models']}个模型、{totals['epochs']}个Epoch、{totals['optimizer_updates']}次优化器更新及同数反向传播。训练墙钟合计{totals['training_wall_seconds']/3600:.3f}小时，包含实际日志和图像刷新，不是纯GPU kernel耗时。正式训练失败数{totals['formal_training_failures']}。预检及第一次只读eval模式梯度检查曾报错，其错误/代码/执行记录保留，修正为train模式、Dropout开启、零optimizer更新的只读诊断后通过；没有修改网络以绕过错误。

A/B独立核验检查了真实初始化、每轮更新/早停、主指标、事件组、尾部数据类型、权重变化及Validation逐点重放。主指标精确一致，尾部分位数同时记录原float32误差路径与全float64的微小数值差异，没有用容差隐藏样本身份错位。

输入和预处理仍为3843行、[0,2690)/[2690,3074)/[3074,3843)、1750/218个Train/Validation起点、Top10、48→24、0.5h、EWMA .3、Train-only scaler、原始ΔV。研究期没有新后留窗口、输入变换或评分。图中恢复电压使用原始起点电压，Step24曲线不拼接重叠预测；全部24步正式指标不变。

`FINAL_DECISION.json`是停止判定，`TRAINING_TOTALS.json`是实际训练预算，`FIGURE_INDEX.csv`和`ARTIFACT_SHA256.csv`分别提供图片/代码/CSV索引及SHA256。`FINAL_INTEGRITY.json`与`HISTORY_PROTECTION.json`在最终验收生成，用于确认旧成果保留；不要把保留的第一次失败回执误作当前状态，应看LATEST_A_EXECUTION/LATEST_B_EXECUTION及独立审计PASS。

仅新增F2_Performance_Improvement目录；不覆盖正式主程序、旧模型、LSTM、历史预测或论文文件。没有新增SOH/RUL、物理网络或backbone。完成此轮后停止训练。
''')
    text(run/'05_文件与复现导航.md','''# 打开顺序

1. 04_最终研究总结.md：通俗答案和真实总表。
2. 01_阶段A_训练配置研究.md：九次原生F2训练。
3. 02_阶段B_残差融合研究.md：六次配对训练、CNN关闭和重复性。
4. 03_阶段C_D_停止决定.md：为什么没有加权/六种子后留成绩。
5. tables/FINAL_VALIDATION_STUDY_SUMMARY.csv、B_PLANNED_NATIVE_REPEATABILITY.csv及各逐种子表。

figures/A与figures/B各图均有PNG/PDF/同名_data.csv；FINAL只有本轮真实Validation汇总，不生成未执行的六种子后留图。FIGURE_INDEX.csv会列出实际存在的图片、绘图CSV、脚本和协议；TRAINING_FIGURE_INDEX.csv列出每个新模型真实训练曲线。

models/A及models/B各实例保存best_model.pt、TRAINING_LEDGER/BEFORE_TRAINING配置和身份、逐Epoch日志、Train/Validation逐点预测、paths、完整指标及CNN关闭反事实。保存权重均来自本次真实新训练。不要把v2权重加载到原融合类后冒称同一模型；应使用configuration与load_new。

VS Code使用Performance_Interactive.py，Python选择项目3.11/CUDA。默认续做当前运行，已完成模型不会重新训练；START_NEW_RUN=True才建立全新有限实验。Run Cell/Run All的图像由真实IPython.display显示，日志和图像并非历史训练回放。actual executed.ipynb记录真实验收，未伪称人工点击VS Code GUI。

verify_stage.verify(run,'A'/'B')可用现有新权重独立复核，不训练或更新参数；已冻结的审计回执不应覆盖，重复核验请另存新回执并保持B预注册引用的SHA。最终历史保护和SHA清单由final_integrity.finalize生成。
''')
    return totals
