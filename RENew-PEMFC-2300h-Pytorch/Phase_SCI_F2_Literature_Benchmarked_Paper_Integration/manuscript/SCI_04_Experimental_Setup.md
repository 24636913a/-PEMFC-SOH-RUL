# 4. Experimental setup

## 4.1 冻结训练协议与模型选择

正式模型比较采用预定Seed42、52和62，每个seed分别训练F0、F1和F2，共9次历史运行。它们共享时间划分、Top10掩码、EWMA、scaler及24步目标。训练环境为Python3.11.9、PyTorch2.6.0、pytorch-forecasting1.8.0，GPU为RTX3070 Laptop。AdamW学习率为0.0009605449687795866，weight decay为6.95942232671908×10⁻⁶，batch size32，最多450 epoch，early-stopping patience40，梯度范数裁剪1。隐藏维度64、连续变量嵌入维度32、多头数4、LSTM层数1、dropout0.1。模型以标准化增量的MSE训练，并按冻结Validation准则保留最佳checkpoint。[表T03](../tables/T03_FROZEN_SETTINGS.md)；自有证据C07及H冻结协议。

三个seed用于描述同一时间划分下的初始化差异，不是三个时间fold，也不是三台独立电堆。重复性探查Repeat-A/B不加入正式三seed统计。各模型先在每个seed上计算指标，再报告等seed均值及样本标准差；标准差不作为置信区间或模型预测区间。历史正常GPU设置不宣称严格跨执行环境逐位确定性，本轮数值审查采用保存缓存、身份和独立算术复核。[图F04](../figures/F04_formal_model_comparison.png)；自有证据C07、C25、C30。

六个优质F2实例为42、103、193、202、210和212。除来自正式实验的42外，两轮筛选共新尝试32个seed，30个完成、2个失败，失败身份为101和151；第二轮在找到3个新合格实例后停止，尚有38个已注册候选未启动。其筛选使用同一Validation，不属于无偏随机初始化样本。本文完整披露预算和停止规则，将六实例用于预测示例及筛选后诊断，不用其平均值与未经过相同搜索的F0/F1平均值比较结构性能。[图F09](../figures/F09_seed_selection.png)、[表T06](../tables/T06_SELECTED_F2_INSTANCES.md)；自有证据C16–C18。

## 4.2 比较模型与消融层次

主比较包含同协议F0/F1/F2及Persistence。对于增量目标，Persistence令所有\(\widehat y_{t,h}=0\)，对应绝对电压一直保持起点原始值；它不需拟合，使用与模型完全相同的205个起点和4920个目标。既有其他模型若采用不同fold、scaler或训练预算，不拼接为本协议公平基线。当前证据尚不包含相同协议下重新训练的简单LSTM/TCN对照，本轮不通过未经授权训练补齐。[图F04](../figures/F04_formal_model_comparison.png)、[表T04](../tables/T04_FORMAL_MODEL_COMPARISON.md)；自有证据C21–C22。

模块证据分为独立训练和推理干预。既有G0/G1/G2消融使用各自从头训练的模型，保留其失败于预设收益标准的结论。六实例推理诊断则固定同一checkpoint：B0使用原g，B1临时固定0.5，B2固定各模型Train历史输入产生的平均g；另用预先固定的10次置换改变窗口与g的对应关系，并保留g分布。B1不是重新训练F1，置换也不单独证明因果精度收益。本文不以筛选实例的推理变化覆盖独立训练的负面证据。[图F07](../figures/F07_gate_evidence.png)、[表T07](../tables/T07_INDEPENDENT_TRAINING_ABLATION.md)；自有证据C11–C15。

## 4.3 指标、重叠窗口与历史工况分组

令物理增量残差\(e_{t,h}=\widehat y_{t,h}-y_{t,h}\)。主RMSE为全部查询残差平方均值的平方根，乘1000后以mV报告；MAE、P95、P99和最大绝对误差同样使用物理mV。R²对ΔV真值计算，不能与另行对绝对电压计算的R²混用。逐horizon各含205项，第24步定义为12 h端点。该曲线评价同一个固定24步模型在不同预测位置的误差，并非分别训练多个输出长度的实验。[图F05](../figures/F05_horizons_endpoint.png)、[表T04](../tables/T04_FORMAL_MODEL_COMPARISON.md)；自有证据C08、C28。

4920个查询包含251个唯一目标时刻，较多重叠使不同目标在 pooled RMSE中获得不同权重。为诊断这种影响，target-time-equal RMSE先对同一目标时刻的所有查询取残差平方均值，再对各目标等权平均并开方；该指标没有先平均预测值。本文同时保留全部残差及极值，不删除低重数目标或不利窗口。去重目标仍来自同一电堆和连续时间序列，不能将251个时刻当作251个独立电池进行统计检验。[图F10](../figures/F10_metric_sensitivity.png)、[图F12](../figures/F12_target_concentration.png)；自有证据C08–C09。

工况分组采用Train1380个历史窗口定义的三分位规则。电流密度标准差槽位的历史均值阈值为0.2673383和0.2760350 A/cm²，转换次数为18.895833和19.708333次/bin，电流密度均值为0.6939044和0.6988030 A/cm²。低组满足\(x\le q_1\)，中组满足\(q_1<x\le q_2\)，高组满足\(x>q_2\)。阈值在Validation误差计算前固定，分组仅描述相对历史状态，不将其命名为物理安全边界，也不从未来实际工况反推分组。[图F06](../figures/F06_history_regimes.png)、[TRAIN_THRESHOLDS.csv](../outputs/condition_analysis/TRAIN_THRESHOLDS.csv)；自有证据C23。

本轮计算只读取许可的正式Validation缓存及Train历史汇总，严格核对origin、target、horizon键与逐点真值，并通过独立程序复算指标。历史Assessment仅以已有汇总解释研究边界，既有Test资格审查也不构成本轮Test评价。因而本文的性能主张限定在已复用Development设置，不能作为独立确认性泛化证据。[独立复核记录](../outputs/audits/INDEPENDENT_RESULT_AUDIT.json)、[执行状态](../outputs/audits/COMPUTATION_STATUS.json)；自有证据C19–C25。

工况比较首先保留每组起点和查询数量，再在同seed、同组样本上计算模型差异，最后等seed汇总。这些组是同一时间序列的条件子集，不是另外三套独立运行试验。三种分组变量可能相关，其分别出现的差异不能累加成三个独立增益，也不意味着历史波动导致预测误差发生因果变化。困难窗口依据完整正式池中预先写明的诊断规则选择，并与固定共同起点的优质实例图分开，避免将最优曲线作为总体性能的替代。[图F06](../figures/F06_history_regimes.png)、[图F08](../figures/F08_selected_voltage_example.png)、[图F11](../figures/F11_difficult_window.png)；[配对工况结果](../outputs/condition_analysis/PAIRED_REGIME_COMPARISONS.csv)及自有证据C23、C27。

### 工作稿溯源（非投稿正文）

训练参数取自H冻结协议及[表T03](../tables/T03_FROZEN_SETTINGS.md)，不是本轮新训练。筛选次数取两轮完整状态和预算记录；独立训练消融取P2正式结果，推理机制取SixSeed审计。正式指标、逐步及工况分别对应`outputs/experiment_tables/FORMAL_VALIDATION_METRICS.csv`、`HORIZON_METRICS.csv`及`outputs/condition_analysis/CONDITION_METRICS.csv`。已完成的C22–C24见`outputs/audits/COMPUTATION_STATUS.json`及`INDEPENDENT_RESULT_AUDIT.json`：独立程序完成800项指标检查，最大浮点差异为5.68434×10⁻¹⁴，未执行训练或模型前向。未补做简单强基线、独立Test或预测区间校准，不宣称这些证据已经完成。
