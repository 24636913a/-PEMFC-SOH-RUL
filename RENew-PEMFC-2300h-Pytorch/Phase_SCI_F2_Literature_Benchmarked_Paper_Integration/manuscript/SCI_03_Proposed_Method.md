# 3. Proposed dual-path TFT

## 3.1 原生TFT组件与任务适配

本文的基础模型保留TFT的连续变量选择网络、门控残差网络、LSTM encoder/decoder、可解释多头attention及残差归一化。历史物理槽位和相对时间坐标分别嵌入，并经变量选择形成共同的历史表示\(E\in\mathbb R^{B\times48\times64}\)。模型无实际静态协变量，静态上下文通路使用零信息输入。未来decoder仅处理确定的相对步长。这些结构是对已有TFT实现的继承与严格历史任务适配，不属于本稿新增模块。[TFT](https://doi.org/10.1016/j.ijforecast.2021.03.012)；[图F02](../figures/F02_architecture.png)；自有证据C03–C04。

与原方法常用的分位数输出不同，当前任务采用单通道点预测头，每个未来位置输出一个标准化ΔV，并以外部MSE训练。这里的F0是共同预处理、输入及输出接口下的基础TFT实现，不能据名称认为它复现了原TFT论文的所有数据集、概率损失和实验设置。F1、F2继承同一主体，使结构比较聚焦历史编码的变化。[表T03](../tables/T03_FROZEN_SETTINGS.md)、[图F02](../figures/F02_architecture.png)；自有证据C03–C04；[TFT](https://doi.org/10.1016/j.ijforecast.2021.03.012)。

## 3.2 并行历史编码与融合位置

LSTM路径从共同表示\(E\)提取顺序信息，其输出经过原生post-LSTM门控及Add-Norm得到\(L\)。新增卷积分支也接收\(E\)，并不接收LSTM路径的预测结果或未来decoder位置。它并行执行核宽3、膨胀率分别为1和4的depthwise一维卷积，通过显式左侧填充保持48步长度。每个位置的卷积访问限于该位置及此前历史，不采用对称填充。该设计在同一历史表示上形成不同局部时间尺度的特征。[图F02](../figures/F02_architecture.png)；自有证据C04；源码`causal_multiscale_branch.py`。

两路卷积结果沿通道拼接，再经1×1卷积从128通道投影回64通道，与输入表示执行dropout、残差相加及LayerNorm，得到\(C\in\mathbb R^{B\times48\times64}\)。可写为\(C=\operatorname{LN}[E+\operatorname{Dropout}(\operatorname{PW}(\operatorname{Concat}(D_1(E),D_4(E))))]\)。它与LSTM路径在相同时间位置和隐藏维度对齐，不是分别预测两个最终电压后再加权，也不同于先TCN、后LSTM的串联模型。[图F02](../figures/F02_architecture.png)；自有证据C04；[P3](https://doi.org/10.1016/j.ijhydene.2024.06.095)，官方摘要。

融合发生于encoder的post-LSTM结果与后续静态增强之间。F1使用\(H=0.5L+0.5C\)，F2使用历史工况门控确定比例；decoder的post-LSTM表示保持原接口，再与融合历史序列连接，进入原生attention和点预测头。因而比较F1/F0考察增加分支及固定融合的整体变化，比较F2/F1考察可学习门控替代固定比例后的整体变化。参数量分别为323483、332379和332436，三者不是严格等容量模型，应在解释结构收益时保留这一差别。[图F02](../figures/F02_architecture.png)；自有证据C04、C10、C15及[结构规格](../outputs/experiment_tables/DIAGRAM_SPEC.json)。

encoder融合并不替代原生LSTM传给decoder的hidden/cell。源码先用原LSTM encoder的状态初始化LSTM decoder，decoder的外部输入只有相对步长；随后分别形成encoder和decoder的post-LSTM表示。门控只改变前者，融合后的48步历史表示与未被该融合改写的24步decoder表示连接，才进入静态增强、attention及输出层。因此模型内同时存在原生状态传递和历史表征融合两条路由，不能将架构简画成“融合结果送入decoder”。[图F02](../figures/F02_architecture.png)；源码`dualpath_strict_history_tft.py::forward`；自有证据C03–C04。

这种融合比例的解释需要结合张量位置。较大的g仅说明当前encoder表示在该混合节点更偏向LSTM特征；两条路径的尺度、方向及后续attention响应共同决定最终输出。不同seed即便产生相似g，也未必使用相同隐藏表征；反之，不同的g响应可能在后续层得到接近的预测。本文因此同时观察g、预测变化与误差变化，不仅凭门控曲线或其与电流波动的相关系数认定路径的实际精度贡献。[图F07](../figures/F07_gate_evidence.png)；自有证据C05、C11–C15、C27。

## 3.3 历史工况门控与直接输出

门控上下文由五个标准化历史槽位各自沿48步求均值得到，按固定顺序为负载转换次数、零电流比例、电流密度标准差、电流标准差和电流密度均值。槽位内的标准差先描述半小时bin内的电流波动，再沿历史窗口求均值，不能将它误写为直接计算整个24 h原始电流的标准差。该上下文来自输入端的历史统计，而不是未来工况或当前预测误差。[表T01](../tables/T01_FEATURES.md)、[图F02](../figures/F02_architecture.png)；自有证据C05；源码`regime_aware_dualpath_tft.py`。

令五维上下文为\(r_t\)，门控计算为\(g_t=\sigma[W_2\operatorname{ELU}(W_1r_t+b_1)+b_2]\)，隐藏维度为8，随后\(H_t=g_tL_t+(1-g_t)C_t\)。每个origin生成一个标量，在encoder时间位置和隐藏通道上广播；它不是未来24步各自独立的24个门控。\(g_t\)表示进入下游网络之前的LSTM特征比例，\(1-g_t\)表示卷积特征比例。由于后续网络包含非线性处理，不能将两者直接解释成最终预测电压的可加贡献百分比。[图F02](../figures/F02_architecture.png)；自有证据C05、C27。

门控输出层在原始训练初始化时置零，使初始\(g=0.5\)，之后的权重由历史训练过程确定。动态响应是否实际存在、是否改变预测以及是否稳定降低误差是三个不同问题。本文用已完成的门控统计、固定门控和置换推理诊断回答前两项，并结合独立训练消融评价第三项，不将设计公式本身作为独立精度增益证明。所有门控机制结论限于冻结实例和对应评价范围。[图F07](../figures/F07_gate_evidence.png)、[表T07](../tables/T07_INDEPENDENT_TRAINING_ABLATION.md)；自有证据C11–C15。

### 工作稿溯源（非投稿正文）

C03–C05对应`canonical_tft.py`及F的`dualpath_strict_history_tft.py`、`causal_multiscale_branch.py`、`regime_aware_dualpath_tft.py`。本轮只读源码，未实例化模型或执行前向。[DIAGRAM_SPEC.json](../outputs/experiment_tables/DIAGRAM_SPEC.json)锁定张量、分支与参数量；图F02由自有源码绘制，未复制发表论文插图。门控独立收益边界对应C10–C15，原TFT门控继承关系对应C26。没有将attention权重或g解释为电化学因果参数。
