# 基于历史双路径时间融合Transformer的PEMFC动态电压多步预测

中文可追溯工作稿，日期2026-10-17。供研究负责人审核，不是投稿终版。正文六章与自有图表/原始CSV对应，章末工作稿溯源不属于拟投稿正文。


---

# 1. Introduction

> 中文可追溯工作稿。文中引文指向核验过的文献，图表编号依本轮索引；章末“工作稿溯源”不属于拟投稿正文。

## 1.1 动态电压多步预测问题

质子交换膜燃料电池的电压是运行性能的直接观测量，但动态负载下的电压变化并不只对应老化。电流切换、运行条件变化和停机后的性能恢复都会改变观测序列。预测模型因此需要同时处理短时波动和较慢的性能演变，并明确其预测对象。Benaggoune等将有新测量反馈的多步预测与连续外推的长期预后分开，说明两者的信息条件和误差传播方式不同。RealHyFC的动态耐久试验也包含变负载与停机过程，适合研究运行电压预测，但不能仅凭电压预测精度认定已验证健康状态或失效寿命。[P1](https://doi.org/10.1016/j.apenergy.2022.118835)；[DATA_PAPER](https://doi.org/10.1016/j.ijhydene.2026.156060)

本文研究的问题是在预测起点仅有历史测量、未来实际工况未知时，直接预测随后12 h的电压增量。任务使用过去24 h的48个半小时统计区间，输出未来24步，不在块内接入新获得的目标电压。这里的“未来工况未知”限定模型接口：预测器不接收未来实测电流、温度或压力，并不意味着它可以准确响应任意未知负载安排。将这一边界与输出跨度同时规定，才能解释误差来源并讨论预测的适用范围。[图F01](../figures/F01_workflow.png)、[图F03](../figures/F03_data_task.png)；自有证据C01–C03。

## 1.2 相关研究与近邻方法

卷积、循环网络及其组合已用于PEMFC性能预测。Benaggoune等采用膨胀卷积与attention进行多步电压预测，并以ARIMA产生的趋势作为另一条条件路径辅助长期外推。Tu等研究TCN提取高层时间特征后连接LSTM，并在公开数据和自测动态循环数据上评价。因此，多尺度卷积或卷积与长时记忆结合不是尚未研究的问题。本文需要区分的是两条路径处理何种信息、在网络何处融合，以及输出是否在同一预测起点直接生成。[P1](https://doi.org/10.1016/j.apenergy.2022.118835)，Figs7–10、16；[P3](https://doi.org/10.1016/j.ijhydene.2024.06.095)，官方摘要及方法片段。

TFT将局部序列处理、变量选择与时间attention结合，原方法已经包含GRN、GLU及已知和未知时变变量的接口。Li等将其用于PEMFC电压退化预测，分析协变量的重要性及时间attention。其正式设置为5点输入、1点输出，未来已知变量表还列有入口温度、压力及气体流量。Chou与Wang进一步使用BiLSTM改进TFT，比较原生TFT、特征选择及不同滚动步长。这些工作意味着本文不能将“首次将TFT、工况信息或门控用于PEMFC”作为贡献，而应说明当前历史编码与严格历史输入实现的具体差异。[TFT](https://doi.org/10.1016/j.ijforecast.2021.03.012)；[P2](https://doi.org/10.1016/j.ijhydene.2023.03.316)，Tables2–3、Figs3–4；[P4](https://doi.org/10.1016/j.renene.2025.123019)，Fig7、Tables6–11。

预测配置与验证方式同样影响对方法的判断。Yakut比较单步和多步LSTM，并在动态片段上测试；目前可得摘要不足以认证其多步实现和物理预测间隔，不能将配置比值直接换算成本稿的12 h任务。Wei等将主模型比较、预测跨度和模块消融分开报告，为结果组织提供参考。不同研究的电堆规模、采样、平滑、输入信息和滚动策略不同，本文据此比较实验设计，不按跨论文RMSE大小建立优劣排名。[P5](https://doi.org/10.1016/j.ijhydene.2025.02.317)，官方摘要；[P6](https://doi.org/10.1016/j.jpowsour.2026.239941)，Tables1–3。

现有论文中的同名“多步”也可能对应不同的信息更新方式。P1在每块预测之后接入届时的新测量，P4使用滚动预测并以阈值交点评价寿命，P2正式实验则是单步输出。这些设置都有各自用途，但不能仅按步数相同判断任务等价。本文将预测起点、输入截止时间和目标身份一并固定，使未来24步误差可以追溯到同一历史信息集；对于原文没有说明的块内策略或未来变量闭合，保持未知，不用模型常识代替作者的实施证据。[P1](https://doi.org/10.1016/j.apenergy.2022.118835)，Fig7；[P2](https://doi.org/10.1016/j.ijhydene.2023.03.316)，数据预处理；[P4](https://doi.org/10.1016/j.renene.2025.123019)，§4。

参考研究也提示应把完整模型比较与组件贡献分开。不同架构总体误差较低，可以支持该配置在相应任务上的应用价值，却不能自动确定是哪一层、哪一类输入或哪种融合带来收益。P4包含原生TFT和特征选择对照，P6区分卷积及物理模块变体，说明消融应与具体主张对应。本稿保留不利模块结果，并把重复初始化、模型筛选及推理干预各自呈现，以减少对整体平均成绩的过度归因。[P4](https://doi.org/10.1016/j.renene.2025.123019)，Tables6–7、10–11；[P6](https://doi.org/10.1016/j.jpowsour.2026.239941)，Table3；自有证据C10–C18。

辅助研究进一步说明，模型名称相近并不代表预测任务或证据层级相同。Jiang等研究TCN与BiLSTM组合，Yi等采用matrix LSTM，其400单电池、1374 h自建试验中的老化目标来自极化曲线0.7 A/cm²选点序列，不能视为本文未知未来工况下的动态电压标签。Ko等则研究具有物理先验的长期预后，并明确假定预测期电流来自训练末段复制。这些研究可借鉴结构对照、重复实验及输入假设的披露方式，但不要求本稿扩展到寿命或物理建模，也不能以它们的健康或寿命指标替代本稿的动态电压评价。[A1](https://doi.org/10.1016/j.ijhydene.2026.153481)，Figs5–8、Table2；[A2](https://doi.org/10.1016/j.ijhydene.2025.02.302)，§2、§4；[A3](https://doi.org/10.1016/j.apenergy.2025.125318)，Fig7、Tables2–3。

## 1.3 本文研究内容与贡献边界

本文在同一TFT历史变量选择表示上保留LSTM路径，增加因果多尺度卷积分支，并在encoder的post-LSTM位置融合两类表征。一个由五项历史统计驱动的标量门控控制路径比例，未来decoder只使用确定的相对步长。研究首先评价该完整配置在固定任务上的经验表现，再分别检验结构差异、门控行为与预测敏感性。这一设计定位为可复现的应用实现，不以模块名称本身替代贡献证据。[图F02](../figures/F02_architecture.png)；自有证据C03–C05、C26。

实验区分三组预定随机初始化下的正式比较和随后选出的六个优质实例。前者在共同起点、目标和预处理下比较F0/F1/F2，后者用于模型选择披露和机制诊断。除整体误差外，本文报告逐步、终点、历史工况分组、尾部及目标时间等权结果，并保留独立训练门控消融的不利证据。目标是判断方法在哪些指标和运行区段有效，而不是将门控变化直接解释为稳定精度收益或电化学因果作用。[图F04](../figures/F04_formal_model_comparison.png)、[图F07](../figures/F07_gate_evidence.png)、[图F10](../figures/F10_metric_sensitivity.png)；自有证据C06–C18、C27。

### 工作稿溯源（非投稿正文）

文献身份、访问范围及具体页码见[文献矩阵](../literature/01_已发表PEMFC论文横向比较矩阵.md)和[参考文献](../literature/REFERENCES.md)。P3/P5未取得全文，不据摘要宣称其缺少实验。实现与证据层级对应[CLAIM_EVIDENCE_MAP.csv](../outputs/evidence_index/CLAIM_EVIDENCE_MAP.csv)的C01–C05、C06–C18、C26–C29。数据论文仅核对前4页的方法范围。本文未建立独立Test、跨电堆或SOH/RUL证据。


---

# 2. Data and forecasting problem

## 2.1 RealHyFC数据及评价区域

研究使用RealHyFC公开动态耐久试验。原始数据论文描述的耐久性装置为10单电池短堆，活性面积280 cm²，采用面向重载应用的动态循环。标称循环约12 h，包含短停和长停，因此运行时间与日历时间不能混同，实际周期也不应被假定为严格等长。论文另有8单电池fresh stack的BoL敏感性测量，它与耐久性电堆的身份不同，本文不将两者组合成额外寿命轨迹。[DATA_PAPER](https://doi.org/10.1016/j.ijhydene.2026.156060)，PDF pp2–4 §2.1/§2.3；官方[数据记录](https://zenodo.org/records/20715542)。

本文沿用已有Development接口，其读取协议为先读取规定CSV前3007条记录，再筛选`bin_end_ophrs2_h < 1819.997738`，得到3005条。时间顺序划分为Train行区间[0,2103)、Validation[2103,2404)、Assessment[2404,3005)。合法窗口分别为1380、205及378个，窗口数不能由区间长度简单相减推定。本轮使用冻结的合法缓存及身份记录，不重新打开完整原始表，也不读取Test。Assessment只复用既有汇总，其已参与历史研究反馈，不能作为新的独立盲测。[表T02](../tables/T02_SPLITS.md)、[图F03](../figures/F03_data_task.png)；自有证据C01、C19–C20。

冻结输入包含14个电压、电流及负载统计槽位，包括电压均值/极值、电流与电流密度的均值/波动、转换计数和零电流占比。它们不是包含温度、湿度、压力的完整运行状态向量。本文的“工况”主要指这些可观测电流和负载统计，不据此宣称已控制所有热、水、气状态。各字段名称、单位和掩码身份见表T01。[表T01](../tables/T01_FEATURES.md)；自有证据C02、C05。

本文的半小时统计接口与原试验的高频测量分辨率不同。模型预测的是这些统计区间上的电压变化，不承担亚秒级负载切换尖峰的重建任务。与此同时，标称约12 h运行循环与未来12 h预测区间只是时间长度相近，两者不是同一个身份：一个预测块未必覆盖恰好一个完整物理循环，也不能将半小时步长解释成循环编号。预测和评价均沿用缓存中的真实起点、目标和运行小时，不借名义循环时长补造对应关系。[图F03](../figures/F03_data_task.png)；[DATA_PAPER](https://doi.org/10.1016/j.ijhydene.2026.156060)，§2.3；自有证据C01。

## 2.2 直接多步增量目标

令\(V_t\)表示冻结字段`cell_voltage_mean`在起点区间对应的原始电压均值，单位为V。起点\(t\)的预测标签定义为\(y_{t,h}=V_{t+h}-V_t\)，其中\(h=1,\ldots,24\)。历史输入由截至起点的48个半小时bin组成，覆盖24 h历史区间。输出是同一起点未来0.5–12 h的24个增量，而不是以未来实测值逐步更新的24次单步预测。不同起点可以预测相同目标时刻，评价时保留其origin、target及horizon三个身份。[图F03](../figures/F03_data_task.png)；自有证据C01、C08及本轮[缓存身份表](../outputs/audits/FORMAL_CACHE_IDENTITY.csv)。

模型输出标准化增量\(\widehat z_{t,h}\)，物理增量仅通过\(\widehat y_{t,h}=s_V\widehat z_{t,h}\)恢复，其中\(s_V=0.13680318714138442\) V为冻结Train电压标准差。目标变换不减去增量均值，也不使用Validation统计。绝对预测电压为\(\widehat V_{t+h}=V_t+\widehat y_{t,h}\)，所加起点必须是原始电压，不能替换为EWMA值。现有预测CSV已经保存物理单位V的增量，指标复算直接使用该列，不再次乘scale。[表T03](../tables/T03_FROZEN_SETTINGS.md)、[图F08](../figures/F08_selected_voltage_example.png)；自有证据C01–C03及冻结SCALER记录。

## 2.3 预处理与可用信息

所有模型共享已确定的Train特征筛选及标准化。历史电压均值在各48步窗口内执行因果EWMA，系数为0.3，其他槽位不进行该电压滤波。公共scaler由未滤波的合法Train历史拟合，滤波后的输入使用同一套均值与标准差。Top10索引固定为[0,3,13,11,4,9,6,7,10,8]，未选槽位置为标准化空间的零值，张量仍保留14个物理输入槽位。因此“Top10”不意味着模型只接收10维张量，也不表示本轮再次进行了特征选择。[表T01](../tables/T01_FEATURES.md)、[表T03](../tables/T03_FROZEN_SETTINGS.md)；自有证据C02。

模型接口另加入确定的相对步长坐标，使encoder连续张量为48×15维。decoder虽然保留统一容器中的物理通道占位，但其变量选择只接入相对步长，未来实测物理变量不进入有效计算。encoder步长为−47至0除以24，decoder步长为1至24除以24；这些坐标在起点即可生成。该接口与文献中显式使用future-known物理协变量的设置不同，但不排除预测受到未观测未来负载的限制。[图F01](../figures/F01_workflow.png)、[图F02](../figures/F02_architecture.png)；自有证据C03；[P2](https://doi.org/10.1016/j.ijhydene.2023.03.316)，Tables2–3。

### 工作稿溯源（非投稿正文）

C01–C03/C19–C20对应H冻结协议及`Phase_TFT_Original_Baseline/models/canonical_tft.py`。字段来自`Phase_F_TFT_DualPath_RegimeFusion/models/frozen_feature_schema.py`；滤波及scaler权威为`Phase_TFT_FullFeature_CausalFilter_Comparison/scripts/run_e_comparison.py`及其冻结配置。本轮只复用已获许可的Train/Validation缓存。数据论文方法核验范围及8/10-cell区别见[literature/search_log.md](../literature/search_log.md)。本轮[执行状态](../outputs/audits/COMPUTATION_STATUS.json)明确未读取Test、未做模型前向。


---

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


---

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


---

# 5. Results and discussion

## 5.1 同协议整体预测表现

双路径F2在冻结Development Validation上取得了较低的整体预测误差。表T04保留F0、F1、F2的全部三个预定Seed和同样本Persistence结果，每个模型均评价205个起点、4920个预测查询及251个唯一目标时刻。三个Seed的F2 RMSE为6.2652、7.0346和8.4364 mV，均低于相同Seed的F0与F1。按每Seed RMSE等权汇总，F0、F1和F2分别为9.067563±1.142630、9.155598±1.122666和7.245400±1.100853 mV。这里的离散量为三个初始化结果的样本标准差，不是置信区间。

**表T04　共同Validation上的完整模型与Seed比较。Persistence只有一次确定性计算，Seed=-1仅为存储标识。**

|模型|Seed|RMSE/mV|MAE/mV|R²(ΔV)|
|---|---|---|---|---|
|F0|42|8.3072|6.4135|0.998187|
|F0|52|10.3815|7.0297|0.997168|
|F0|62|8.5140|3.3163|0.998095|
|F1|42|9.2127|4.3654|0.997770|
|F1|52|8.0055|4.8620|0.998316|
|F1|62|10.2486|5.7261|0.997240|
|F2|42|6.2652|4.7042|0.998969|
|F2|52|7.0346|5.0334|0.998700|
|F2|62|8.4364|3.0851|0.998130|
|Persistence|-1|195.0786|119.7985|-0.000032|

F2相对F0和F1的等Seed平均RMSE分别降低20.0954%和20.8637%。这两个比例由模型RMSE均值之比计算，不能与逐Seed相对改善率的均值混用。图F04同时显示单个Seed与汇总结果，使整体优势及初始化差异均可见。F1均值略高于F0，且仅一个Seed的整体RMSE更低，说明增加卷积分支本身尚未形成一致收益。F2是整体配置的经验优势，组件归因还需独立对照。

![图F04 正式三Seed与Persistence比较](../figures/F04_formal_model_comparison.png)

同一组预测键上的Persistence采用零电压增量，RMSE为195.078555 mV，MAE为119.798476 mV。保持起点电压难以跟随动态负载造成的电压变化，因此该对照主要验证模型确实超越简单静态保持。它属于较弱基线，不能替代同协议LSTM或TCN。表中的R²针对ΔV，其较高取值也不能代替绝对误差、末端误差或健康状态验证。

平均MAE也支持F2的整体误差降低：F0、F1、F2依次为5.5865、4.9845和4.2743 mV。但MAE对少量大残差的敏感程度低于RMSE，不能据此忽略尾部。恢复绝对电压时，预测与真值共享同一个原始起点电压，因此它们的残差与ΔV残差相同，图中电压单位和表中误差单位可以一致转换。高R²主要描述该目标方差下的拟合程度，不能作为健康信息已被识别的证明。

这一比较沿用已冻结特征、缩放和权重，不重新训练。本轮800项独立指标检查的最大数值差为5.6843×10⁻¹⁴，核验对象为九模型与Persistence的49200个查询。复算一致支持结果可追溯性，不能增加评价数据的独立性。

## 5.2 固定24步预测的距离误差与终点表现

图F05评价同一个24步输出模型在不同预测位置的误差，每一步均有205个预测值。它检验误差随预测距离的变化，不代表分别训练了24种输出长度。F2三个Seed在12h终点的RMSE为7.1461、8.5556和29.8174 mV，等Seed均值为15.1731 mV。前两个Seed的终点误差较低，第三个Seed则出现明显恶化。整体7.2454 mV无法完整表达12h终点的风险。

![图F05 逐步误差与12h终点](../figures/F05_horizons_endpoint.png)

[Benaggoune等（2022）](https://doi.org/10.1016/j.apenergy.2022.118835)将多步块预测与后续真实测量更新结合，[Chou与Wang（2025）](https://doi.org/10.1016/j.renene.2025.123019)研究滚动预测，[Wei等（2026）](https://doi.org/10.1016/j.jpowsour.2026.239941)的多步讨论采用自回归描述。这些工作有助于组织跨度分析，但与本文同起点直接24步输出的信息条件不同。本研究的12h长度是固定任务范围，现有结果不足以支持任意远期预测。

## 5.3 历史工况分组下的表现

F2的平均优势在九个历史工况组中均存在，但逐Seed方向并不完全一致。分组分别依据过去48步的电流密度标准差、负载转换次数和平均电流密度，其阈值只由1380个Train历史窗口确定。三组低/中/高起点数量分别为81/74/50、63/73/69和71/56/78。每一种划组均覆盖同一批205个Validation起点，跨划组的窗口重复出现。

**表T09　全部历史组的同Seed差值。负值表示F2平均RMSE更低，胜出数保留三个初始化的方向。**

|历史划组|等级|起点数|F2−F0均值/mV|对F0胜出|F2−F1均值/mV|对F1胜出|
|---|---|---|---|---|---|---|
|电流密度波动|LOW|81|-1.6225|2/3|-2.7135|2/3|
|电流密度波动|MID|74|-2.1382|3/3|-1.3403|2/3|
|电流密度波动|HIGH|50|-1.8456|3/3|-1.7143|2/3|
|负载转换次数|LOW|63|-1.7850|3/3|-1.7780|2/3|
|负载转换次数|MID|73|-1.9850|2/3|-2.9297|3/3|
|负载转换次数|HIGH|69|-1.7193|3/3|-0.5039|1/3|
|平均电流密度|LOW|71|-2.2361|3/3|-3.7124|3/3|
|平均电流密度|MID|56|-1.5846|3/3|-0.5694|1/3|
|平均电流密度|HIGH|78|-1.5734|3/3|-0.6339|2/3|

九组F2等Seed平均RMSE均低于F0和F1，表明整体优势并非只由某一个历史负载层级形成。然而，电流波动LOW组对F0只有2/3胜出，平均负载MID及转换次数HIGH组对F1都只有1/3胜出。因此，相同组内的均值优势可与多数Seed不占优同时发生，不能写成每种工况均稳定提升。图F06展示组别、样本数及差异，并未依据Validation误差重新调整边界。

![图F06 Train定义的历史工况分组](../figures/F06_history_regimes.png)

平均电流密度的Train三分位边界约为0.6939和0.6988 A/cm²，说明LOW与HIGH只是历史均值的相对排序，二者不等于瞬时负载始终处于低值或高值。历史均值还会压缩窗口内的负载差异，因此本研究同时保留波动与转换次数两种划组。各组数量由冻结阈值自然形成，并未强制Validation等样本，也没有依据结果合并不利组。

这些阈值是相对Train分布的统计等级，不能作为通用低负载或高波动的物理标准。分组也没有控制温度、湿度和压力。与[Li等（2023）](https://doi.org/10.1016/j.ijhydene.2023.03.316)按实际负载级呈现预测曲线不同，本节研究的是预测起点已知的历史状态。它避免将未来真实工况作为在线输入，但不构成九套独立工况数据或电化学因果识别。

## 5.4 双路径与动态门控的证据强弱

门控确实参与了预测计算，其独立精度贡献尚未获得支持。六个筛选实例的g都随历史输入变化，固定或置换g会改变预测。g控制的是encoder特征混合比例，不能直接解释为两条路径对最终电压预测的贡献百分比。不同初始化面对同一历史输入的响应方向也有差异，15对门控秩相关范围为−0.996603至0.987941，其中Seed210与多数模型的电流波动响应方向相反。

对同一checkpoint固定g=0.5后，4/6实例RMSE变差，平均增加0.010890 mV。使用各模型Train历史平均g时，仅3/6变差，平均增加0.003566 mV。十次预定置换中43/60次误差增加，各Seed置换均值有5/6变差，总体平均增加0.016604 mV。差异规模很小，且方向不一致。固定0.5的推理扰动没有重新学习其他参数，不能直接称为重新训练的F1。

独立训练G0、G1和G2分别采用固定融合、全局可学习比例和历史条件门控，其三个Seed的均值RMSE为8.387048、8.349936和9.143434 mV。G2相对两对照平均分别增加0.756386和0.793498 mV，配对均只有1/3胜出，未达到事先规定的0.5mV平均改善和逐Seed限制。图F07将训练对照与推理扰动分开，避免以选后实例的小幅优势覆盖独立训练的负面证据。

![图F07 独立训练消融与筛选实例门控扰动](../figures/F07_gate_evidence.png)

门控幅度也不直接衡量其价值。两条路径表征的尺度、冗余程度以及后续非线性处理，都可能使较明显的g变化只产生较小的最终预测变化。反之，某个固定比例在现有权重下表现较好，也不能推导出从头训练时该比例最优。推理反事实在此用于揭示已训练函数的依赖，而独立训练对照用于评价设计选择，二者的问题和解释范围应保持区分。

因此，当前可陈述历史工况依赖的双路径融合已实现并被检验，不能陈述动态门控独立、稳定提高精度。相关性和推理敏感性揭示模型依赖，受控训练才更直接回答设计收益。两类证据共同约束创新主张，而不是择取其中有利部分。

## 5.5 六个优质实例与选择偏差

已有六实例表明冻结F2可以产生误差较低的训练结果，但它们来自Validation筛选。表T06完整列出六个实例，最佳Seed202的RMSE为3.9123 mV。图F08采用筛选阶段已经固定的起点2205，展示单一12h预测，不拼接重叠窗口，也不以该曲线重新选择模型。

**表T06　筛选后实例，不能作为无偏随机Seed总体或正式结构比较。**

|筛选Seed|RMSE/mV|MAE/mV|12h终点RMSE/mV|最大绝对误差/mV|
|---|---|---|---|---|
|42|6.2652|4.7042|7.1461|30.5175|
|103|4.9959|3.8295|4.7940|24.8627|
|193|5.3954|4.0322|5.4291|33.6604|
|202|3.9123|3.0092|4.4606|23.0644|
|210|5.6800|4.3661|6.2041|25.4494|
|212|4.1475|3.1533|4.2869|24.8128|

![图F08 已筛选Seed202的固定起点预测示例](../figures/F08_selected_voltage_example.png)

第一轮设置8h总运行安全预算，注册并启动20个新Seed，18个完成。Seed101因首批日志序列化错误在零训练步时失败，151因训练中的Windows状态文件替换错误中断，失败记录及151的partial checkpoint均保留且未重跑，不能将它们记为预测精度未达标。第二轮设置12h安全预算，注册201至250共50个候选，实际依次完成201至212的12个，取得202、210和212三个新合格实例后停止，余38个未启动。两轮共32次新增尝试、30次成功，Seed42另来自历史H。图F09保留完整完成分布与筛选结果，未把失败或未启动候选隐去。

六实例经回顾性同口径核对，均满足第二轮预先冻结的A级与B级条件，即整体RMSE≤6.3mV、终点RMSE≤10mV、最大误差≤50mV。这些阈值制定时第一轮结果已经可见，不能将对历史实例的回查追认为第一轮的预注册成功。条件核对与两轮筛选也都使用了同一Validation。将其均值与未经相同筛选的F0/F1均值比较会混合架构和选择效应。本文主比较始终使用H的三个预定Seed，六实例仅证明存在良好实现并提供机制诊断对象，不能证明一般初始化的期望表现。

## 5.6 尾部、困难窗口与目标等权敏感性

Seed62说明整体误差降低与末端可靠性并不等价。表T04-S保留同Seed三个模型的全部敏感性对照。F2整体RMSE低于两者，但终点和目标等权均更高，最大误差也略高于F1。较低的P99与较大的最大误差并存，表明少量极端残差不能由分位数概括消除。

**表T04-S　Seed62同样本敏感性对照，单位均为mV。**

|模型|Seed|整体RMSE/mV|12h终点RMSE/mV|目标等权RMSE/mV|P99/mV|最大绝对误差/mV|
|---|---|---|---|---|---|---|
|F0|62|8.5140|29.5735|26.7988|13.5583|309.0627|
|F1|62|10.2486|29.5625|24.6631|24.1329|306.4147|
|F2|62|8.4364|29.8174|27.0539|12.0616|307.9849|

![图F10 正式误差口径敏感性](../figures/F10_metric_sensitivity.png)

图F11按三个正式F2模型平均窗口RMSE最大这一诊断规则选择起点2224，展示相同起点的完整24步预测。Seed62在目标2247和2248，即第23、24步出现大幅错配，最大绝对误差达到307.9849 mV。该规则只用于事后展示困难样本，不参与训练或模型选择。现有缓存能够定位误差，却不能唯一判定其来自某种物理故障、运行转换或软件原因。

![图F11 正式F2共同困难窗口](../figures/F11_difficult_window.png)

图F12进一步保留各目标的SSE贡献。4920个查询包含重复目标，正式查询等权评价会赋予高重数目标更多权重。目标时间等权先在每个目标内平均平方误差，再对目标等权，因而能揭示另一种风险分布。F2 Seed62在此口径升至27.0539 mV，说明少量低重数大误差对整体查询RMSE的影响可能受稀释。两种口径回答的问题不同，不能事后择取有利指标，也不能先平均预测制造集成性能。251个去重时刻仍具有时间依赖，本文不据重叠查询进行显著性检验。

## 5.7 可用范围与泛化限制

本文结果限于一条RealHyFC轨迹及已复用的Development评价。历史I2汇总中F2等Seed RMSE为12.1971±6.3938 mV，三个Seed范围5.1653至17.6615 mV。它在该区域仍优于同Seed对照，但绝对误差和离散性提示Validation上的表现不能直接外推。表T10仅复用已保存汇总，本轮未读取其逐点预测、重新推理或据此选择新模型。

该Assessment已经参与历史研究反馈，不具独立Test身份。现有资格审计也未确认另一个合法、未受设计反馈影响的区域。由此不能宣称独立泛化已经成立，同样不能把独立性未知写成F2使用Test标签训练已被证实。与文献进行比较时，必须保留数据集、电堆尺度、采样、平滑及未来输入的差异，不按跨论文绝对RMSE排名。Tu与Yakut尚缺完整正文，其未知实验设计不能被叙述为文献空白。

预测不确定性也尚未经过独立校准。三个初始化的标准差反映训练随机性的一部分，六个筛选实例则已经受到选择条件影响，二者都不能直接转换为给定预测时刻的可信覆盖区间。本文据此报告点预测误差和风险分布，不额外宣称在线决策保障。

现有证据支持冻结配置的Development经验价值，仍缺同协议简单LSTM/TCN基线及合格独立验证。旧N2虽有这些模型，却使用不同划分和预算，不能直接拼入本文排名。补齐这两项证据比扩大Seed筛选或重新优化门控更能回答方法是否具有可重复的应用价值。

## 工作稿溯源（投稿整理时移入补充材料）

主数值：`outputs/experiment_tables/PAPER_NUMERIC_FACTS.json`、`FORMAL_VALIDATION_METRICS.csv`、`EQUAL_SEED_SUMMARY.csv`、`HORIZON_METRICS.csv`。分组：`outputs/condition_analysis/CONDITION_METRICS.csv`和`PAIRED_REGIME_COMPARISONS.csv`，表T09。消融/筛选：表T06、T07、`SIX_GATE_INTERVENTIONS.csv`、`ALL_COMPLETED_SCREENED_SEEDS.csv`。案例/集中性：`CASE_SELECTION.csv`、`DIFFICULT_WINDOW_DATA.csv`、`TARGET_SSE.csv`。历史Assessment仅表T10。图号和源码由`outputs/evidence_index/FIGURE_INDEX.csv`固定。

对应Claim IDs：C01–C25、C28–C32。所有性能图为Development，六实例为筛选后诊断，SD不作为置信区间。原始预测、极端误差和失败筛选均保留。


---

# 6. Conclusions

## 6.1 实现与性能结论

本研究在RealHyFC上建立了仅使用历史实测信息的动态电压多步预测任务。输入为过去48个半小时步，输出为未来24步原始电压增量，预测电压由未经EWMA平滑的起点电压恢复。双路径TFT在共享历史表示上保留LSTM并加入因果多尺度卷积分支，以五项历史工况统计控制encoder特征混合。该设计提供了清楚的输入可用性和结构接口，其适用目标是未来12h动态电压。

研究得到三方面结果。首先，在三个预定Seed的共同Development Validation上，F2等Seed RMSE为7.245400±1.100853 mV，相对原生TFT及固定等比例双路径的均值分别降低20.0954%和20.8637%。这一结果支持整体配置的经验价值，尚不足以分别归因于卷积或动态门控。其次，逐步、历史工况、尾部和目标时间等权评价补充了单一平均误差的局限。九个历史组的F2平均误差均较低，但逐Seed并非一致占优，Seed62仍有明显终点风险和大幅极端误差：终点及目标时间等权RMSE较两对照更差，最大绝对误差307.9849 mV低于F0的309.0627 mV、略高于F1的306.4147 mV。因此，整体优势不能表述为所有工况与预测位置的稳定优势。第三，完整保留了初始化筛选、门控扰动和独立训练对照。门控确实变化并影响预测，但固定Train平均比例时六实例只有一半变差，P2独立训练也未达到预设增益门槛，动态门控独立提升精度的主张未获支持。

六个优质实例中Seed202取得3.9123 mV RMSE，仅代表同一Validation筛选得到的良好实现。其结果与三Seed公平比较分开呈现，避免把筛选收益当作模型总体性能。本文的贡献在于冻结任务下的双路径实现、可追溯经验比较，以及同时报告收益、不利样本和模块证据边界。它不构成已验证的电化学机理、SOH或RUL预测。

## 6.2 最小必要后续验证

后续首先需要同协议简单LSTM与TCN对照，以固定输入、相同时间划分、相同预测目标和评分补齐常规时序基线。应预先确定配置、Seed和预算，保留全部结果，而不是持续搜索直到获得有利排名。Persistence已提供零增量参照，但无法单独回答双路径TFT相对标准时序模型的价值。

另一项必要条件是合格的独立评价来源。现有Assessment已经复用，不能重新命名为Test，也不能通过同一区域后验切分恢复确认性资格。未来评价需先核清数据访问史与任务兼容性，再冻结预测流程和判据。单RealHyFC数据的限制若无法解除，应保持探索性Development结论，不承诺跨电堆或任意未来负载泛化。

当前证据不要求继续筛选Seed、扩大骨干网络或为门控追求正面结果反复训练。更有价值的推进方式是补齐最小公平对照和独立验证，同时保留已观察到的负面事实。本轮没有启动上述训练或新评价。

## 工作稿溯源（投稿整理时移入补充材料）

数值对应表T04–T09及`outputs/experiment_tables/PAPER_NUMERIC_FACTS.json`。模块结论对应表T07、`SIX_GATE_INTERVENTIONS.csv`及Claim C10–C18。泛化限制对应Claim C19–C21、C29–C32及历史L资格结论。新计算完整性依据`outputs/audits/INDEPENDENT_RESULT_AUDIT.json`。结论未增加正文以外的实验或数字。


---

# 工作稿声明与待确认事项

## Data and code availability

RealHyFC原始耐久性数据来自[Fensterle等数据记录](https://doi.org/10.5281/zenodo.20715542)，官方元数据标记CC BY4.0。对应原始数据论文为[10.1016/j.ijhydene.2026.156060](https://doi.org/10.1016/j.ijhydene.2026.156060)。本文仅使用已冻结合法Development接口；本阶段没有读取或发布新的Test数值。代码、缓存复算、图源数据及SHA索引保存于本项目独立整合阶段，发布状态以本阶段Git回执为准。不能把数据许可自动解释成所有第三方软件或论文PDF的许可。

## AI assistance

AI辅助工具用于定向文献定位与整理、代码编写、既有缓存数值复核、原创图表生成和中文工作稿起草。文献已按实际可访问范围分级；不同代理的内部检查不等于真人同行评议。研究负责人应审阅实验解释与全文、核验引用并对最终投稿内容负责。AI工具不列为作者。

## Author contributions / funding / conflicts / ethics

本工作稿尚未确认作者名单和CRediT分工、基金项目及编号、利益冲突真值；这些信息须由研究负责人和实际作者填写，不能默认“无资助”或“无冲突”。本阶段未开展人体或动物实验，仅处理公开电化学实验资料和既有预测缓存。正式伦理声明以作者单位与目标期刊要求核定。本文件是待审核声明，不是已签署投稿声明。

## Scientific scope

本文为单RealHyFC记录上反复使用Development的探索性点预测研究，不构成独立确认性Test、跨堆泛化、因果机理或SOH/RUL验证。两篇论文尚无全文；其未知细节不是否定证据。门控设计和数值依赖可以报告，独立稳定精度收益不能报告为已证实。


---

# 核验参考文献

[P1]–[P6]核心；[A1]–[A3]辅助；[TFT]原方法；[DATA_PAPER]数据来源。实际证据等级及页码见矩阵/索引，不将元数据核验等同全文阅读。

## P1

Khaled Benaggoune; Meiling Yue; Samir Jemei; Noureddine Zerhouni. (2022). A data-driven method for multi-step-ahead prediction and long-term prognostics of proton exchange membrane fuel cell. *Applied Energy*, 313, 118835. [DOI](https://doi.org/10.1016/j.apenergy.2022.118835).

## P2

Li Hongwei; Qiao Binxin; Hou Zhicheng; Liu Junnan; Yang Yue; Lu Guolong. (2023). An interpretable data-driven method for degradation prediction of proton exchange membrane fuel cells based on temporal fusion transformer and covariates. *International Journal of Hydrogen Energy*, 48, 25958-25971. [DOI](https://doi.org/10.1016/j.ijhydene.2023.03.316).

## P3

Chongyang Tu; Fen Zhou; Mu Pan. (2024). A model with high-precision on proton exchange membrane fuel cell performance degradation prediction based on temporal convolutional network-long short-term memory. *International Journal of Hydrogen Energy*, 74, 414-422. [DOI](https://doi.org/10.1016/j.ijhydene.2024.06.095).

## P4

Jia-Hong Chou; Fu-Kwun Wang. (2025). Future prediction on the remaining useful life of proton exchange membrane fuel using temporal fusion transformer model. *Renewable Energy*, 247, 123019. [DOI](https://doi.org/10.1016/j.renene.2025.123019).

## P5

Yurdagül Benteşen Yakut. (2025). Short-term voltage loss prediction in PEMFCs using single-step and multi-step LSTM models. *International Journal of Hydrogen Energy*, 144, 652-664. [DOI](https://doi.org/10.1016/j.ijhydene.2025.02.317).

## P6

Lin Wei; Shangrui Gao; Zonghai Chen; Yujie Wang; Zhendong Sun. (2026). A hybrid framework for PEMFC degradation prediction integrating Crossformer and physical constraints. *Journal of Power Sources*, 677, 239941. [DOI](https://doi.org/10.1016/j.jpowsour.2026.239941).

## A1

Jian Jiang; Dongsheng Du; Lin Su; Yifan Du. (2026). A TCN-BiLSTM-based approach for remaining useful life prediction of PEMFC. *International Journal of Hydrogen Energy*, 207, 153481. [DOI](https://doi.org/10.1016/j.ijhydene.2026.153481).

## A2

Fengyan Yi; Xing Shu; Jiaming Zhou; Jinming Zhang; Chunxiao Feng; Hongtao Gong; Caizhi Zhang; Wenhao Yu. (2025). Remaining useful life prediction of PEMFC based on matrix long short-term memory. *International Journal of Hydrogen Energy*, 111, 228-237. [DOI](https://doi.org/10.1016/j.ijhydene.2025.02.302).

## A3

Taehwan Ko; Dukyong Kim; Jaewoong Park; Seung Hwan Lee. (2025). Physics-informed neural network for long-term prognostics of proton exchange membrane fuel cells. *Applied Energy*, 382, 125318. [DOI](https://doi.org/10.1016/j.apenergy.2025.125318).

## TFT

Bryan Lim; Sercan Ö. Arık; Nicolas Loeff; Tomas Pfister. (2021). Temporal Fusion Transformers for interpretable multi-horizon time series forecasting. *International Journal of Forecasting*, 37, 1748-1764. [DOI](https://doi.org/10.1016/j.ijforecast.2021.03.012).

## DATA_PAPER

Eva Fensterle; Jürgen Hunger; Frank Rabsahl; Florian Wilhelm; Christian Kolle Christensen; Adam Benson; Rami Al-Khamissi; Marine Tregaro; Sylvie Escribano; Joachim Scholta; Markus Hölzle. (2026). Spatially resolved electrochemical characterization of a PEM fuel cell stack during durability testing. *International Journal of Hydrogen Energy*, 250, 156060. [DOI](https://doi.org/10.1016/j.ijhydene.2026.156060).
