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
