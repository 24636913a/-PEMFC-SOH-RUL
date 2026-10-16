# 本轮核心P1–P5真实文献核验

核验日期：2026-10-17。仅审查文献，不读取任何项目数据、预测缓存或权重，不训练。P编号按本轮用户指令；历史Phase O中Benaggoune是P4、Chou是P1，不能按旧编号直接拼表。

## 证据分级与实际访问

|本轮ID|论文|本轮实际访问|历史Phase O ID|
|---|---|---|---|
|P1|[Khaled Benaggoune等，2022](https://doi.org/10.1016/j.apenergy.2022.118835)|FULL_TEXT_READ_AND_KEY_PAGES_RENDERED|P4|
|P2|[Li Hongwei等，2023](https://doi.org/10.1016/j.ijhydene.2023.03.316)|FULL_TEXT_READ_AND_KEY_PAGES_RENDERED|P2|
|P3|[Chongyang Tu等，2024](https://doi.org/10.1016/j.ijhydene.2024.06.095)|PUBLISHER_ABSTRACT_AND_LIMITED_SNIPPETS_ONLY|无对应核心记录|
|P4|[Jia-Hong Chou等，2025](https://doi.org/10.1016/j.renene.2025.123019)|FULL_TEXT_READ_AND_KEY_PAGES_RENDERED|P1|
|P5|[Yurdagül Benteşen Yakut，2025](https://doi.org/10.1016/j.ijhydene.2025.02.317)|PUBLISHER_ABSTRACT_AND_LIMITED_SNIPPETS_ONLY|无对应核心记录|

本地P1/P2/P4全文SHA与Phase O证据账本完全一致。本轮重新提取通读相关全文，按PDF物理页定位，并渲染查看关键表图。P3/P5无本地PDF，官方页面直接打开返回403；搜索索引仍提供官方摘要/有限章节片段，Elsevier公开API返回coredata与openaccess=false，未取得正文。因此其章节结构、模块消融等必须保留UNKNOWN。Crossref官方接口核实P5唯一作者为Yurdagül Benteşen Yakut。

## 对本稿最重要的比较结论

1. **优先借鉴P1的多步任务区分与结果组织，P2的协变量接口及解释性呈现，P4的native/improved TFT对照与跨度敏感性。** 这些工作已覆盖多步预测、CNN长时感受野、PEMFC TFT和改进TFT，不能虚构“从未应用”的研究空白。
2. 本项目差异应写成具体接口与实现：RealHyFC、24 h历史、未来12 h直接24步ΔV、历史工况统计驱动的并行LSTM/CNN encoder融合、推理不使用未来实际工况。差异本身不等于已证明性能增益。
3. P1短期每块预测后引入届时真实观测；P2正式5→1；P4滚动至阈值，未来已知主要hour/cycle index。三者均不能直接当作与本稿完全等价的48→24直接多步。P1的预测step物理间隔、P3/P5的具体horizon映射未充分确认，不能补齐。
4. P2 Tables2–3的未来已知工况含T/P/流量。原文没有足够控制命令可用性证据，不等于可指控数据泄漏，但也不能将其任务混称本稿future-unknown。
5. P4 Table6/7与Tables10/11明确存在特征选择对照及native TFT对照。文献“没有消融”这一说法不成立；但同容量/同调参预算/独立多seed是否控制仍不能认证。
6. P3摘要及方法片段足以确认串联TCN→LSTM以及自测动态数据。未取得全文，不得称其无消融、无稳健性或没有多步实验。P5同理。
7. 全部文献的RMSE受堆数、输出尺度、采样、平滑、任务及未来信息影响，**本轮不建立跨论文RMSE排行榜**。P4的RUL RE尤其不能与本稿mV误差比较。

## 可供整合器直接使用的17字段逐篇记录

### P1 — Khaled Benaggoune（2022）

A data-driven method for multi-step-ahead prediction and long-term prognostics of proton exchange membrane fuel cell。Applied Energy 313，118835。[DOI](https://doi.org/10.1016/j.apenergy.2022.118835)。

1. **研究问题与预测目标**：以堆电压为性能观测，区分短期多步预测与无新测量反馈的长期预后。〔FULL_TEXT_VERIFIED；PDF pp4–5 §2.4; p6 Fig7〕
2. **数据来源及工况类型**：FCLAB/PHM两台5-cell、100 cm²堆：恒流与5 kHz小纹波；另有PRAGMA 15-cell、33.625 cm²恒流堆。并非RealHyFC动态负载轨迹。〔FULL_TEXT_VERIFIED；PDF p4 §2.3, Tables2–3; pp4–5 Figs5–6〕
3. **输入变量**：短期直接使用历史堆电压；长期条件分支使用Train电压趋势分解及由Train拟合ARIMA产生的未来趋势，不是未来实测工况。〔FULL_TEXT_VERIFIED；PDF p7 §3.4 step1; pp10–12 §4.1–4.2 Fig16〕
4. **预测方式**：短期每块预测n点，再以届时新获得真实测量更新下一块；长期将预测值接回输入直至阈值。短期块内direct/recursive实现细节不能完全认证。〔PARTIALLY_VERIFIED；PDF p6 Fig7; p8 §3.5.2; p11 §4.3.2〕
5. **预测步长及实际物理时间**：正式比较5/10/15/24步；图横轴为小时，但全文未充分定义平滑后各预测步的统一时间间隔，不能擅将24步换算为本项目12 h。输入17/20/25及attention尺寸12/32/12都不是预测跨度。〔PARTIALLY_VERIFIED；PDF p8 Table4 §3.5.2; pp8–11 Figs12–14 Table5; Appendix pp13–14〕
6. **主要模型**：带skip connection和Luong attention的两层膨胀CNN；长期为ARIMA引导的conditional CNN。〔FULL_TEXT_VERIFIED；PDF pp5–7 §§3.1–3.4 Eqs4–9 Fig10; pp10–12 §4 Fig16〕
7. **方法创新点**：因果膨胀卷积扩大时间感受野，attention筛选历史特征；长期另加条件趋势路径。双路径/多尺度卷积/attention本身不能再宣称首次用于PEMFC。〔FULL_TEXT_VERIFIED；PDF pp6–7 Figs8–10; pp10–12 Fig16 Eqs13–14〕
8. **对比模型**：短期MLP、LSTM、BiLSTM；长期ConCNN、DilCNN、ESN、LSTM。〔FULL_TEXT_VERIFIED；PDF p8 §3.5.2; p11 Table5; p12 Table6〕
9. **模块消融方式**：Table6比较有无条件分支；未在该全文中定位到dilation、skip、attention分别移除且等容量等调参预算的逐模块消融，不能把模型间比较当作这类独立消融。〔FULL_TEXT_VERIFIED；PDF pp8–12 Tables5–7; §3.5 and §4.3〕
10. **不同工况分析**：三堆/恒流与纹波分别给曲线和指标；讨论启停、恢复以及后段困难区间。不是Train阈值定义的历史工况分组误差。〔FULL_TEXT_VERIFIED；PDF pp8–9 Figs12–14 §3.5.2; pp11–13 Fig17〕
11. **稳健性和泛化验证**：跨三个实验数据、多个起点和预测跨度；长期MC dropout重复50次及箱线图。50次dropout不等于50次独立训练seed，也未据此认证跨堆迁移。〔FULL_TEXT_VERIFIED；PDF pp11–13 §4.3.2 Fig17; p12 Table6〕
12. **实际评价指标**：短期报告MAPE和名为RMSE的指标；p7 Eq11印刷为归一化绝对误差开方，非标准RMSE。实现UNKNOWN。长期RUL绝对偏差/箱线图及训练推理时间。〔FULL_TEXT_VERIFIED；PDF p7 Eqs10–11; pp11–14 Tables5–7 Fig17〕
13. **核心图表类型**：任务信息流Fig7、卷积/attention结构Fig10、多horizon预测叠图Figs12–14、跨堆×horizon指标Table5、长期条件结构Fig16、起点箱线图Fig17、时间开销Table7。〔FULL_TEXT_VERIFIED；PDF pp6–14〕
14. **论文章节结构**：1 Introduction; 2 PEMFC degradation and performance prediction; 3 Multi-step-ahead prediction based on dilated CNN; 4 Prognostics based on conditional CNN; 5 Conclusion; Appendix Data preparation。〔FULL_TEXT_VERIFIED；PDF p3组织段; pp3,5,9,12–14 headings〕
15. **可借鉴做法**：借鉴明确区分块预测和滚动反馈、跨度×模型误差表、困难恢复区段解释、计算开销说明。保留本稿单一动态电压目标，不复制RUL章节。〔REVIEWER_SYNTHESIS；依据PDF p6 Fig7,p11 Table5,p14 Table7; 本任务48→24协议〕
16. **不能直接借鉴的原因**：数据/堆数/电压尺度/平滑/预测反馈条件不同；附录预处理整体边界不充分；印刷RMSE公式异于标准。不得跨文献误差排名，也不能将其ARIMA条件分支等同F2 encoder内LSTM/CNN融合。〔REVIEWER_SYNTHESIS；依据PDF pp4,6–8,10–14〕
17. **具体来源索引**：本轮本地全文15页实际提取通读，关键p6 Fig7、p8 Table4/Fig12、p11 Table5已渲染核对；Phase O仅作交叉核对。〔FULL_TEXT_VERIFIED；runtime/core/P1_fulltext.txt; P1_p6.png/P1_p8.png/P1_p11.png〕

### P2 — Li Hongwei（2023）

An interpretable data-driven method for degradation prediction of proton exchange membrane fuel cells based on temporal fusion transformer and covariates。International Journal of Hydrogen Energy 48(66)，25958–25971。[DOI](https://doi.org/10.1016/j.ijhydene.2023.03.316)。

1. **研究问题与预测目标**：使用TFT和协变量预测动态运行单电池电压，提供变量重要性及时间attention解释。〔FULL_TEXT_VERIFIED；PDF pp3–4 contributions; pp8–12 methods/results〕
2. **数据来源及工况类型**：同济大学已发表的汽车单PEMFC动态耐久试验：25 cm²，3076循环约1008 h，每周期35次电流变化、9负载级；周期性停机后有性能恢复。〔FULL_TEXT_VERIFIED；PDF pp8–9 dataset/preprocessing; p9 Fig8〕
3. **输入变量**：作者称17项协变量，Tables2–3区分过去测量和未来已知；后者列hour/cycle position、入口压力/温度及氢氧流量。其未来可知的排程/控制命令证明未充分报告，不能断言作弊，也不能视为history-only。〔FULL_TEXT_VERIFIED；PDF p6 Table1; p8 Tables2–3; p10 parameter definition〕
4. **预测方式**：正式设置moving window5、forecast horizon1，即5→1单步；长时间曲线是逐步评价，不能据TFT通用多horizon能力假定本文已完成直接24步。〔FULL_TEXT_VERIFIED；PDF p9 Data preprocessing; p10 step4; pp11–12 Fig8〕
5. **预测步长及实际物理时间**：5输入点预测1点；按每电流平台倒数第5个原始记录采样、每周期35点，各平台时长不等，单步不能固定换算0.5 h。原文周期时长文字与1008 h/3076 cycles存在明显表达歧义，保留不补写。〔PARTIALLY_VERIFIED；PDF p9 Data preprocessing; p8 dataset〕
6. **主要模型**：原生TFT的LSTM、GRN/GLU、VSN及interpretable multi-head attention。〔FULL_TEXT_VERIFIED；PDF pp5–8 Figs3–4 Eqs9–21〕
7. **方法创新点**：将不同时间可用性协变量接入TFT并分析变量重要性与attention；PEMFC中使用TFT、门控与工况解释已有明确先例。〔FULL_TEXT_VERIFIED；PDF p3贡献; pp6–8 architecture; pp11–12 interpretation〕
8. **对比模型**：LSTM、GRU、LSTM-attention、GRU-attention；作者称对照结果来自已有研究，不能自动当作完全统一重训练预算。〔FULL_TEXT_VERIFIED；PDF p10 Tables5–6; p12 Comparison of different models〕
9. **模块消融方式**：Tables5–6为模型对照，Tables7–8是变量重要性；未在本文定位到F2式LSTM/CNN双路径gate独立训练消融，不能把变量重要性当作模块因果收益。〔FULL_TEXT_VERIFIED；PDF pp10–12 Tables5–8 Figs9–12〕
10. **不同工况分析**：Fig8分9个电流负载级展示测量/预测曲线；不等同本项目按过去48步波动阈值分组。〔FULL_TEXT_VERIFIED；PDF p9 Fig8; p11 Performance discussion〕
11. **稳健性和泛化验证**：单动态单电池6:4顺序切分；全文未明确给出独立多训练seed分布/跨电堆验证。表4 hidden最优154超出列示16–128，Train-only minmax拟合范围不够明确。〔FULL_TEXT_VERIFIED；PDF p8 Table4; p9 preprocessing; pp10–12 results〕
12. **实际评价指标**：正式Tables5–6列RMSE/MSE/MAE/R²；文字/公式23–26含MAPE且名称有混用，不能把未列成结果表的MAPE虚构为已报告数值。〔FULL_TEXT_VERIFIED；PDF p10 Tables5–6; p11 Evaluation Eqs23–26〕
13. **核心图表类型**：Fig3原生结构、Fig4历史/未来已知接口、Fig6训练流程、Fig8九负载曲线、Fig9模型RMSE、Figs10–11变量重要性、Fig12时间attention；Tables5–8结果与重要性。〔FULL_TEXT_VERIFIED；PDF pp5–12〕
14. **论文章节结构**：1 Introduction; 2 Prediction methods; 3 Interpretable degradation prediction of PEM fuel cells; 4 Results and discussions; 5 Conclusion。〔FULL_TEXT_VERIFIED；PDF p4组织段及pp4,8,11–12 headings〕
15. **可借鉴做法**：借鉴原生TFT模块与新增模块的明确划分、历史/未来可用接口图、分负载呈现和解释性证据单独陈述；注意相关性不升级为电化学因果。〔REVIEWER_SYNTHESIS；依据PDF Figs3–4,8,10–12; Tables2–3〕
16. **不能直接借鉴的原因**：该任务允许作者声明为future-known的运行参数，本稿不使用未来真实工况；采样/目标/跨度不同，不能用其单步误差与本稿12 h多步误差排序。解释性非门控独立精度增益。〔REVIEWER_SYNTHESIS；依据PDF pp8–12; 本稿history-only48→24协议〕
17. **具体来源索引**：本轮实际提取通读14页；渲染核对p5 Fig3、p8 Tables2–4、p10 Tables5–6、p11 Tables7–8；作者姓名保留出版社/首页印刷顺序，不擅改姓氏。〔FULL_TEXT_VERIFIED；runtime/core/P2_fulltext.txt; P2_p5.png/P2_p8.png/P2_p10.png/P2_p11.png〕

### P3 — Chongyang Tu（2024）

A model with high-precision on proton exchange membrane fuel cell performance degradation prediction based on temporal convolutional network-long short-term memory。International Journal of Hydrogen Energy 74，414–422。[DOI](https://doi.org/10.1016/j.ijhydene.2024.06.095)。

1. **研究问题与预测目标**：预测PEMFC性能/电压退化。〔PUBLISHER_ABSTRACT_VERIFIED；Official Abstract〕
2. **数据来源及工况类型**：IEEE2014 PHM数据及2673 h自测动态循环负载数据；出版社方法片段说明自测5 kW堆、10单电池、300 cm²。〔PUBLISHER_SNIPPET_VERIFIED；Official Abstract; section snippet PEMFC system configuration〕
3. **输入变量**：UNKNOWN：摘要提到原始数据/电压退化，完整模型输入变量与未来工况接口未获全文核实。〔UNKNOWN；Official Abstract scope only〕
4. **预测方式**：UNKNOWN：无法从TCN-LSTM名称确认单步、递归多步还是直接多步。〔UNKNOWN；Full text unavailable〕
5. **预测步长及实际物理时间**：UNKNOWN：2673 h是试验总长，不是预测跨度；50%训练长度是切分描述，不能换算为直接预测horizon。〔UNKNOWN；Official Abstract〕
6. **主要模型**：串联TCN高层特征后由LSTM提取长时关联的混合模型。〔PUBLISHER_ABSTRACT_VERIFIED；Official Abstract〕
7. **方法创新点**：以TCN局部/高层时间特征与LSTM长时相关结合；非本稿同一TFT内并行历史特征融合。具体容量和实现UNKNOWN。〔PUBLISHER_ABSTRACT_VERIFIED；Official Abstract〕
8. **对比模型**：UNKNOWN：摘要称与已有先进算法比较，所比较模型完整名称/同预算公平性需全文。〔UNKNOWN；Official Abstract; full comparison tables unavailable〕
9. **模块消融方式**：UNKNOWN：未获全文，不得称没有消融。〔UNKNOWN；Full text unavailable〕
10. **不同工况分析**：公开PHM及自测动态循环条件均验证；具体按负载/历史波动分组及分组表UNKNOWN。〔PUBLISHER_ABSTRACT_VERIFIED；Official Abstract〕
11. **稳健性和泛化验证**：两个公开PHM堆和另一个自测动态堆；摘要提及50%训练长度验证。重复训练seed及迁移设置UNKNOWN，不能直接称跨堆泛化已被证明。〔PUBLISHER_ABSTRACT_VERIFIED；Official Abstract〕
12. **实际评价指标**：RMSE、MAE、MAPE、R²可由出版社Evaluation index片段确认。具体统一单位、尾误差与重复统计UNKNOWN。〔PUBLISHER_SNIPPET_VERIFIED；Official section snippet Evaluation index of prediction accuracy〕
13. **核心图表类型**：Fig1试验工况、Fig2a因果卷积及Table1运行条件在公开片段中有指向；完整结果表图编号/版面UNKNOWN，不能臆造。〔PUBLISHER_SNIPPET_VERIFIED；Official PEMFC system configuration; TCN model snippets〕
14. **论文章节结构**：仅可核对Introduction、PEMFC system configuration、TCN model、Evaluation index、Conclusion等片段；完整章序UNKNOWN。〔PUBLISHER_SNIPPET_VERIFIED；Official section snippets〕
15. **可借鉴做法**：借鉴独立动态数据与不同训练长度的验证逻辑，比较局部卷积和长时记忆的互补性；这只是建议，不代表本轮要训练。〔REVIEWER_SYNTHESIS；依据Official Abstract〕
16. **不能直接借鉴的原因**：缺全文不能复刻输入/步长/基线公平性；自测2673 h不是RealHyFC，未据摘要把其RMSE与本稿排名。TCN+LSTM和动态工况均非空白。〔REVIEWER_SYNTHESIS；依据Official Abstract; access limitations〕
17. **具体来源索引**：官方摘要/有限方法片段+Crossref书目信息；ScienceDirect直接open403，Elsevier公开API仅coredata/openaccess=false；无本地PDF，未声称全文阅读。〔PUBLISHER_SNIPPET_VERIFIED；Official URL; https://api.crossref.org/works/10.1016/j.ijhydene.2024.06.095; runtime/core/P3_publisher.xml〕

### P4 — Jia-Hong Chou（2025）

Future prediction on the remaining useful life of proton exchange membrane fuel using temporal fusion transformer model。Renewable Energy 247，123019。[DOI](https://doi.org/10.1016/j.renene.2025.123019)。

1. **研究问题与预测目标**：以堆总电压滚动预测至设定阈值以推算条件RUL，分析起点和步长。不是本稿固定未来12 h ΔV任务。〔FULL_TEXT_VERIFIED；PDF pp2–4 §§2–3; p10 §4 Eq37〕
2. **数据来源及工况类型**：IEEE2014 PHM两台5-cell/100 cm²堆；FC1恒70 A，FC2以70 A为基准5 kHz纹波；设3个阈值/起点场景。〔FULL_TEXT_VERIFIED；PDF pp2–4 §§2.1–2.2 Tables1–3〕
3. **输入变量**：过去选出的流量/冷却温度/电流等历史特征；未来已知侧图示cycle number、文字为hour index。完整未知工况滚动闭合细节未给足，不能自行补造。〔PARTIALLY_VERIFIED；PDF p2说明; p7 Table4; p8 Fig7; pp10–11 §4〕
4. **预测方式**：序列预测块后更新Sp的step-ahead rolling，预测继续至EOL阈值；与固定origin一次输出24步不同。未完全认证每次滚动未知协变量如何闭合。〔PARTIALLY_VERIFIED；PDF p10 §4; p10 Fig9; pp11–12 rolling description〕
5. **预测步长及实际物理时间**：实际分析1/3/5/7/9步且文字明确1–9 h；Table8及Fig12题注误写12种steps，表中为5种。不得将题注数字当12个实验。〔FULL_TEXT_VERIFIED；PDF pp11–13 Table8 Fig12; p14 Conclusions〕
6. **主要模型**：改进TFT，以BiLSTM替换encoder/decoder LSTM，保留VSN/GRN/attention/quantile输出。图和方法为双侧替换，摘要/结论部分仅说encoder，应保留差异。〔FULL_TEXT_VERIFIED；PDF p8 Fig7; p9 §3.2.2; p10 §§3.2.3–3.2.5; p14结论〕
7. **方法创新点**：BW/DWBW确定训练区间与预测起点、RF特征筛选/PCA可视化、TPE调参、BiLSTM增强TFT和分位数输出。〔FULL_TEXT_VERIFIED；PDF pp3–11 §§3–4, Figs2–10, Tables4–5〕
8. **对比模型**：Tables10–11含native TFT及improved TFT，同时列已有研究LSTM/BiLSTM/ESN。历史文献结果与当前重训练结果须区分。〔FULL_TEXT_VERIFIED；PDF p13 Tables10–11及正文〕
9. **模块消融方式**：Tables6–7有/无特征选择，Tables10–11有native TFT对照。未认证等参数容量、相同调参预算与独立多seed；不可声称完全无消融。〔FULL_TEXT_VERIFIED；PDF pp11–13 Tables6–7,10–11〕
10. **不同工况分析**：FC1和FC2两纹波场景、不同起点/训练长度，Fig12展示恢复/复杂段；不是多种真实动态驾驶工况或Train历史阈值分组。〔FULL_TEXT_VERIFIED；PDF pp11–13 Figs11–12 Tables6,8,10–11〕
11. **稳健性和泛化验证**：3个场景，5种步长，FC1不同训练长度及FC2比较；输出10/50/90%分位预测。未报告独立多seed分布，quantile窄区间不等于校准后的可靠覆盖率。〔FULL_TEXT_VERIFIED；PDF p10 §3.2.4; pp11–14 Tables6–11〕
12. **实际评价指标**：主要RUL AE与RE%，Tables6–11；Table9训练/预测时间；quantile输出。其RUL相对误差不是电压RMSE，不能跨指标比精度。〔FULL_TEXT_VERIFIED；PDF pp10–13 Eqs37–39 Tables6–11〕
13. **核心图表类型**：Fig2流程、Fig7架构、Fig10超参并行坐标、Fig11单步滚动曲线、Fig12各步长曲线、Fig13特征重要性；Table8步长指标、Tables10–11原生TFT/改进TFT对照。〔FULL_TEXT_VERIFIED；PDF pp5,8,11–14〕
14. **论文章节结构**：1 Introduction; 2 Dataset description; 3 Proposed method; 4 Future RUL prediction results and discussion; 5 Conclusions。〔FULL_TEXT_VERIFIED；PDF p2组织段及pp2,3,10,14 headings〕
15. **可借鉴做法**：借鉴原生/改进TFT明确对照、预测长度敏感性、数据接口图和模块对照表。本文可借鉴实验证据形式，无需复制其RUL任务。〔REVIEWER_SYNTHESIS；依据PDF Tables6–11 Figs7,11–12〕
16. **不能直接借鉴的原因**：Eq3利用后续极化点替换异常值，不满足本项目在线因果预处理；EOL/Sp文字与表有冲突；其原生TFT对照及改进TFT已有先例，不能宣称首次改进TFT或误差尺度可直接比较。〔REVIEWER_SYNTHESIS；PDF p4 Eq3/Table3,p6 Sp叙述,p11正文及Tables6–7,p13Tables10–11〕
17. **具体来源索引**：本轮15页提取通读，视觉核对p8 Fig7、p11 Tables5–7和p13 Tables10–11；原文Sp/EOL/attention范围矛盾保留。〔FULL_TEXT_VERIFIED；runtime/core/P4_fulltext.txt; P4_p8.png/P4_p11.png/P4_p13.png〕

### P5 — Yurdagül Benteşen Yakut（2025）

Short-term voltage loss prediction in PEMFCs using single-step and multi-step LSTM models。International Journal of Hydrogen Energy 144，652–664。[DOI](https://doi.org/10.1016/j.ijhydene.2025.02.317)。

1. **研究问题与预测目标**：比较单步和多步LSTM的短期堆电压/电压损失预测。〔PUBLISHER_ABSTRACT_VERIFIED；Official Abstract/Highlights〕
2. **数据来源及工况类型**：摘要区分FC1稳态与FC2动态，训练前650 h/850 h并测试剩余及动态片段。完整数据来源/电池数需全文；不能仅凭FC1/FC2名字强行认定实验同于某PHM版本。〔PUBLISHER_ABSTRACT_VERIFIED；Official Abstract〕
3. **输入变量**：UNKNOWN：摘要没有足以认证的完整输入列表、平滑、未来协变量及scaler信息。〔UNKNOWN；Full text unavailable〕
4. **预测方式**：摘要明确single-step与multi-step；direct/recursive多步实现UNKNOWN。〔PUBLISHER_ABSTRACT_VERIFIED；Official Abstract〕
5. **预测步长及实际物理时间**：摘要列5/1、25/1、25/5、1000/50配置；比值含义/重采样间隔未获全文，不能将50当50 h，也不能自动等同本稿48→24。〔PARTIALLY_VERIFIED；Official Abstract/Highlights〕
6. **主要模型**：Single-step及multi-step LSTM。〔PUBLISHER_ABSTRACT_VERIFIED；Official Abstract〕
7. **方法创新点**：比较短期电压预测配置和动态片段；具体相对原生LSTM的结构改动UNKNOWN。不得称首次多步电压预测。〔PUBLISHER_ABSTRACT_VERIFIED；Official Abstract/Highlights〕
8. **对比模型**：摘要可确认单步/多步配置互比；其他基线模型列表UNKNOWN。〔PUBLISHER_ABSTRACT_VERIFIED；Official Abstract〕
9. **模块消融方式**：UNKNOWN：未获全文，无法判断是否存在模块消融或其设计。〔UNKNOWN；Full text unavailable〕
10. **不同工况分析**：动态片段分别随机测试。片段抽样、重复次数和冻结规则UNKNOWN；摘要同时称FC2前850 h训练与dynamic FC2 exclusively testing，需全文消歧，不能选择有利解释。〔PARTIALLY_VERIFIED；Official Abstract〕
11. **稳健性和泛化验证**：摘要称650 h和850 h训练长度差异影响较小；独立多seed、跨堆迁移与真正未来边界UNKNOWN。〔PUBLISHER_ABSTRACT_VERIFIED；Official Abstract〕
12. **实际评价指标**：MAPE、RMSE、R²。完整单位/逐步误差/尾部误差和数值表UNKNOWN。〔PUBLISHER_ABSTRACT_VERIFIED；Official Abstract〕
13. **核心图表类型**：UNKNOWN：没有合法取得全文结果表图，不能凭摘要生成图号或认定缺少图。〔UNKNOWN；Full text unavailable〕
14. **论文章节结构**：UNKNOWN：未取得完整目录/章节。〔UNKNOWN；Full text unavailable〕
15. **可借鉴做法**：借鉴预测窗口配置明确对比与动态片段测试；本稿采用预先固定典型窗口并保留不利窗口，减少只展示有利片段风险。〔REVIEWER_SYNTHESIS；依据Official Abstract/Highlights〕
16. **不能直接借鉴的原因**：未知预测物理跨度和多步实现，不能直接比较RMSE；随机片段验证不自动等于本稿时间顺序Validation，不能用其结果替代本稿独立泛化评价。〔REVIEWER_SYNTHESIS；依据Official Abstract及访问边界〕
17. **具体来源索引**：官方摘要/Highlights+Crossref作者书目；ScienceDirect open403，Elsevier公开API仅coredata/openaccess=false；无本地全文。检索结果混入cited-by的Wei/Crossformer摘要，已排除、不归属本篇。〔PUBLISHER_ABSTRACT_VERIFIED；Official URL; https://api.crossref.org/works/10.1016/j.ijhydene.2025.02.317; runtime/core/P5_publisher.xml〕

## 访问与证据边界

- 源PDF为仓库参考论文目录中的既有受版权保护文件，仅只读；本轮不复制到论文或提交其提取内容。runtime/core保存全文提取和页面渲染供本机核验，须排除Git。
- P3/P5直接页面返回403是访问限制，不能称论文没有全文，也不能据此否定其未公开实验。
- ScienceDirect检索页面可能连带展示cited-by文章。P5页面中Crossformer/physical constraints段落属于其他论文，已排除，不能误当LSTM本文结论。
- 目录/参数/指标原文矛盾单独记录：P1印刷RMSE公式；P2 hidden size154超范围、周期文字；P4 Sp/EOL、五种steps题注称十二种、encoder/decoder替换范围。仅说明可重复性边界，不推断作者代码或主观行为。
- 本轮文件均为中文转述及定位，无超过25词的逐源直接引文；不复用竞争论文图像。
