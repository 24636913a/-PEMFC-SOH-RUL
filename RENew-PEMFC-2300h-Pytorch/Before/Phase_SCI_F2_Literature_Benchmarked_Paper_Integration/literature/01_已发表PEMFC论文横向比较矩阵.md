# 已发表PEMFC论文横向比较矩阵

核验日期2026-10-17；定向六篇核心+三篇辅助，另核对原TFT与RealHyFC数据论文。不是系统综述，不支持全领域排他性首创。7篇本地全文本轮直接重核并查看关键图表；Tu(P3)及Yakut(P5)未取得正文，仅官方摘要/有限片段，UNKNOWN不是未做。所有页码为PDF从1起的页序。每字段等级与定位保存在长表CSV/原JSON；不把不同论文RMSE直接排名。

|ID|作者|年份|期刊|DOI|证据状态|
|---|---|---|---|---|---|
|P1|Khaled Benaggoune; Meiling Yue; Samir Jemei; Noureddine Zerhouni|2022|Applied Energy|10.1016/j.apenergy.2022.118835|本轮原PDF方法/实验全文重核+关键页面视觉检查|
|P2|Li Hongwei; Qiao Binxin; Hou Zhicheng; Liu Junnan; Yang Yue; Lu Guolong|2023|International Journal of Hydrogen Energy|10.1016/j.ijhydene.2023.03.316|本轮原PDF方法/实验全文重核+关键页面视觉检查|
|P3|Chongyang Tu; Fen Zhou; Mu Pan|2024|International Journal of Hydrogen Energy|10.1016/j.ijhydene.2024.06.095|官方摘要/有限片段；全文不可访问|
|P4|Jia-Hong Chou; Fu-Kwun Wang|2025|Renewable Energy|10.1016/j.renene.2025.123019|本轮原PDF方法/实验全文重核+关键页面视觉检查|
|P5|Yurdagül Benteşen Yakut|2025|International Journal of Hydrogen Energy|10.1016/j.ijhydene.2025.02.317|官方摘要/有限片段；全文不可访问|
|P6|Lin Wei; Shangrui Gao; Zonghai Chen; Yujie Wang; Zhendong Sun|2026|Journal of Power Sources|10.1016/j.jpowsour.2026.239941|本轮原PDF方法/实验全文重核+关键页面视觉检查|
|A1|Jian Jiang; Dongsheng Du; Lin Su; Yifan Du|2026|International Journal of Hydrogen Energy|10.1016/j.ijhydene.2026.153481|本轮原PDF方法/实验全文重核+关键页面视觉检查|
|A2|Fengyan Yi; Xing Shu; Jiaming Zhou; Jinming Zhang; Chunxiao Feng; Hongtao Gong; Caizhi Zhang; Wenhao Yu|2025|International Journal of Hydrogen Energy|10.1016/j.ijhydene.2025.02.302|本轮原PDF方法/实验全文重核+关键页面视觉检查|
|A3|Taehwan Ko; Dukyong Kim; Jaewoong Park; Seung Hwan Lee|2025|Applied Energy|10.1016/j.apenergy.2025.125318|本轮原PDF方法/实验全文重核+关键页面视觉检查|

## P1 · Khaled Benaggoune等 (2022)

*A data-driven method for multi-step-ahead prediction and long-term prognostics of proton exchange membrane fuel cell*，Applied Energy 313，118835。[DOI](https://doi.org/10.1016/j.apenergy.2022.118835)。

证据：本轮原PDF方法/实验全文重核+关键页面视觉检查。

|核对项|已核实内容/未知边界|证据等级与定位|
|---|---|---|
|研究问题与目标|以堆电压为性能观测，区分短期多步预测与无新测量反馈的长期预后。|FULL_TEXT_VERIFIED; PDF pp4–5 §2.4; p6 Fig7|
|数据及工况|FCLAB/PHM两台5-cell、100 cm²堆：恒流与5 kHz小纹波；另有PRAGMA 15-cell、33.625 cm²恒流堆。并非RealHyFC动态负载轨迹。|FULL_TEXT_VERIFIED; PDF p4 §2.3, Tables2–3; pp4–5 Figs5–6|
|输入信息|短期直接使用历史堆电压；长期条件分支使用Train电压趋势分解及由Train拟合ARIMA产生的未来趋势，不是未来实测工况。|FULL_TEXT_VERIFIED; PDF p7 §3.4 step1; pp10–12 §4.1–4.2 Fig16|
|单步/多步方式|短期每块预测n点，再以届时新获得真实测量更新下一块；长期将预测值接回输入直至阈值。短期块内direct/recursive实现细节不能完全认证。|PARTIALLY_VERIFIED; PDF p6 Fig7; p8 §3.5.2; p11 §4.3.2|
|跨度及物理时间|正式比较5/10/15/24步；图横轴为小时，但全文未充分定义平滑后各预测步的统一时间间隔，不能擅将24步换算为本项目12 h。输入17/20/25及attention尺寸12/32/12都不是预测跨度。|PARTIALLY_VERIFIED; PDF p8 Table4 §3.5.2; pp8–11 Figs12–14 Table5; Appendix pp13–14|
|主要模型|带skip connection和Luong attention的两层膨胀CNN；长期为ARIMA引导的conditional CNN。|FULL_TEXT_VERIFIED; PDF pp5–7 §§3.1–3.4 Eqs4–9 Fig10; pp10–12 §4 Fig16|
|方法创新|因果膨胀卷积扩大时间感受野，attention筛选历史特征；长期另加条件趋势路径。双路径/多尺度卷积/attention本身不能再宣称首次用于PEMFC。|FULL_TEXT_VERIFIED; PDF pp6–7 Figs8–10; pp10–12 Fig16 Eqs13–14|
|比较模型|短期MLP、LSTM、BiLSTM；长期ConCNN、DilCNN、ESN、LSTM。|FULL_TEXT_VERIFIED; PDF p8 §3.5.2; p11 Table5; p12 Table6|
|模块消融|Table6比较有无条件分支；未在该全文中定位到dilation、skip、attention分别移除且等容量等调参预算的逐模块消融，不能把模型间比较当作这类独立消融。|FULL_TEXT_VERIFIED; PDF pp8–12 Tables5–7; §3.5 and §4.3|
|不同工况分析|三堆/恒流与纹波分别给曲线和指标；讨论启停、恢复以及后段困难区间。不是Train阈值定义的历史工况分组误差。|FULL_TEXT_VERIFIED; PDF pp8–9 Figs12–14 §3.5.2; pp11–13 Fig17|
|稳健性及泛化|跨三个实验数据、多个起点和预测跨度；长期MC dropout重复50次及箱线图。50次dropout不等于50次独立训练seed，也未据此认证跨堆迁移。|FULL_TEXT_VERIFIED; PDF pp11–13 §4.3.2 Fig17; p12 Table6|
|评价指标|短期报告MAPE和名为RMSE的指标；p7 Eq11印刷为归一化绝对误差开方，非标准RMSE。实现UNKNOWN。长期RUL绝对偏差/箱线图及训练推理时间。|FULL_TEXT_VERIFIED; PDF p7 Eqs10–11; pp11–14 Tables5–7 Fig17|
|核心图表|任务信息流Fig7、卷积/attention结构Fig10、多horizon预测叠图Figs12–14、跨堆×horizon指标Table5、长期条件结构Fig16、起点箱线图Fig17、时间开销Table7。|FULL_TEXT_VERIFIED; PDF pp6–14|
|章节结构|1 Introduction; 2 PEMFC degradation and performance prediction; 3 Multi-step-ahead prediction based on dilated CNN; 4 Prognostics based on conditional CNN; 5 Conclusion; Appendix Data preparation。|FULL_TEXT_VERIFIED; PDF p3组织段; pp3,5,9,12–14 headings|
|本项目可借鉴|借鉴明确区分块预测和滚动反馈、跨度×模型误差表、困难恢复区段解释、计算开销说明。保留本稿单一动态电压目标，不复制RUL章节。|REVIEWER_SYNTHESIS; 依据PDF p6 Fig7,p11 Table5,p14 Table7; 本任务48→24协议|
|不能直接照搬|数据/堆数/电压尺度/平滑/预测反馈条件不同；附录预处理整体边界不充分；印刷RMSE公式异于标准。不得跨文献误差排名，也不能将其ARIMA条件分支等同F2 encoder内LSTM/CNN融合。|REVIEWER_SYNTHESIS; 依据PDF pp4,6–8,10–14|
|定位汇总|本轮本地全文15页实际提取通读，关键p6 Fig7、p8 Table4/Fig12、p11 Table5已渲染核对；Phase O仅作交叉核对。|FULL_TEXT_VERIFIED; runtime/core/P1_fulltext.txt; P1_p6.png/P1_p8.png/P1_p11.png|

## P2 · Li Hongwei等 (2023)

*An interpretable data-driven method for degradation prediction of proton exchange membrane fuel cells based on temporal fusion transformer and covariates*，International Journal of Hydrogen Energy 48，25958-25971。[DOI](https://doi.org/10.1016/j.ijhydene.2023.03.316)。

证据：本轮原PDF方法/实验全文重核+关键页面视觉检查。

|核对项|已核实内容/未知边界|证据等级与定位|
|---|---|---|
|研究问题与目标|使用TFT和协变量预测动态运行单电池电压，提供变量重要性及时间attention解释。|FULL_TEXT_VERIFIED; PDF pp3–4 contributions; pp8–12 methods/results|
|数据及工况|同济大学已发表的汽车单PEMFC动态耐久试验：25 cm²，3076循环约1008 h，每周期35次电流变化、9负载级；周期性停机后有性能恢复。|FULL_TEXT_VERIFIED; PDF pp8–9 dataset/preprocessing; p9 Fig8|
|输入信息|作者称17项协变量，Tables2–3区分过去测量和未来已知；后者列hour/cycle position、入口压力/温度及氢氧流量。其未来可知的排程/控制命令证明未充分报告，不能断言作弊，也不能视为history-only。|FULL_TEXT_VERIFIED; PDF p6 Table1; p8 Tables2–3; p10 parameter definition|
|单步/多步方式|正式设置moving window5、forecast horizon1，即5→1单步；长时间曲线是逐步评价，不能据TFT通用多horizon能力假定本文已完成直接24步。|FULL_TEXT_VERIFIED; PDF p9 Data preprocessing; p10 step4; pp11–12 Fig8|
|跨度及物理时间|5输入点预测1点；按每电流平台倒数第5个原始记录采样、每周期35点，各平台时长不等，单步不能固定换算0.5 h。原文周期时长文字与1008 h/3076 cycles存在明显表达歧义，保留不补写。|PARTIALLY_VERIFIED; PDF p9 Data preprocessing; p8 dataset|
|主要模型|原生TFT的LSTM、GRN/GLU、VSN及interpretable multi-head attention。|FULL_TEXT_VERIFIED; PDF pp5–8 Figs3–4 Eqs9–21|
|方法创新|将不同时间可用性协变量接入TFT并分析变量重要性与attention；PEMFC中使用TFT、门控与工况解释已有明确先例。|FULL_TEXT_VERIFIED; PDF p3贡献; pp6–8 architecture; pp11–12 interpretation|
|比较模型|LSTM、GRU、LSTM-attention、GRU-attention；作者称对照结果来自已有研究，不能自动当作完全统一重训练预算。|FULL_TEXT_VERIFIED; PDF p10 Tables5–6; p12 Comparison of different models|
|模块消融|Tables5–6为模型对照，Tables7–8是变量重要性；未在本文定位到F2式LSTM/CNN双路径gate独立训练消融，不能把变量重要性当作模块因果收益。|FULL_TEXT_VERIFIED; PDF pp10–12 Tables5–8 Figs9–12|
|不同工况分析|Fig8分9个电流负载级展示测量/预测曲线；不等同本项目按过去48步波动阈值分组。|FULL_TEXT_VERIFIED; PDF p9 Fig8; p11 Performance discussion|
|稳健性及泛化|单动态单电池6:4顺序切分；全文未明确给出独立多训练seed分布/跨电堆验证。表4 hidden最优154超出列示16–128，Train-only minmax拟合范围不够明确。|FULL_TEXT_VERIFIED; PDF p8 Table4; p9 preprocessing; pp10–12 results|
|评价指标|正式Tables5–6列RMSE/MSE/MAE/R²；文字/公式23–26含MAPE且名称有混用，不能把未列成结果表的MAPE虚构为已报告数值。|FULL_TEXT_VERIFIED; PDF p10 Tables5–6; p11 Evaluation Eqs23–26|
|核心图表|Fig3原生结构、Fig4历史/未来已知接口、Fig6训练流程、Fig8九负载曲线、Fig9模型RMSE、Figs10–11变量重要性、Fig12时间attention；Tables5–8结果与重要性。|FULL_TEXT_VERIFIED; PDF pp5–12|
|章节结构|1 Introduction; 2 Prediction methods; 3 Interpretable degradation prediction of PEM fuel cells; 4 Results and discussions; 5 Conclusion。|FULL_TEXT_VERIFIED; PDF p4组织段及pp4,8,11–12 headings|
|本项目可借鉴|借鉴原生TFT模块与新增模块的明确划分、历史/未来可用接口图、分负载呈现和解释性证据单独陈述；注意相关性不升级为电化学因果。|REVIEWER_SYNTHESIS; 依据PDF Figs3–4,8,10–12; Tables2–3|
|不能直接照搬|该任务允许作者声明为future-known的运行参数，本稿不使用未来真实工况；采样/目标/跨度不同，不能用其单步误差与本稿12 h多步误差排序。解释性非门控独立精度增益。|REVIEWER_SYNTHESIS; 依据PDF pp8–12; 本稿history-only48→24协议|
|定位汇总|本轮实际提取通读14页；渲染核对p5 Fig3、p8 Tables2–4、p10 Tables5–6、p11 Tables7–8；作者姓名保留出版社/首页印刷顺序，不擅改姓氏。|FULL_TEXT_VERIFIED; runtime/core/P2_fulltext.txt; P2_p5.png/P2_p8.png/P2_p10.png/P2_p11.png|

## P3 · Chongyang Tu等 (2024)

*A model with high-precision on proton exchange membrane fuel cell performance degradation prediction based on temporal convolutional network-long short-term memory*，International Journal of Hydrogen Energy 74，414-422。[DOI](https://doi.org/10.1016/j.ijhydene.2024.06.095)。

证据：官方摘要/有限片段；全文不可访问。

|核对项|已核实内容/未知边界|证据等级与定位|
|---|---|---|
|研究问题与目标|预测PEMFC性能/电压退化。|PUBLISHER_ABSTRACT_VERIFIED; Official Abstract|
|数据及工况|IEEE2014 PHM数据及2673 h自测动态循环负载数据；出版社方法片段说明自测5 kW堆、10单电池、300 cm²。|PUBLISHER_SNIPPET_VERIFIED; Official Abstract; section snippet PEMFC system configuration|
|输入信息|UNKNOWN：摘要提到原始数据/电压退化，完整模型输入变量与未来工况接口未获全文核实。|UNKNOWN; Official Abstract scope only|
|单步/多步方式|UNKNOWN：无法从TCN-LSTM名称确认单步、递归多步还是直接多步。|UNKNOWN; Full text unavailable|
|跨度及物理时间|UNKNOWN：2673 h是试验总长，不是预测跨度；50%训练长度是切分描述，不能换算为直接预测horizon。|UNKNOWN; Official Abstract|
|主要模型|串联TCN高层特征后由LSTM提取长时关联的混合模型。|PUBLISHER_ABSTRACT_VERIFIED; Official Abstract|
|方法创新|以TCN局部/高层时间特征与LSTM长时相关结合；非本稿同一TFT内并行历史特征融合。具体容量和实现UNKNOWN。|PUBLISHER_ABSTRACT_VERIFIED; Official Abstract|
|比较模型|UNKNOWN：摘要称与已有先进算法比较，所比较模型完整名称/同预算公平性需全文。|UNKNOWN; Official Abstract; full comparison tables unavailable|
|模块消融|UNKNOWN：未获全文，不得称没有消融。|UNKNOWN; Full text unavailable|
|不同工况分析|公开PHM及自测动态循环条件均验证；具体按负载/历史波动分组及分组表UNKNOWN。|PUBLISHER_ABSTRACT_VERIFIED; Official Abstract|
|稳健性及泛化|两个公开PHM堆和另一个自测动态堆；摘要提及50%训练长度验证。重复训练seed及迁移设置UNKNOWN，不能直接称跨堆泛化已被证明。|PUBLISHER_ABSTRACT_VERIFIED; Official Abstract|
|评价指标|RMSE、MAE、MAPE、R²可由出版社Evaluation index片段确认。具体统一单位、尾误差与重复统计UNKNOWN。|PUBLISHER_SNIPPET_VERIFIED; Official section snippet Evaluation index of prediction accuracy|
|核心图表|Fig1试验工况、Fig2a因果卷积及Table1运行条件在公开片段中有指向；完整结果表图编号/版面UNKNOWN，不能臆造。|PUBLISHER_SNIPPET_VERIFIED; Official PEMFC system configuration; TCN model snippets|
|章节结构|仅可核对Introduction、PEMFC system configuration、TCN model、Evaluation index、Conclusion等片段；完整章序UNKNOWN。|PUBLISHER_SNIPPET_VERIFIED; Official section snippets|
|本项目可借鉴|借鉴独立动态数据与不同训练长度的验证逻辑，比较局部卷积和长时记忆的互补性；这只是建议，不代表本轮要训练。|REVIEWER_SYNTHESIS; 依据Official Abstract|
|不能直接照搬|缺全文不能复刻输入/步长/基线公平性；自测2673 h不是RealHyFC，未据摘要把其RMSE与本稿排名。TCN+LSTM和动态工况均非空白。|REVIEWER_SYNTHESIS; 依据Official Abstract; access limitations|
|定位汇总|官方摘要/有限方法片段+Crossref书目信息；ScienceDirect直接open403，Elsevier公开API仅coredata/openaccess=false；无本地PDF，未声称全文阅读。|PUBLISHER_SNIPPET_VERIFIED; Official URL; https://api.crossref.org/works/10.1016/j.ijhydene.2024.06.095; runtime/core/P3_publisher.xml|

## P4 · Jia-Hong Chou等 (2025)

*Future prediction on the remaining useful life of proton exchange membrane fuel using temporal fusion transformer model*，Renewable Energy 247，123019。[DOI](https://doi.org/10.1016/j.renene.2025.123019)。

证据：本轮原PDF方法/实验全文重核+关键页面视觉检查。

|核对项|已核实内容/未知边界|证据等级与定位|
|---|---|---|
|研究问题与目标|以堆总电压滚动预测至设定阈值以推算条件RUL，分析起点和步长。不是本稿固定未来12 h ΔV任务。|FULL_TEXT_VERIFIED; PDF pp2–4 §§2–3; p10 §4 Eq37|
|数据及工况|IEEE2014 PHM两台5-cell/100 cm²堆；FC1恒70 A，FC2以70 A为基准5 kHz纹波；设3个阈值/起点场景。|FULL_TEXT_VERIFIED; PDF pp2–4 §§2.1–2.2 Tables1–3|
|输入信息|过去选出的流量/冷却温度/电流等历史特征；未来已知侧图示cycle number、文字为hour index。完整未知工况滚动闭合细节未给足，不能自行补造。|PARTIALLY_VERIFIED; PDF p2说明; p7 Table4; p8 Fig7; pp10–11 §4|
|单步/多步方式|序列预测块后更新Sp的step-ahead rolling，预测继续至EOL阈值；与固定origin一次输出24步不同。未完全认证每次滚动未知协变量如何闭合。|PARTIALLY_VERIFIED; PDF p10 §4; p10 Fig9; pp11–12 rolling description|
|跨度及物理时间|实际分析1/3/5/7/9步且文字明确1–9 h；Table8及Fig12题注误写12种steps，表中为5种。不得将题注数字当12个实验。|FULL_TEXT_VERIFIED; PDF pp11–13 Table8 Fig12; p14 Conclusions|
|主要模型|改进TFT，以BiLSTM替换encoder/decoder LSTM，保留VSN/GRN/attention/quantile输出。图和方法为双侧替换，摘要/结论部分仅说encoder，应保留差异。|FULL_TEXT_VERIFIED; PDF p8 Fig7; p9 §3.2.2; p10 §§3.2.3–3.2.5; p14结论|
|方法创新|BW/DWBW确定训练区间与预测起点、RF特征筛选/PCA可视化、TPE调参、BiLSTM增强TFT和分位数输出。|FULL_TEXT_VERIFIED; PDF pp3–11 §§3–4, Figs2–10, Tables4–5|
|比较模型|Tables10–11含native TFT及improved TFT，同时列已有研究LSTM/BiLSTM/ESN。历史文献结果与当前重训练结果须区分。|FULL_TEXT_VERIFIED; PDF p13 Tables10–11及正文|
|模块消融|Tables6–7有/无特征选择，Tables10–11有native TFT对照。未认证等参数容量、相同调参预算与独立多seed；不可声称完全无消融。|FULL_TEXT_VERIFIED; PDF pp11–13 Tables6–7,10–11|
|不同工况分析|FC1和FC2两纹波场景、不同起点/训练长度，Fig12展示恢复/复杂段；不是多种真实动态驾驶工况或Train历史阈值分组。|FULL_TEXT_VERIFIED; PDF pp11–13 Figs11–12 Tables6,8,10–11|
|稳健性及泛化|3个场景，5种步长，FC1不同训练长度及FC2比较；输出10/50/90%分位预测。未报告独立多seed分布，quantile窄区间不等于校准后的可靠覆盖率。|FULL_TEXT_VERIFIED; PDF p10 §3.2.4; pp11–14 Tables6–11|
|评价指标|主要RUL AE与RE%，Tables6–11；Table9训练/预测时间；quantile输出。其RUL相对误差不是电压RMSE，不能跨指标比精度。|FULL_TEXT_VERIFIED; PDF pp10–13 Eqs37–39 Tables6–11|
|核心图表|Fig2流程、Fig7架构、Fig10超参并行坐标、Fig11单步滚动曲线、Fig12各步长曲线、Fig13特征重要性；Table8步长指标、Tables10–11原生TFT/改进TFT对照。|FULL_TEXT_VERIFIED; PDF pp5,8,11–14|
|章节结构|1 Introduction; 2 Dataset description; 3 Proposed method; 4 Future RUL prediction results and discussion; 5 Conclusions。|FULL_TEXT_VERIFIED; PDF p2组织段及pp2,3,10,14 headings|
|本项目可借鉴|借鉴原生/改进TFT明确对照、预测长度敏感性、数据接口图和模块对照表。本文可借鉴实验证据形式，无需复制其RUL任务。|REVIEWER_SYNTHESIS; 依据PDF Tables6–11 Figs7,11–12|
|不能直接照搬|Eq3利用后续极化点替换异常值，不满足本项目在线因果预处理；EOL/Sp文字与表有冲突；其原生TFT对照及改进TFT已有先例，不能宣称首次改进TFT或误差尺度可直接比较。|REVIEWER_SYNTHESIS; PDF p4 Eq3/Table3,p6 Sp叙述,p11正文及Tables6–7,p13Tables10–11|
|定位汇总|本轮15页提取通读，视觉核对p8 Fig7、p11 Tables5–7和p13 Tables10–11；原文Sp/EOL/attention范围矛盾保留。|FULL_TEXT_VERIFIED; runtime/core/P4_fulltext.txt; P4_p8.png/P4_p11.png/P4_p13.png|

## P5 · Yurdagül Benteşen Yakut (2025)

*Short-term voltage loss prediction in PEMFCs using single-step and multi-step LSTM models*，International Journal of Hydrogen Energy 144，652-664。[DOI](https://doi.org/10.1016/j.ijhydene.2025.02.317)。

证据：官方摘要/有限片段；全文不可访问。

|核对项|已核实内容/未知边界|证据等级与定位|
|---|---|---|
|研究问题与目标|比较单步和多步LSTM的短期堆电压/电压损失预测。|PUBLISHER_ABSTRACT_VERIFIED; Official Abstract/Highlights|
|数据及工况|摘要区分FC1稳态与FC2动态，训练前650 h/850 h并测试剩余及动态片段。完整数据来源/电池数需全文；不能仅凭FC1/FC2名字强行认定实验同于某PHM版本。|PUBLISHER_ABSTRACT_VERIFIED; Official Abstract|
|输入信息|UNKNOWN：摘要没有足以认证的完整输入列表、平滑、未来协变量及scaler信息。|UNKNOWN; Full text unavailable|
|单步/多步方式|摘要明确single-step与multi-step；direct/recursive多步实现UNKNOWN。|PUBLISHER_ABSTRACT_VERIFIED; Official Abstract|
|跨度及物理时间|摘要列5/1、25/1、25/5、1000/50配置；比值含义/重采样间隔未获全文，不能将50当50 h，也不能自动等同本稿48→24。|PARTIALLY_VERIFIED; Official Abstract/Highlights|
|主要模型|Single-step及multi-step LSTM。|PUBLISHER_ABSTRACT_VERIFIED; Official Abstract|
|方法创新|比较短期电压预测配置和动态片段；具体相对原生LSTM的结构改动UNKNOWN。不得称首次多步电压预测。|PUBLISHER_ABSTRACT_VERIFIED; Official Abstract/Highlights|
|比较模型|摘要可确认单步/多步配置互比；其他基线模型列表UNKNOWN。|PUBLISHER_ABSTRACT_VERIFIED; Official Abstract|
|模块消融|UNKNOWN：未获全文，无法判断是否存在模块消融或其设计。|UNKNOWN; Full text unavailable|
|不同工况分析|动态片段分别随机测试。片段抽样、重复次数和冻结规则UNKNOWN；摘要同时称FC2前850 h训练与dynamic FC2 exclusively testing，需全文消歧，不能选择有利解释。|PARTIALLY_VERIFIED; Official Abstract|
|稳健性及泛化|摘要称650 h和850 h训练长度差异影响较小；独立多seed、跨堆迁移与真正未来边界UNKNOWN。|PUBLISHER_ABSTRACT_VERIFIED; Official Abstract|
|评价指标|MAPE、RMSE、R²。完整单位/逐步误差/尾部误差和数值表UNKNOWN。|PUBLISHER_ABSTRACT_VERIFIED; Official Abstract|
|核心图表|UNKNOWN：没有合法取得全文结果表图，不能凭摘要生成图号或认定缺少图。|UNKNOWN; Full text unavailable|
|章节结构|UNKNOWN：未取得完整目录/章节。|UNKNOWN; Full text unavailable|
|本项目可借鉴|借鉴预测窗口配置明确对比与动态片段测试；本稿采用预先固定典型窗口并保留不利窗口，减少只展示有利片段风险。|REVIEWER_SYNTHESIS; 依据Official Abstract/Highlights|
|不能直接照搬|未知预测物理跨度和多步实现，不能直接比较RMSE；随机片段验证不自动等于本稿时间顺序Validation，不能用其结果替代本稿独立泛化评价。|REVIEWER_SYNTHESIS; 依据Official Abstract及访问边界|
|定位汇总|官方摘要/Highlights+Crossref作者书目；ScienceDirect open403，Elsevier公开API仅coredata/openaccess=false；无本地全文。检索结果混入cited-by的Wei/Crossformer摘要，已排除、不归属本篇。|PUBLISHER_ABSTRACT_VERIFIED; Official URL; https://api.crossref.org/works/10.1016/j.ijhydene.2025.02.317; runtime/core/P5_publisher.xml|

## P6 · Lin Wei等 (2026)

*A hybrid framework for PEMFC degradation prediction integrating Crossformer and physical constraints*，Journal of Power Sources 677，239941。[DOI](https://doi.org/10.1016/j.jpowsour.2026.239941)。

证据：本轮原PDF方法/实验全文重核+关键页面视觉检查。

|核对项|已核实内容/未知边界|证据等级与定位|
|---|---|---|
|研究问题与目标|多工况PEMFC电压退化预测；主要一阶段24窗MISO单步，另有多步实验；不是已验证标准SOH或RealHyFC真实RUL任务。|FULL_TEXT_RECHECKED_THIS_PHASE; PDF p7 §3.5/§4.2; p10 §5|
|数据及工况|FC1/FC2为PHM2014两个5cell、100 cm²堆，分别70 A稳态1140h与70 A±7 A、5 kHz纹波1020h；FC3为同济单电池3076动态周期约1008h，周期1180s、35平台。FC1/2清洗平滑后1h重采样，FC3每平台倒数第五值。|FULL_TEXT_RECHECKED_THIS_PHASE; PDF p2 §2.1; p3 Fig1/§2.2|
|输入信息|历史电压及运行参数经Spearman筛选（排除弱于0.3的变量）送卷积与Crossformer；物理电压还涉及j、T、氢氧/水分压、面积及老化参数。预测目标时刻这些物理工况如何获得未形成明确信息表；不能默认在线已知，也不能据此断言使用未来实测值。|FULL_TEXT_RECHECKED_WITH_UNRESOLVED_INTERFACE; PDF p3 §2.3/Fig2; p5 Fig3; p6 Eqs12–15|
|单步/多步方式|卷积先提局部特征，Crossformer预测电压；§3.5主要MISO单步；§4.2明确以autoregressive解释1/5/10步误差累积。不是F2冻结48→24同起点直接向量输出。|FULL_TEXT_RECHECKED_THIS_PHASE; PDF p7 §3.5/§4.2|
|跨度及物理时间|1、5、10步；FC1/2在1h重采样下对应1/5/10h。FC3为非等时长测试平台的代表点，不能将每步强换1h，亦非35平台均匀平分1180s的已证实物理时间。|FULL_TEXT_RECHECKED_WITH_UNRESOLVED_INTERFACE; PDF p3 §2.2; p7 §4.2; p9 Table2|
|主要模型|一维卷积→Crossformer的DSW嵌入、TSA、层级encoder–decoder；预测电压与半经验物理电压构成损失。i0、iL指数衰减、Rc线性增长，六个初值/老化参数参与训练；data、physics、joint三阶段。不是LSTM/CNN并行TFT。|FULL_TEXT_RECHECKED_THIS_PHASE; PDF p4 §3.1–3.2; p5 Fig3; p6 Fig4/Eqs12–15; p7 Eq16/§3.5|
|方法创新|作者主张卷积增强跨变量/跨时间建模和物理先验对少样本稳健性的协同；属于本文主张，不能移植成我们已完成的物理创新。|FULL_TEXT_RECHECKED_THIS_PHASE; PDF p2 introduction contributions; p10 §5|
|比较模型|CNN、Transformer、Informer、CNN-Attention；FC1/2/3和70/50/30%训练比例下比较RMSE/MAPE。|FULL_TEXT_RECHECKED_THIS_PHASE; PDF p7 §4.2; p8 Table1/Fig5|
|模块消融|Crossformer、C-Crossformer、PI-Crossformer、PIC-Crossformer四变体×3工况×3训练比例。模块消融确实存在；等参数量、等额外信息和训练预算未给出完整证据。|FULL_TEXT_RECHECKED_WITH_UNRESOLVED_INTERFACE; PDF p7 §4.3; p9 Fig6; p10 Table3|
|不同工况分析|稳态、准动态、动态三组分别给曲线与表；是多个数据集内分别训练/评价，不能自动解释为在一工况训练、完全跨工况零样本迁移。|FULL_TEXT_RECHECKED_THIS_PHASE; PDF p2 §2.1; p8 Fig5/Table1; p9 Fig6; p10 Table3|
|稳健性及泛化|训练比例70/50/30%和1/5/10步敏感性；没有从当前全文找到固定seed清单、跨seed方差或独立极化参数辨识诊断。此为本PDF报告范围，不能推断作者未公开实现的行为。|FULL_TEXT_RECHECKED_WITH_REPORTING_GAPS; PDF p7 §§4.2–4.3; p8–10 Tables1–3|
|评价指标|RMSE(V)、MAPE(%)，Eq17/18；主70%表PIC分别FC1=.00191、FC2=.00324、FC3=.00305 V。摘要.00191–.00305的范围遗漏表中FC2，应以表为准；不用于跨论文绝对精度排名。Eq16无求和印刷与Eq17不同，实际损失实现未核实。|FULL_TEXT_RECHECKED_WITH_SOURCE_AMBIGUITY; PDF p1 abstract; p7 Eqs16–18; p8 Table1; p10 Table3|
|核心图表|Fig1原/处理数据与工况、Fig2 Spearman、Fig3总框图、Fig4 Crossformer、Fig5多模型曲线、Fig6模块与训练比例；Table1主比较、Table2多跨度、Table3消融。|FULL_TEXT_RECHECKED_THIS_PHASE; PDF pp3–10 Figs1–6/Tables1–3|
|章节结构|§1背景和三点贡献；§2数据与预处理；§3模块、物理方程、损失、预测策略；§4指标→基线→跨度→消融/少样本；§5结论。|FULL_TEXT_RECHECKED_THIS_PHASE; PDF p2 organization; pp3–10 section headings|
|本项目可借鉴|借鉴数据协议→结构→常规对照→跨度→模块证据→局限的组织；图表分离数据场景、模型比较与机制证据；我们的已有逐步误差/尾部误差可更透明呈现。|REVIEWER_SYNTHESIS_FROM_VERIFIED_FULLTEXT; Derived synthesis from PDF p7 §§4.1–4.3; pp8–10 Tables1–3|
|不能直接照搬|不能因其用了PINN就在本项目加损失；不能把其同济/PHM的mV、固定工况或自回归步骤与RealHyFC未知未来工况直接24步排名；不能把未报告的seed/信息接口写成已核实。|REVIEWER_SYNTHESIS_FROM_VERIFIED_FULLTEXT; PDF p2 §2.1; p3 §2.2; p7 §§3.5/4.2; derived comparison|
|定位汇总|核心定位p5 Fig3、p6 Eqs12–15、p7 Eqs16–18/§3.5/§4.2/§4.3、p8 Table1、p9 Table2/Fig6、p10 Table3；渲染p5/p7/p10已看。|FULL_TEXT_RECHECKED_THIS_PHASE; PDF pages as listed|

## A1 · Jian Jiang等 (2026)

*A TCN-BiLSTM-based approach for remaining useful life prediction of PEMFC*，International Journal of Hydrogen Energy 207，153481。[DOI](https://doi.org/10.1016/j.ijhydene.2026.153481)。

证据：本轮原PDF方法/实验全文重核+关键页面视觉检查。

|核对项|已核实内容/未知边界|证据等级与定位|
|---|---|---|
|研究问题与目标|标题称RUL，实际结果围绕电压信号预测、MSE/MAE/R²，没有足以重建真实失效时刻预测协议的独立RUL误差表。|FULL_TEXT_RECHECKED_WITH_REPORTING_GAPS; PDF p9 Table2/§6; p10 §7|
|数据及工况|FCLAB公开数据，前500个初始电压点；p2文字写5 stacks each100cm²，堆/单电池身份措辞有歧义，不替作者修正为5cell。|FULL_TEXT_RECHECKED_WITH_SOURCE_AMBIGUITY; PDF p2 §2/Remark1/Table1|
|输入信息|一维电压→VMD→MI/Pearson选择IMF→KF→中心窗口MF；MF初窗9再二级选择15。切分内独立拟合、部署时仅历史滤波接口未交代完整。|FULL_TEXT_RECHECKED_WITH_REPORTING_GAPS; PDF pp2–3 §3.1–3.3; p8 Fig7/§5|
|单步/多步方式|TCN与IALA优化的BiLSTM-attention分别产生输出；式39计算与当前真实目标的绝对误差，式40选择较小分支。该已发表公式描述含目标依赖；不是可直接部署的历史输入门控。实现若另有在线选择器，此全文未说明。|FULL_TEXT_RECHECKED_WITH_TARGET_DEPENDENT_FORMULA; PDF p8 Eqs39–40/§5; p9 Fig8; p10 §6|
|跨度及物理时间|前500点，但点间物理时间、历史窗长、固定直接多步长度及时间顺序切分不足以复建；不得把500点说成500h或与F2未来12h对等。|FULL_TEXT_RECHECKED_WITH_REPORTING_GAPS; PDF p2 Remark1; pp8–10 §§5–6|
|主要模型|并行的是独立TCN和BiLSTM-attention预测器，TCN含因果/膨胀卷积与残差，IALA优化BiLSTM权重，最后逐点输出选择；不是TFT encoder内特征融合。|FULL_TEXT_RECHECKED_THIS_PHASE; PDF pp5–8 Eqs16–40/Figs5–8|
|方法创新|指数衰减能量的IALA、优化BiLSTM权重、并行互补；应把作者创新与部署信息资格分开。|FULL_TEXT_RECHECKED_THIS_PHASE; PDF p2 contributions; p7–8 §4/§5|
|比较模型|TCN、BiLSTM、PSO/GWO/ALA/IALA-BiLSTM、残差TCN、IALA-BiLSTM-Attention及组合；Table2共9行。|FULL_TEXT_RECHECKED_THIS_PHASE; PDF p9 Table2|
|模块消融|Table2包含残差TCN、attention与优化器变体，不能称完全无消融；但逐点目标择优不能作为同信息在线组合器的独立训练增益证明。|FULL_TEXT_RECHECKED_WITH_UNRESOLVED_INTERFACE; PDF p9 Table2; p8 Eqs39–40|
|不同工况分析|Table1设备可控范围；结论承认只FCLAB验证，尚未测极温湿/突然负载等真实复杂环境。|FULL_TEXT_RECHECKED_THIS_PHASE; PDF p2 Table1; p10 §7|
|稳健性及泛化|Figs9–11 loss/误差直方/散点；正文称多次实验稳定但具体seed表和随机重复的均值/方差未完整给出；不等于验证RealHyFC分布外泛化。|FULL_TEXT_RECHECKED_WITH_REPORTING_GAPS; PDF pp9–12 §6/Figs9–11|
|评价指标|MSE/MAE/R² Eq41；99.35%是Table2 R²，非真实RUL准确率；组合值MSE1.4211e−9、MAE2.6636e−5依据表，不和F2物理mV排名。|FULL_TEXT_RECHECKED_THIS_PHASE; PDF p9 Eq41/Table2|
|核心图表|Figs1–4分解滤波，Fig5 TCN，Fig6 BiLSTM-attention，Fig7 IALA，Fig8流程，Fig9 loss，Figs10–11误差分布/散点，Fig12各方法曲线；Tables1环境范围、2结果。|FULL_TEXT_RECHECKED_THIS_PHASE; PDF pp2–13 Figs1–12/Tables1–2|
|章节结构|§1引言；§2系统；§3分解/滤波/TCN/BiLSTM/attention理论；§4 IALA；§5组合；§6实验；§7结论。|FULL_TEXT_RECHECKED_THIS_PHASE; PDF section headings pp1–10|
|本项目可借鉴|借鉴结构对照、误差分布与预测曲线联合呈现；明确区分两路径表征与最终输出融合；本项目用历史输入gate而不使用目标误差选择输出。|REVIEWER_SYNTHESIS_FROM_VERIFIED_FULLTEXT; PDF p8 §5; p9 Table2; pp11–13 Figs10–12|
|不能直接照搬|不照搬VMD+中心平滑、不扩展优化器/模型池；不把R²当RUL精度；只准确指出式39/40的目标依赖，不指控未公开代码或作者意图。|REVIEWER_SYNTHESIS_FROM_VERIFIED_FULLTEXT; PDF p3 §3.3; p8 Eqs39–40; p9 Eq41|
|定位汇总|关键p2 Remark1/Table1，p3中心MF，p8 Eqs39–40，p9 Fig8/Table2/Eq41，p10局限；p8/p9已渲染视觉核对。|FULL_TEXT_RECHECKED_THIS_PHASE; PDF pages as listed|

## A2 · Fengyan Yi等 (2025)

*Remaining useful life prediction of PEMFC based on matrix long short-term memory*，International Journal of Hydrogen Energy 111，228-237。[DOI](https://doi.org/10.1016/j.ijhydene.2025.02.302)。

证据：本轮原PDF方法/实验全文重核+关键页面视觉检查。

|核对项|已核实内容/未知边界|证据等级与定位|
|---|---|---|
|研究问题与目标|标题RUL，实验证据主要为固定电流密度下电压老化单/多步预测；未发现可独立复建EOL与RUL误差的实验指标表。|FULL_TEXT_RECHECKED_WITH_REPORTING_GAPS; PDF pp7–10 §4/Eqs24–26/Figs9–14|
|数据及工况|自建90kW平台、400cell串联、1374h；试验包括怠速/启停/额定/变载，环境25°C/50%RH/标准大气；从极化曲线取j=.7A/cm²电压老化数据。不是PHM。Phase O将其列PHM属历史账本错误，本轮单独更正而不改旧文件。|FULL_TEXT_RECHECKED_THIS_PHASE; PDF p2 §2; p4 Table2|
|输入信息|电压与温度、压力、流量、电流等相关参数；db5/8层/minimax软阈波形预处理、[0,1]归一化、Spearman排除\|rho\|<.2弱变量；Train-only拟合范围与未来变量闭合未足够说明。|FULL_TEXT_RECHECKED_WITH_REPORTING_GAPS; PDF pp2–5 §2/Figs4–5/Table2|
|单步/多步方式|对LSTM、LSTM-Transformer、S-LSTM、M-LSTM做单步与20步；递归/直接输出以及滚动时协变量获取细节未完整说明。|FULL_TEXT_RECHECKED_WITH_UNRESOLVED_INTERFACE; PDF p6 steps7–9; pp7–9 §4|
|跨度及物理时间|每1h一组点，单步与20步；按文中采样名义是1h/20h，但窗口/缺测时间身份未给精细追踪，不可视为本项目直接24步协议同构。|FULL_TEXT_RECHECKED_WITH_UNRESOLVED_INTERFACE; PDF p5 process1; p6 process9|
|主要模型|矩阵记忆Ct、指数输入门与sigmoid或指数遗忘门、q/k/v投影，因果卷积/skip/归一化块。不是TFT两路径，也不是物理约束。|FULL_TEXT_RECHECKED_THIS_PHASE; PDF pp5–6 Eqs2–10/Fig6|
|方法创新|作者主张矩阵存储和指数门增强长序列/高维特征学习及并行计算；当前论文可作PEMFC数据驱动架构背景，不要求本项目复现新backbone。|FULL_TEXT_RECHECKED_THIS_PHASE; PDF p2 contributions/Table1; p9 §5|
|比较模型|LSTM、LSTM-Transformer、S-LSTM、M-LSTM；训练比例70%和50%；单步和多步分别给指标。|FULL_TEXT_RECHECKED_THIS_PHASE; PDF pp7–10 Figs9–14|
|模块消融|架构比较存在；没有定位到逐项去矩阵/去指数门/等容量训练对照。不能把多个完整网络比较直接当每个组件的独立贡献证明。|FULL_TEXT_RECHECKED_WITH_REPORTING_GAPS; PDF pp5–9 §3–4/Figs9–14|
|不同工况分析|源试验四类负载，目标却是极化j=.7的选点老化；未见按动态工况分组的单独误差表，不能把数据来源变载等同于未知未来动态电压预测。|FULL_TEXT_RECHECKED_WITH_REPORTING_GAPS; PDF p2 §2; pp7–9 §4|
|稳健性及泛化|70%/50%训练比例和单/20步；正文称S-LSTM与M-LSTM接近，保留这一事实。50%实验p6流程写最后30%、p9写50%测试；p10 Fig14题注One-step而正文称multi-step，歧义保留。没有具体seed分布报告。|FULL_TEXT_RECHECKED_WITH_SOURCE_AMBIGUITY; PDF p6 step6; p9 §4; p10 Fig14|
|评价指标|MAE/MSE/R² Eqs24–26；p7 70%单步M-LSTM MAE=.5076e−3、MSE=.1464e−5、R²=.9997。不可与未经同尺度审计的F2 mV直接排序，亦非真实RUL误差。|FULL_TEXT_RECHECKED_THIS_PHASE; PDF p7 Eqs24–26/Fig10/§4|
|核心图表|Fig1总体、Figs2–3台架/采集、Fig4去噪、Fig5相关性、Figs6–8网络、Fig9/12电压曲线、Fig10/11/13/14指标柱图；Table2变量。|FULL_TEXT_RECHECKED_THIS_PHASE; PDF pp3–10 Figs1–14/Table2|
|章节结构|§1引言，§2实验与数据，§3 M-LSTM与基线，§4不同训练比例及单/多步结果，§5结论与复杂度局限。|FULL_TEXT_RECHECKED_THIS_PHASE; PDF pp1–10 section headings|
|本项目可借鉴|采用独立数据定义、单/多步分表、减少训练量的敏感性、复杂度限制；尤其借鉴作者明确承认M-LSTM计算成本，而非把所有模块包装成无代价创新。|REVIEWER_SYNTHESIS_FROM_VERIFIED_FULLTEXT; PDF p9 §4/§5|
|不能直接照搬|不能把标题RUL当已给EOL标签，不把原文400cell定状态老化成绩移植到RealHyFC动态多步；不仿照不清楚的切分或非因果滤波；不新增M-LSTM搜索。|REVIEWER_SYNTHESIS_FROM_VERIFIED_FULLTEXT; PDF p2 §2; p5–6流程; p9–10|
|定位汇总|p2§2是数据纠正核心；p5–6 Eqs2–10/Fig6和流程；p7–10 Figs9–14；渲染p6/p9已看。PDF p2对应印刷229，p6对应233。|FULL_TEXT_RECHECKED_THIS_PHASE; PDF pages as listed|

## A3 · Taehwan Ko等 (2025)

*Physics-informed neural network for long-term prognostics of proton exchange membrane fuel cells*，Applied Energy 382，125318。[DOI](https://doi.org/10.1016/j.apenergy.2025.125318)。

证据：本轮原PDF方法/实验全文重核+关键页面视觉检查。

|核对项|已核实内容/未知边界|证据等级与定位|
|---|---|---|
|研究问题与目标|以时间和电流密度预测长时电压，再通过预设VEOL交点推RUL，强调降低输入数据长度IDL。不能把文中预设阈值转换成RealHyFC已观测失效。|FULL_TEXT_RECHECKED_THIS_PHASE; PDF p9 §4.4–4.5/Eqs5–7|
|数据及工况|PHM2014/FCLAB两组5cell、100cm²；FC1 .7A/cm²近稳态，FC2 .63–.77A/cm²、5kHz三角纹波。不是RealHyFC10cell宽变载。|FULL_TEXT_RECHECKED_THIS_PHASE; PDF pp4–5 §3/Figs3/5/Table1|
|输入信息|t和j两输入；预测期j由训练阶段从最大电压测量到tinput的电流段复制。是明示假定的未来工况，不能称读取真实未来j，亦不适用于任意未知未来动态负载。|FULL_TEXT_RECHECKED_THIS_PHASE; PDF p5 §4.1; p6 Fig7; p9 §4.4 step3|
|单步/多步方式|DNN映射(t,j)→V，外推到预测阶段，VEOL交点得寿命；不是带48步历史的直接24步神经预测器。|FULL_TEXT_RECHECKED_THIS_PHASE; PDF p4 Fig4; p5 §4.1; p9 §4.4|
|跨度及物理时间|由tinput到VEOL的长期预测，例FC1输入210/330h，FC2 244/330h，IDL26/40.8/35.2%；不是固定1/3/24步预测。|FULL_TEXT_RECHECKED_THIS_PHASE; PDF p7 Tables2–3; p8 Tables4–5; p9 Eq7|
|主要模型|8层每层20节点tanh DNN；数据、膜、催化层电压残差相加；Adam两阶段各5000epoch，lr1e−3后1e−6。膜方程引用[39]49/53/56，催化引用[41]1–7，本PDF未完整展开及标定。|FULL_TEXT_RECHECKED_WITH_REPORTING_GAPS; PDF pp4–5 Fig4/§4.1; p8 §4.3|
|方法创新|作者主张用膜/催化先验及复制j处理系统波动，以较少老化数据达到可接受RUL误差；并非本项目应恢复PINN的依据。|FULL_TEXT_RECHECKED_THIS_PHASE; PDF p3 introduction; p10 §6|
|比较模型|与既有SOTA混合方法的IDL/RUL比较；同时无物理方程、无电流输入两变体。|FULL_TEXT_RECHECKED_THIS_PHASE; PDF p7 Tables2–3; p8 Tables4–5; p10 §5.2|
|模块消融|Tables2/3物理与j输入有去项对照；20次重复给均值/标准差，FC2无方程未收敛如实保留；未单独拆膜/催化损失或识别未来电流复制策略的独立增益。|FULL_TEXT_RECHECKED_WITH_REPORTING_GAPS; PDF p7 Tables2–3; p9 §5.1; p10 continuation|
|不同工况分析|两种电流条件分别评价，并展示波动；不是训练一工况后跨另一工况零样本泛化。|FULL_TEXT_RECHECKED_THIS_PHASE; PDF p4 Fig3; p5 Fig5; p7 Figs8–9|
|稳健性及泛化|每配置20次重复均值/标准差、不同IDL；随机seed细表与完整电化学参数标定未在当前全文定位；结论承认真实工况噪声/离群是未来工作。|FULL_TEXT_RECHECKED_WITH_REPORTING_GAPS; PDF p9 §5.1; p7 Tables2–3; p10 §6|
|评价指标|有符号RUL误差=(真实RUL−预测RUL)/真实RUL；accuracy=1−相对绝对误差；IDL=tinput/tEOL。不是电压R²/RMSE。Fig4根号逐点平方后求和与文本RMSE名称有数学差异，实际实现未核实。|FULL_TEXT_RECHECKED_WITH_SOURCE_AMBIGUITY; PDF p4 Fig4; p5 §4.1; p9 Eqs5–7|
|核心图表|Fig4结构与损失、Fig7训练/预测/电流复制/RUL流程、Figs8–11电压外推与阈值；Tables2/3重复消融，Tables4/5 IDL比较。|FULL_TEXT_RECHECKED_THIS_PHASE; PDF pp4–9 Figs4/7–11/Tables2–5|
|章节结构|机制背景→公开数据→PINN与输入/方程选择→明确预测步骤/指标→消融→少数据对标→限制。|FULL_TEXT_RECHECKED_THIS_PHASE; PDF §§2–6 pp3–10|
|本项目可借鉴|借鉴训练与预测信息分离、把反事实/消融失败和不收敛保留、逐配置报告重复分布；对本项目更重要的是明示未知未来负载边界。|REVIEWER_SYNTHESIS_FROM_VERIFIED_FULLTEXT; PDF p6 Fig7; p7 Tables2–3; p9 steps1–4|
|不能直接照搬|不移植膜/催化参数或复制j为合法真实未来工况；不把RUL accuracy跨任务和F2 mV排名；不将10%示例当RealHyFC真实EOL，不增加PINN或RUL实验。|REVIEWER_SYNTHESIS_FROM_VERIFIED_FULLTEXT; PDF p8 §4.3; p9 §4.4/Eqs5–7|
|定位汇总|p1四位作者；p4 Fig4；p5 §4.1；p7 Tables2/3；p8 §4.3/Tables4/5；p9 step3/Eqs5–7；p4/p7/p9已渲染看。|FULL_TEXT_RECHECKED_THIS_PHASE; PDF pages as listed|

## 本项目任务身份

RealHyFC的10-cell耐久记录；24h历史输入（48×0.5h），同一起点直接输出未来12h（24步）原始ΔV。推理decoder只接收确定的相对步长，不接未来真实电流、温度、压力或其他实际工况。信息边界可区别具体实现，但原TFT本就区分已知/未知变量，不能声称首创未知未来预测。我们的误差来自Development，既有Assessment被复用，筛选六实例不是随机总体。上述文献不同数据/采样/目标/滚动方式/输入信息条件不能直接比较绝对误差。
