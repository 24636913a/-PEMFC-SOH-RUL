# O-01 — TFT直接竞争全文对照

基准 `9723ed75910594155b4c0ae4888d9bbe8519ec11`。页码为PDF页序，证据等级见账本；本报告不运行竞争代码或F2。

## P1：不是“把BiLSTM换成CNN”这么简单，也不是完全不同思想

Chou与Wang，*Future prediction on the remaining useful life of proton exchange membrane fuel using temporal fusion transformer model*，Renewable Energy247(2025)123019，[DOI](https://doi.org/10.1016/j.renene.2025.123019)。

全文p8 Fig7与p9 §3.2.2把TFT encoder、decoder两侧改为BiLSTM，保留GRN/VSN、注意力和quantile回归。p2引言/p14结论主要称encoder，p9有反向替换措辞错误；以详细结构图和方法为较强依据，同时保留文本矛盾。其图中没有F2式并行causal depthwise CNN。

F2保留原encoder LSTM，并将**同一个encoder VSN序列**送入因果卷积分支，融合**post-LSTM encoder特征**后接原decoder与attention。P1是替换RNN，F2是新增历史特征路径。两者共同继承TFT框架，均属增量扩展；结构差别不能自动证明F2更有价值。

BW/DWBW用于描述平滑衰退曲线膝点并确定训练区间；RF按重要性选输入；TPE搜索超参数；三者不是电化学物理约束。p4极化插值式3使用后续诊断点，全文预处理无法直接当作F2的在线因果协议。p11调参为300次TPE，网络容量/训练预算与F2不同，不比较跨文献RMSE。

p13表10/11确有native TFT与improved TFT对照，不能说作者“完全没有消融”。它支持该论文的整体替换比较，但等容量、分离encoder/decoder替换、多seed稳健性证据不足。P1未来known主要hour/cycle index，未来实际工况不可得的设定与F2有功能相似；p10–12递归step-ahead直到96%阈值，不是F2同origin48→24直接向量。滚动过程中其他未知协变量如何闭合UNKNOWN，不能推测偷偷取真值。

RUL为预测EOL减起点，AE小时/相对误差；不是本项目ΔV的mV指标。Sp、EOL文字及表格存在冲突，详见账本，不据其最小RUL误差声称更高一般精度。

## P2：工况信息进TFT已存在；新增gate要说明控制对象

Li等，*An interpretable data-driven method for degradation prediction of proton exchange membrane fuel cells based on temporal fusion transformer and covariates*，IJHE48(2023)25958–25971，[DOI](https://doi.org/10.1016/j.ijhydene.2023.03.316)。

p5 Fig3、p6–8式9–21是VSN/GRN/原生门控/attention；历史多变量融入encoder，未来known进decoder。p8表3把入口T/P、气体流量等列为future-known。作者是否将控制计划与未来actual严格区别，NOT_REPORTED。它不提供F2这样显式“actual全future-unknown、仅horizon index”的相同信息契约。

P2没有图示的LSTM/卷积双分支，也没有五项历史运行统计控制两条分支混合的gate。F2新增gate**控制特征编码路径**，VSN控制输入变量，GLU/GRN控制组件输出，两者对象不同。因此结构可区分，但“使用工况自适应加权”与“门控”思想本身不是首次。注意力重要性不等于电化学机理因果解释。

P2实测同济单cell动态数据，5步→1步，最后阶段6:4切分；F2为RealHyFC10-cell、24h历史→12h直接输出。p10模型表和p11重要性表不足以证明新增F2式分支gate的独立价值；F2自己的独立收益也必须以Phase J为准，不能靠P2未做相同消融来回避。

## 三者及原TFT

|维度|原TFT S1|P1|P2|本项目F2|
|---|---|---|---|---|
|局部时间编码|LSTM encoder/decoder|BiLSTM替换两侧|原TFT LSTM|保留LSTM；并行历史depthwise CNN|
|新增融合对象|原生变量/组件|未给F2式路径gate|VSN/GRN原生融合|post-LSTM encoder路径融合|
|decoder|未来known变量|主要时间/cycle|时间和入口工况/flow|仅相对horizon index|
|输出|multi-horizon quantiles|递归step-ahead/阈值RUL|主要5→1电压|48→24 ΔV点预测|
|独立模块证据|Fig3模块移除|整体native/improved对照|模型对照/变量重要性|F1不稳；learned-vs-fixed增益未证实|

S1：Lim等，[10.1016/j.ijforecast.2021.03.012](https://doi.org/10.1016/j.ijforecast.2021.03.012)，p3式1、p4 Fig2、p4–6式2–8/17、p9–10 Fig3。原TFT已经处理未来未知与已知信息，不能宣称F2首次实现未来未知预测。

结论：P1最接近TFT扩展定位，P2最接近工况协变量应用。未发现与冻结F2完整同构的这两篇方法；这不是全领域排他性首创证明。F2值得保留的差异是可复现的**历史路径融合位置和严格部署输入契约**，价值强弱由自身实验决定。
