# 全文证据账本

## 证据规则与来源

基准 `9723ed75910594155b4c0ae4888d9bbe8519ec11`。PDF身份/文件SHA见 `O_PDF_IDENTITY_AND_SOURCE_LEDGER.csv`；来源为仓库 `参考论文/` 中的原PDF，而非旧中文总结。DOI链接用于定位正式来源，具体方法判断来自本地全文和实际页面渲染。

`FULL_TEXT_VERIFIED`表示对应主张已在全文定位，不表示论文全部实现均可复现。`PARTIALLY_VERIFIED`表示只核实部分方法/信息接口；`AMBIGUOUS_FROM_SOURCE`表示原文自身冲突或缺少足以消歧的信息；`NOT_REPORTED`表示未提供该项。后两者不得用模型常识补齐。

核心P1/P2/P3/P6由主审读取全部15/14/13/11页，并视觉检查下列关键页面；P4/P5/P7由并行提取代理读取全部15/24/11页及关键视觉页面，主审综合其逐页证据。五份补充论文精读方法及实验相关章节。其他九份只进行首页身份核对，不列为完整方法验证。共21份物理PDF，其中一份为重复论文的中文转述；不是21项独立全文研究。

## P1 — Chou & Wang

[10.1016/j.renene.2025.123019](https://doi.org/10.1016/j.renene.2025.123019)，本地“利用时间融合变压器模型对质子交换膜燃料剩余使用寿命的未来预测.pdf”。

|主张|实际定位|等级|
|---|---|---|
|图示BiLSTM替换encoder和decoder两侧LSTM|p8 Fig7；p9 §3.2.2；p14结论仅称encoder，不能据结论删去图中decoder|FULL_TEXT_VERIFIED；摘要/结论范围冲突另记AMBIGUOUS_FROM_SOURCE|
|没有F2式并行卷积分支|p8 Fig7，p9–10式13–36|FULL_TEXT_VERIFIED|
|未来已知主要为hour/cycle index，实际运行参数未来不可用|p2–3、§4讨论|FULL_TEXT_VERIFIED|
|BW/DWBW膝点定义训练时期、RF筛选、TPE调参|p4式3/表3；p6；p11表5，300次TPE|FULL_TEXT_VERIFIED|
|递归step-ahead直到阈值，不是F2冻结48→24直接向量输出|p10–12，p10式37、p11式38–39|FULL_TEXT_VERIFIED；滚动过程中未知工况闭合细节NOT_REPORTED|
|存在native TFT对照，不是完全没有消融|p13表10/11|FULL_TEXT_VERIFIED；容量/独立多seed控制NOT_REPORTED|

实际视觉检查p8 Fig7/8、p11表5–7。首页为Renewable Energy247(2025)123019，online 2025-04-01。PHM5-cell、100cm²，FC1稳态70A、FC2高频纹波，不等同RealHyFC10-cell动态任务。极化插值式3使用后续极化点，不能直接当作在线因果标签构建。p6 Sp=690与表3的609等冲突保留；RUL百分比误差不与电压mV混合。

## P2 — Li等

[10.1016/j.ijhydene.2023.03.316](https://doi.org/10.1016/j.ijhydene.2023.03.316)，本地“一种基于时间融合转换器和协变量的可解释数据驱动质子交换膜燃料电池衰退预测方法.pdf”。

p5 Fig3是原生TFT流程；p6式9–12 GRN/GLU；p7–8式13–21 VSN/attention；p8表2/3分别列过去观测变量与未来已知变量。后者包含入口T/P和氢氧流量，**不只是时间索引**。是否有预先协议/控制命令证明其在预测起点可知：NOT_REPORTED。不能把未来known栏目直接判为作弊，也不能将它与F2未知未来工况的任务混称。

p8–9：同济单cell25cm²、3076cycles约1008h，35steps/cycle；每稳定电流段倒数第五点；5步输入预测1步；末期6:4切分。p9式22归一化训练极值拟合范围未明确，完整因果性只能PARTIALLY_VERIFIED。p10表5/6是模型对照，p11表7/8是变量重要性，不是新增双路径gate的独立消融。p8表4最优hidden154超出所列16–128范围，记录为原文歧义，不推测实际搜索。

视觉检查p5 Fig3、p6 Fig4、p8表2–4。明确原生TFT已有多变量融合和门控。

## P3 — Jiang等

[10.1016/j.ijhydene.2026.153481](https://doi.org/10.1016/j.ijhydene.2026.153481)，`1-s2.0-S0360319926001187-main.pdf`。

p2–3 VMD→MI/Pearson选IMF→KF→MF；MF式采用中心邻域，分解/平滑是否在训练切分内单独拟合NOT_REPORTED。p5–7式16–29、Fig5/6为TCN与BiLSTM-attention，p7–8式30–38是IALA权重优化。p9 Fig8显示**两条独立预测输出**，不是同一TFT encoder内部特征融合。

**关键视觉核验：p8式39/40，p9 Fig8/表2。** 式39计算每点预测相对原信号真实值的绝对误差，式40选择误差更小的TCN或BiLSTM输出。p8步骤4、p10实验叙述再次采用该选择。全文未提供不用目标真值的独立门控器。因此文中描述的正式组合方法记 `TARGET_DEPENDENT_SELECTION_IN_PUBLISHED_METHOD`。这证明的是**已发表公式/流程的目标依赖**，不是对未公开源代码的行为取证或故意造假的指控。

若解释为离线互补性oracle诊断，它可以描述理想上界，但不能成为可部署预测器；若作者另有部署gate，当前文中未说明，实施层仍UNKNOWN。单点选择的误差等于两模型误差的最小值是公式推论，不是本轮重新运行科研指标。500初始电压点、切分/窗口/seed不足以重建部署接口；表2的99.35%为R²口径，不是RUL独立精度证明。

## P4 — Benaggoune等

[10.1016/j.apenergy.2022.118835](https://doi.org/10.1016/j.apenergy.2022.118835)，`1-s2.0-S0306261922002756-main.pdf`。

p5–8 Fig8–10、式4–9、表4：dilation1/2、kernel2、skip与Luong attention；不是新近才引入PEMFC的多尺度CNN。输入17/20/25，attention尺寸12/32/12**不是预测horizon**。p6 Fig7与p8 §3.5.2：每轮5/10/15/24步预测后，用新真实测量更新下一轮。因测量此时已经发生，不能直接称为泄漏；但不等于F2同一origin直接24步任务。块内直接/递归接口未完全明确，PARTIALLY_VERIFIED。

p10–12 Fig16、式13/14：历史电压和条件趋势两条卷积路径；未来趋势由Train拟合ARIMA获得，再add/attention融合。这与F2“双路径”概念相近，但分支对象和工况来源不同。p11表6是有/无条件预测对照，非dilation、skip、attention逐项等预算消融。p13–14附录有移动均值/outlier流程，不可假定Train-only因果。p7式11印刷RMSE不是标准平方误差均值形式，实现UNKNOWN；不据其数字与F2横向排名。MC-dropout50次不等于50训练seeds。

## P5 — Sahajpal等

[10.1016/j.ijhydene.2023.04.143](https://doi.org/10.1016/j.ijhydene.2023.04.143)，`1-s2.0-S0360319923019031-main.pdf`。

p5–6 Fig2、式1–10：CNN后接LSTM/GRU/BiLSTM/BiGRU，是**串联**。p11表2/3、流程：前一小时24项测量→当前堆电压；测试各步真实历史输入，未来协变量递归闭合NOT_REPORTED。SG(order2/window21)是否因果不明；Train-only minmax明确。50:10:40切分，不等同F2。

p7–11：每模型/数据集500TPE trials/Hyperband，max150epochs、patience30；50次重复训练，但seed列表未报。表4–6有min/mean/max，卷积混合的均值不总优于纯RNN。p20–21表7与其他文章任务/最优与均值口径不同，不可比较绝对RMSE排序。p9提供代码仓库链接；本轮未运行外部代码。中文`main(源代码).pdf`是转述，不是该仓库代码，不作为独立原文证据。

## P6 — Wei等

[10.1016/j.jpowsour.2026.239941](https://doi.org/10.1016/j.jpowsour.2026.239941)，`1-s2.0-S0378775326006919-main.pdf`。

视觉核对p5 Fig3、p6式12–15、p7式16/三阶段训练。卷积先于Crossformer，DSW/TSA/HED不是F2 LSTM/卷积并行结构。式12堆电压为cell数乘Nernst减三极化项；式14涉及j、T、i0、Rc、Ac、iL；式15让i0/iL指数下降、Rc线性增加，六个初始/老化参数参与优化。j A/cm²、Rc Ω、Ac cm²形成V；压力对数需标准态约定，文中bar约定不等于已证明RealHyFC压力语义可移植。

式16比较神经网络预测与物理电压；图3直接连到神经网络反传；§3.5为数据、物理、联合三阶段。**A（约束神经输出）成立，B（只约束独立物理模型）不足以解释完整方法；C（单独先用真实V拟合物理参数）未给明确独立流程；D（预测目标处物理工况来源）UNKNOWN。** 式16印刷单点/N根号缺少式17的求和，准确loss实现AMBIGUOUS_FROM_SOURCE，不能替作者补成已知MSE。

p7–10：24窗、主要MISO1步，1/5/10步实验用autoregressive叙述；参数容量/seed/λ/可辨识性诊断未完整报告。p10表3有Crossformer/C/PI/PIC模块对照，不能称“没有物理消融”，也不能因此证明参数可辨识。PHM两5cell与同济单cell，不是RealHyFC。本轮不提出新loss。

## P7 — Ko等

[10.1016/j.apenergy.2025.125318](https://doi.org/10.1016/j.apenergy.2025.125318)，`808537.pdf`。

p4 Fig4、p5 §4.1：8×20 tanh DNN，输入t和j→V；不是时间窗口TFT。data、membrane、catalyst目标电压残差。p8 §4.3将膜公式指向[39]式49/53/56、催化层指向[41]式1–7；**本PDF未完整展开这些方程和本实验参数标定表**，不能借通用PINN知识补造。这里只核实残差接口，细参数/外部方程实施NOT_REPORTED。

p6 Fig7、p9 §4.4 step3明确复制Train末段j到预测期。这是**假定未来电流**，不是偷看真实未来电流，也不是任意未知工况预测器。p7表2/3有去方程、去j对照，未单独分离膜/催化层。p9式5/6/7为有符号RUL相对误差、accuracy、IDL；真实EOL定义不能从说明中的10%例子推定。图示loss称RMSE但逐点根号平方后均值等价MAE，实现不详。两个PHM堆，Adam两段lr各5000epoch；不误写为Wei的三阶段。

## 五份补充原文

|ID|DOI与定位|确证内容/边界|
|---|---|---|
|S1 原TFT|10.1016/j.ijforecast.2021.03.012；p3式1、p4 Fig2、p4–6式2–8/17、p9–10 Fig3|原生门控、VSN及future-known接口已经存在；移除GLU有消融；不是F2新增独占概念|
|S2 RF-TCN|10.1016/j.apenergy.2025.125569；p4–8 Fig2–5/§3.3.1；p14–15表4/Fig18；p18表7|RF-RFE串联TCN；新测量反馈；42.53%只为FC2特定输入对照MAE改善|
|S3 稳态模型RNN|10.3390/en18102665；p14–16 Fig7、式38、表6|物理Vsm作为输入，已知本步I/标称其余条件；不是自动可迁移到F2的未来未知输入接口；结论中的attention/residual未在方法充分落实|
|S4 恢复融合|10.1109/TEC.2023.3311460；p3式4/5，p5 Fig4/5/式22，p6式23/24，p8表I|PA-VR与GRU-A独立输出融合。p5先Train窗口N，再用已发生N+1误差求权重，预测N+2；不是P3未来逐点oracle。未来启停预设；作者把RUL列未来工作|
|S5 M-LSTM|10.1016/j.ijhydene.2025.02.302；p5–6式2–10/Fig6，p7–10 Fig9–14/式24–26|矩阵记忆和指数门控；主要验证电压MAE/MSE/R²，非可信失效时间标签；切分50:30与50:50/图14题注冲突保留|

S1本地为article-in-press排版，2021 DOI/版权可核；最终卷页不能冒充本地首页所印卷页。P1/P2/P3/P6还用出版社DOI页面核对期刊身份/出版信息。Supplement：P4附录实际存在；未见嵌入附件不代表出版社外部Supplement不存在，未核实的一律UNKNOWN。文献图表只在ignored runtime内部检查，不用于投稿插图。
