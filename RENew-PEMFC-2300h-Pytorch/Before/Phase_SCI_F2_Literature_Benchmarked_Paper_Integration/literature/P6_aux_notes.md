# P6与三篇辅助论文：本轮原文重核

本文件属于第一阶段文献审查，不是新增仿真实验。没有加载项目模型、运行推理、训练或读取Assessment/Test。详细68条字段证据见同目录`P6_aux_evidence.json`。每条含证据等级与PDF页/章/图表定位。

## 来源与审核身份

读取了Phase O身份CSV及全文证据账本后，重新从仓库`参考论文/`的四份原PDF提取并阅读方法与实验，不能把本轮工作描述为仅复述Phase O。P6/Jiang/Ko本轮共11/13/11页，Yi10页。关键图表渲染保留在ignored `runtime/support/`，不复制原PDF或原图进入论文成果。原PDF SHA均与Phase O身份记录相同。

- Wei等，2026，Journal of Power Sources 677,239941，DOI [10.1016/j.jpowsour.2026.239941](https://www.sciencedirect.com/science/article/pii/S0378775326006919)。作者：Lin Wei; Shangrui Gao; Zonghai Chen; Yujie Wang; Zhendong Sun。
- Jiang等，2026，International Journal of Hydrogen Energy 207,153481，DOI [10.1016/j.ijhydene.2026.153481](https://www.sciencedirect.com/science/article/abs/pii/S0360319926001187)。作者：Jian Jiang; Dongsheng Du; Lin Su; Yifan Du。
- Yi等，2025，International Journal of Hydrogen Energy 111,228–237，DOI [10.1016/j.ijhydene.2025.02.302](https://www.sciencedirect.com/science/article/pii/S0360319925008808)。作者：Fengyan Yi; Xing Shu; Jiaming Zhou; Jinming Zhang; Chunxiao Feng; Hongtao Gong; Caizhi Zhang; Wenhao Yu。
- Ko等，2025，Applied Energy 382,125318，DOI [10.1016/j.apenergy.2025.125318](https://www.sciencedirect.com/science/article/abs/pii/S0306261925000480)。原PDF首页四作者：Taehwan Ko; Dukyong Kim; Jaewoong Park; Seung Hwan Lee。

2026-10-17使用出版社搜索结果复核标题、卷年页及DOI。直接打开部分ScienceDirect页面403，Yi DOI解析访问失败；故方法实验依据是本地原PDF，不把搜索摘要升级为全文。CoLab等聚合结果给Ko多加一位作者，与原PDF及作者CRediT不符，没有采用。

## 最有价值的借鉴

1. **P6提供结果组织范例，不提供本项目PINN授权。** 方法总图p5 Fig3，基础比较p8 Table1，多跨度p9 Table2，模块/数据比例p10 Table3。这种“任务→模型→主比较→跨度→机制/局限”组织能用现有RealHyFC成果实现。P6卷积串联Crossformer，我们的LSTM/CNN并行TFT信息流不同，不能把名称相近当创新相同或绝对首次。
2. **Jiang帮助厘清融合信息条件。** p8式39按真实目标计算逐点绝对误差，式40选误差小的分支；p9 Fig8/Table2、p10实验再次说明选择。这是对已发表公式的准确描述，不是调查未公开程序或作者意图。F2历史工况gate无需目标真值，但现有机制诊断也没有证明其稳定独立精度增益；两个事实应同时写。
3. **Yi体现预处理、单/多步及训练量对照的叙述结构。** p5–6方法、p7–10实验可借鉴；其S-LSTM与M-LSTM结果接近和计算开销限制应保留。无需因此增加M-LSTM、分解或超参搜索。
4. **Ko强调部署信息与消融失败的透明披露。** p9 step3明确未来电流由历史复制；p7 Table3明确去物理方程不收敛。我们可以借鉴信息接口表及不利结果呈现，但不能移植其未来工况假设、物理参数或RUL成绩。

## 本轮发现的历史记录错误及原文歧义

- **必须修正新矩阵中Yi数据身份。** 原PDF p2§2写90kW/400串联cell自建台架、1374h，试验含idle/start-stop/rated/variable load，选极化j=0.7 A/cm²的电压老化序列。Phase O把它列作PHM，属于旧账本错误。这里新证据纠正，原Phase O保持不动。该研究不是与RealHyFC未知未来负载任务直接可比的动态预测。
- P6主70%表1/3的PIC-Crossformer FC1/FC2/FC3 RMSE分别0.00191/0.00324/0.00305 V。摘要的0.00191–0.00305 V范围没有涵盖FC2。正式引用以表中分数据集值为准，不用摘要范围构造横向精度排名。
- P6 p7 Eq16的损失印刷缺少Eq17明确求和形式，真实实现未核验。物理参数辨识和目标处j/T/P来源也未完整给出，不能替作者补成已知实现。图3/§3.5却明确物理电压约束神经输出，不能反过来写其只拟合独立物理模型或没有物理消融。
- Jiang p2写“5 stacks(each100cm²)”且只说明500初始电压点；不擅自纠正为5cell、500h或某个已知预测跨度。99.35%为Table2的R²，不是独立RUL准确率。
- Yi p6流程称50%训练/最后30%预测，p9说50/50；p10 Fig14题注One-step，p9正文把它解释为multi-step。保留原文歧义，不拼成无冲突协议。其1h抽样与20步可写名义20h，但缺少直接/递归与协变量闭合细节。
- Ko p4 Fig4逐点根号平方再均值的式子与p5称RMSE有数学口径差异；p8把膜/催化方程指向外部引用，没有完整实验标定表。RUL评价基于预设VEOL，不会自动补齐RealHyFC真实EOL标签。

## 可比性与SCI写作边界

四篇论文与RealHyFC在cell数、目标状态、平滑、采样、物理时间、未来条件和直接/递归方式上存在实质差异，不能按RMSE绝对大小排名。横向表应对标**研究设计和证据结构**。我们的亮点应围绕严格历史输入、固定24步输出、可追溯数据/权重、逐步与尾部误差、筛选身份公开、反事实机制诊断及失败事实。不能将Spearman、EWMA、Optuna、TFT原生门控逐个写成首次创新，也不能写未经现有P2支持的动态门控独立稳定精度增益。

不建议从本四篇引出新RUL/PINN/物理约束训练、新backbone或文献数字竞赛。当前论文只聚焦RealHyFC动态多步电压预测。

## 实际视觉核验记录

- P6：PDF p5 Fig3信息流，p7 Eq16与预测/消融协议，p10 Table3。
- Jiang：p8 Eq39/40及流程，p9 Fig8/Table2。
- Yi：p6 Fig6和实验流程，p9结果与计算局限。
- Ko：p4 Fig4、p7 Tables2/3、p9电流复制与指标。

共10张页面PNG实际打开检查；渲染和逐页全文仅留runtime/support。未拿论文截图充当本项目图表。
