# N — 公平实验与停止规则（条件设计，当前NO-GO）

## 原任务，不换评价问题

任务仍是0.5h，history48→future24 ΔV，即24h过去→12h未来，strict future-unknown。F2输入/scaler/Top10/EWMA/anchor复用H/E契约。原Development为单次blocked划分：Train rows `[0,2103)`至1310h，Validation `[2103,2404)`至1460.5h，Assessment `[2404,3005)`至1819.5h；不是三TimesNet expanding folds。1380/205/378 legal windows只是历史manifest数量，本轮不重新构建或计算。

合法query：目标24步完整位于相应split，同temporal segment、连续0.5h、valid/finite；仅origin及以前history允许作为context，严格沿用原规则，不误写成禁止合法过去跨评分区起点。缺失键/真值/锚点不一致必须停止，不静默求交集。

## 最小四组以及可归因差值

|组|架构/训练信息|目的|
|---|---|---|
|F0|原生Canonical TFT，原合法历史与普通ΔV监督|原基线；采用同预处理/预算的新matched run身份，原历史权重保留|
|F2|冻结工况感知双路径设计、普通ΔV监督|结构基线；旧F2正式数值不被新版本覆盖|
|F2-Control|与Physics-F2同F2容量，同Z、teacher容量、crossfit、soft-loss、总预算，无电化学先验|隔离额外信息/teacher蒸馏/容量收益|
|Physics-F2|同F2学生加合格物理teacher/残差|只有相对Control差异才可能归因独立物理机制|

如果没有额外Train Z，Control不应加Z；若有，二者同时且只在训练使用。每组相同legal keys、scaler、data supervision、初始化规则、停止规则与学生budget。含teacher的两组包括teacher拟合/crossfit预算，记录参数数/步数/wall-time与全部失败，禁止把teacher计算藏在“offline免费”。F0/F2无teacher时总计算差异明确列出，不能用Physics对F0/F2的差值单独证明物理作用。

不以历史checkpoint warm-start一组而其他随机开始；后续若资格成立，应有全部新matched训练身份，原九权重用于只读历史参照。N本轮没有重训这些组。

## 数据与选择防线

- Seed42/52/62全部列出，固定paired seed方向 Physics−Control，负RMSE差表示新方法更好。不挑最佳seed；三Seed sampleSD不是独立重复样本显著性证据。
- 教师选择/物理参数/尺度/caliper/λ只能由认证Train证据预先决定，再冻结；Validation最多承担原早停职责。不得将已暴露Assessment用于反复挑λ、结构、预算或门槛。
- 不因为物理support/rare error在评估后删除target；物理supervision eligibility可在训练前定义，评分仍完整原合法keys。新teacher适用域若不覆盖所有Train，Control采用相同soft-target mask。
- 当前joint support不足，不借dwell次数当独立多负载样本。所有raw测量读取与查参另外授权；本轮只有报告证据。
- Assessment只能标 `DEVELOPMENT_EXPLORATORY_REUSED_ASSESSMENT`，晚期Test仍 `PHASE_L_CONFIRMATORY_TEST_NOT_ELIGIBLE`。本轮不开放。没有新数据集可引入来“补”本轮；只对现有独立F2论文明确限制。

## 指标事前固定

正式：全部 `(origin,horizon)` query等权平方误差开根转mV，**forecast_query_weighted_RMSE**。各seed分别计算，再等Seed算均值与sampleSD。

强制敏感性：同target_row内先平均**平方误差**，再target时间等权平均开根，**target_time_equal_weight_RMSE**。不是先平均预测的ensemble。P95/P99/max绝对误差、12h端点、24步曲线、MAE、target SSE集中度及数量一起记录。pooled跨seed/区域仅描述，不能混称等Seed或等Fold。这里没有三fold正式评价，不编造equal-fold指标。

收益归因先用平方误差检查再展示RMSE；重叠query不独立，不宣称bootstrap显著性。对λ、gate、teacher输出的泄漏/单位/梯度/常数塌缩必须通过生产测试；通过不等于性能收益。

## 晋级和硬停止

**当前训练准入硬门槛：** 方程、结构及实用参数辨识、RealHyFC稳定适用域、信息集合、完整目标/诊断隔离、非平凡输出梯度、独立物理机制全部有证据。当前A/B均不满足，training_enabled=false。

任何未来训练performance gate的具体改善量/容差尚未合法预注册。不能本轮无证据编一个λ或改善百分比就宣称“下一阶段可直接执行”。在所有科学资格满足后，另一次明确授权才能冻结执行版；不是自动延长P3审查。

若以后合格实验显示Physics不优于等信息Control，物理先验增益未成立，即使优于F0也不升级；单位/来源/keys/梯度/参数辨识任一失效为技术/科学阻断，不能改用一个容易改善的新loss。

本阶段停止：两路线均不合格，停止新增未认证极化残差、未经认证pair或新物理head。保留F2作为动态电压预测独立论文；不新增模型搜索、SOH、EOL、RUL、PINN试验或下一自动阶段。
