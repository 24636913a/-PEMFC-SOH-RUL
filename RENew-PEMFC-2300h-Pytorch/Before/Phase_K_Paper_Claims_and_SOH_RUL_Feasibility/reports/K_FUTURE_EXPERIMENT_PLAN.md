# 下一步计划（仅规划）

F2继续冻结。`PHASE_J_F2_ARCHITECTURE_FROZEN`和`PHASE_J_GATE_DYNAMIC_EFFECT_UNRESOLVED`同时保留；不重启backbone search、不新增门控扫描。

## 必须做

|计划|回答的问题|预先约束|
|---|---|---|
|论文证据与复现材料收口|已有优势在哪一任务/协议成立？|复用H/I2/J既有冻结数字，报告复用Assessment及C1_D0_E0复现限制；不同split/scaler/目标不能横向拼排名|
|任务级独立评价方案审定|Development优势是否能外推？|具体候选Test所有历史访问与选择影响先建registry；已暴露的目标/区域不能自动改称pristine；本轮不打开|
|共同评价单位与模型身份冻结|重叠多步预测到底如何加权？|主指标forecast_query_weighted_RMSE_mV；target_time_equal_weight_RMSE强制敏感性；同一target先平均平方误差而非平均预测；端点12h/P99逐评价单元报告|
|Seed/checkpoint及停止规则固定|是否避免挑seed与盲测后反复修改？|所有42/52/62及现checkpoint hash；多seed结果用逐seed指标后等seed汇总，pooled/ensemble另标，不选最佳；如未来refit需另行显式注册|
|若保留健康章节，先冻结proxy研究口径|是否有真实映射而非看答案？|同配置稀疏标签、train-period calibration、测量时间/来源隔离、support/recovery/uncertainty；没有continuous标签就不能训练dense监督|

以上是建议的未来方案约束，不覆盖历史冻结协议，也不是新的Test授权。新增统计显著性分析若要开展，必须先定义独立单位/temporal dependence，不能把窗口当独立样本。本轮不执行bootstrap或重新计算模型指标。

## 可以做（需单独决策授权）

文献最邻近结构比较，检验组合新颖性而非先声称全球首创；寻找来源清晰、兼容输入与盲测安排的外部动态数据；增加独立reference-condition检查点以验证健康映射。外部dataset应按任务兼容性选择，不为追逐更低Development误差；此阶段没有推荐未经核验的数据源。

## 暂时不做

连续SOH插值标签、重搜j*/滤波/seed、F2改结构、重训F1固定门控、SOH/RUL head、任意90%/95%EOL、2300−t、短时ΔV无限滚动寿命外推。完整RUL先补可信多轨迹failure/censoring与长期health evidence，不能通过神经网络复杂度补数据缺口。

## Test开启条件（所有满足，仍需明确授权）

论文任务与科学主张固定；源路径/边界/暴露史核验，明确是否任务级independent还是posthoc holdout；模型/输入处理/scaler/seed/checkpoint、合法窗口/同键truth、所有指标与聚合固定；runner完整可回读、异常fail-closed、一次性ledger与pre-target prediction freeze；技术验收不读Test；结果后不重新选择模型。独立资格不足时只可提出明确标签的holdout计划，不能洗白既有暴露。

E04–E07的多Seed不是三fold；T1-R三fold与H Development7:1:2不是同协议。N1寿命坐标split与TFT Development内部split亦不同。记录1819.997738边界不等于允许查看其后的特征/target。
