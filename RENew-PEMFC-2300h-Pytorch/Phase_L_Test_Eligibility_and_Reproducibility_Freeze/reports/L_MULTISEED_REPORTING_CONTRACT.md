# 三Seed汇总与配对契约

正式集合：每模型 Seed42/52/62，n=3。先独立计算全部24步指标，再报告三个Seed算术均值及样本SD（ddof=1）、逐Seed原值。Repeat-A/B不进n=3；不选最小RMSE或Assessment Seed62作为主成绩。

同Seed配对 F1−F0、F2−F1、F2−F0；负=新模型误差更低。报告每Seed差、mean/sampleSD、改善方向计数/3；relative improvement若列出，明确是逐Seed百分比的均值还是meanRMSE之比，不混称。主结论不依据新Test另定晋级阈值或删Seed。

三种量严格不同：

1. **主报告**：mean(per-seed RMSE)。
2. **描述**：pooled RMSE = sqrt(sum all errors² / all query counts)；同query数时sqrt(mean(seed RMSE²))。不称等Seed RMSE均值。
3. **预测集成**：先平均对应预测再计误差；本轮不设为正式模型、不执行、不能以它替换以上量。

三个Seed是同数据/同stack的随机训练重复，不是三独立电堆；重叠查询不是独立寿命轨迹；不做基于query独立性的显著性宣称。未来完整九模型共同评价；图可展示全部Seed或三Seed线，若需单Seed代表必须另行事前固定非性能规则；L不挑Seed。

当前协议是单次Development Train/Validation/Assessment划分，不把历史三 expanding fold均值套进这里。独立数据存在多区域/stack时，需新增预注册外推协议，不自动拼pooled排行榜。
