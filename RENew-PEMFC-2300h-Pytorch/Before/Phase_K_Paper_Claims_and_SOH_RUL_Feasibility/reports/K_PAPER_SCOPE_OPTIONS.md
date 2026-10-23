# 候选论文定位

## A — 动态工况下PEMFC多步电压预测（推荐主线）

科学问题：只有历史运行信息时，工况感知双路径TFT能否改进12h动态电压多步预测？
方法：冻结F2联合结构，成熟Top10/EWMA/TPE为输入处理与配置工具，不列三项新算法。
数据：合法连续48历史/24target、严格时间分割、train-only scaler、未知未来工况。
成果：H/I2限定Development多Seed证据与J工程/主张边界；不需要连续SOH或EOL才能研究该任务。
缺失：任务级独立评价、外部泛化、完整暴露史与cross-protocol复现材料。
风险：复用Assessment选择偏差、少数目标/seed驱动、非bitwiseCUDA重复性。
贡献：可复现、因果信息集下的经验动态预测设计；不是寿命预测或电化学机制发现。
适用性：当前最强。外部dataset对跨堆主张必要，但不应以不存在外部验证否认Development事实。

## B — 动态预测与稀疏健康表征联合研究（有条件辅助）

科学问题：动态预测表征与reference-condition retention在合法支持域中是否存在可验证联系？
F2仍做动态预测；健康部分必须独立定义、稀疏事件级评价、训练期校准。不能把两种任务的误差或标签拼为同一ground truth。
成果：N1 sparse proxy及HA/CSTL负面边界。
缺失：可靠映射与独立健康检验；7点已经用于proxy选择，不足以强独立确认。
风险：operating-condition混杂、回弹、锚点太少、proxy校准不确定性。
贡献可限于health-proxy可辨识性/数据资格审查，不能预告正面SOH性能。需要新增reference-condition健康观测或独立life验证强泛化。
当前适用性：可作为有明确限制的辅助章节；不让它延误A或诱导补伪标签。

## C — SOH估计与RUL预测（当前不推荐直接执行）

科学问题：健康轨迹及failuretime可否被独立预测？
必需：明确health supervision/reference condition、long-range验证、failure阈值/事件或合法censoring、多独立lifetimes、future-load与uncertainty。
当前缺口：动态→健康映射、dense真实监督、EOL和独立寿命评价未成立。
主要风险：将记录终止当failure、将插值当真值、用短期动态预测外推健康而不验证。
预期贡献只有在缺口补齐后才可讨论。新增合格寿命数据是必要研究条件，不建议为了标题强行执行。

推荐A为论文主任务，B只保留实证/资格边界，C延后。`RESEARCH_HYPOTHESIS`不是已完成成果；方案高级程度、标题长度或期刊影响不能替代可验证性。来源E03–E07、E15–E23。
