# 窗口与信息集合（只定义，不构建真实Test清单）

冻结任务：0.5h，X[row o−47:o]共48点，Y[o+1:o+24]共24点；origin为bin end的availability时刻。target_row=o+h，ΔV=V[o+h]−V[o]；origin锚点实际电压已知。未来真实current/density/temperature/RH/pressure等全部future-unknown，不能输入decoder；decoder仅h/24相对索引。P1S、SOH、age不作为新增输入。

输入14slots按H_CONFIG原始顺序；Top10 `[0,3,13,11,4,9,6,7,10,8]`；排除slot在standardized space置0。仅历史voltage per-window EWMA α=.3，首点原值，递推不跨窗或用未来；CommonScaler从unfiltered Train legal histories冻结，不重fit。

## 合法窗口（唯一事前规则）

1. source身份、row映射、OpHrs2/end时刻唯一且严格递增；排序不能掩盖duplicate，重复或不明跨文件坐标 fail closed。不最近时间配对，不重采样/插值。
2. 72 bins同temporal_segment_id，全 `bin_valid_for_windows`，原schema finite；相邻bin end差0.5±1e−6h。依原冻结规则缺口无填补，wholebin物理范围不得跨评价边界。gap_before仅首bin已有段边界允许，内部有断点必须形成不同segment/不连续从而拒绝。
3. **允许Development历史context**，但仅o及以前48点；这与原C_CODE history可先于Validation/Assessment起点一致。不把历史全部位于评价区误写成历史原规则。origin须是评价区已观察合法起点，24未来target bins全部完整在事先声明的评价region内；越界wholewindow拒绝。合成test可检验规则，但实际Test row/origin count未知。
4. coverage等只消费已有valid/schema质量规则，不临时调coverage threshold，不按误差、rare condition、support结果删样本。未来finite/quality是评分source质量屏障，不是未来covariate输入或可用于选模型的特征。
5. 若未来source/region无法提供合法同schema时间坐标/模型单位，不执行；不把外部不同采样协议硬套同row任务。所有九checkpoint同window key集合；缺键/重复/真值/anchor/年龄不一致则停止，不求交集静默删样本。

## 读取与逻辑隔离限制

本轮连N1合法prefix也无需读；禁止mixed CSV fullread/hash/Git blob。未来真实mixed文件不能仅靠“StageP”函数名宣称target-blind。48→24 rolling的某较早target会成为更晚origin的合法past；因此**at-origin未来不见**不等于“所有预测冻结前任何target列值都从未物理读取”。必须事先定义可信数据broker、逐originsnapshot与target-scoring隔离，记录共享列的物理读取/逻辑可见范围。

L没有实现/验收真实 broker，synthetic window checks不是已可运行Test runner。真实I/O隔离仍未解决、资格BLOCKED；不能以这些规则合成通过就READY。此契约适用于以后合格独立源或明确探索性方案；不改变当前已暴露区域资格。
