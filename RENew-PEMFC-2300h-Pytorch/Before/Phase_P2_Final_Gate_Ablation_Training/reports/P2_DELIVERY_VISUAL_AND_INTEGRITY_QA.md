# P2 交付核验

训练结束后在新独立Python进程中，仅重读P2协议、源码身份、当前P2缓存与checkpoint SHA；再次执行 `audit_completed(require_all=True)`，结果9/9 PASS。没有重训、真实数据推理、原N1加载或Test访问。

五张规定PNG均实际打开检查：RMSE、24步horizon、12h endpoint、target-time-equal、gate distribution。标签/图例可读，未裁切；数据来自真实九次结果，全Seed保留。PNG为300dpi，另附对应PDF。风格采用scientific-figure-making的简洁spines、蓝/红/灰语义配色、打印友好hatch、全部Seed散点与样本SD，不将SD写成CI。

门控直方图使用统一0..1范围与30 bins，G0/G1可能处于同bin并重叠，G2的小幅变化也可能位于一个bin；不能将其解释为权重完全相等或G2没有变化。准确分布参数以 `outputs/metrics/P2_GATE_WEIGHT_SUMMARY.csv` 和逐origin权重CSV为准：G1三个Seed全局权重约0.516826、0.530668、0.503865；G2的origin-level SD约0.015139、0.139351、0.006388。

11项执行防线测试记录于outputs/audits/P2_EXECUTION_GUARD_TESTS.json。九组真实训练分别在170/413/179/219/261/234/302/363/137 epoch停止；总2278 epoch，每epoch44个batch，优化器更新总100232次，均在原450/pat40预算内。它们不包含P1的合成检查或P2的纯防线测试。

本地九个新checkpoint均存在，约1.49 MB/个；未删除唯一副本、未伪称有独立备份。大型batch journal也本地完整保留；Git忽略这两类文件，使用完整路径、SHA和备份状态清单保存可核验身份。预测、metrics、完整epoch日志、实际origin序列、运行记录及图表提交到P2目录。

冻结后没有改动任何训练源码、测试源码、协议、参数、数据定义或门槛。最终研究报告补充只解释本次真实结果与交付范围。历史阶段只读，checkpoint未读取或覆盖。冻结前Git LF/工作区CRLF表示误报单独保留于P2_PREFREEZE_IMPLEMENTATION_ERRATUM.md，不删除历史记录，不将其写成原数据变化或有效训练失败。
