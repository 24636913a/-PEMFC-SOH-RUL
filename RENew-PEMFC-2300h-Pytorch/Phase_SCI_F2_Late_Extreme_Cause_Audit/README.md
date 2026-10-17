# F2后段极端误差专项诊断（探索性）

基准提交：`f14a4dec70a3b8a67e866fa7fa3f07843b9ed443`。只读五模型×固定42/52/62，沿用原548个后段起点、13152条查询、640个唯一目标时刻。两组事件由研究负责人指定；不是独立Test，也不是事前确认性假设检验。

A. **已证实：误差高度集中于两次启停混合桶的相邻状态切换。** 四个目标时刻占640个唯一目标的0.625%，每Seed涉及96/13152条查询（0.730%），却贡献F2的80.828%～83.613%后段平方误差。按唯一目标等权后仍为73.118%～75.447%。这是两个物理事件，不是96个独立事件。

B. **已证实：Train和Validation有可比先例。** 按“零电流占比>0.5、前后桶零电流占比为0、有效且同segment”这一描述性定义，Train有107个，84个进入合法预测目标；Validation有16个，12个进入合法预测目标。最近历史输入匹配也找到相似的48步工况，但对应未来低谷有第12步和第13步两种身份。不能称为完全未见过的新退化模式。

C. **已证实：这两次事件出现一致的局部提前一格。** 每Seed、每事件23/23个完整高低配对起点发生预测高低顺序颠倒；19/19个覆盖六点邻域的起点，其预测低谷比真实低谷早1个聚合行，即0.5运行小时。三个Seed合计114个重叠诊断窗口均如此，不能当作114个独立事件。其他启停事件的F2诊断窗口1514/1524（99.344%）为零滞后，因此不是全文件标签整体错一行，也不能称为后段普遍提前。

D. **已证实“F2在这些事件中更差”，尚无法判断“双路径融合或门控导致更差”。** 完全匹配的四目标查询中，F2在三个Seed上的平方误差都高于F0、F1、LSTM、TCN；但LSTM Seed52也有同类提前，F1的部分Seed亦失败。F1/F2冻结权重经过不同训练轨迹，不能用它们的差值单独识别门控因果作用。只读观察没有发现gate饱和或超出Train范围；隐藏路径差异也在Train既有范围内。本轮未做门控替换、分支移除或新训练，不能把潜在机制解释升级成已证实原因。

E. **较可信的解释、值得下一步核查：运行时钟冻结、停机样本混合和下一次启停相位的可预测性。** 先审计仅依赖过去观测的时钟/周期相位线索能否区分“下一桶低谷”和“再下一桶低谷”，以及启停计划是否在起点实际可获得。使用现有Train/Validation的原始时间、OpHrs2、电流及既有窗口即可做最小只读研究；不先换网络或训练。若相似过去信息对应不同未来启停位置，或可用计划无记录，就停止声称精确切换可在线确定；保持正式协议和成绩不变。

## 复现与文件

```powershell
# 使用既有项目Python环境，PYTHONUTF8=1、PYTHONDONTWRITEBYTECODE=1
python -B scripts/analyze.py
python -B scripts/raw_event_check.py
python -B scripts/extract_frozen_gates.py
python -B scripts/complete_diagnostics.py
python -B scripts/verify.py
python -B scripts/plot_figures.py
python -B scripts/write_reports.py
python -B scripts/seal.py
```

以上在本阶段目录运行。configs已保存身份，不能重冻结；所有写入限制在本目录。原始读取只流式扫描两个已暴露事件的现有Loadcycle文件，仅保留每事件四桶12210条真实样本；不运行旧一次性评价程序。门控仅采集原始输出，先要求Train/Validation/后段预测逐点严格复现，再验证hook不改变预测与权重。

主证据为EVENT_SSE_CONTRIBUTIONS.csv、POINTWISE_ERRORS.csv、RAW_BIN_MIXTURE_RECOMPUTATION.csv、PAIR_INVERSION_SUMMARY.csv、ALL_STOP_LOCAL_TIMING.csv、NEAREST_HISTORY_ONLY_MATCHES.csv和FORWARD_IDENTITY_AUDIT.csv。所有查询均保留。七套图均来自真实CSV，PNG300dpi及PDF配套，代码及最终SHA见DELIVERY_MANIFEST。
