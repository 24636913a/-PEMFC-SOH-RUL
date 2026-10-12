# 当前研究状态：已经证明什么，还缺什么

基准 `ffcd9d836b1dfe12feeee220423f50be8f8ca241`。

## 通俗结论

现在已经有一个表现较好的**动态电压预测模型F2**，还没有证明“历史工况门控本身必然有效”，也没有合格的Physics-F2。下一步最值得明确回答的是门控究竟依赖工况信息，还是只需学到一个更好的全局融合比例。不能为了让最终模型带physics标签而加入没有适用性证据的loss。

任务不变：RealHyFC、0.5h、过去48步（24h）→同起点直接未来24步（12h）ΔV；推理不使用未来actual工况。Raw14槽位经冻结Top10 mask、因果EWMA和Train-only scaler；不是SOH/RUL任务。[H配置；O-04；O1最终报告]

## 历史结果（直接引用保存证据，不重新计算缓存）

|模型|Validation等Seed正式RMSE mV|sample SD mV|12h endpoint等SeedRMSE mV|target等权RMSE等Seed均值 mV|
|---|---:|---:|---:|---:|
|F0 Canonical TFT|9.067563|1.142630|16.868942|15.509111|
|F1 固定融合双路径|9.155598|1.122666|25.703238|22.812287|
|F2 工况门控双路径|7.245400|1.100853|15.173052|13.939445|

来源：H `outputs/metrics/H_MULTISEED_STATISTICS.csv`、O1 `outputs/O1_EQUAL_SEED_SUMMARY.csv`。约20.10%正式RMSE改善描述的是F2相对F0**三Seed均值**，不是每Seed、每horizon或独立Test都改善。P只整理保存值，不重跑O1。

O1：`VALIDATION_CACHE_IDENTITY_PASS`、`F2_ADVANTAGE_METRIC_DEPENDENT`、`VALIDATION_EVIDENCE_COMPLETED`。Seed62 F2 formal8.436399、endpoint29.817395、target等权27.053894mV；两个目标2247/2248的3个queries占正式SSE78.262216%。相同目标也影响F0/F1；仅凭残差不能解释物理故障。[O1 Seed62报告]

## 门控现状

J同一F2权重固定.5推理对照：学习/固定三Seed均值7.245400185/7.247419090mV，差−.002018905mV；方向混合。它证明数值依赖，不证明独立训练受控收益。独立训练F1与F2历史轨迹也不等于G0/G1/G2完整公平比较。[J fixed-gate报告]

## 物理及独立性限制

M2：7273个保持片段、8个统计相对安静，物理认证0；真正可用稳定dwell未知，不能写成“不存在任何稳态”。低高负载状态可比、RH语义与内状态未闭合。N：`NO_PHYSICS_F2_TRAINING_ROUTE_QUALIFIED`，P3暂停。这不是Physics-F2训练失败，而是尚不具训练资格。[M2/N最终决策]

Assessment复用，L确认性Test资格否决；O1补证不能修复这一点。O的“可以条件性写稿”仅是技术写作就绪；**按本次指令，先完成最终方案/获准实验，再给导师审核，通过后才正式写论文**，本轮不写稿。

## 来源定位

上文路径均相对研究根目录；完整相对路径和当前SHA见 `evidence/P_STATIC_SOURCE_MANIFEST.json`。O/O1结果不撤销，F2不修改；没有重新读取预测CSV、raw/N1、模型权重或Test内容。
