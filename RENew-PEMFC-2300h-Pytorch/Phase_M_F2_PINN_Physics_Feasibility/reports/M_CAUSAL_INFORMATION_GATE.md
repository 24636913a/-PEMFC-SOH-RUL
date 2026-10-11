# 因果信息门槛

定义 \(\mathcal F_t\) 为合法历史48步及 origin-known 元数据。未来实际 \(u_{t+h}=(j,T,p,RH,flow,\ldots)\) 不属于它。F2 的相对 decoder step 已知，但不携带这些工况。

|信息类型|本轮地位|未来实验必须声明|
|---|---|---|
|past-known actual|F2 仅冻结14槽位/Top10；额外历史条件不暗入网络|过去工况辅助监督是额外训练信息，公平控制要同信息集合|
|origin-known|horizon坐标、原始 Vt、原 Train scaler|真实原始 Vt 与 EWMA编码末值分开|
|future-known by protocol|尚未证明未来set schedule在起点确知|独立 known-duty 情景，不能等同实际执行量|
|future-assumed|例如复制过去 j/恒负载|改变估计对象为 scenario-conditioned forecast，披露假设|
|future-forecasted|G-STOP 类独立预测模块|输入预测值与传播误差，不能替换为 oracle；是新模块/协议|
|future-actual extra training supervision|本轮不授权；POT候选不准使用其选对|不自动等于泄漏，但改变监督和匹配集合，需要专门 train/inference 与公平协议|
|future actual inference/oracle|严格禁止|不能包装为物理 residual|
|diagnostic/offline|现有报告、字段字典、源/权重身份|不把 Test 暴露史 metadata 当新目标数值读取|

## 未知未来负载反例

项目数学推断：同一 \(\mathcal F_t\) 可以接续两个合法但不同的未来负载 j_a、j_b。在同状态/正常极化域内，\(V(j_a)\ne V(j_b)\)。F2 history-only 点预测只输出一个值，所以不能同时满足两个未来状态的点态确定性方程。额外 T/P/RH 和膜状态更增加不闭合性。

这不证明所有 history-only 物理学习都不可能：可研究经验证的条件分布积分、稳健工作域或状态动态；但需要独立的未来工况分布/状态闭合及参数证据，不能直接写 \(f(E[j])=E[f(j)]\)。本轮未具备、未实施。

判定规则：残差需使用 \(\mathcal F_t\)、预测输出及已合法冻结常数；缺未来实际条件、未知参数、非平凡输出梯度或适用域任一项，都不能作为当前严格预测的合格 PINN residual。假定未来 nominal current 的论文不是反例，因为其任务信息集不同。

本轮对 Test 数值、Train 数值和混合源均零访问。读取 L 的暴露史只继承资格结论，不重新请求任何晚期数据。
