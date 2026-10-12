# N — 两条物理融合路线的收口

## 结论先行

当前 A、B 均未获 `PHYSICS_METHOD_ADMISSIBLE`。最终为 `NO_PHYSICS_F2_TRAINING_ROUTE_QUALIFIED`。最有希望的**概念**是 A：交叉拟合的训练期物理教师；并不是马上可以训练的新模型。当前不授权后续训练实施方案进入执行，停止新增没有独立物理依据的 loss。

|审查维度|A：物理教师蒸馏|B：联合电化学残差|
|---|---|---|
|数学形式|MSE(s,Y)+λ MSE(s,T_phys)，T_phys 必须独立合格|MSE(s,Y)+λ mean(r_phys²)，残差必须闭合|
|信息可用性|可以事先注册 Train-only privileged Z；学生只见历史 X|直接用未来真实 j/T/RH/P 算残差不能成为推理物理定律|
|参数|物理教师仍需 j0/ASR/jlim/界面状态及辨识证据|同样需要；端到端联合拟合不消除结构不可辨识|
|RealHyFC支持|M2 partial Train身份恢复，但没有认证状态可比、多负载幅值支持|不能从动态30min统计获得稳态瞬时方程闭合|
|输出梯度|教师固定且不等于学生时可非零；不是物理独立性的证明|电压残差可非零；I/A 为零，order-only 有退化解|
|非物理对照|同 Z、容量、蒸馏损失、交叉拟合和总预算的非物理教师|同信息/容量/预算的辅助监督；不得仅用 plain MSE 输给额外信息|
|当前等级|`PHYSICAL_PARAMETER_NOT_IDENTIFIABLE`，域证据不足；纯电压教师另为 `PHYSICS_PRIOR_NOT_INDEPENDENT`|`NOT_ADMISSIBLE`；未来直接残差同时 `INFORMATION_SET_MISMATCH`；复制电压残差 `PHYSICS_PRIOR_NOT_INDEPENDENT`|
|概念前景|`CONDITIONALLY_ADMISSIBLE`，仅设计合法 LUPI 信息接口|只有新独立方程/状态证据才能重新论证；当前不建议继续增加损失|

## 继承证据，不重做历史实验

G-STOP 的约280cm²支持 j=I/A 及混合统计身份，不支持未来电压方程。POT 的合成梯度曾通过但未来 actual 配对信息 gate 失败；N 的新 LUPI 设计不能改写 POT 当年的信息契约或结论。

M：`NO_ADMISSIBLE_PINN_CONSTRAINT_YET`。M1：整bin无转移必要条件未通过，raw dwell当时未知。M2 已恢复7273个保持片段，但不是7273个稳态物理测量：calibration 中8个 relative quiet，冻结 evidence 子域0个；认证 dwell/pair 数本轮历史记录为0，真正可用物理稳态总数仍未知。`M2_STATE_COMPARABILITY_UNRESOLVED` 和 `RAW_DWELL_EVIDENCE_SUPPORT_INSUFFICIENT` 保留。不给 P50 放宽，不将校准 quiet 转充证据，不重配 dwell；P3 暂停。

H 的 F2 多Seed经验优势可以保留；J 的固定0.5与学习gate结果接近，不能把 gate 命名为已验证的电化学机制。保持 `PHASE_J_F2_ARCHITECTURE_FROZEN`，九checkpoint不加载不修改。K 的论文定位仍是动态48→24 ΔV预测，不是已完成 SOH/RUL。

L：`PHASE_L_CONFIRMATORY_TEST_NOT_ELIGIBLE`。Development Assessment 已复用，晚期区域无确认性资格；这不等于本轮读取Test，也不能断言历史存在监督标签泄漏。

## 训练教师的三个概念不得混淆

1. 普通 neural teacher 用 Train电压拟合、向学生蒸馏：可以是统计学习，但不是独立物理证据。
2. 物理教师拟合 Train电压并嵌入有依据的结构：可以是灰箱，但仍必须检验方程/参数/独立域；拟合误差小不保证参数真值或物理可辨识。
3. 经独立 Train诊断约束且在隔离块检验的物理教师：可能提供条件电流响应/电阻/动力学结构，而不是复制标签。RealHyFC 当前没有这套合格证据。

未来能够证伪的主张只能是：“同信息、同容量、同总预算下，**已认证的**电化学结构是否比无结构教师改善严格历史学生的多步电压泛化？”不是“增加KD是否改善”，也不是“PINN+TFT首次”。

具体历史引用路径及 SHA 见 `evidence/N_SOURCE_MANIFEST.json`，方程反例见第二份报告。没有新真实时序计算或性能指标。
