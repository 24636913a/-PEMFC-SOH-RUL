# O-10 — 最终投稿方向决策

## 三项独立正式判断

|维度|判断|含义|
|---|---|---|
|创新性|`INCREMENTAL_BUT_DEFENSIBLE`|核读集合未见完整同构，但TFT、CNN-RNN、多尺度、gate均有先例；具体集成设计可以诚实界定|
|实验证据|`EVIDENCE_INCOMPLETE`|整体Development正面，模块/完整Validation诊断不足，独立确认性资格未成立|
|写作就绪|`READY_FOR_CONDITIONAL_DRAFT`|可开始受限主张的初稿；不是已经可无条件投稿或保证录用|

本结论是内部科研判断，不替代编辑/真实同行评议。对需要独立Test的强确认性稿，存在关键验证缺口；不以“可以写”改变 `PHASE_L_CONFIRMATORY_TEST_NOT_ELIGIBLE`。

## 最接近三篇

1. **Chou & Wang2025，BiLSTM-TFT**：[10.1016/j.renene.2025.123019](https://doi.org/10.1016/j.renene.2025.123019)，p8 Fig7、p9方法、p13表10/11。最接近TFT增量扩展定位；替换两侧LSTM，而F2新增历史并行CNN与encoder融合，不是简单BiLSTM替换版本。
2. **Li等2023，TFT与协变量**：[10.1016/j.ijhydene.2023.03.316](https://doi.org/10.1016/j.ijhydene.2023.03.316)，p5 Fig3、p8表2/3。最接近工况信息融合；原生VSN/GRN不同于F2路径gate，future-known工况与F2仅历史契约不同。
3. **Jiang等2026，TCN–BiLSTM**：[10.1016/j.ijhydene.2026.153481](https://doi.org/10.1016/j.ijhydene.2026.153481)，p8式39/40、p9 Fig8。双表征思想相近，但独立输出按目标真值误差选择，与F2网络内历史特征融合不同。只记录发表接口的目标依赖，不指控故意造假。

**不能遗漏P4 Benaggoune2022**：dilCNN和长期双条件路径是多尺度/互补思想的重要先例。不是只有上面三篇与F2相关，三篇选择按本项目TFT主定位与并行混合争点，而非忽略更早先例。

## 可以保留的贡献与定位

- 具体的TFT历史encoder双路径实现：共享VSN、causal depthwise d1/d4、post-LSTM融合、原decoder接口不变。
- 无未来actual工况的同origin48→24 ΔV任务和共同冻结预处理，三seed整体配置的Development实证。
- 透明报告null/negative消融、重叠评价敏感性、尾部/端点和确认性限制；这是可审计的应用研究价值。

继续保留F2为论文主模型，**弱化gate为整体结构组成，不宣称已证实独立gain**。优先定位“应用型预测/严格协议与实证”，不是基础新架构理论。不能写首次门控、首次CNN-RNN、多尺度首创、跨stack泛化、机理解释、PINN、真实SOH/RUL。

## 距正式投稿还缺什么

最强五项质疑及回应详见O-08：组合创新有限；gate独立价值未证实；Assessment复用/无独立Test；尾部和重叠权重敏感；电压预测不可等同健康寿命。A类材料整理已完成。B类完整九模型Validation尾部/horizon/目标SSE及身份一致的图可能需要另行只读重算，**本轮没有执行**。C类模块反事实/等容量等预算比较仍缺，**没有授权新增训练**。D类独立确认数据无法通过重新切已暴露RealHyFC区域修复。

若不补B/C/D，就只能以明确探索性Development范围形成初稿并由负责人判断目标期刊是否接受这一证据强度；不能声称问题已经解决。未开展的SOH/RUL与物理研究应排除，不是为了稿件完整性必须填的“性能章节”。

## 历史保持与交接

H正面整体结果、F1不稳定、J gate独立收益未证明、CUDA strict determinism限制、Assessment复用、L确认性Test资格否决、K健康寿命不足、M/M1/M2物理支持不足、N `NO_PHYSICS_F2_TRAINING_ROUTE_QUALIFIED` 均保持。P3辅助极化继续暂停，Physics-F2未训练。本轮没有读晚期Test数值、预测cache、模型权重或raw/N1。

**建议现在开始条件性论文初稿；不是自动撰写完整论文、投稿或增加实验。** O提交main并推送后停止。后续是否补证、选刊或写稿由人工研究决策，不自动恢复PINN或backbone搜索。
