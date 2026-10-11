# Phase M 最终物理研究决策

**科学状态：`NO_ADMISSIBLE_PINN_CONSTRAINT_YET`。唯一首选技术方向：P3，历史条件匹配极化辅助监督；等级 `CONDITIONALLY_ADMISSIBLE`。当前不允许进入 PINN 训练。**

任务级只读审计的 PASS/FAIL、来源/代码/权重前后身份、合成测试证据见 `outputs/M_READONLY_AUDIT.json`。技术审计通过不将科学状态升级为方程合格。没有新实验性能、推理、参数辨识、SOH或RUL结果。

## 十二项回答

1. **可靠物理观测有哪些？** 字段与既有审计支持有效十cell电压、actual I/j 的bin统计、load transitions及zero-current ratio；G-STOP支持面积关系。温度、入口/出口表压、入口RH/露点、流量/计量比在N1 schema中存在，但不属于F2输入；本轮不验证实际样本完整性/校准可靠性。
2. **真正可计算的方程？** 历史j=I/A及相关输入统计身份可作为校验；不是F2电压loss。完整历史电化学式只有在补齐状态/参数并验证稳态域后才能计算。当前没有全部门槛通过的严格未来电压 residual。
3. **哪些需要未来未知工况？** 未来Nernst/全极化电压、target-condition极化次序、每步热力学边界与物理预测残差融合，均需未来j或T/界面组成等。假定/预测负载与实际已知负载必须单列。
4. **为何不能直接约束24步？** 方程值依赖未来负载/反应状态；F2只有过去且输出ΔV。相同历史可接不同未来负载，电压不同；30min平均非线性关系又不闭合。入口压力/RH不等于界面分压/膜水。可微性不能弥补这些缺口。
5. **POT/G-STOP的限制？** POT已证明其future-actual配对不合法；不得换TFT后重提同方案。G-STOP面积/zero-current物理身份是过去的数据证据，不是未来电压定律或新PINN创新。负面结论不被删除，也不解释成永久禁止所有物理学习。
6. **参数辨识证据？** 面积约280.001183cm²有历史支持。当前白名单没有j0/α/ASR/jlim/界面状态的联合可辨识证据。EIS/极化文件存在与曲线能拟合，不等于参数唯一识别；本轮没有读这些观测。
7. **无未来actual的非平凡残差？** 历史条件化辅助输出可以设计梯度路径；但共同支持、稳态/状态可比、测量隔离与防退化解尚未验证。尚无已合格的严格未来输出残差。宽边界或平滑至多physics-inspired，I/A输出梯度0。
8. **首选P1/P2/P3？** 仅推荐P3作证据优先方向；其历史时间域较符合当前任务，不需偷偷使用未来actual。不是自动多任务训练批准；先做已有RealHyFC Train补证，并等待新授权。
9. **保留哪些F2结构？** 原14槽位/Top10、EWMA(0.3)、Train scaler、48→24、Canonical TFT、历史causal多尺度卷积、5→8→1 gate和原decoder均保留为对照authority。本轮源码和checkpoint未改。后续新版本可以保留参数化设计，不可把新权重称原冻结权重。
10. **哪些必须独立？** 辅助头、历史物理pair/eligibility/测量隔离、unit adapter、物理尺度/λ、独立训练checkpoint及访问/验证协议。P2若加物理预测器或future-condition predictor也是新模块，但本轮不推荐、不实施。
11. **如何公平验证贡献？** 相同窗口/信息/预算/初始化/Seed42/52/62；新matched data-only与同容量非物理辅助控制；原正式query-weighted RMSE，强制target-time-equal SSE-first敏感性；开关消融、梯度/量级/违约率、端点/P99/过平滑/极端工况均完整报告。复用Assessment只探索，不作新盲测。
12. **最低训练条件已经满足？NO。** 没有任何候选达到PHYSICS_EQUATION_ADMISSIBLE；只有有条件补证方向。当前允许的工作已完成，后续需明确授权Train-only证据核实；本阶段不自动创建或执行下一实验。

## 最小证据补全与停止

唯一建议：在既有RealHyFC合法Train内证明历史稳定条件匹配极化的支持、状态控制和辅助测量隔离，必要时用已有Train极化/EIS核查半经验适用性。先冻结访问路径/边界及验收规则，再申请授权；不是更换dataset，也不是强行定义健康状态。支持不足就停止P3，不能把未测状态设常数升级严格PINN。

最终维持 PHASE_J_F2_ARCHITECTURE_FROZEN、PHASE_J_GATE_DYNAMIC_EFFECT_UNRESOLVED、PHASE_L_CONFIRMATORY_TEST_NOT_ELIGIBLE、PHYSICS_LOSS_NOT_ESTABLISHED、HEALTH_EVIDENCE_NOT_IDENTIFIABLE。潜在贡献仅是future-unknown任务下的条件物理辅助学习；没有“PINN+TFT首创”“机理已验证”“健康/RUL已解决”结论。

deep-research 的来源分级使全文未核实方法保持 UNKNOWN，并用数学反例拒绝仅凭公式升级；write-like-me 按用户正式中文/协议式写法组织材料，未检索私人文档。这些技能影响了文献证据边界与报告表述，没有启动完整论文或模型流程。

**完成本阶段审计、仅提交Phase M并推送main后停止；不训练、不部署物理loss、不自动开启Test。**
