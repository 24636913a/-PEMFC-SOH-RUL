# Phase K 最终研究推荐

**选择论文定位A：动态工况下的PEMFC多步电压预测。冻结F2；采用统一的工况感知双路径时序融合TFT主张。稀疏健康研究至多作为独立、受限辅助路线，完整RUL当前不具备条件。**

这是Research/Planning结论，不是新增模型成绩。见[来源台账](../evidence/K_SOURCE_EVIDENCE_LEDGER.md)及其他8份专题报告。

## 十一项明确回答

1. **双路径TFT能否作为核心？可以，限动态48→24 ΔV任务。** H中F2等Seed Validation RMSE 7.245400mV，F0 9.067563/F1 9.155598；I2复用Assessment 12.197052 vs21.745493/22.892878mV。均是PROJECT_VERIFIED_EVIDENCE，不是独立Test结论。（E04–E07）
2. **创新独立还是联合？联合。** F1独立增益不稳定、J动态门控独立精度收益未证实；保留两个模块，但不拆成两个均已证明机制的贡献。（E03–E07、E12–E14）
3. **Spearman/EWMA/TPE角色？成熟预处理/配置工具。** 训练期特征筛选、单侧电压递推平滑、冻结优化配置；不列为三个新算法。T1-R正式INSUFFICIENT_GAIN必须保留。（E01/E02/E09–E11、L1–L5）
4. **稀疏锚点支持范围？参考条件性能保持率与事件级健康proxy研究。** nominal七点，主1.0/辅助1.6A/cm²；不支持dense真实SOH或独立multi-life健康验证。（E15–E18）
5. **完整RUL标签？没有。** 既无可验证failuretime，亦不能用2300−t。（E21）
6. **EOL还是删失？EOL_NOT_ESTABLISHED。** 有条件特定500mV限值摘录，但不能当j*=1.0健康EOL；右删失须先定义failure准则、核实未failure与停止原因，不能把它当精确寿命。（E21、L6）
7. **关键缺环？动态电压→可验证reference-condition健康状态/长期轨迹的映射。** 还缺EOL/合法censoring及future-load/长期uncertainty，短期ΔV本身无法完成连接。（E15–E23）
8. **哪种论文最有基础？A。** B有资格审查/有限辅助潜力，C暂不推荐。不是因论文标题级别或保证发表。（全部相关证据）
9. **外部寿命dataset？做完整/泛化RUL需要合格独立寿命数据；A的跨堆泛化也需外部验证。** 当前单条stack记录与多cell/window不是独立life。（E17/E21）
10. **下阶段优先证据？独立评价资格与可复现协议收口。** 先固定主张、暴露史、指标单位、全部seed/checkpoint；不是继续门控扫描。健康章节若保留，再审定reference映射和独立锚点；本轮均不执行。（E04–E07）
11. **Test何时开启？科学/执行身份与评价单位冻结、暴露史证明适当独立资格、技术runner验收通过，再获明确一次性授权。** 已暴露区域不可重新命名pristine；本轮不打开。（未来方案，不是实验结果）

## 负面证据必须保留

PHASE_J_GATE_DYNAMIC_EFFECT_UNRESOLVED；PHASE_F_ASSESSMENT_INHERITED_SELECTION_BIAS；C1_D0_E0_REPRODUCIBILITY_UNRESOLVED；HA FORECAST_ONLY_REPRESENTATION / HEALTH_EVIDENCE_NOT_IDENTIFIABLE；EOL_NOT_ESTABLISHED。没有独立泛化、物理机制、完整SOH/RUL或发表保证。

J原wrapper报告接口BLOCKED和独立缓存补全史不删除：E06/E08分别记录原异常与31测试、9checkpoint身份、完整validation缓存核验；本轮只读取文档和身份，不重复推理。

## 研究技能与人工责任

deep-research 的来源分级与academic-paper的“主张不超出证据”约束用于上述证据分类、反驳检查与保守措辞。未执行完整paper pipeline、未模拟不存在的多agent审稿、未生成论文/伪引用。自动辅助编制材料仍需研究者核对来源与最终投稿表述。

## 停止

PHASE_K_CLAIMS_AND_FEASIBILITY_AUDIT_COMPLETED。
New training=NO；new inference=NO；new Assessment inference=NO；Test content=NO；checkpoint changes=NO；new SOH/RUL labels=NO；automatic next phase=NO。
