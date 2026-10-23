# 必须保留的限制

- F2_VALIDATION_MULTISEED_GAIN_OBSERVED
- F2_REUSED_ASSESSMENT_MULTISEED_GAIN_OBSERVED
- PHASE_F_ASSESSMENT_INHERITED_SELECTION_BIAS
- C1_D0_E0_REPRODUCIBILITY_UNRESOLVED
- TIMESNET_FREEZE
- BACKBONE_SEARCH_CLOSED

F2 Assessment SD=6.3938192307 mV，大于F0的4.6130226020 mV；Seed62在均值收益中贡献明显。F1固定融合并未获得稳定收益。F2三个seed均胜出不等于训练稳定性已解决。F2误差集中于少数目标，历史I2报告五目标SSE占比37.54%–84.70%，最大SSE目标row2952/1793.5 h、multiplicity19。这里只引用已保存报告，没有再读取Assessment输入或进行Assessment forward。

原Development Assessment存在历史选择复用，非独立Test；Validation也曾用于训练/选择。重叠窗口和同数据三个seed不能提供独立样本显著性。门控与工况的数学依赖不证明物理因果机理，固定.5替换有训练/推理条件差异。C1/D0/E0历史复现未决，不能与不同协议TimesNet直接排名。

当前动态ΔV输出不是独立SOH ground truth，不完成RUL。未来需要明确健康监督和参考条件、动态电压与健康状态映射、未来健康轨迹、观测/外推EOL与右删失说明、阈值和不确定性校准及独立验证。无自动下一阶段、正式Test打开或额外架构训练。
