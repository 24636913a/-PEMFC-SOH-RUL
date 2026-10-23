# 历史证据与不可撤销的研究边界

以下均为既有文档的读取结果，未重新计算历史性能。实际相对路径、当前 SHA256、最后修改提交及基准提交身份见 `M_SOURCE_IDENTITY_MANIFEST.json`。

|阶段|读取的核心文件|继承的事实与限制|
|---|---|---|
|E|`reports/E_SERIES_EXPERIMENT_REPORT.md`、协议、scaler、静态预处理代码|Top10 为 Train 派生；仅历史电压逐窗口 EWMA(0.3)，再公共 Train scaler，再零掩码。Assessment 已暴露；滤波不等于机理辨识。|
|F|`reports/F_FINAL_EXPERIMENT_SUMMARY.md`、三个模型源码|双路径仅改历史 encoder；门控依赖五个缩放后历史统计。单 Seed 结构差异不能拆解为已证明的独立物理贡献。|
|H|`reports/H_FINAL_CONCLUSION.md`、`configs/H_FROZEN_PROTOCOL.yaml`|九个 F0/F1/F2 × Seed42/52/62 checkpoint；三 Seed Validation 中 F2 有经验优势；非 CUDA 位级确定性。不是旧 TimesNet expanding-fold 三折协议。|
|I-R|`reports/IR_FINAL_RECOMMENDATION.md`|有限 Validation 输入上，保留原 requires_grad 标志、eval/no_grad、batch32 才复现 H authority。改变执行语义可能改变数值；本阶段没有推理。|
|I2|`reports/I2_FINAL_SCIENTIFIC_CONCLUSION.md`|复用 Development Assessment 中 F2 三 Seed 有方向一致的结构比较优势；不是独立 Test，也不证明机理。|
|J|`reports/J_FINAL_ARCHITECTURE_DECISION.md`|`PHASE_J_F2_ARCHITECTURE_FROZEN`；`PHASE_J_GATE_DYNAMIC_EFFECT_UNRESOLVED`。门控有数值作用，但固定门控比较方向混合、均值差很小。|
|K|`reports/K_FINAL_RESEARCH_RECOMMENDATION.md`|主张限动态 48→24 ΔV；健康、EOL、RUL 缺环仍在。本阶段只用 RealHyFC，不继承其外部数据建议作为新行动。|
|L|`reports/L_FINAL_EVALUATION_READINESS_DECISION.md`、九权重身份清单|`PHASE_L_CONFIRMATORY_TEST_NOT_ELIGIBLE`。读取暴露史结论，不读取晚期数值；不能证明 F2 曾用 Test 标签训练，也不能洗白既有晚期区域。|
|POT|`reports/STAGE0_PHYSICAL_FEASIBILITY_REPORT.md`|已注册候选依赖未来实际 j/T/RH/P 匹配。历史 Train 9452 对、梯度测试通过，仍信息 gate 失败；`PHYSICS_LOSS_NOT_ESTABLISHED`。这是该候选的否决，不是所有 PINN 的不可能定理。|
|G-STOP|`PHYSICAL_RELATION_AUDIT.md`、`FUTURE_CONDITION_FORECAST_AUDIT.md`|Fold1 Train 的面积约 280.001183 cm²；j=I/A、零电流混合统计获得支持。未来工况预测/投影是独立模块；消除输入一致性违约不等于提升电压预测。|
|HA|`reports/FINAL_HEALTH_REPRESENTATION_AUDIT.md`|`FORECAST_ONLY_REPRESENTATION` / `HEALTH_EVIDENCE_NOT_IDENTIFIABLE`。无 positive-current stratum 通过匹配支持；条件调整不可辨识；EMD endpoint 单位 V，而不是 SOH。|

## 对新方案的直接限制

POT 的失败发生在信息可用性层面，不是因为公式不可微。因此换成 TFT、调 λ 或增加层数，均不能补足未来真实工况。G-STOP 的面积只支持单位关系；不能据此获得膜电阻、交换电流密度、界面气体分压或未来负载真值。HA 的负面证据不允许将低频电压当不可逆健康状态，也不意味着所有参考条件健康研究永久不可能。

M 的数据授权更窄：连 Development 实际观测也没有读取；只读字典、代码、协议及上述报告。继承报告的数值是历史证据，不是本轮新验证。不能将“字段存在”升级为“当前样本完整、可靠、可辨识”。
