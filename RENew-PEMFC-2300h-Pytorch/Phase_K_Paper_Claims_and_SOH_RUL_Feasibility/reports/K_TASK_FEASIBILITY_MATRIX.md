# 三条研究路线可行性

|路线|已有支持|数据可能支持、尚未执行|额外数据/假设|当前不能成立|
|---|---|---|---|---|
|A 动态电压多步预测|E03–E07：冻结F2、48×14槽位/Top10 mask/0.5h历史、24步ΔV；H三Seed匹配对照、I2复用Assessment；J身份与门控诊断|独立任务级封存评价、复现材料整理|具体评估集暴露史、预先固定评价单位/seed/checkpoint；跨堆声明需要外部动态数据|pristine独立泛化、纯degradation increment、门控因果收益|
|B 稀疏参考健康辅助研究|E15–E18：same nominal配置7个Development锚点；E23证明chronology-safe calibration有必要|预定义参考条件映射的事件级误差/支持域检验；只作为独立健康proxy研究|dwellscatter/denominator不确定性、recovery区分、同条件支持；独立健康锚点|dense supervised SOH、F2动态输出直接等于polarization SOH|
|C 完整SOH/RUL|E21仅支持终止/限值的资料审计，没有可验证failuretime|可先规划有失效准则的longitudinal/censoring数据研究|长期健康映射、未来load场景、外部独立life、failure定义、长期uncertainty与外推验证|真实RUL性能、2300−t、从12hΔV无限递推得可靠寿命|

`PROJECT_VERIFIED_EVIDENCE`：F2预测 y_h=V_(t+h)−V_t，V是**动态运行单体平均电压**。V̂_(t+h)=V_t+ŷ_h是短时输出重建，不是reference-condition health或退化率。未知未来实测current/temperature/RH/pressure不能偷看；decoder仅使用origin-known relative forecast step。

`RESEARCH_HYPOTHESIS`：若将动态预测在指定参考条件下与独立极化proxy连接，或在可验证future-duty-cycle情景下建立long-range health trajectory，才可能进一步讨论阈值crossing。这两项均未由现F2证明，不是本轮实施任务。

过去24h、未来12h的短时尺度含负载动态。连续滚动多次会积累误差、输入分布漂移、缺未来工况与未识别的恢复效应，不能把模型的滚动输出当数千小时真实健康轨迹。RUL即便后来有threshold也是条件于阈值/负载情景的外推，需要独立校准，不等于failuretime监督。
