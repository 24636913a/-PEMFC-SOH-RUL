# 稀疏极化健康参考的能力边界

## 已冻结参考与读取范围

`PROJECT_VERIFIED_EVIDENCE`：E15–E18，主 j*=1.0 A/cm²、辅助1.6，按非 after-Air-Air Train+Validation选取。same nominal 10-cell BoL参考 V_ref=0.6989533305168152 V。本轮混合锚点CSV仅读取最前8条Development记录：7 nominal检查点（BoL/300/600/850/1150/1500/1750），另1条1500 recovery记录，后者不进入nominal选择集合。未读取Test数值、未全文件哈希。

定义保持：
SOH_V(t_i)=V(j*,t_i)/V(j*,BoL)。
1−[V_ref−V(t_i)]/V_ref 与它代数相同，不是第二个独立标签。
更准确名称：**参考条件单体极化电压保持率 / reference-condition performance-retention SOH proxy**。无EOL归一化、无真实RUL。

## 配置可比性

E17总体结论 **CONDITIONAL**。标称极化同durability 10-cell、280cm²、per-cell定义及相近协议支持比较；但独立BoL sensitivity/EIS可能是8-cell短堆，不能无条件与10-cell绝对电压拼接。论文引用的标称条件约75°C入口/70°C冷却、RH50/35%、出口160/150kPaG、stoichiometry1.5/1.7以及极化dwell/ramp必须与每个参考来源对应。该条件记录来自旧配置审计，不是本轮重读原始文件。

`SOURCE_DOCUMENTED_FACT`：E17为原资料条件的项目内摘录；本轮只核对审计来源与适用范围，没有重新测量所有条件。MEA系列/面积相同不等于同一物理堆；温度/RH/pressure、稳态程度、极化方向、recoveries均可能混杂。

## 七项判断

1. 可表征标准化参考点的**性能变化**，不是独立验证的不可逆损伤或绝对真实SOH。
2. 主参考7个nominal锚点，N1 Train5/Validation2；辅助j*来自相同检查点，不是增加7条独立寿命。
3. 数千raw dwell samples是一个检查点内相关观测；std是局部散布代理，不能称独立重复测量方差。BoL denominator不确定性共享于所有比值。
4. 不足以训练并独立评价逐0.5h continuous SOH；线性插值只能是额外模型假设，不能标为ground truth，本轮不生成。
5. 选择j*已用这7个Development锚点；再用同7点“独立验证”其优越性存在选择依赖。单堆多个时间点不是多个独立个体。
6. F2动态输出 V(t+h)−V(t) 未标准化到j*，不能直接除BoL就获得以上标签。必须另有可核验的参考条件映射、匹配支持与可逆波动处理。
7. 稀疏辅助研究可以检查性能保持率、条件兼容性和预定义映射的事件级误差；监督构造与校准必须按时间只用当时可用训练锚点，不能把后续健康信息注入早期fold。

E15的主参考ρ≈−0.964、non-increasing fraction≈0.833及1500 recovery shift≈3.916 mV说明趋势与回弹并存。相关好不等于连续标签可靠，不能强制单调化掩盖恢复。

## 与其他健康报告的区别

E23 CSTL chronology-safe P1S-B2 是**另一个事件级校准proxy**，不是N1极化点逐小时标签；其训练only B2与Fold3健康迁移未晋级提示映射需要验证。不得混用参考电压、global B2或proxy误差单位，也不凭健康实验失败否定动态电压预测。

`INSUFFICIENT_EVIDENCE`：真实连续SOH、跨堆可迁移health mapping、失效归一化、独立寿命验证。
