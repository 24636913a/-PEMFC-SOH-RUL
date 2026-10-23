# 方程可辨识性审查

## 1. 结构可辨识与预测可计算不同

依据 [DOE §2.5](https://www.netl.doe.gov/sites/default/files/netl-file/FCHandbook7.pdf) 的损失分解，项目数学审查考虑：

\[
V=E-b\ln(j/j_0)-jASR+B\ln(1-j/j_{lim}).
\]

变量/单位/适用域见候选报告。即使假定观测到稳态 V/j，E 与 b ln(j0) 只以组合截距出现：对任意 c，令 ln(j0')=ln(j0)+c、E'=E−bc，电压不变。因此没有独立 E/交换电流证据时，曲线拟合不能证明二者分别识别。若只覆盖窄 j 区间，浓差项一阶展开与欧姆项高度混淆；同时让状态、温湿压与参数自由变化会进一步恶化。

这是结构/局部灵敏度的反例，不是对当前数据做的新拟合。历史面积估计消除了一个比例问题，但不消除以上参数退化；EIS 存在也不等于模型、等效电路与单位已验证。

## 2. 当前可计算性结论

|候选|主要问题|本阶段等级|
|---|---|---|
|未来完整电压分解|未来 j/T/活动度未知；j0/ASR/jlim/膜水无联合辨识|FUTURE_INFORMATION_REQUIRED（同时 NOT_IDENTIFIABLE）|
|历史电压分解|历史工况在 N1 schema 中存在，但界面状态/参数与稳态域尚未验证|CONDITIONALLY_ADMISSIBLE；当前不能实施|
|j=I/A 输入身份|历史关系有支持，输出梯度为0；平均 I 被 Top10 掩码|NOT_ADMISSIBLE，限“作为 F2 电压损失”；物理身份本身不被否定|
|未来目标条件匹配极化|需要 future actual 选对；已被 POT 否决|FUTURE_INFORMATION_REQUIRED|
|历史条件匹配极化辅助|理论导数/梯度路径存在；条件共同支持/稳定性/非复制任务待补证|CONDITIONALLY_ADMISSIBLE|
|热力学每步上界|未来反应状态无闭合，不能把标准态常数普遍化|FUTURE_INFORMATION_REQUIRED|
|工程/经验范围或平滑|可计算但非严格电化学残差，范围/尺度尚未注册|PHYSICS_INSPIRED_ONLY|
|观测电压随时间单调|不同工况与可逆恢复不满足条件|NOT_ADMISSIBLE|
|不可逆健康状态方程|H 与工况/可逆状态未分离；HA 健康证据不足|NOT_IDENTIFIABLE|

没有候选在全部十四项证明字段上达到 PHYSICS_EQUATION_ADMISSIBLE。表中的 CONDITIONALLY_ADMISSIBLE 是允许论证/补证，不是运行批准。

## 3. 最小补证计划：仅已有 RealHyFC Train，尚未执行

一是源语义证据：cell配置、绝压/表压、气体组分/标准流量、温度传感器位置、polarization protocol 的可比性。二是稳定域证据：Train 内同段、同测试配置、低转移 dwell 的联合温湿压/流量支持，不仅各变量边际 overlap；以工况元数据定义筛选，不能按物理 loss 或模型输赢筛。三是物理适用证据：固定模型/参数化，保留独立 Train 子域检验极化次序/半经验误差，必要时合法 EIS/极化 cross-check，而非全部点同时拟合兼验收。四是聚合误差证据：仅 Train raw-resolution 稳定片段核查 mean(f) 与 f(mean)；若只能取得30min统计，应承认验证能力有限。

未来补证必须另行授权真实数据访问，冻结允许路径、prefix firewall、样本范围、匹配 caliper/最低支持数、状态稳定假设和验收指标。**本轮不挑阈值、不读 Train、不拟合参数、不重新提取数据、不创健康标签。** 若共同支持、状态控制或测量语义任一不能成立，就停止该辅助物理路线，而不是强行指定常数。
