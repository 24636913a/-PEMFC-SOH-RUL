# EIC Review Report — IJHE Associate Editor

## Reviewer Identity
IJHE Associate Editor, PEMFC durability & prognostics方向

## Overall Recommendation: **Minor Revision**

## Confidence Score: 4 (High)

## Summary Assessment

本文提出将时序融合Transformer (TFT) 结合 L=1+多步滞后增强策略应用于PEMFC剩余使用寿命预测。在两工况（FC2纹波、FC1准静态）上进行系统消融实验，采用Loop 50评估协议，并辅以四重特征重要性验证和MC Dropout不确定性量化。论文的核心贡献在于：(1) L=1+滞后增强策略有效回避了滑动窗口在微弱退化信号下的信噪比困境；(2) 首次将TFT引入PEMFC退化预测并系统验证其跨工况稳定性；(3) 发现"模型-工况适配"现象——DLinear在FC1上第二优、FC2上彻底失效，TSMixer则完全反转。论文选题契合IJHE的读者群，方法创新性中等偏上，实验设计扎实。主要不足在于：仅基于IEEE 2014单一数据集的两个电堆验证，泛化性证据不足；L=1范式在理论层面与"预测"任务的定义存在张力。

## Strengths

1. **S1: 系统化的实验设计**: Loop 50协议+两工况交叉验证+5模型消融实验，比该领域90%的论文更严谨
2. **S2: "模型-工况适配"发现**: DLinear/TSMixer在两工况下性能完全反转的现象具有重要的工程启示
3. **S3: 四重可解释性验证**: Spearman+RF+VS+SHAP构成完整的可解释性证据链
4. **S4: 论文结构清晰**: 六章结构递进合理，逻辑链条完整
5. **S5: L=1范式**: 针对PEMFC退化信号微弱的特殊性，放弃滑动窗口的设计有深刻物理洞察

## Weaknesses

1. **W1: 仅有2个同批次电堆的验证 (Major)**: 数据集仅包含2片电堆，每片~1000小时。工况差异与电堆个体差异完全混杂
2. **W2: L=1"预测"的语义问题 (Major)**: L=1本质是"状态映射"而非"前向预测"，术语使用可能导致方法论争议
3. **W3: 与PINN/物理约束方法的对比缺失 (Minor)**: 相关工作中未讨论物理信息机器学习方法

## Detailed Comments

### Journal Fit
论文主题与IJHE的Aims & Scope高度匹配。IJHE近年发表过多篇PEMFC RUL预测的深度学习论文，本文的方法学深度超过该刊大部分同类论文。

### Originality
L=1+滞后增强在PEMFC领域是新的。TFT在PEMFC上的首次适配有贡献但增量有限。最具原创性的是"模型-工况适配"的系统发现。

### Structural Coherence
引言的三点贡献与后续章节对应良好。但第5章的讨论（5.8节）与第6章的结论存在约30%的内容重复。

### Title
建议最终标题: "Temporal Fusion Transformer with Lag-Augmented Single-Step Mapping for PEMFC Degradation State Estimation and Remaining Useful Life Prediction"

## Questions for Authors

1. FC1和FC2是否来自同一批次生产的电堆？
2. 在RUL预测中（5.7节），105h的"真实失效"是如何确定的？

## Dimension Scores

| Dimension | Score | Descriptor |
|-----------|-------|------------|
| Originality (20%) | 78 | Strong |
| Methodological Rigor (25%) | 82 | Strong |
| Evidence Sufficiency (25%) | 65 | Adequate |
| Argument Coherence (15%) | 82 | Strong |
| Writing Quality (15%) | 76 | Strong |
| **Weighted Average** | **76** | **Minor Revision** |
