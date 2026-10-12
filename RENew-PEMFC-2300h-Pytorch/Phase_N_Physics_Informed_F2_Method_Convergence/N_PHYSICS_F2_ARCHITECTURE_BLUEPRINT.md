# N — Physics-F2 架构蓝图（不执行）

## 冻结基线的真实结构

依据H协议、F源码及E scaler metadata。原F2和九checkpoint只读；N未加载权重、未forward。保持J架构冻结及gate独立增益未证实。

输入14 slots固定，依原Top10 `[0,3,13,11,4,9,6,7,10,8]` 在standardized space置零其余slots；**不是新增14个全部非零特征模型**。仅历史voltage per-window causal EWMA α=.3。CommonScaler来自unfiltered Train legal histories，不从teacher或Validation重fit。

Canonical TFT encoder VSN → 原LSTM与历史因果卷积分支(k3,d1/4，左padding2/8) → 原regime gate融合。gate使用历史5变量均值，经5→8 ELU→1 sigmoid；不是未来工况门控。原decoder只接确定性相对horizon，14个physical slots占位为零。static enrichment、attention及输出路径不改变。输出24步standardized ΔV，用未滤波raw origin锚点重建绝对平均cell V。

冻结参数：hidden64、hidden continuous32、heads4、LSTM1、dropout.1；AdamW lr.0009605449687795866、weight_decay6.95942232671908e−6、batch32、max450、patience40、clip1。Seed42/52/62，非早期TimesNet42/2024/3407。

## 方案A：唯一保留的概念结构

```text
仅合法Train已发生、认证的工况/诊断
         ↓  前置时间块拟合 + 完整source依赖purge
   frozen-for-receiving-block physics teacher
         ↓  OOF Train软目标（可额外条件化Train future actual Z）
 原合法48步历史 X → 独立新F2学生 → 24步ΔV
                                ↑ data loss + distilled loss
推理：只有 X / origin anchor / deterministic horizon；teacher与Z全部移除
```

不得用接收块的目标voltage来反演teacher状态后声称OOF；材料状态若不能仅由合格先行/历史信息确定，就不产生teacher软目标。参数辨识通过前，不建立teacher代码或占位物理网络。

学生容量保持F2不变：A只改训练监督，不添加inference层。F2-Control也用完全相同F2学生；同Z、同teacher容量/预算/交叉拟合、同soft-target形式，但teacher不施加电化学结构。若将来选parameter-network teacher，两种teacher活动参数数、拟合次数/停止规则、输入Z、总计算成本均须事先匹配，无法匹配时必须报告容量/预算混杂，不能称纯物理消融。

**未冻结、因此阻断执行：** teacher模型具体参数化、独立诊断约束、稳定域/认证测量、真实crossfit块及支持、λ与loss尺度、匹配teacher实现及总预算。这些不是可在看Validation后自由选择的TODO；任何一项无证据即no-go。

## 方案B：不存在可执行的公式插槽

形式 `L=L_data+λL_phys` 可以写，但必须有 `r_phys(u,X,θ)` 的合法闭合。当前未来j/T/活动度未知，输入历史统计不能决定每个实际未来电压。替换为future actual Z的Train penalty实质回到A的特权监督，应按A的非物理Control审查，不能双算为另一个独立机制。

添加历史auxiliary头属于新独立版本、多任务/辅助路线；P3状态支持未通过，N不建立该头。物理参数网络/状态ODE/condition predictor同样不在本轮实现。不能偷偷加qH、单调HI、宽范围clamp、j-only曲线或健康标签使蓝图“可跑”。

## 软件执行边界

本轮代码仅纯数学contract与静态审计，不import原模型/runner。`require_training_qualification()` 对 equation/identifiability/domain/information/isolation/independence任何缺失、False或Unknown均拒绝；配置额外固定training_enabled=false。因此蓝图不是READY，不创建下一pre-open/训练阶段。

如果未来获得全新证据与授权，学生必须是新checkpoint，新实验目录；原F2继续作为冻结对照，不fine-tune/覆盖九权重。N不自动推进该步骤。
