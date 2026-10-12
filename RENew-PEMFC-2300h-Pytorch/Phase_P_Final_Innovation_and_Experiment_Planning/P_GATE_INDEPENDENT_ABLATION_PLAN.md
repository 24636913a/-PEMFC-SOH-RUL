# 最后一次工况门控独立训练消融方案

状态：`GATE_ABLATION_READY_FOR_AUTHORIZATION`；不是训练已获授权。研究问题：**同一个双路径TFT中，窗口工况变化是否比可学习的全局比例更有用？** 不修改历史F2或权重；以后另行批准独立实验版本，不能写回H/F/J。

## A1：冻结源码事实

`Phase_F_TFT_DualPath_RegimeFusion/models/regime_aware_dualpath_tft.py`：五项历史变量为load_transition_count、zero_current_fraction、j_std、I_std、j_mean；来自scaled/masked encoder_cont的48步均值，而非未来工况或目标V。

`c=mean_history(X_regime)`；`g(c)=sigmoid(w2ᵀ ELU(W1c+b1)+b2)`；`E=g E_LSTM+(1-g)E_CNN`。g为每窗口一个标量，广播到48×64编码特征；不是两模型输出选择。融合在post-LSTM encoder、原decoder/attention之前。[regime源码；dualpath源码]

参数：W1 8×5=40、b1 8、w2 8、b2 1，共57；均归入AdamW训练，不是手工固定物理系数。原代码w2/b2零初始化，使所有窗口起始g=.5。首步数据梯度可更新输出层，隐藏层数据梯度因w2=0初始为0；随后可变，AdamW权重衰减另计，不将初始零数据梯度写成参数永不更新。[源码及链式法则]

J gate均值/SD：Seed42 .427577/.043966；52 .488762/.004588；62 .456182/.031680。J learned−fixed RMSE为+.012179、−.003166、−.015070mV，三Seed均值−.002019mV。Seed52接近全局固定比例，不能由相关性推导独立物理机制。当前不能证明门控独立有效的原因是：同权重推理扰动不是重新训练反事实；缺可学习全局权重控制；初始化/训练轨迹和容量未被同一新实验控制。[J统计/对照报告]

## A2：唯一拟议三组

|组|融合规则|gate参数化|可训练参数总数|回答问题|
|---|---|---|---:|---|
|G0|固定g=.5|无可训练gate|332379|固定双路径基线|
|G1|所有窗口共享同一g|**同G2的5→8→1网络，输入固定全1的5维无量纲上下文**|332436|可学习全局融合能否解释收益|
|G2|g取实际历史五项均值|原冻结F2 gate|332436|历史条件依赖是否比全局权重有帮助|

G1的**有效输出只有一个全局权重**，不是57个不同融合比例。采用57参数而非单scalar logit，是为使主要G2−G1比较在相同参数化、初始化、非线性、优化器和容量计数下仅改变条件输入。常量1是PROJECT_EXPERIMENT_DESIGN，不含电化学意义、不用Validation估计；不另试常量、全局gate结构或第四组挑最好。G1参数有冗余，所有窗口梯度沿固定方向，必须报告；参数量相等不意味着能证明任意单一工况变量的因果作用。

单scalar `g=sigmoid(a)`虽更简单（仅1参数），不是本版主要G1：它会把G2的输入依赖与56个参数/参数化差异混合。若负责人坚持scalar版本，须在新实验前明确替换本提案及判定范围，不能新结果出来后更换。当前计划仍为**三组九次**，不同时追加两种G1。

G0无需虚设57个unused参数来声称公平。332379与332436的57参数差是“是否学习权重”的真实处理；主要工况信息增益由G2−G1（同57参数）判断，G0辅助判断可学习融合收益。[历史参数量：L模型manifest；新G1数目为结构算术预期，待授权后的实例化验收，不是本轮实测]

## 初始化、随机性及训练公平

每seed先建立一套**全新随机初始化**的共同TFT+CNN核心初始state，三组使用完全相同的该核心起点；不是从历史checkpoint复制。G1/G2 gate初始state完全相同，输出层仍零初始化。用隔离RNG创建gate，不能额外初始化改变共同核心、DataLoader或dropout的随机序列。

三组初始g=.5，授权后应在合成输入eval模式核对初始输出/共同state、参数数目及G1不同窗口输出同值；这属于未来工程验收，本轮未实例化/forward。共同seed42/52/62，DataLoader专用同seed generator；train/eval切换一致，num_workers=0，shuffle=true。保持H正常CUDA设置，不强称逐位确定性；禁止为某组反复重跑到好成绩。

核心、两路径、head都正常训练；G2绝不冻结encoder或获得warm-start优势。算法/超参/数据完全沿用H，具体见P_MATCHED_EXPERIMENT_DESIGN。max450/pat40；各组独立early stopping、各自最佳Validation epoch正常冻结，不强迫同epoch，也不把其它组最佳epoch借过来。没有额外full-train refit；继承H训练与Validation选epoch规则。

## 预算与一次性停止

3组×3Seed=9次；不追加F0重训（历史F0仅背景）；不热启动历史G2，不用老F1代替新G0。GPU串行，组顺序按Seed轮换G0/G1/G2、G1/G2/G0、G2/G0/G1，控制机器热/运行顺序影响，不使用该顺序选择结果。

H保存mean time：F1约901s、F2约1279s/run。按同环境九次约2.3–3.2 GPUh仅粗略外推；为max450全跑和I/O/验证预留**4–8h排期**，不是ETA保证。本轮不探测GPU、不跑epoch；实际运行预算先单次批准，超过8h保留现场申请决定，不更改epoch上限。最大4050epochs，ceil(1380/32)=44batch/epoch，对应最多178200 optimizer steps。硬件RTX3070 Laptop8GB，新增gate57参数低成本，但显存可运行性须授权后验收，不能由历史结果保证。

只开展一次，完整九结果后按预定门槛收口。技术失败是技术失败，不改称“无增益”；不删除现场、补Seed、换gate、借Assessment反复改方案。
