# 软物理方法与适用条件

研究模型仍为**TFT动态双路径模型（TFT-based Dynamic Dual-Path Model）**，使用原F2及P3 G2；保留原生TFT路径、因果多尺度路径、历史工况动态融合、48×14到24步DeltaV预测。没有修改backbone、门控或参数数目。

## 唯一候选和单位

`Vhat(t+h) = V_origin_raw + s * DeltaVhat_std(t+h)`。

DeltaV目标仅除以冻结目标尺度；不加输入电压scaler均值。起点用未EWMA的原始单体均值，不能用滤波输入替代。`s=0.13680318714138442 V`，来自原Train窗口未滤波电压std，不是新估计的未来DeltaV std。

通用实现：`Lphys = mean(ReLU((Vmin-Vhat)/s)^2 + ReLU((Vhat-Vmax)/s)^2)`；未注册某侧边界时省略该项。`Ltotal=MSE(DeltaVhat_std,DeltaV_std)+0.01*Lphys`。两项无量纲。

本次唯一实际筛查候选是`Vmin=0 V, Vmax=None`。lambda=0.01在真实前向检查前冻结；未开展另一阈值、公式或权重搜索。API合成测试的1.0 V只是测试夹具，绝不是RealHyFC注册上限。

## 物理依据及为什么未获资格

DOE/NETL《Fuel Cell Handbook》第7版§2.2、表2-2（印刷页2-4至2-6）说明可逆电位随温度和反应物/产物状态改变；1.229 V对应298 K、液态水标准条件，不能自动作为所有动态工况上限。[官方手册](https://www.netl.doe.gov/sites/default/files/netl-file/FCHandbook7.pdf)

只有已认证正常发电域内，非负电压才可作为很弱的合理性筛查。反转实验报告说明燃料/空气匮乏下可以出现负电压；这不是“不可能测量”。[Liang等，Behavior of PEMFC in starvation](https://www.sciencedirect.com/science/article/pii/S037877530501089X)（原始实验论文；本轮仅获取摘要，全文请求403，未声称全文核验）。

本项目未来工况未知，也缺未来界面分压、水状态及故障域认证。入口历史均值不能保证未来状态，当前未认证统一非平凡边界；这是一项针对当前信息条件的研究判断，不是证明理论上永远不存在可行域。历史0.5 V是特定测量/极化语境限值，不能移作动态均值电压下限。

## 活性与数值筛查

冻结工程筛查要求：Train全部合法1380起点；初始化及至少一个保存角色激活比例≥0.001且loss>1e-12；加权物理项/数据MSE≤0.1。这些是工程准入标准，不是物理界限。本次9项全部激活比例=0，即使只要求任意非零也失败；没有用Validation选择规则。

单侧ReLU平方区间内梯度为0，越界有限梯度直接作用于预测输出。合成梯度有效并不证明真实适用性。判定：`SOFT_PHYSICS_NOT_QUALIFIED`，具体含`PHYSICS_LOSS_INACTIVE`；停止正式训练。

保留Phase P对历史重建、未认证极化斜率、数据教师和短期单调性的否定证据，见[原审查](../Phase_P_Final_Innovation_and_Experiment_Planning/P_PHYSICS_LITE_CANDIDATE_REVIEW.md)。本轮不推翻该结论。
