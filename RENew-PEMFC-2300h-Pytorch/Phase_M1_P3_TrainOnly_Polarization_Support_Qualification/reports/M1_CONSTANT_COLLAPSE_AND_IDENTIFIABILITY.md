# 常数塌缩与可辨识性

L_order=max(0,q_high−q_low)²。任意常数q=C均给0损失。固定下降线q(j)=C−aj(a>=0)也为0，能忽略工况和材料状态；同曲线身份记忆及电压后继复制也可制造虚假成功。生产order_loss与合成测试明确展示常数/下降线/反向线。

非退化监督需要合法独立实测幅度或独立半经验校准，明确V/mV单位，标签从visible view彻底排除，校准/证据时间隔离、无共享dwell，不使用未来actual。幅度项可在数学上排除常数，但必须至少存在可验证的幅度变化及跨j/条件证据；单调次序不能证明这些。

本轮没有已认证dwell或独立幅度证据，故`P3_NONDEGENERATE_SUPERVISION_NOT_ESTABLISHED`。不是把未测膜水设常数，不拟合j0/ASR/jlim，也不运行神经网络。不能由拟议safe view和一个可写公式推出任务已经可学习。

最小支持floor为每时间域20个无共享dwell pairs、3负载水平、2独立段，均为冻结项目设计而非充分物理证明。未来若执行，需报告常数预测误差、j-only线误差、condition/state忽略、curve ID记忆与voltage-copy控制；不得先看结果再换criterion。本轮仅作数学/数据资格审查。
