# 周期感知现代递归架构

## 实现边界

本阶段实现为 `local auditable implementation（本地可审计实现）`。结构思想来自 Beck 等人的 xLSTM：sLSTM 使用指数门和稳定化标量记忆，mLSTM 使用矩阵记忆与 covariance-style update；代码针对 48 步、多变量连续时序预测重新实现，不是 NX-AI 官方语言模型仓库的逐行复现。

## sLSTM 编码器

每一步生成 candidate、input、forget 和 output pre-activation。input/forget gate 使用指数形式；实现把标量 numerator/normalizer recurrence 写成严格因果的 log-domain softmax 等价形式，以 log-sum-exp 完成稳定化，避免指数门尺度直接传入隐藏状态。两层均使用 hidden dimension 64，层间 dropout 为 0.1。

## mLSTM 编码器

每一步将递归状态映射为 4 个 head 的 query、key、value。每个 head 维护 `16×16` 矩阵记忆和长度 16 的 normalizer；使用遗忘门和指数输入门执行 rank-one covariance-style update，再以 query 读取矩阵记忆。两层均使用 hidden dimension 64。

## Hybrid xLSTM 编码器

固定顺序为 `sLSTM → mLSTM`。该顺序先进行标量门控的状态追踪，再由矩阵记忆建模更丰富的状态交互，不搜索反向顺序。

## Cycle-Aware Memory（周期感知记忆）

编码器输出 48 个隐藏状态。真实周期 `P=19` 时，取索引 `47, 28, 9`，即当前、上一周期同相位和上上周期同相位状态。共享 scorer：

`Linear(64,32) → Tanh → Linear(32,1) → Softmax`

得到 `α0, α1, α2`，构成加权周期上下文。融合门为：

`g = sigmoid(Linear([h_t;c_cycle],64))`

`z = g*h_t + (1-g)*c_cycle`

最终使用 `LayerNorm(64) → Linear(64,24)` 直接输出未来 24 个电压增量。错误周期对照只把 `P=19` 替换为预注册的 `P=15`，其余结构与参数完全相同。

## 参考来源

- xLSTM paper: https://arxiv.org/abs/2405.04517
- NX-AI official repository: https://github.com/NX-AI/xlstm
- inspected upstream release: v2.0.4 (`add5001`), no upstream code copied or modified.
