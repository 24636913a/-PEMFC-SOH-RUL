# 物理损失的非平凡性与接口审查

严格方程约束必须同时满足物理/信息/可辨识/梯度/量纲门槛。半经验模型需限定已验证的参数域；工程范围、平滑和幅度正则只能称 physics-inspired。此分级以 [PINN 作者框架](https://arxiv.org/abs/1711.10561) 为概念参照，但本项目不会把所有代数正则等同 PDE-PINN。

## 输出接口：只读设计，没有导入/调用模型

现有 `CanonicalStrictHistoryTFT.predict_delta` 输出 `[B,24]` standardized ΔV；F2 在 F1 `fuse_encoder` 处以历史 context 门控混合 encoder LSTM/conv。decoder 只有 relative step，14个raw占位零且不参与decoder VSN。F2源码、权重和 API 均不改变。

拟议物理 adapter 接收 `delta_std`、`origin_voltage_raw_V`、冻结 `target_scale_V`、合法物理 context 与适用性 mask。其输出可定义 residual_V、eligibility_reason、unit_contract；但这些只是设计契约，**没有新增可运行 physics adapter/loss**。

\[
\hat V_h=V_t^{raw}+s_V\hat\Delta_{std,h},\quad
L_{phys}=N^{-1}\sum (r_h/s_{phys})^2.
\]

sV 是既有 Train V尺度；sphys 只能是将来预注册的 Train 派生/物理参考尺度，未选定。残差 r 为 V，则上述 loss 无量纲。任意 λ 均未选定。origin V 未滤波，不能以 EWMA 值取代。

## 梯度检查（解析而非模型 forward）

已知真实合法电化学 context 时，\(r_A=\hat V_h-f(u_h,p)\) 对 standardized ΔV 的导数是 sV，非零；但目前 u_h/p 缺失，梯度存在不意味着计算合法。输入恒等式 rB=j−I/A 对输出导数0；它的loss对只预测电压的网络参数梯度也0。范围惩罚仅边界激活时有梯度，宽域内梯度0；即使有梯度也不等于有信息或物理正确。

历史 P3 辅助 rC(qφ(encθ(Xhistory))) 可通过 φ/θ反传，不直接约束最终24步输出；若 encoder 固定，仅新增头获得辅助梯度，不能改变既有未来预测。因此未来若研究迁移贡献，需要独立训练的新版本/相同初始化对照，而不是宣称当前冻结 checkpoint 已被物理增强。

## 三个反驳检查

1. “公式已知即可PINN”：缺界面状态/参数/未来负载，残差还没有函数值；不成立。
2. “输入已满足I/A所以物理损失有效”：梯度为0且输入冗余；不成立。
3. “历史极化成立必能提高未来ΔV”：共享表征可能无用甚至损害负载变化预测；只有受控消融能够反驳/支持。

这三项为本代理的自我反证，不声称有三个独立审稿人。结论：尚无已验证、无需未来真实工况、对冻结 F2 未来电压形成非平凡约束的严格机理残差。
