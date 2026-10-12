# O-03 — 物理融合全文审查（不启动物理开发）

## P6：确有约束神经输出的物理项，不自动保证可辨识

Wei等，[10.1016/j.jpowsour.2026.239941](https://doi.org/10.1016/j.jpowsour.2026.239941)，PDF p5 Fig3、p6式12–15、p7式16–17与§3.5，p10表3。卷积串联Crossformer；不是F2双路径TFT。

式12：`Vstack=ncell(En-eta_a-eta_o-eta_c)`。式13为T/P相关Nernst项；式14活化log(j/i0)、欧姆j*Rc*Ac、浓差log(1-j/iL)；式15 `i0=i00 exp(-k_i0 t)`、`Rc=Rc0+k_R t`、`iL=iL0 exp(-k_iL t)`。必须满足j/iL<1、正交换电流和明确压强标准态；这些前提不能靠自动微分替代。j单位A/cm²、面积cm²、RcΩ形成电压V；t为h。单位与参数域核对不等于RealHyFC的材料、传感器与状态可比性认证。

六个初始/老化参数进入训练。图3将Vhat、Vphysics及真实电压分别送到物理/数据loss并反向传播；三阶段为纯数据、物理、联合。判别：

|备选理解|本轮结论|依据/缺口|
|---|---|---|
|A 直接约束神经输出|FULL_TEXT_VERIFIED|Fig3、式16、§3.5|
|B 仅训练独立物理子模型|不足以描述整个方法|存在神经输出与物理预测的直接损失连接|
|C 真V单独先拟合物理参数|UNKNOWN|数据阶段/联合监督明确，未给独立先验物理标定完整流程|
|D 未来物理工况假设/额外actual|UNKNOWN|j/T/P如何在目标时刻提供没有可复现信息清单；不能说一定偷看actual|

式16印刷缺少式17明确求和形式，准确loss实现AMBIGUOUS_FROM_SOURCE；不替作者填成标准MSE。参数识别秩、独立极化拟合、参数协方差、各参数物理区间与传感器语义没有完整证据。p10表3存在C/PI/PIC消融，但没有等容量/同信息/同预算证明。主要24窗MISO1步，多步autoregressive；不等同F2直接24步。

## P7：膜/催化层先验与假定未来电流

Ko等，[10.1016/j.apenergy.2025.125318](https://doi.org/10.1016/j.apenergy.2025.125318)，p4 Fig4、p5 §4.1、p6 Fig7、p7表2/3、p8 §4.3、p9 §4.4/式5–7。

输入(t,j)的8×20 tanh DNN输出电压；两种物理模型给电压目标，加数据loss，非TFT历史表征或时间PDE residual。膜方程指向其参考[39] Han等式49/53/56，催化层指向[41] Fowler等式1–7，本PDF未完全给出方程及本实验参数集。**不能编造膜厚、活性面积等具体标定值，也不能宣称逐参数可辨识已经核实。**

未来j由Train历史段复制，属于明示假定未来工况，不是获取真实未来actual。PHMFC1稳态/FC2高频小纹波环境与RealHyFC未知负载不同。去所有物理方程、去j对照支持论文内部物理先验作用，未独立分离膜/催化层、参数识别或未来复制策略收益。

图4逐点根号平方后平均与文字RMSE存在数学口径落差；实际代码未核实。两段Adam各5000epoch，20次重复的训练随机性范围UNKNOWN。RUL accuracy的阈值、真实EOL不能从方法例子推出本项目EOL；不移植“10%”作为RealHyFC失败标签。

## 补充：输入增强、输出融合与残差约束不是一回事

S3 Liu等，Energies2025，[10.3390/en18102665](https://doi.org/10.3390/en18102665)：p14–15 Fig7/式38把Vsm作为RNN输入，本步I已知、其他条件用标称值；不等于 `Vpred=Vphys+RNN_residual`。p16表6有同RNN有/无物理输入对照。结论中的注意力/残差/双向措辞未充分落实到方法，保留歧义。

S4 Wu等，[10.1109/TEC.2023.3311460](https://doi.org/10.1109/TEC.2023.3311460)：物理PA-VR/AUKF和GRU-A输出并行融合；Rec(t)在物理分支内，权重取已测窗口误差；未来启停预设，不是F2未知未来工况。不是物理loss，也不是物理预测先交GRU串联修正。

本轮未发现可以不经认证直接迁移的RealHyFC物理残差。继承M/M1/M2/N：`NO_ADMISSIBLE_PINN_CONSTRAINT_YET`、`M2_STATE_COMPARABILITY_UNRESOLVED`、`RAW_DWELL_EVIDENCE_SUPPORT_INSUFFICIENT`、`NO_PHYSICS_F2_TRAINING_ROUTE_QUALIFIED`。P3辅助极化训练暂停，不调P50，不重配dwell。文献说明有相关方向，**不构成新的Physics-F2训练授权，也不构成F2已完成物理创新**。
