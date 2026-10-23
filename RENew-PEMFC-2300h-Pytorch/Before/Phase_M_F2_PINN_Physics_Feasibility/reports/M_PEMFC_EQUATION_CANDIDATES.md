# PEMFC 方程候选与量纲契约

理论依据：[DOE Fuel Cell Handbook 第七版，2004，第2章 §2.1–2.6、第3章](https://www.netl.doe.gov/sites/default/files/netl-file/FCHandbook7.pdf)。本报告的可辨识性、梯度和聚合讨论是本项目数学审查，不是手册对 RealHyFC 的验证。损失表达式是设计，未部署。

## A：电化学电压分解

\[
V=E_{rev}-\eta_{act}-\eta_{ohm}-\eta_{conc},\qquad
E_{rev}=E^\circ(T)+\frac{RT}{2F}\ln\frac{a_{H_2}a_{O_2}^{1/2}}{a_{H_2O}}.
\]

V、E、η 为单 cell V；R 为 J/(mol·K)，F 为 C/mol，T 为 K；活动度 a 无量纲。理想气体时气体活动度近似 p/p°，纯液态水活动度近似 1 的假设必须明确；水蒸气条件不能无条件置 1。入口总压不足以确定反应界面活动度。

在阳极/阴极有效 Tafel 区间、稳态且机制不变的近似下：

\[
\eta_{act}=b\ln(j/j_0),\quad b=RT/(\alpha n_{eff}F),\qquad
\eta_{ohm}=j\,ASR=I\,R_{cell},\qquad
\eta_{conc}=-B\ln(1-j/j_{lim}).
\]

j/j0/jlim：A/cm²；b/B：V；α、n_eff 无量纲；ASR：Ω·cm²；Rcell：Ω。浓差式要求 0<j<jlim、传质近似成立；Tafel 式不是零电流或交换电流附近的通用式。多电极有效参数不是任意固定的材料常数。

项目支持：历史 I/j/A 支持单位部分；没有证据闭合 T、界面活动度、j0/α/ASR/jlim/膜水的联合估计。将参数写成 trainable 不能证明其可辨识，也不能任意设常数制造“真实 residual”。未来 24 步实际 j/T/P/RH 既未知又不在 decoder。

30 min 平均一般满足 \(\overline{f(j,u)}\ne f(\bar j,\bar u)\)。这是非线性与协方差带来的聚合问题；输入的 j std/p2p 不足以还原 joint distribution。只能在另行验证的低转移、稳定 dwell 子域采用准稳态近似，不可把全部动态 bin 当极化测试。

## B：电流—电流密度恒等式

\[
j=I/A,\quad r_B=j-I/A.
\]

I：A，A：cm²，j/rB：A/cm²；同 raw samples、同面积及同单位时成立。G-STOP 的 Fold1 Train 支持约 280.001183 cm²。均值/std/p2p 的对应关系有历史支持；不是独立电压机理。

F2 仅输出 ΔV，故 \(\partial r_B/\partial\widehat{\Delta V}=0\)，只处罚输入身份无法更新电压模型。新电流头会改架构/任务，且该恒等式仍可能仅重编码已经冗余的输入。本轮不采用。

## C：条件化极化次序

\[
\left.\frac{\partial V}{\partial j}\right|_{T,a,ASR,j_0,j_{lim},\,state}<0.
\]

在上述模型正参数/有效域中，导数为 \(-b/j-ASR-B/(j_{lim}-j)<0\)，单位 V/(A/cm²)。推导条件包括工况、材料状态和稳态近似相同；不能把恢复后的不同状态、异温/异湿/异压曲线直接配对。

未来输出次序损失例：\(r_C=\max(0,\hat V_{high\,j}-\hat V_{low\,j})\)，单位 V。它可有非零输出梯度，但 POT 已发现按未来实际工况筛对不被当前协议授权。相同未来 h 不意味着相同条件。

历史辅助版本只考虑全部已发生的 Train dwell，先独立验证条件/状态可比，再用辅助输出 qφ 代替未来 Vhat；其梯度可进入历史 encoder，但不是直接未来机理 residual。详见 P3。

## D：边界

已知局部反应状态、正常发电及非负损失时，理论上 V≤Erev(T,a)。没有未来状态可用便不能获得每步上界。常数 1.229 V 是特定标准态参考，不是对全部未来条件的无条件上界。

工程 penalty 可写 \(r_D=\max(0,\hat V-U)+\max(0,L-\hat V)\)，V。U/L 必须分别注明设备工作域、热力学状态或经验分位数来源；需区分安全阈值与不可违反的定律。Train max/min 或 quantile 只是经验边界；零电流不保证 V=0，异常/反转单体不能被统一正常态下界覆盖。无可靠工作域时归类 PHYSICS_INSPIRED_ONLY，不是严格 PINN。

## E：不可逆状态

概念式 \(V(t)=g(H(t),u(t),w(t))+\epsilon(t)\)，V 为 V，H 为尚未定义/辨识的状态，u 为工况、w 为可逆水/热状态，ε 为 V。若另有不可逆损伤定义才可能假设 \(dH/dt\le0\)，量纲依 H 定义。本项目尚未闭合 H→V 或 irreversible/reversible 分离。

HA 支持 HEALTH_EVIDENCE_NOT_IDENTIFIABLE。不能直接从此概念式制造健康标签，不能以 EWMA/EMD 或观测 dV/dt≤0 替代；本轮不定义 SOH/EOL/RUL。

## 统一输出接口

冻结 F2 输出标准化 ΔV。拟议物理层须先使用原 target scale sV 还原：\(\hat V_{t+h}=V_t^{raw}+s_V\hat{\Delta V}_{std,h}\)。sV=0.13680318714138442 V（既有 Train scaler）；Vt 是 origin-known 未滤波平均电压，不是 EWMA 末值。loss 必须明确 V² 或除以仅 Train 冻结的参考尺度后无量纲；不得把 standardized voltage 直接代入电化学式。
