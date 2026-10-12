# N — 方程、单位、参数及输出梯度审查

## 1. 明确审查对象

以下是依据 [DOE/NETL Fuel Cell Handbook，第7版](https://www.netl.doe.gov/sites/default/files/netl-file/FCHandbook7.pdf) 的电压损失分解构造的**项目候选表达式**，不是宣称任何竞争论文逐字采用它。正式典型稳态关系需独立确认状态：

\[
V_{cell}=E(T,a)-b\ln(j/j_0)-j\,ASR+B\ln(1-j/j_{lim}).
\]

|量|单位与域|RealHyFC证据/缺口|
|---|---|---|
|Vcell、E|V，每有效单体|十cell均值，U6排除；stack voltage不得直接代入|
|T|K|入口/出口℃可换K，但不等于界面T|
|a|无量纲相对标准态活动度|入口表压/湿度不能直接得到界面 H2/O2/水活动度|
|j、j0、jlim|A/cm²；0<j<jlim；对数比无量纲|actual j历史可观测；j0/jlim无联合独立辨识|
|b、B|V，非负|需动力学/传质适用域与独立约束|
|ASR|Ω·cm²，非负|j·ASR为V；面积身份不等于ASR已知|

Nernst 可写为 `E=E0(T)+(RT/(2F)) ln(a_H2 a_O2^(1/2)/a_H2O)`，R为J/(mol·K)，F为C/mol。标准态、产物水相及活动度定义必须注册；不能把液态水假设、入口总表压和过饱和RH混拼。Tafel近似还要求相应动力学电流域，不因形式上j>0就适用于低负载/启动；稳态传质形式不能自动覆盖整个动态周期。没有在本轮计算 Nernst值。

M2 raw的高RH已经由历史重新聚合证实来源一致，但物理解释/仪器校准未知；不将RH clip到100来让公式成立。标准流量 nl/min 也不能自动等于电极局部消耗流量。缺膜含水/热迟滞/干预控制，不能将它们设常数升级资格。

## 2. 结构可辨识反例

固定b，`ln(j0')=ln(j0)+c`、`E'=E−bc` 给出完全相同V。无独立E或活动度证据时，j0与截距不可分别辨识。把网络输出命名为j0不改变这个对称性。可以将合并截距作为有效经验参数，但不得冒充独立电化学识别，也仍须证明跨条件可移植性。

窄j域中 `B ln(1−j/jlim)≈−(B/jlim)j`，与ASR混淆。动态水/温度变化与材料老化同样可能解释电压变化；额外自由参数更易欠识别，不因训练NLL/MSE更小就通过。

需要的辨识证据是：有明确状态与协议的 Train测量、受约束参数化及结构rank/灵敏度、独立诊断约束、隔离域预测检验与不确定性/支持范围。本轮全部不拟合、不调用 EIS/极化文件。M2已显示认证支持未闭合。

## 3. 梯度成立与物理知识成立是两回事

F2输出u为standardized ΔV，固定scaler `sV=0.13680318714138442 V`：

\[
\hat V_{o+h}=V_o+s_V u_h,\quad
r_h=(\hat V_{o+h}-V_{phys})/s_V,\quad
\partial r_h/\partial u_h=1.
\]

`r²` 的梯度为 `2r`，非零残差处非零；在满足方程处为零是正常的。但若 `Vphys=Ytrue`，它就是标准化电压MSE，不能称额外物理约束。若教师固定，蒸馏加数据loss有恒等式：

\[
(s-y)^2+\lambda(s-t)^2
=(1+\lambda)\left[s-\frac{y+\lambda t}{1+\lambda}\right]^2
+\frac{\lambda}{1+\lambda}(y-t)^2.
\]

第二项对学生不求梯度。对学生这相当于混合软目标；当t=y，只是MSE重加权。它并不否定所有灰箱蒸馏，但**独立物理价值必须来自教师中可认证、可证伪的结构与等信息对照**，不能来自这个平方损失本身。

## 4. 被明确拒绝的“物理损失”

|候选|数学问题|决策|
|---|---|---|
|j=I/A|仅输入身份，对F2电压输出梯度0|`NOT_ADMISSIBLE`作为电压训练约束|
|全局随年龄电压单调下降|动态负载/可逆恢复不满足，无工况闭合|`NOT_ADMISSIBLE`|
|未经认证dwell排序|常数q给0-loss；固定j-only下降也给0-loss|`NOT_ADMISSIBLE`；P3暂停|
|未来actual条件化残差|实际未来负载不可由同一历史唯一确定|`INFORMATION_SET_MISMATCH`作为预测物理闭合|
|真实未来电压当“物理”目标|同一Y再复制，与MSE等价|`PHYSICS_PRIOR_NOT_INDEPENDENT`|
|30min均值代入瞬时非线性方程|mean(f(j))通常不等于f(mean(j))，统计不足|适用域未成立|
|把F2ΔV损失解释成不可逆H ODE|H未观测、工况/可逆波动未分离|`PHYSICAL_PARAMETER_NOT_IDENTIFIABLE`|

联合训练 θ 与神经学生不能绕过上述反例；θ和输出同时自由变化还可能吸收残差。不存在本轮已通过所有门槛的B路线。

## 5. 合成证据含义

`scripts/n_math_contracts.py` 与 `tests/test_n_contracts.py` 测试截距/j0对称、解析导数、平方完成、t=y退化、常数/j-only排序退化、聚合非闭合、单位适配及未来负载歧义。这些只证明反例或形式梯度，**不是RealHyFC参数识别、教师精度、PINN收益或机制验证**。实际测试数量/日志由独立审计入口写入outputs，不在文档预设成功。
