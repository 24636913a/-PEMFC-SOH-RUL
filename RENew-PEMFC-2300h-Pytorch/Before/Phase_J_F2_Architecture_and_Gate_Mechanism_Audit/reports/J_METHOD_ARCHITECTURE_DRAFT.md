# 可用于论文的方法架构草稿

## 整体方法与信息集

本仓库采用工况感知双路径严格历史TFT预测动态运行电压变化。输入仅为预测起点及此前已观测的多变量历史，主输出是24步ΔV，而非直接SOH或RUL。方法描述以冻结实现为限，不主张文献范围中的首次提出。

令采样间隔δ=0.5 h，历史长度L=48、预测长度H=24。原始单体平均电压记为V，历史输入为

\[
X_t=[x_{t-47},\ldots,x_t]\in\mathbb R^{48\times14},\quad
y_{t,h}=\Delta V_{t,h}=V_{t+h}-V_t,\ h=1,\ldots,24.
\]

## Train限定特征处理

历史Train窗中第j变量的起点值与每个horizon的ΔV计算Spearman相关，冻结的排序分数为

\[
S_j=\frac{1}{H}\sum_{h=1}^{H}|\rho_S(x_{t,j},y_{t,h})|.
\]

实现对undefined相关使用nanmean，稳定按Raw索引解决同分，电压为必留变量。只消费历史已经冻结的Top10（0,3,13,11,4,9,6,7,10,8），本轮不重算排名。掩码在标准化空间置零未选槽位，保留14槽模型接口。

每个历史窗内仅对电压独立进行因果EWMA，首点以原始电压初始化：

\[
\bar V_1=V_1,\qquad \bar V_\ell=0.3V_\ell+0.7\bar V_{\ell-1}.
\]

其他13变量保持原值。输入scaler由未滤波合法Train历史拟合并保存，当前直接复用：

\[
\tilde x_{\ell,j}=m_j\frac{x^{\mathrm{filtered}}_{\ell,j}-\mu_j}{\sigma_j},\qquad
\tilde y_{t,h}=\frac{\Delta V_{t,h}}{\sigma_V}.
\]

目标没有额外减均值，σV为原输入电压scale。不要把标准化目标误写为原始电压V。

## TFT变量选择与历史编码

原生TFT对14个连续槽位和确定性历史相对坐标进行prescaling与变量选择，形成E∈R^{48×64}。可概括为

\[
E_\ell=\sum_j w_{\ell,j}\mathrm{GRN}_j(\tilde x_{\ell,j}),\qquad \sum_jw_{\ell,j}=1.
\]

该式仅概括源码VSN，不替代实际GRN/GLU实现。无真实静态协变量，静态embedding为零，保留原生static-context网络的已训练常量变换。原生LSTM历史编码与其后GLU、AddNorm产生H_LSTM∈R^{48×64}。编码器最终hidden/cell仍交给原生decoder，与后续fusion分开。

## 多尺度因果卷积

卷积分支同样消费E，而不是消费H_LSTM。对每个通道c，k=3的depthwise卷积写为

\[
C_{d,\ell,c}=b_{d,c}+\sum_{u=0}^{2}w_{d,u,c}E_{\ell-du,c},\quad d\in\{1,4\}.
\]

负索引用左侧零padding，右侧不补值，因此仅访问过去/当前位置。感受野跨度为3/9个采样点；对应最早到当前时间跨度1/4 h。d=4分支采样的是0、4、8步间隔，不等同于9点连续平均。

\[
H_{\mathrm{Conv}}=\mathrm{LayerNorm}\{E+\mathrm{Dropout}[\mathrm{PWConv}(C_1\Vert C_4)]\}.
\]

通道64+64拼接128，经kernel1 pointwise压回64。没有另加卷积激活函数。

## 固定与工况感知融合

F1历史融合为

\[
H_{F1}=0.5H_{\mathrm{LSTM}}+0.5H_{\mathrm{Conv}}.
\]

F2按Raw索引J=(11,13,9,6,8)，从标准化masked历史输入取工况上下文

\[
z_{t,r}=\frac1{48}\sum_{\ell=1}^{48}\tilde x_{t,\ell,J_r},\quad z_t\in\mathbb R^5.
\]

五变量为load_transition_count、zero_current_fraction、current-density std、current std、current-density mean。z不是原始电流单位；因这五项均被保留且未被电压滤波，物理窗口均值可按μJ+σJz恢复并独立核验。

\[
g_t=\mathrm{sigmoid}\{W_2\mathrm{ELU}(W_1z_t+b_1)+b_2\},
\quad W_1\in\mathbb R^{8\times5},\ b_1\in\mathbb R^8,
\quad W_2\in\mathbb R^{1\times8},\ b_2\in\mathbb R.
\]

\[
H_{F2}=g_tH_{\mathrm{LSTM}}+(1-g_t)H_{\mathrm{Conv}}.
\]

单个g在历史时间和特征维广播，不是每个horizon一个g，也不是最终预测混合。初始化输出层零使初始g=.5，训练后的g必须实际采集，不假设保持初值。

因果性相对于预测起点定义：g使用起点之前全部48步历史，因此某个较早历史位置的融合表示也可受窗口内较晚、但已经观测的工况影响。这不等于在每个历史位置进行在线逐点因果更新，也不构成访问起点之后的未来信息。卷积分支自身仅使用当前位置及更早历史；两种因果性描述须区分。

## 原生解码与预测输出

Decoder只输入h/24的相对坐标，未来14个物理槽为零且不进入decoder VSN。原LSTM encoder的hidden/cell初始化decoder，随后decoder GLU/AddNorm保持。融合历史与decoder表示拼接，经过static enrichment GRN，原生interpretable attention（Q为decoder，K/V为完整历史+decoder，因果decoder mask）、post-attention GateNorm、position-wise GRN、pre-output GateNorm及逐步scalar projection：

\[
\widehat{\Delta V}_{t,1:24}=\sigma_V\,f_{\theta}(\tilde X_t),\qquad
\hat V_{t+h}=V_t+\widehat{\Delta V}_{t,h}.
\]

pre-output GateNorm仍使用未被融合改动的decoder residual。没有新增SOH/RUL头、未来真实工况或物理退化微分方程。

## 评价、复杂度与证据边界

\[
\mathrm{RMSE}=1000\sqrt{N^{-1}\sum_{(t,h)}(\widehat{\Delta V}_{t,h}-\Delta V_{t,h})^2},\quad
\mathrm{MAE}=1000N^{-1}\sum_{(t,h)}|\widehat{\Delta V}_{t,h}-\Delta V_{t,h}|.
\]

上式单位mV，预测查询等权，重叠窗口不独立。目标时刻等权只作为既有敏感性指标，先平均同一target的平方误差，再跨target平均开根，不能先平均预测。三Seed算术均值与pooled RMSE不同。

F0/F1/F2参数量323483/332379/332436。卷积分支新增8896，门控新增5×8+8+8×1+1=57。没有测量延迟或效率提升宣称。

Spearman、EWMA、Optuna-TPE是成熟的筛选、预处理和配置优化工具，不称为新算法。结构贡献是保留原TFT并增加双路径历史表示及历史工况自适应融合。F1无稳定增益，不能声称卷积必然提高精度。F2的历史Validation和复用Assessment三Seed方向性改善支持限定的经验预测优势。门控相关与推理时固定替换仅解释数学/数值依赖，不能单独证明不可逆退化的物理机理。


Base: `427d4dc402f083a5b9e094b12981c5e231df8100`. Protocol SHA256: `0244f05fd777ccd615fda7eb9533d129323dc27fbb58212e091130011d0d9b24`. Code identity SHA256: `67b0d3588f75db56d435ddc5277d38324708f660ac10cc1850a5908d771612c8`.

Validation only, 205 origins/seed, 615 gates; all 4920 queries per seed retained. New training/Assessment inference/Test/seed selection: NO.
