# 公平实验与事前判定方案（待授权，不执行）

## 唯一有理由申请的实验：门控9次

三组G0/G1/G2定义见P_GATE_INDEPENDENT_ABLATION_PLAN，其中G1为固定上下文的同型57参数全局gate，G2为真实历史上下文gate。G2−G1为主要工况条件依赖比较；G1−G0帮助区分学习全局比例。G0−历史F1不能当复现成功证明，三组新权重都应独立保存。

共享历史随机核心initial-state哈希、G1/G2 gate初始-state哈希、optimizer/DataLoader配置、代码/依赖完整身份和初始输出一致性，应在**未来获授权后、训练前**验收。原F2源码/九checkpoint/主协议不修改；新实现只允许指定实验版本的融合输入开关。当前没有创建模型或训练适配器。

## 完全沿用H的冻结配置

|项|值/authority|
|---|---|
|数据/输入/输出|RealHyFC，0.5h，48×14 slots/Top10，直接24 ΔV|
|Train/Validation|rows[0,2103)/[2103,2404)，1380/205合法windows|
|N1读取规则（未来执行时）|read_csv nrows3007→end_age<1819.997738→3005；禁止full-N1哈希；需继承prefix-safe读取，不执行当前loader|
|scaler/滤波|unfiltered合法Train histories fit；对照冻结E mean/scale；EWMA alpha.3 voltage-only；标准化空间Top10零mask|
|backbone|hidden64、continuous32、attention heads4、LSTM1、dropout.1；原causal d1/d4分支|
|optimizer/loss|AdamW；standardized ΔV MSE；lr .0009605449687795866；weight_decay 6.95942232671908e−6|
|batch/epochs/stop|32；max450；patience40；Validation RMSE改善>1e−12；gradient clip1|
|seed|42/52/62；专用DataLoader generator；不热启动、不挑Seed|
|runtime|继承H正常CUDA/cuDNN/TF32设置；RTX3070 Laptop8GB；不要求不受支持的bitwise确定性|

训练early stopping继续使用旧Validation，但该区域已反复反馈设计，结果**仍为探索性Development**。不利用Assessment选epoch/模型/gate/常量或阈值；不读Test。数据边界、key宇宙和role不改变，所有极端targets保留。

## 主要与强制辅助指标

Formal Query-Weighted RMSE(mV)保持主地位。逐Seed与等Seedmean/sampleSD；MAE；h1…24 RMSE/MAE/P95/P99/max；h24 endpoint；target等权RMSE（每目标平均SSE→等目标平均→sqrt）；top1/5/10 target SSE share、具体keys/multiplicity；参数量、训练耗时、epoch、峰值内存记录。

query重叠、251唯一target不是251个独立实验。禁止p值、bootstrap显著性或“三Seed足够一般泛化”宣称。只做描述性配对，新减对照，RMSE差不当作可加和归因；不删2247/2248，也不以新结果修改主指标。

## 拟议成功标准（PROJECT_EXPERIMENT_DESIGN，不是文献通用门槛）

在任何新训练前固定如下规则；本P提交制定时**已知道J/O1历史结果**，不是盲确认性预注册。这些门槛只规定最后一次实验怎样收口。

1. 技术门槛：9/9完成、共同初始化/数据keys/真值/协议/数值finite、完整持久化/重载审计通过；否则技术失败，不能改写成无增益。
2. G2正式三Seed均值必须分别低于G0和G1，改善至少 `max(0.5mV, 5%×对应对照mean)`；对两个对照独立计算。
3. 同Seed正式RMSE须3/3分别优于G0、G1。不凭最好Seed晋级。
4. 每Seed的endpoint和target等权RMSE，G2不得超过对应G0/G1的1.05倍；各Seed MAE/P99/max以及等Seed均值全部报告，其中MAE和P99均值不得超过任一对照1.05倍。若对照为0，相应G2必须也为0，否则FAIL；浮点核验容差1e−10mV只用于一致性，不改门槛。
5. 全24horizon和SSE集中度公开，不要求每步必胜，也不事后挑有效h段作为主要结果。完整gate分布可描述，不以图相关性取代受控G2−G1。

全部通过才允许“历史工况条件依赖在本次受控Development实验有稳定经验增益”；不能写电化学因果机制。如果只正式均值改善但稳健性门槛未过，写“收益依赖指标/Seed”，弱化独立机制主张。若G2不优G1，则全局学习比例足以成为替代解释；若G1也不优G0，则学习融合整体收益未建立。任何负结果都停止进一步改gate。**不自动以获胜组替换正式冻结F2**，最终模型版本由负责人决定。

0.5mV+5%的要求是控制一次性计算是否值得的工程效应标准，不是显著性证据；即使过线仍有CUDA非逐位、3Seed和Validation复用局限。过不了不能降低阈值重跑。

## 物理增强公平原则：尚未获准实例化

当前B1/B2/B3均不通过，因此没有可执行的“F2 / nonphysics control / Physics-F2”三版本协议，训练次数0，不冻结λ或head。若将来有新的合格物理证据且另行批准，才需同辅助信息、容量、总预算（含teacher拟合/crossfit）的三方对照；单优于原F2不足以证明独立物理价值。

不借本条自动提出补证任务、重新读raw、恢复P3或重启N。只保存公平性条件，不执行或扩展候选。
