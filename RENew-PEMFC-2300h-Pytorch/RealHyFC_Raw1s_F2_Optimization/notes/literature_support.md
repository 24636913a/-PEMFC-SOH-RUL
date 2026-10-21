# 原始秒级 F2 优化：文献支持与适用边界

检索日期：2026-10-21。用途是为本轮固定实验提供依据和边界，最终数值结论由本轮真实运行结果决定。这里不作系统综述，不写预定胜负，也不授权额外训练。逐源核查记录在 `outputs/literature_verification.json`。

## 核查方法与诚实边界

先通过 Semantic Scholar API 查询 DIR 论文的 arXiv ID，实际收到3次 HTTP 429，并在重试前各退避2秒。随后全批降级为 DOI 信息与原出版商、会议论文集、作者机构页面核查。没有声称取得 S2 ID、匹配成功或确认“未收录”。

本轮重新用网络工具实际检索并核查了7个来源，未将上一阶段的引用清单直接当作本轮核验结果。若 doi.org 跳转失败，核对的是出版商页面写出的 DOI、题名和作者，不把这说成 HTTP 解析成功。部分原始全文仅能通过出版商/作者页面的搜索节选读取，JSON逐项说明。

核验范围是论文存在、书目信息和本文所用主张。没有完成全部期刊索引、撤稿记录、作者履历或财务利益冲突审查，也没有独立复现实验。来源论文及预印本只计一篇。

## 7个一手来源及它们实际支持的内容

| 来源与准确链接 | 实际支持 | 本任务中不支持的外推 |
|---|---|---|
| Steininger 等（2021），[Density-based weighting for imbalanced regression](https://link.springer.com/article/10.1007/s10994-021-06023-5)，DOI `10.1007/s10994-021-06023-5` <!--ref:steininger2021dense--><!--anchor:section:3.1%20DenseWeight%20and%203.2%20DenseLoss--> | 以训练目标密度构造权重，研究稀少目标与常见目标间的损失取舍。原式(6)为 `1/N * sum(w * M)`。 | 不是本轮的事件前缀标签，不证明权重3最优，不证明F2获益；逐批以 `sum(w)` 归一化也不是原式的原样实现。 |
| Rudy 与 Sapsis（2023），[Output-weighted and relative entropy loss functions for deep learning precursors of extreme events](https://www.sciencedirect.com/science/article/pii/S0167278922002743)，DOI `10.1016/j.physd.2022.133570` <!--ref:rudy2023extreme--><!--anchor:section:Abstract%20and%20Introduction--> | 在动力系统实例中研究强调极端输出的损失和相对熵目标，适合作为“值得检验事件敏感目标”的机理性方法依据。 | 不等于所有重加权都有效，不证明本轮二元1:3规则，不提供RealHyFC切换时间必然可预测的证据。 |
| Yang 等（2021），[Delving into Deep Imbalanced Regression](https://proceedings.mlr.press/v139/yang21m.html)，ICML/PMLR 139，11842–11851 <!--ref:yang2021dir--><!--anchor:section:Abstract--> | 连续回归的不平衡需要考虑相邻目标关系，论文研究标签/特征分布平滑。 | 目标值稀少与动态事件稀少并非同一概念；不能自动移植分类权重，不能把LDS/FDS或新模块加到本轮。 |
| Kendall、Gal 与 Cipolla（2018），[Multi-Task Learning Using Uncertainty to Weigh Losses for Scene Geometry and Semantics](https://openaccess.thecvf.com/content_cvpr_2018/html/Kendall_Multi-Task_Learning_Using_CVPR_2018_paper.html)，CVPR，7482–7491 <!--ref:kendall2018multitask--><!--anchor:section:Abstract%20and%20Introduction--> | 多任务损失的相对尺度和权重需要设计，论文给出同方差不确定性权重并检验视觉任务。 | 任务间权重不等于目标秒的事件权重；不能证明事件辅助头一定有益。该方法仅供下一阶段讨论。 |
| Koutnik 等（2014），[A Clockwork RNN](https://proceedings.mlr.press/v32/koutnik14.html)，ICML/PMLR 32(2)，1863–1871 <!--ref:koutnik2014clockwork--><!--anchor:section:Abstract--> | 不同隐状态模块可以在不同时间粒度更新，提供多时间尺度建模的基础例子。 | 其音频、语音及手写实验不证明PEMFC改进，也不确定应选哪些秒级尺度。不能据此改本轮冻结结构。 |
| Challu 等（2023），[NHITS: Neural Hierarchical Interpolation for Time Series Forecasting](https://ojs.aaai.org/index.php/AAAI/article/view/25854)，DOI `10.1609/aaai.v37i6.25854` <!--ref:challu2023nhits--><!--anchor:section:Abstract%20and%20How%20to%20Cite--> | 分层插值、多速率采样可作为长跨度预测表示设计的候选，文中的近似讨论带有平滑性条件。 | 不证明下采样能保留每次电压突变，不允许插值制造原始标签，不把论文长跨度基准收益移植成本任务收益。 |
| Lim 等（2021），[Temporal Fusion Transformers for interpretable multi-horizon time series forecasting](https://www.sciencedirect.com/science/article/pii/S0169207021000637)，DOI `10.1016/j.ijforecast.2021.03.012` <!--ref:lim2021tft--><!--anchor:section:3%20Multi-horizon%20forecasting--> | 区分只能历史观测的量和起点时真正已知的未来协变量，支持明确多步信息集。 | 不能把未来真实电流、真实事件标记或日志中的未来设定值当作预测时可得输入，也不证明F2等于TFT或具有同样表现。 |

Rudy 论文正式卷年是2023，DOI中的2022以及2021预印本日期不应混作正式出版年。NHITS正式AAAI题名与早期“N-HiTS”预印本对应同一个工作，不重复计数。上述基础方法论文的年代不代表它们是截至检索日最好的模型。

## 本轮哪些是用户预设，哪些才是文献依据

普通目标权重1、事件目标权重3，是用户明确预先指定的一次工程检验，**没有文献证明1:3对RealHyFC最优**。事件阈值 `0.038477648942 V`、约75%均匀/25%补充事件采样、最多8192起点、最多50轮，以及B1的3%/5%/10%准入条件，都来自本轮用户协议或其继承的训练段冻结协议，不是由论文推导出的最优值。

本轮事件标记的准确含义是：对目标提前量 `h`，若 `t+1` 到 `t+h` 内任一真实单秒电压跳变超过冻结阈值，则该起点×提前量训练查询的权重为3，否则为1。权重只作用于训练损失。它不是对“目标秒自身是否跳变”逐点加权，也不是按目标绝对电压的密度加权。这个定义应在代码和报告保持一致。

`sum(w * squared_error) / sum(w)` 是用户指定公式。不要将它替换成 `mean(w * squared_error)` 并声称等价。对全体固定样本而言，两者只是一个固定因子；对权重组成不同的小批次，因子会随批次变化，进而改变更新尺度。本段为对公式的直接代数解释，不是引用论文已验证的RealHyFC结论。

约25%的事件起点补充和事件目标权重3会同时增强事件的训练影响。必须保留用户方案，同时报告实际事件起点、事件目标秒占比、独立事件数和同一事件重复覆盖，防止把窗口重复当作大量独立事件。这是本项目统计解释要求，不是新采样或调参建议。

## 如何解释成功、失败及下一阶段

事件加权有依据值得做一次，不代表它必然改善全样本RMSE。若改进集中于突变，但普通区间代价超过冻结要求，应按用户规则判为未通过。即使B1合格，也只是Seed202单次条件性证据，不能推广成稳定优势；B1未合格应取消B2，不借文献理由搜索其他权重。

后续是否增加辅助任务，首先取决于当前误差与历史前兆的审计。辅助任务的真值可以在训练中作为监督，但预测时不能喂入真实未来事件标签。若讨论多尺度模型，应保留原始秒级目标、真实毫秒时间戳和切换信息；只压缩历史表示并不自动保证突变被保存，仍需独立验证。以上均为研究建议，当前固定试验不新增辅助头、门控或多尺度结构。

预测起点每60秒滚动、每个预测输出逐秒以及原始记录约1Hz是三件不同的事。本轮提高起点密度后，120秒任务的不同起点可能预测同一个目标秒，不能把这些查询算作独立物理事件或独立实验重复。文献也不能把历史中没有可提前观测线索的外部命令变成可确定预测量。

## 透明性声明

本说明使用AI协助定向检索、原始网站书目信息匹配和证据边界整理。JSON记录了真实S2失败、网页读取范围和未执行的完整venue/COI核查。原文没有下载到本地，故技能信任链中的 `source_acquired` 与 `source_verified_against_original` 保持false；它们与已执行的一手网页核查是不同声明。研究者仍须复核后使用。本文件不包含新训练结果，也未修改旧研究。
