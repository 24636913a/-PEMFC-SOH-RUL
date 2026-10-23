# 创新组成与可用主张

证据编号与精确路径见 [台账](../evidence/K_SOURCE_EVIDENCE_LEDGER.md)。以下数字全部来自已冻结报告，没有本轮新评估。`PROJECT_VERIFIED_EVIDENCE` 是项目内审计证据，不等于外部独立验证。

|组成|来源与当前实现|成熟方法应用 / 项目结构设计|历史证据与已证明内容|未证明内容|允许 / 禁止表述|
|---|---|---|---|---|---|
|Spearman Top10|L1；E09、E10、E11。Train 合法窗口末时刻各变量与 24 个未来 ΔV 的秩相关绝对值取均值，稳定排序，强制保留电压；14 通道中的非 Top10 标准化后置零|成熟统计量应用；不是新 Spearman 算法|E 系列同阶段 Top10/Raw14 对照是探索性依据；索引 [0,3,13,11,4,9,6,7,10,8] 已冻结|相关不识别退化机制；选择优势未独立盲测确认|允许“训练期监督特征筛选”；禁止“提出新的物理因果特征发现算法”|
|因果 EWMA|L2；E10、E11。每窗口首电压保持原值，此后 s_t=0.3V_t+0.7s_(t−1)，仅电压通道，之后用未滤波 Train 公共 scaler|成熟递推平滑应用；因果实现是工程约束|E02 中 E2 Top10 EWMA RMSE 7.7858 mV，E0 Top10 raw 19.3188 mV；仅同阶段复用 Assessment、Seed42|去噪不等于去除可逆影响；因果不等于无全部选择偏差；不证明 SOH 提取|允许“单侧输入滤波，在该探索对照中改善预测”；禁止“新退化分解算法”“不可逆退化净化”|
|Optuna-TPE|L3/L4；E01。每折 16 attempted / 16 completed，Seed42；Fold2 trial13 配置由后续 E/F 固定使用|成熟优化工具应用；非网络创新|T1-R 最终是 TFT_OPTUNA_INSUFFICIENT_GAIN；C 部分改善，但竞争性/Fold3 门槛未全部通过|不能把搜索框架存在等同于优化路线已晋级；更大预算与搜索效果需区分|允许“使用冻结搜索产生并固定配置”；禁止“Optuna 证明 TFT 全面最优”|
|LSTM+多尺度因果 Conv|L5 原 TFT 已含 LSTM；E03、E11–E14。encoder VSN 两个 depthwise Conv(k=3,d=1/4)，左填充2/8，concat+1×1投影、dropout、残差 LayerNorm，与 post-LSTM encoder 融合，decoder 保持原路|本项目结构组合/插入位置设计；LSTM/Conv/残差均非新算法|工程等价/因果路径证据已审计；F0/F1 参数323483/332379。H 中 F1 对 F0 仅1/3 seed改善，均值9.155598 vs9.067563 mV|不支持卷积分支单独稳定提高预测精度；不支持两个尺度对应独立电化学过程|允许“构建双路径历史编码”；禁止“已证明多尺度分支独立稳定增益”|
|历史工况感知动态融合|E13、E14：48步内5个历史变量均值→5→8 ELU→1 sigmoid，窗口级 g；g H_LSTM+(1−g)H_Conv；57新增参数，F2总332436。无未来实测工况|本项目新增融合位置与条件上下文组合；原 TFT 本身已有 gating，不是首次门控 TFT|H 中 F2对F0/F1均3/3 seed更好；I2复用 Assessment也3/3。J证实 g动态、会改变预测数值|J学习/固定0.5门控等Seed均值7.245400/7.247419 mV，差约−0.002019、方向混合；独立精度增益/物理机制未证实|允许“工况感知联合架构在复用 Development 上呈现经验优势”；禁止“门控独立收益已验证”“识别不可逆退化机制”|

## 主张边界

`PROJECT_VERIFIED_EVIDENCE`：F2 Validation 等 Seed RMSE 7.245400 mV（样本 SD 1.100853），F0 9.067563，F1 9.155598。I2 Assessment 等 Seed均值 F2 12.197052 mV，F0 21.745493，F1 22.892878；这是 **reused Development Assessment**，不是 pristine Test，不是三折均值，也不是预测集成/pooled RMSE。来源 E04–E07。

F2 Seed62 Validation 对 F0 只改善约0.077586 mV；Assessment F2 SD 6.393819 mV，且少数目标贡献大。不得写全工况/全种子普遍稳定，不能把重叠查询作为独立重复。

`INSUFFICIENT_EVIDENCE`：两模块各自独立贡献、统计显著性、外部堆泛化、健康与寿命机制。J固定门控是**同一个训练后 F2 的推理诊断**，不是独立重训 F1，不能替代训练受控消融。

`METHOD_LITERATURE_SUPPORT`：L1–L5支持成熟方法归属；本轮做了定点来源核查，不是完整全球 novelty 检索，不能声称“首创”或保证发表。
