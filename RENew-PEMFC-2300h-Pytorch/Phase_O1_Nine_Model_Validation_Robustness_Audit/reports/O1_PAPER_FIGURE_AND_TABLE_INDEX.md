# O1 — 论文候选图表

六组原创图，每组PNG300dpi+PDF矢量，均来自身份全部通过的九个历史Validation缓存：

1. Fig1_Validation_Horizon_RMSE_Three_Seeds：3个Seed panel、全部24horizon、统一y尺度。
2. Fig2_Validation_12h_Endpoint：三个Seed三模型h24 endpoint，起点数量205/run。
3. Fig3_Formal_vs_Target_Time_Equal：正式query与target等权敏感性；不改变主指标。
4. Fig4_Validation_Tail_and_Max_Error：P95/P99/max完整对照，max独立轴避免隐藏极端值。
5. Fig5_Validation_Target_SSE_Concentration：Top1/Top5目标贡献，比例0–1，不等同RMSE指标。
6. Fig6_Validation_Horizon_Equal_Seed_Mean_SD：等Seed均值和sampleSD，不是CI或ensemble。

数据总表 `O1_NINE_MODEL_VALIDATION_METRICS.csv`；逐horizon `O1_HORIZON_METRICS.csv`；等Seed及horizon汇总两个CSV；目标SSE、配对模型/步长/query差与Seed62极端queries完整保存。CSV列名明确mV、mV²、share和counts，全部真实数值未格式化成字符串。spreadsheets技能用于单位/来源/去重纪律；用户要求机器可复现CSV和Python计算，未额外创建Excel工作簿。scientific-figure-making技能用于语义色、印刷线型/纹理、统一尺度和矢量输出。

图题均Validation；没有Independent Test、Cross-Stack Generalization或RUL Accuracy主张。可用作条件性F2论文实证图；图表完整不改变独立Test资格。正式主指标和全部Seed保留，不能只引用最有利图。
