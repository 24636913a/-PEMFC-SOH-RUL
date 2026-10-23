# 辅助电压完整血缘

raw有效10cells(U1–U11排除U6) → instantaneous cell mean / cross-cell std / range / min/max → bin统计；exported U.S.AveCell为并行电压来源。

这些观测 → EWMA或其他邻接/滚动缓存 → 后续历史槽位；以及matched_condition_voltage、运行点电压reference、P1S/retention/B2/latent，都可能携带标签信息。仅mask cell_voltage_mean不足：其余cell通道、min/max、range和EWMA后继仍可复制。

本轮拟议独立辅助view采用显式9个NON-voltage Raw14通道，所有时间的所有电压血缘及voltage-derived cache都不暴露。没有仅删除当前dwell电压后保留污染EWMA的做法。age/cycle/curve ID不在可见列。真实Train列值扰动和synthetic raw/exported/EWMA perturbation均调用生产view验证不变。

这是一项充分保守的列级隔离设计，不是已经构造完成的真实dwell训练集。真实label dwell的source/time身份尚未恢复，不能声称完成完整标签窗口联接。原F2主输入/Top10/EWMA/scaler均不修改；未来同容量非物理对照必须采用同一辅助view。未建立辅助网络、编码器或训练任务。
