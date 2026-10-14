# 身份分层
 旧N1报告保留，不作改写。本轮补充按“1月名义BoL”和“10–11月BoL敏感性”拆分的证据，避免把旧报告关于独立8-cell敏感性的结论扩大到所有BoL文件。
 原IJHE方法页明确10-cell durability和独立8-cell fresh-sensitivity。[原论文](https://doi.org/10.1016/j.ijhydene.2026.156060)；名义BoL方法的前作说明10-cell及CDDM位置，且实际1月cell-EIS有10个正电压通道。[前作](https://onlinelibrary.wiley.com/doi/10.1002/fuce.70057)。这些支持同配置，但原始文件无物理电堆serial，仍不宣布SAME_STACK_VERIFIED。
 新鲜8-cell敏感性系列对durability10-cell属DIFFERENT_STACK_CONFIRMED；其晚于老化实验的测量不可在早期预测起点作校准。名义BoL对1150h标CONFIGURATION_CONDITIONAL，同物理堆绑定仍UNKNOWN；不能计算同堆老化归因。
 cell-EIS1150的文件00为文献所述CDDM上方5-cell复合参考，不是整个10-cell阻抗，也没有除5/除10移作单电池ASR。空间文件00是参考单电池；01–16涉及局部电流响应，而不是16块独立MEA。
 文献图给出7入口、2中央、7出口的区域定义；但所发布文件没有独立接线表证明导出index与图中区域或U1–U11的一一关系。CHxx和unit329xx是仪器通道/设备ID，不是电堆serial。未用电压接近来猜U通道，不翻转空间负实部，不用全280cm²作为局部区面积。
 逐对身份状态、证据和限制见EIS_STACK_IDENTITY_MATRIX.csv，逐通道见MEASUREMENT_CHANNEL_AND_UNIT_MAP.csv。
