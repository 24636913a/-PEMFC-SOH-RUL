# O-05 — O_ARCHITECTURE_OVERLAP_MATRIX

“同构”应同时满足输入对象、连接位置、融合算子、训练/推理信息及输出契约。共享一个模块不代表整模型重复；不同框架名字也不代表没有思想重叠。以下为已核对文献范围内判断，不是全领域排他证明。

|对象|核心/融合|尺度与gate|信息/预测/目标|数据与贡献证据|对F2的等级|
|---|---|---|---|---|---|
|S1 原TFT|LSTM encoder/decoder、VSN/GRN/attention|组件门控、变量选择；无F2 CNN|past-only和future-known分流；multi-horizon|非PEMFC原方法；GLU等消融|原生模块DIRECT_STRUCTURAL_OVERLAP；新增路径DISTINCT_DESIGN|
|P1 BiLSTM-TFT|替换encoder/decoder RNN|原TFT门控，无CNN分支|历史参数+future时间；递归电压到阈值RUL|PHM5cell；native/improved整体对照；等容量/seed未知|FUNCTIONALLY_SIMILAR；具体路径DISTINCT_DESIGN|
|P2 TFT协变量|原生TFT多变量编码|VSN/GRN原生权重|future-known含T/P/flow；5→1电压|同济singlecell；模型对照/重要性非路径gate消融|原生TFT DIRECT_STRUCTURAL_OVERLAP；工况利用FUNCTIONALLY_SIMILAR|
|P3 TCN-BiLSTM|独立TCN与BiLSTM输出选择|TCN dilation、attn；目标误差oracle|真实目标参与式40；电压/R²验证|PHM500点；整体比较，部署gate未报|双表征DIFFERENT_IMPLEMENTATION_SAME_IDEA；融合DISTINCT_DESIGN|
|P4 DilCNN-attn|短期CNN/skip/attn；长期V/趋势双分支|dilation1/2；add+Luongattn|新测量后更新；长期ARIMA条件及递归|PHM5cell+Pragma15cell；有/无条件对照|dilation FUNCTIONALLY_SIMILAR；双路径DIFFERENT_IMPLEMENTATION_SAME_IDEA|
|P5 CNN-RNN|CNN→RNN串联|CNN细kernel未充分报；RNNgate|一小时lag24变量→V；滚动闭合UNKNOWN|PHM；500trials、50重复，容量不齐|DIFFERENT_IMPLEMENTATION_SAME_IDEA；连接DISTINCT_DESIGN|
|P6 PICCrossformer|CNN→DSW/TSA/HED；物理输出loss|层次尺度；attention非F2gate|j/T/P物理目标；future来源UNKNOWN；主1步/递归多步|PHM+同济；C/PI/PIC对照|局部+全局FUNCTIONALLY_SIMILAR；PINN与F2 DISTINCT_DESIGN|
|P7 PINN|t,j→DNN；模型电压监督|无历史双路径|未来复制j情景；电压到阈值RUL|PHM；去方程/去j，参数识别不全|DISTINCT_DESIGN；实际参数INSUFFICIENT_EVIDENCE|
|S2 RF-TCN|特征筛选→TCN|RF-RFE+TCN+PSO|新测量反馈|PHM；输入与TCN对照|FUNCTIONALLY_SIMILAR局部编码；非同构|
|S3 Vsm-RNN|物理V作为RNN特征|非F2路径gate|已知本步I/标称工况→V|动态实验；有/无Vsm对照|DISTINCT_DESIGN；扩展attention实现INSUFFICIENT_EVIDENCE|
|S4 recovery hybrid|PA-VR/AUKF与GRU独立输出并联|已测窗口误差权重|预定启停、N+1更新→N+2预测|PHM；模型/混合比较|DIFFERENT_IMPLEMENTATION_SAME_IDEA；权重对象DISTINCT_DESIGN|
|S5 M-LSTM|矩阵RNN记忆替换|指数gate|20步电压；future闭合UNKNOWN|PHM；整体模型对比，非模块消融|DISTINCT_DESIGN；多步接口INSUFFICIENT_EVIDENCE|

## 按F2主张分解

1. TFT/VSN/GRN/attention：直接继承S1，不列原创。
2. 因果膨胀CNN编码历史：P4/S2有实质先例，不能首创。
3. CNN/RNN互补：P3/P5已有思想；并行VSN输入与post-LSTM特征位置是具体增量。
4. 历史工况控制路径融合：与输入VSN/组件GLU、误差加权、目标oracle控制对象不同；可描述结构区别，独立性能价值未证实。
5. unknown-future direct ΔV：任务协议明确，不是新的预测理论或首个无future方法。
6. Physics-F2：未完成，不能计算为已发表竞争中的本项目贡献。

最接近三篇按直接研究定位为P1、P2、P3；**多尺度/双路径思想必须额外突出P4**，不能只对比有接口缺口的P3来夸大差异。未发现核读集合中的完整同构F2，但不主张“不存在其他相似文章”。
