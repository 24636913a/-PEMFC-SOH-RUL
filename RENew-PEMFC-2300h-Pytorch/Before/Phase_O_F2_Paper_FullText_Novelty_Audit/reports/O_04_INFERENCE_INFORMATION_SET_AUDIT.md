# O-04 — 训练与推理信息集合

## 冻结F2契约

`X_origin=[48,14]`来自预测起点及以前，每0.5h，原始schema经因果电压EWMA、Train-only scaler、Top10零mask。F2网络不接收未来真实电流、温湿压、流量、极化、checkpoint年龄或目标真值。decoder只接收确定的相对horizon索引；其他placeholder为0且不在decoder variable selection集合。gate为五项历史标准化工况的48步均值。ΔV24维点预测通过共同原点电压还原绝对V。

源码依据：`Phase_TFT_Original_Baseline/models/canonical_tft.py`，`Phase_F_TFT_DualPath_RegimeFusion/models/dualpath_strict_history_tft.py`、`regime_aware_dualpath_tft.py`、`frozen_feature_schema.py`；历史训练输入依据 `Phase_H_Determinism_MultiSeed_Stability/scripts/h_common.py` 与 `Phase_TFT_FullFeature_CausalFilter_Comparison/scripts/run_e_comparison.py`。路径相对研究根目录。本阶段只读，不执行loader/scaler/model。

## 逐论文推理边界

|方法|训练额外信息/预处理|预测起点可用信息|目标处/滚动信息|判断|
|---|---|---|---|---|
|P1 BiLSTM-TFT|极化插值、平滑、膝点、RF/TPE|历史多变量；未来时间/cycle|递归阈值预测，实际工况未来不可用；其他协变量闭合细节UNKNOWN|功能相似；不是F2直出协议|
|P2 TFT|历史和future-known分组|历史状态；入口工况/flow列known|known实际可得性证明NOT_REPORTED|信息集不同；不推断作弊|
|P3 TCN-BiLSTM|VMD/KF/中心MF；IALA|两独立预测器输入|式40选择需同点真V|TARGET_DEPENDENT_SELECTION_IN_PUBLISHED_METHOD|
|P4 DilCNN|平滑；长期条件ARIMA|历史电压及Train预测趋势|短期新测量到达后更新；长期递归|新已发生测量不是泄漏；与固定origin直出不同|
|P5 CNN-RNN|SG非因果性未排除；Train-only minmax|前一小时24观测|测试逐起点实际lag，未知未来闭合UNKNOWN|one-step在线证据，不是自由运行全未来|
|P6 PICCrossformer|物理参数/数据loss联合优化|处理后多变量24窗|物理目标j/T/P来源UNKNOWN，多步递归|物理项成立；future接口不足|
|P7 PINN|膜/催化层模型电压监督|t及历史j复制方案|假定未来j，不是真实未来actual|条件情景预测；非F2信息集|
|S4 recovery hybrid|PA/GRU窗口N训练|已发生N+1计算权重|预测N+2；未来启停预设|历史误差加权，不是P3同目标oracle|
|F2|Train监督ΔV；无物理teacher|过去24h合法历史、原点V|24步一起输出，不用未来actual|静态冻结契约清晰，当前未重推理|

P1/P4/P7已经涵盖不直接获取真实未来运行参数的若干预测情景；原TFT也定义unknown exogenous只在过去可用。**禁止“F2首次提出无未来信息预测”。** F2可强调一个明确且可部署的同origin直接12hΔV任务，而非把条件预测/online update当作同任务比较对象。

## 信息泄漏与可比性需要分开

训练loss读取未来Y是正常监督；推理融合读取未来Y则不合法。训练期privileged工况若另行注册并隔离可能成为额外监督，但F2目前未使用该机制。未来known setpoint、假定复制j、模型生成趋势、真实future actual是四种不同信息，不相互替代。

本文仅从公开公式判断P3的目标依赖；对P2/P6未报告接口留UNKNOWN。预处理风险也不等于证明作者访问了不允许文件。本项目已复用Assessment的事实不因严格推理接口而消失：因果输入不自动带来独立确认性Test。
