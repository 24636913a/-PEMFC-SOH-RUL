# F2 结构冻结审计

Base: `427d4dc402f083a5b9e094b12981c5e231df8100`. Protocol SHA256: `0244f05fd777ccd615fda7eb9533d129323dc27fbb58212e091130011d0d9b24`. Code identity SHA256: `67b0d3588f75db56d435ddc5277d38324708f660ac10cc1850a5908d771612c8`.

Validation only, 205 origins/seed, 615 gates; all 4920 queries per seed retained. New training/Assessment inference/Test/seed selection: NO.

`PHASE_J_F2_ARCHITECTURE_FROZEN`

|model|class|parameters|branch_parameters|gate_parameters|
|---|---|---|---|---|
|F0|CanonicalStrictHistoryTFT|323483|0|0|
|F1|DualPathStrictHistoryTFT|332379|8896|0|
|F2|RegimeAwareDualPathTFT|332436|8896|57|

F1 相比 F0 新增 8896 个卷积分支参数，F2 相比 F1 新增 57 个门控参数。F0 保持原生 strict-history TFT；F1 在历史编码表示上等权融合；F2 用历史工况门控替换该等权融合。不是最终预测值的加权平均。

卷积分支直接消费 encoder VSN 的 [B,48,64]，而非原始14变量或 post-LSTM 表示。两条 depthwise 路径 k=3,d=1/4，分别仅左补2/8点，感受野3/9点（最早到当前跨度1/4 h）。没有额外卷积激活函数。拼接128通道后 pointwise 映射64，Dropout(.1)、VSN残差、LayerNorm。门控的LSTM侧为 post-LSTM GLU+AddNorm 后的表示。

原生 LSTM encoder 的 hidden/cell 继续初始化原生 decoder。融合不改这个状态交接，也不改 decoder residual。融合后的历史与原生decoder拼接后才进入 static enrichment、attention、position-wise GRN 和输出头。Decoder只接确定性相对步长；原始14个物理占位槽为零且不进入 decoder VSN。

超参数（历史冻结而非本轮重搜）：48→24，0.5 h，Raw F2 14槽，hidden64，continuous32，attention4，LSTM1，dropout .1，batch32；AdamW lr=0.0009605449687795866, weight_decay=6.95942232671908e-06。本轮不创建 optimizer。使用原Train scaler、Top10与电压EWMA α=.3。

身份：九个主实验 checkpoint 9/9 唯一身份核验；三个 F2 固定32 Validation参考逐元素精确相等；被动hook和恢复正常门控均3/3 bitwise exact。完整参数/buffer及requires_grad保持，grad=None。原Phase F enabled full-forward证据来自 `F_IMPLEMENTATION_AUDIT.md`，15项历史合成测试；不将其eval一致性夸大为训练bitwise确定性。

源码权威：T0 `canonical_tft.py`，F `dualpath_strict_history_tft.py`、`causal_multiscale_branch.py`、`regime_aware_dualpath_tft.py`、`frozen_feature_schema.py`。逐文件SHA见冻结协议 sources，逐参数名/shape见 J_PARAMETER_SCHEMA.csv。上游 pytorch-forecasting 1.8.0 mirrored forward 版本敏感，源代码SHA必须保持。

无新增模型模块、无推理延迟或GPU效率宣称。结构身份可以冻结，与门控物理解释是否成立是不同问题。
