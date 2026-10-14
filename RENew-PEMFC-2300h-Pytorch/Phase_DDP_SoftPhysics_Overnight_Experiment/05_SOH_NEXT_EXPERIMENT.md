# SOH已有证据与下一项最小实验

## 冻结量和已有成果

参考定义`SOH_V(t_i)=V(j*=1.0 A/cm²,t_i)/V(j*,BoL)`；辅助j*=1.6仅敏感性。沿用既有nominal 10-cell per-cell定义、BoL authority及曲线来源，不混接独立8-cell EIS。Development normal参考为BoL、300/600/850/1150/1500/1750七点；条件可比性仍有前提，不生成逐小时真实SOH。

P1S固定token `221||221.200 A`；dwell≥120s，primary middle_50，last_25仅已冻结敏感性，active10-cell均值的中位数、周期内再汇总。前10个complete周期暖启动，`Vref=0.71383023635 V`在171.82091h后可得；首个合法目标COMPLETE_011在181.34366h，之后131个因果观测。

|来源|真实已保存证据|当前含义|
|---|---|---|
|PAA/PAV|六点preanchor时间先于极化；3/6 median来源在共同支持内|有方向兼容，不代表全部状态可比|
|AHC|六锚点LOAO MAE0.0004768699；严格前三个后续锚点MAE0.0007646870|简单affine有有限支持，LOAO非在线证据|
|PHI|全六点B2：1.1988114938324186R−0.2090362463118213；141密集代理|全锚点校准回顾性；拟合一致性不是预测精度|
|CTA/CTAE|全序列SG/isotonic/PCHIP会改写过去；raw有35次回升|在线用raw与当时已知参考，不能硬单调|
|CNP1S|OLS/Ridge LOAO约0.00036947/0.00043175，总gate未全过|不再挑公式救回条件归一化|
|CMRF|H6 TimesNet0.716930 vs Persistence0.604104 mV-equivalent|复杂点预测未优于强基线|
|PSDM|H6 MAE0.707559，CRPS0.000665065 vs GP0.000631906|物理退化率与概率效用未成立|

这些是历史结果，来源SHA在REPORT_SOURCE_MANIFEST.json。PAA/PAV、AHC、PHI、CTA/CTAE、CNP1S、CMRF和PSDM报告均已阅读；不重建raw、参考或P1S。

## 明天只做一项：前缀校准到下一稀疏参考的SOH估计

任务是**有P1S测量时估计当前参考工况健康proxy**，区别于提前预测未来健康。只用保存Development数据；在参考可得后用raw R，不使用全锚点B2或全序列平滑。

固定三次前缀：已获得300/600/850参考→检验1150；加入1150→检验1500；加入1500→检验1750。以参考结果实际发布时刻为准；若只有nominal availability，明确降为nominal chronology replay，不冒充真实在线。已核实`Phase_AHC_Sparse_Anchor_Affine_Health_Calibration/outputs/B2_CHRONOLOGICAL_PREFIX_EVALUATION.csv`包含完全相同三前缀：1150/1500/1750预测分别0.9715868347291073、0.9740582586162393、0.9728474747386423。明天先复用该缓存，补齐Persistence和年龄趋势对照，不重复affine拟合；本轮仅核对既存值，未计算新对照成绩。

候选只用`Yhat=aR+b, a>=0`的CPU最小二乘，与已有AHC规则一致，不搜权重。每次仅当时已知锚点拟合；目标R由冻结preanchor interval ID/测量窗口取值，先保证测量结束时可得；不能替换为更合适的新dwell。1750目标即使历史上已暴露，当前也只作为既有Development回放，非独立验证。

对照：最近合法SOH参考Persistence；全部既往参考的普通线性年龄趋势；raw R原样作为尺度不等同的诊断对照。目标锚点nominal age只作当前估计时间坐标，不输入未来真实lead或未来工况。若操作年龄与nominal诊断小时不一致，保留身份而不伪造精确OpHrs2。

输出3个锚点逐点AE、有符号误差、MAE、参数、数据/可获得时间身份、in/out-of-support（保留全部样本）。共同BoL不确定性不当作独立误差；small-n不声称显著性。描述性波动区间不冒充95%概率覆盖。

判断：在全部三点保留的条件下，affine若未稳定优于Persistence和年龄趋势，停止复杂SOH模型并优先补独立同参考条件诊断；若有信号，也只支持同堆proxy估计。不要以三个点论证跨堆真实SOH。今晚没有拟合、校准更新或新SOH网络训练。
