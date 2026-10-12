# O-02 — 双路径、多尺度与混合预测对照

## P3：独立输出选择，不是F2特征融合

Jiang等，[10.1016/j.ijhydene.2026.153481](https://doi.org/10.1016/j.ijhydene.2026.153481)，p5–9式16–40、Fig5/6/8。TCN与BiLSTM分别预测；VMD-KF-MF共同预处理；IALA优化BiLSTM权重。F2只有一个共同训练的TFT输出头，两条路径没有独立预测器。

**式39/40，p8；Fig8，p9；实验文字p10实际核对**：设两路误差为 `|y_t-yhat_TCN|` 与 `|y_t-yhat_BiLSTM|`，选择较小者。选择所需真实`y_t`是部署未来时不可得的目标。按发表方法判 `TARGET_DEPENDENT_SELECTION_IN_PUBLISHED_METHOD`；若仅作离线oracle上界则可解释，但不能等同未来盲预测；全文未披露另一不需真值的gate。不能上升为故意数据造假指控，也不推断未公开代码。

F2的gate只用48步历史工况均值，预测起点可得。其特征融合可部署性比式40接口清楚，但不能由竞争方法的接口问题推出F2新增gate有独立性能增益；Phase J恰未证明这一点。

VMD与中心MF的全信号处理范围不清，不能认为TCN层“causal”就使整条预处理因果。500点、窗口长度、数据划分、seed和RUL接口不足，主表R²不等于真实RUL准确率。

## P4：多尺度因果卷积和双条件路径均已有先例

Benaggoune等，[10.1016/j.apenergy.2022.118835](https://doi.org/10.1016/j.apenergy.2022.118835)，p5–8 Fig8–10/式4–9/表4。kernel2、dilation1/2、skip、GELU与Luong attention；并非F2首先将dilation引入PEMFC。

长期版本p10–12 Fig16/式13–14使用voltage与condition-trend两个DilCNN分支，ARIMA生成未来趋势，再add/attention。与F2**双路径互补**同一思想家族，输入对象不同：F2两路共享同一历史VSN，分开学习LSTM和CNN编码，历史工况只控制encoder混合。P4未来趋势预测不是真实未来工况，不可指控泄漏，但属于额外条件建模，应分别界定。

短期p6 Fig7与p8 §3.5.2的5/10/15/24步预测后引入新已发生测量，这是在线更新；不能和同origin一次直接24步混算。块内直出/递归及步长物理小时映射不完全明确，UNKNOWN。表5指标任务/单位不应移植成F2结果。

## P5：串联CNN–RNN先例，不等于并联TFT内部融合

Sahajpal等，[10.1016/j.ijhydene.2023.04.143](https://doi.org/10.1016/j.ijhydene.2023.04.143)，p5–6 Fig2、p11表2/3。CNN输出后接LSTM/GRU/BiLSTM/BiGRU；不是与TFT LSTM并行。500trial、50次重复（seed列表未知）、模型容量不同；表4–6均值并不支持“CNN混合一律更好”。所以F2不能仅按新增卷积理论故事宣称收益，需要如实报告F1独立效果不稳。

前一小时24变量→当前电压，真实历史随起点更新；SG是否causal、未知未来协变量如何闭环未充分报告。其RUL以96%阈值穿越估计，不能替RealHyFC建立EOL。

## 相关补充

- S2 RF-TCN是RF-RFE→TCN串联流程，不是两个输出投票；p8在线新测量更新，p14–15的42.53%为特定FC2输入对照，不作通用精度主张。
- S4 Wu等p5 Fig4/5、式22–24：窗口N训练、已发生N+1求权重、N+2融合。它确为独立物理/GRU输出加权，但不等于P3针对同一未知目标的oracle，也不等于F2一个网络内特征融合。
- S5 M-LSTM是RNN记忆/门控替换；不存在F2 LSTM/CNN encoder分支的同构证据。不以其RUL标题推定验证了真实寿命。

## 本项目可保留与不可保留

保留：共同encoder输入、k3/d1&4因果depthwise两尺度、post-LSTM encoder融合、历史工况控制路径权重、原decoder接口不变。

不可声称：首次双路径、首次CNN–RNN、首次空洞卷积、首次门控。两个卷积分支感受野跨度3/9个离散位置，d4分支只有三个tap；这不是“覆盖连续9点”，更不是经独立电化学测量验证的快/慢物理过程。核心贡献是受控任务下组合设计的实证，不是证明两个物理时间尺度。
