# 单事件查询，保留完整TFT处理流程

新模型位于models/event_tft.py，继承只读的CanonicalStrictHistoryTFT、DualPathStrictHistoryTFT、RegimeAwareDualPathTFT。保留VSN、native encoder/decoder LSTM、因果多尺度CNN、5→8→1历史工况门控、静态增强、解释性注意力、position-wise GRN及pre_output_gate_norm。

输入仍[B,48,14]。历史相对位置仍−47..0除以24，编码半小时历史。新的decoder只有一个全零事件查询，14个物理占位量及坐标均为0，不是0.5h提前量，也不是未来真实时间/电流。H1目标类型由这一个固定任务确定。

事件表示取native pre_output_gate_norm后的[B,1,64]状态，经原生同形状Linear64→1读出，输出[B,1]标准化绝对V。没有平均原24步预测、没有24半小时位置改名为周期、没有插值层、没有用普通LSTM/MLP替换TFT。模型类仍使用完整native forward。

H-F0/H-F1/H-F2可训练参数分别323483/332379/332436，输出层均65参数。相同种子共享native模块与读出头初始化严格一致，F1/F2共享CNN初始化一致。F1固定1:1，F2仅将其换为原历史工况门控。

历史动态任务Top10曾用较晚数据选出，不能回填早期fold。新任务保留同一Spearman-at-origin绝对相关Top10规则及强制meanV，但在innerFit或全Train相应集合内重新计算，按索引稳定处理并列；同fold同阶段三模型使用相同mask/scaler。EWMA α=0.3仅作用历史meanV。输入scaler拟合未滤波训练历史，与原处理合同一致。

新目标标准化用对应训练标签的V均值/标准差；原动态ΔV尺度不适用这一绝对V任务，理由在训练前登记。没有新Optuna、核/门控/层数/目标搜索。

预训练工程验证覆盖全部native模块真实调用、非零梯度、CNN前缀因果性、[B,1]形状、共享初始化、合成AdamW一步以及GPU保存重载精确一致。旧checkpoint从未load用于初始化，只做字节保护。

冻结协议SHA为33ea5aa6014281f6627fc08ccc25e1ec0168175452699d6a34937fbc1ded9d7e，工程及协议先提交推送 `77f60bd5c2de9b00099f2e7937cb1019636e91e7`，随后才开始真实训练。原协议、训练关键代码SHA与数据SHA仍一致。
