# H1健康导向事件级双路径TFT

新任务：完整周期确认后预测下一raw物理周期的冻结middle50固定状态电压。141原标签，新增5周期不入训练。

阶段一：prepare_data.py、engineering.py、freeze_protocol.py，先提交协议；阶段二train.py执行27登记实验，每项内选epoch再独立全Train重拟合，共54个真实GPU拟合pass。原模型/权重只读，无Test数值或原始高频重读。

一次性运行，runID状态存在时不自动重训；控制器跳过COMPLETE，遇FAILED停止并保存错误。恢复状态位于本阶段ignored outputs/runtime，正式inner/finalcheckpoint均保存。

模型输入48×14，保留原骨干及native注意力；仅一个全零事件查询，输出标准化绝对V。Top10和X/Yscaler都在对应Train拟合，不能复用较晚动态任务掩码。外层验证从不选择epoch。

真实训练发生在2026年，训练耗时放入历史标签可用时间之后检查asof可实现性；不声称2025年实际在线部署。等待完成后读08通俗总结和逐点CSV。
