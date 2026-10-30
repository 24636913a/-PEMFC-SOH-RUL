# 数据字典（全部为模拟数据）

## observations.csv：唯一模型观测输入

| 字段 | 单位/类型 | 说明 |
|---|---|---|
| time_h | h | 从 0 开始，每 1 h 一个样本，终点 2200 h |
| stack_voltage_v | V | 5 片串联电堆电压；模型先除以 n_cells |
| n_cells | 整数 | 串联片数，本例假设 5 |
| restart | 0/1 | 在此时刻发生已知重启事件；850 h 后的标志不会用于预测 |
| condition_offset_v | V/单电池 | 工况偏移；本例固定参考工况，全部为 0 |
| is_synthetic | bool | 全部为 True |
| scenario | 字符串 | FC1_like_synthetic，仅情景名，不是真实 FC1 |

## truth_for_evaluation_only.csv：只允许评估模块使用

| 字段 | 单位 | 说明 |
|---|---|---|
| time_h | h | 与观测时间匹配 |
| latent_voltage_v | V/单电池 | 生成器的长期状态真值 |
| q_rev_v | V/单电池 | 生成器的剩余可逆损失真值 |
| condition_offset_v | V/单电池 | 模拟工况偏移 |
| measurement_noise_v | V/单电池 | 本次噪声样本 |
| noise_free_operating_voltage_v | V/单电池 | latent+offset−q；便于比较瞬时运行阈值 |
| is_synthetic | bool | 全部为 True |

该文件不得被添加为训练特征。删除它不会阻止模型运行，只会跳过真值相关指标与图层。

## synthetic_events.csv

人工阅读的模拟事件清单。模型不读取整张事件表，而只根据截至当前时刻已观测到的 `restart` 字段计算恢复龄，避免偷看未来事件。

## 输出时间与评价口径

- `history_estimates.csv`：0–850 h；0–359 h 的模型状态为空，表示仍在初始校准。
- `future_forecasts.csv`：850–2200 h；850 h 是预测起点，评分仅使用 851–2200 h。
- `event_recovery_estimates.csv`：只列初始校准后且不晚于预测起点的事件；观测跃升包含测量噪声。
- 所有 RMSE 按单电池电压计算，展示单位为 mV。
- 所有 RUL 都是从固定预测起点 850 h 到长期状态阈值的剩余小时数。
