# LSTM与完整F2：本次研究导航

正式成绩仅使用models/formal中的12个新实例；models/pilot的Seed202两个实例用于验收，不混入正式均值。P1_existing_F2为旧缓存只读诊断，不冒充新模型成绩。

报告顺序：00_实验协议与划分核验.md → 01_F2_Error_Diagnosis.md → 02_LSTM_vs_DualPath_Results.md → 03_Prediction_Improvement_Plan.md → 04_最终研究总结.md。

图像索引（全部300dpi PNG，并附PDF和绘图CSV）：

| 图 | 用途 |
|---|---|
| P3/01_six_seed_RMSE | 六种子RMSE公平比较 |
| P3/02_six_seed_R2 | 六种子原始ΔV R² |
| P3/03_mean_RMSE_SD | 等权均值和样本SD，非置信区间 |
| P3/04_horizon_RMSE | Step1～24误差变化 |
| P3/05_absolute_voltage_Step24 | 固定Step24绝对电压曲线，不拼接重叠窗口 |
| P3/06_scatter_all_queries | 全部12648查询/模型，保留极端点 |
| P3/07_scatter_actual_current | 目标实际电流密度着色，仅事后分析 |
| P3/08_error_over_time | 619唯一目标时间上的误差，未平滑 |
| P3/09_adverse_events_and_load | P1预先选定的不利事件与原始工况 |
| P3/10_training_loss | 本次Seed202两模型真实训练日志 |
| P3/11_fixed_origin_24step | 固定中位合法起点的单次24步预测 |
| P3/12_paired_worst_query_windows | Seed202两模型各自最差查询所在的完整同起点窗口，全部24步 |

P1图提供旧F2误差、固定10h局部RMSE、分区分布及不利案例。每个新模型实例另有本次实时training_history.png及CSV。

tables/存逐种子、均值±SD、配对差、全分区和逐步指标；formal_diagnostics/存尾部、事件贡献、条件组及原始最大误差记录。指标为ΔV；曲线和散点展示用原始起点电压恢复的绝对电压。

P5_AUDIT.json为独立输入/标签、早停、真实更新、checkpoint重放、指标和图像审计。全部后留结果属于探索性评价，不能称独立盲测；没有执行额外性能优化。
