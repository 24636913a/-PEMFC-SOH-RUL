# F2专属小规模超参数优化

**结论：NO_JOINT_RMSE_MAE_IMPROVEMENT。MAE改善11.6760%，但RMSE恶化43.4943%、R²下降；本轮没有整体改善，不替换原F2，不追加搜索。**

内部时间隔离排除了24个跨目标边界的窗口；该剔除仅用于搜索，最终三个模型仍使用完整1380个原Train窗口。固定起点图不能替代全样本指标。

基准`9aa7ac0c04e1edacdba8a452cb881d5a74aad042`。原始F2结构332436参数、48×14→24 ΔV、Top10与因果EWMA不变。仅搜索学习率、Dropout、weight_decay、batch_size；6个Optuna试验（含原配置），不扩大预算。

Train内部按原始行[0,1472)、[1472,1787)、[1787,2103)分别用于拟合、早停与选参，合法窗口1000/136/220。scaler仅拟合第一段。原冻结Top10来自历史Train研究，不宣称内部调参集是全新独立数据。

选择trial 4，参数：`{'learning_rate': 0.000364001593589285, 'dropout': 0.05, 'weight_decay': 0.0005128483636263388, 'batch_size': 16}`；最终训练轮数335由该trial内部早停最佳轮数冻结。三个最终模型从头使用全部1380个原Train窗口及原Train-only scaler，固定轮数后只评价205个原Validation起点、4920条目标。Validation不选参数、不选最终epoch、不筛选种子。原基准曾使用Validation早停，二者选择历史不同，不能声称这是两套完全相同的epoch选择机制。

三seed等权平均：

| 模型 | RMSE (mV) | MAE (mV) | R²（ΔV） |
|---|---:|---:|---:|
| 原冻结F2 | 7.245400 | 4.274253 | 0.998599 |
| 本轮优化F2 | 10.396737 | 3.775190 | 0.996961 |


RMSE改善-43.4943%，MAE改善11.6760%（正值为改善）。R²按原冻结ΔV标签计算；绝对电压图用未EWMA原始起点电压恢复，未拼接重叠窗口。状态见outputs/RESULT.json。

仅Development探索性评价，不读取Assessment、Late-life或独立Test来选参/评价。历史checkpoint、预测、代码和论文不替换。图为预定中间起点，非挑选误差最低样本。没有新增门控特征、启停诊断或分工况报告。

复现：
```powershell
python -B scripts/refine.py prepare
python -B scripts/refine.py search
python -B scripts/refine.py final
python -B scripts/refine.py report
python -B scripts/verify.py
```
prepare/search拒绝覆盖既有冻结方案或重跑已存在study；final保留已完成实例。环境沿用历史CUDA运行协议。配置、每次训练日志、权重、预测、Optuna完整记录均在本目录。
