# RealHyFC F2主程序：FC2式结果输出

在研究根目录执行 `python -B RealHyFC_F2_Main.py`，默认加载原始F2 Seed42冻结权重，评价Validation，不训练、不调参。控制台仅显示RMSE、MAE、R²三行，电压误差单位为V；完整精度保存在结果文件中。

`python -B RealHyFC_F2_Main.py --seed all` 运行原始42、52、62三个种子，控制台仍只显示一组指标，为三个种子指标的等权平均，不进行预测集成。

每次运行新建 `RealHyFC_F2_Main_Outputs/<UTC时间>_F2_evaluate_seed<种子>/`，保留历史输出。每个 `seed*/` 中保存：

- `training_history.png`：原历史训练日志的Train/Validation损失，不重新训练。原日志Validation标准化RMSE平方转换为标准化MSE，与Train损失同量纲。
- `prediction_vs_true.png`：固定每个合法起点的第24步，即12运行小时预测；按真实目标运行小时排序，缺口断线，不插值、不随意拼接重叠窗口。
- `scatter_true_vs_pred.png`：该评价区间所有合法24步预测查询的绝对电压散点，包含重叠目标时间。
- `best_model.pth`：原最佳checkpoint的逐字节副本，SHA256与源文件一致；保留原checkpoint字典格式和元数据，使用主程序原有`load_checkpoint`加载。
- `preprocess_info.json`、`predictions.csv`、`training_history.csv`及三个图的绘图CSV。

运行目录的 `evaluation_results.json/csv` 保存控制台对应的单组完整精度指标、评价分区和种子。默认Validation为205个合法起点、每种子4920个查询。指标仍使用全部24步原始ΔV，R²也保留原ΔV定义，不改成绝对电压R²，绘图步长不改变正式指标范围。

保留原Train/Validation/Assessment/后段时间划分及独立预测文件；四区间指标写入 `F2_PARTITION_METRICS.csv`，不在控制台重复打印、不混合求分数。可通过 `--partition HISTORICAL_ASSESSMENT` 或 `--partition LATE_LIFE` 指定另一组展示数据，均属于探索性评价，不称独立Test。原有显式 `--mode train` 实现保留，本次未执行。

模型结构、332436个参数、48→24、Raw14、Top10、Train-only scaler、EWMA及原始电压恢复方法均未修改。每次评价先执行历史Validation逐点精确核验，再执行其余原分区的逐点核验；任何不一致立即失败。

回归测试：`python -B -m unittest discover -s tests -p test_f2_main_output.py`。复核已保存结果：`python -B tests/verify_f2_main_saved_outputs.py`，输出一份简短验收JSON，不训练模型。
