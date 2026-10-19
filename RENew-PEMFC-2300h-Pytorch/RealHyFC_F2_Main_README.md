# RealHyFC F2主程序：FC2式结果输出

在研究根目录执行 `python -B RealHyFC_F2_Main.py`，默认加载原始F2 Seed42冻结权重，评价Validation，不训练、不调参。控制台仅显示RMSE、MAE、R²三行，电压误差单位为V；完整精度保存在结果文件中。

主程序只接受42、103、193、202、210、212六个种子。`python -B RealHyFC_F2_Main.py --seed 103`等命令加载对应原有权重。`python -B RealHyFC_F2_Main.py --seed all` 按42、103、193、202、210、212固定顺序运行，控制台仍只显示一组指标，为六个种子指标的等权平均，不加权、不进行预测集成。旧种子52、62在参数解析阶段报错，其历史文件不删除。

`RealHyFC_F2_Main_Seed_Assets.json`明确保存六个种子的checkpoint、Validation预测、训练日志、冻结协议的相对路径及SHA256、最佳epoch、初始化身份和输入指纹；映射本身也受主程序固定SHA256保护。来源为第二轮已冻结的六候选表，没有重新选种子。42来自Phase H，103/193来自第一轮筛选，202/210/212来自第二轮筛选。六个实例经过Validation筛选，不代表随机初始化总体的无偏平均性能。

每次运行新建 `RealHyFC_F2_Main_Outputs/<UTC时间>_F2_evaluate_seed<种子>/`，保留历史输出。每个 `seed*/` 中保存：

- `training_history.png`：原历史训练日志的Train/Validation损失，不重新训练。原日志Validation标准化RMSE平方转换为标准化MSE，与Train损失同量纲。
- `prediction_vs_true.png`：固定每个合法起点的第24步，即12运行小时预测；按真实目标运行小时排序，缺口断线，不插值、不随意拼接重叠窗口。
- `scatter_true_vs_pred.png`：该评价区间所有合法24步预测查询的绝对电压散点，包含重叠目标时间。
- `best_model.pth`：原最佳checkpoint的逐字节副本，SHA256与源文件一致；保留原checkpoint字典格式和元数据，使用主程序原有`load_checkpoint`加载。
- `preprocess_info.json`、`predictions.csv`、`training_history.csv`及三个图的绘图CSV。

运行目录的 `evaluation_results.json/csv` 保存控制台对应的单组完整精度指标、评价分区和种子。默认Validation为205个合法起点、每种子4920个查询。指标仍使用全部24步原始ΔV，R²也保留原ΔV定义，不改成绝对电压R²，绘图步长不改变正式指标范围。

保留原Train/Validation/Assessment/后段时间划分及独立预测文件；四区间指标写入 `F2_PARTITION_METRICS.csv`，不在控制台重复打印、不混合求分数。可通过 `--partition HISTORICAL_ASSESSMENT` 或 `--partition LATE_LIFE` 指定另一组展示数据，均属于探索性评价，不称独立Test。原有显式 `--mode train` 实现保留，本次未执行。

模型结构、332436个参数、48→24、Raw14、Top10、Train-only scaler、EWMA及原始电压恢复方法均未修改。每个种子先执行历史Validation全部4920点及checkpoint固定首批预测的精确核验；任何不一致立即失败。42还有历史全时域缓存，其他区间也逐点核验。其余五个种子没有历史全时域缓存：其他分区沿用原程序只读推理，验收文件明确记录“无历史全时域缓存”，不虚称这些分区已经复现旧结果。

回归测试：`python -B -m unittest discover -s tests -p test_f2_main_output.py`。六个单种子及all的实际运行记录见 `RealHyFC_F2_Main_Outputs/SIX_SEED_EXECUTION.json`，复核命令为 `python -B tests/verify_f2_six_seed_outputs.py`，输出简短验收JSON和七行汇总CSV，不训练模型。`tests/run_f2_six_seed_acceptance.py`是本次一次性运行脚本，已有运行记录时会拒绝重复执行。旧 `verify_f2_main_saved_outputs.py` 保留为此前三种子输出版本的历史核验代码，不用于六种子版本。
