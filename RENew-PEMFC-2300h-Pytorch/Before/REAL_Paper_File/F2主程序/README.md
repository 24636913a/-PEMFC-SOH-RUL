# F2秒级预测正式入口

只打开 **F2_Train_Interactive.py**，在VS Code选择项目Python 3.11 / CUDA环境，执行 **Run All Cells**。本次Codex只做文件归档和静态检查，未运行任何Python程序；所有训练、推理、图像显示和指标功能均**待用户运行确认**。

任务是17份RealHyFC Loadcycle原始CSV的**300条连续历史→120条未来平均单体电压V**。使用U1–U5、U7–U11，不使用Utot，U6仅记录诊断。名义1秒，实际时间以DateTime为准，不补点、不跨断点、不使用未来工况。与旧半小时48→24实验分别存档。

开头修改 `MODEL="F2"` 或 `"LSTM"`，以及 `SEED`、`MAX_EPOCHS`、`PATIENCE` 等参数。一次仅从头训练所选模型；不加载历史checkpoint。LSTM单层64维；F2保留原生TFT、LSTM/CNN双路径与工况门控，24个输入槽位中仅10项筛选特征有效。六项必要电压/门控字段加四项Train-only Spearman字段，**不是旧Top10索引**。

时间顺序70/10/20；**输入及120个目标都留在各自分区**，这是保守的边界隔离策略。程序在模型创建前保存确切行号、DateTime/运行小时边界和全部合法逐秒起点身份。Train/Validation用于训练与早停；后20%仅冻结索引，**不运行性能评价**。后留已被历史研究接触，不能称独立盲测。

原始约860万行，默认Train与Validation均使用每个合法起点，**完整训练及每轮Validation可能非常耗时**。内存映射缓存按本次运行独立保存，需要数GB磁盘；默认microbatch16、梯度累计2，不自动调整预算、不自动处理OOM重训。需要控制训练量时，在运行前明确设置 `TRAIN_ORIGIN_STRIDE`，它只影响Train；Validation仍为stride1全量。筛选/scaler使用最多10000个Train合法起点的固定均匀样本。滚动统计在每个300条窗口内重新初始化，无额外29条预热；沿用已有Raw1s实现，不套用旧半小时EWMA。

输出自动保存至研究根目录 `RealHyFC_Raw1s_Rolling_300to120/runs/<唯一ID>/`。包含最佳权重、真实Epoch记录、三项Validation指标、120步RMSE、三张300dpi PNG/PDF及绘图CSV、源代码快照、配置与身份记录。重叠查询按约定计入主RMSE，不视为独立实验。电压图固定展示首个合法Validation起点的120步，不拼接窗口。图片在IPython中直接显示。

每次Run All Cells创建新目录，失败保留 `failure.txt`、日志及已生成文件，无自动重试或额外实验。普通CUDA不承诺逐位一致。Holdout另行授权后另行实施，当前代码没有开放该评价路径。

旧目录342个文件已整体移至 `RealHyFC_Raw1s_Rolling_300to120/archives/F2_legacy_9ba5daaf/`；64个代码/配置/说明副本和完整哈希清单在 `backups/v1_20261022_9ba5daaf/`。历史入口与审计程序不是新版入口；恢复方法及路径迁移限制见新工作区README。
