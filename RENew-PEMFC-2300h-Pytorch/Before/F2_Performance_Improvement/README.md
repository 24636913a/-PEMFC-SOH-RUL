# F2性能提升研究：完成，未达到稳定升级条件

当前入口为`Performance_Interactive.py`，在VS Code Python Interactive Window中逐个运行`# %%`单元。使用项目Python3.11/CUDA环境。共享模块导入不启动训练，也不导入带有顶层训练代码的旧交互程序。

本轮完成15次真实新GPU训练：A九次、B六次，共2549个Epoch、140195次优化器更新，正式训练失败数0。A/B独立复算及checkpoint逐点重放通过。没有候选通过训练前冻结的稳定性/尾部条件，因此C事件加权和D六种子后留评价均未启动，新后留成绩为NA。保留F2主模型，没有用普通LSTM替代或改名。

| 方法（各3种子） | Validation RMSE均值±SD(mV) | MAE均值(mV) | R²均值（原始ΔV） |
|---|---:|---:|---:|
| A0 原始F2 | 14.825764 ± 9.360868 | 6.945424 | 0.992788086 |
| A1 仅学习率0.0006 | 12.265011 ± 2.484020 | 6.817052 | 0.995993975 |
| A2 仅Dropout0.05 | 25.968248 ± 8.768987 | 9.964981 | 0.981190960 |
| B 新训练原融合对照 | 17.817881 ± 11.424903 | 8.722101 | 0.989514832 |
| B 残差融合F2-v2 | 17.958332 ± 8.461048 | 8.804337 | 0.990403110 |

A1平均RMSE下降17.27%，但只改善2/3种子，Seed202的MAE和P99明显恶化，不能宣布稳定升级。B_v2比本阶段原融合平均RMSE高0.79%，仅1/3种子改善；关闭CNN补充后3/3种子误差变大，说明CNN确实参与，但不等于整体方案更好。

同配置重复暴露了明显数值波动：Seed42两次原配置RMSE为12.702993与30.900509 mV。三对重复的初始化、输入、共同Epoch采样和所记录的CPU/CUDA RNG均相同，但权重最早在第1/6/2轮发生分歧。不能简单归咎于没有设Seed，也尚不能指定某个算子为唯一原因。

这些都是Validation结果，不能与历史后留F2的44.220508 mV或LSTM的24.885444 mV直接排名。没有新的六种子后留证据，未声称超过LSTM。

先读[最终研究总结](outputs/20261020T043811_940866Z_F2_improvement/04_最终研究总结.md)；[完整验证汇总CSV](outputs/20261020T043811_940866Z_F2_improvement/tables/FINAL_VALIDATION_STUDY_SUMMARY.csv)、[逐种子重复性](outputs/20261020T043811_940866Z_F2_improvement/tables/B_PLANNED_NATIVE_REPEATABILITY.csv)、[停止决定](outputs/20261020T043811_940866Z_F2_improvement/FINAL_DECISION.json)提供真实数值和固定判据。

首次运行先重启交互内核，并将仓库、研究根目录或本目录设为工作目录。入口检查同名模块的实际路径，拒绝静默混用其他项目的`io_utils`等缓存；真实验收控制器始终创建全新IPython内核。

本任务保留原生F2为主模型，不训练新的普通LSTM。旧LSTM只作为历史参照，不能改名为F2。数据固定3843行、Train[0,2690)、Validation[2690,3074)、后留[3074,3843)，Top10、48→24、0.5h、原始ΔV、因果EWMA alpha=.3、Train-only scaler。

阶段A为A0原始配置、A1仅学习率=.0006、A2仅Dropout=.05，各Seed42/202/212，共9次从头训练。所有方案保留原生332436个参数。450轮上限、patience40、AdamW/MSE、batch32、weight_decay和clip1保持不变。

三种子均改善RMSE、平均降幅至少5%、平均MAE不升高并通过P99保护，才达到预先登记的探索性升级条件。普通CUDA波动存在，该门槛不是统计显著性。不得靠某一次好结果追加种子或重跑。

`ACTIVE_RUN.json`指向本任务当前运行。默认继续这个运行，跳过已经验收完成的本任务实例；不会加载旧权重继续训练。只有显式设置`START_NEW_RUN=True`才新建完整实验目录。失败或部分完成的训练不会自动重跑，必须保留错误后检查。

阶段A完成并独立核验后，再按冻结资格决定B和C；本轮不会同时叠加多个改进。没有合格方案就不进行D，后留成绩记NA。只有唯一方案及六个全新最佳权重都冻结后，才允许一次探索性后留评价。后留区间早已被历史研究查看，不是独立盲测。

`execute_stage_a.py`是在真实IPython内执行单元并保存图像/进度的验收控制器，不是训练回放。执行记录按attempt独立保存；此前eval模式梯度检查被cuDNN拒绝的失败也保留。修正为train模式、Dropout开启但无optimizer的只读梯度诊断，没有修改网络。五项CPU/真实数据合同测试已通过。

在入口顶部设置`STAGE_LIMIT='A'`只运行/续做A；设置`STAGE_LIMIT='B'`先验收A再运行/续做B。阶段B固定对照为新训练原凸组合与一个有界残差公式：`HL+0.25*(1-g)*RMS(HL)/max(RMS(HC),1e-6)*HC`，各三个种子，不增加参数。方案、阈值、代码SHA和登记时间保存于`B_REGISTRATION.json`和`B_预注册.md`；三项结构/梯度测试均通过，测试优化器更新数为0。

真实环境命令：`D:\ruanjian\python3.11.9\python.exe -B execute_interactive_stage.py --stage B`。此控制器固定继续已注册的ACTIVE运行；创建新运行请直接在交互入口显式设置START_NEW_RUN=True。控制器执行同一份代码单元、保存真实IPython图像与Epoch输出；验收没有伪称点击过VS Code界面。可直接打开保存的executed.ipynb查看当次图像/进度。只读反事实及已完成运行的入口复核不作为新的正式训练计数。

真实新训练的原回执分别由当前输出目录的`A_FRESH_TRAINING_EXECUTION.json`和`B_FRESH_TRAINING_EXECUTION.json`定位，记录了170/90次实时损失图刷新。`LATEST_B_EXECUTION.json`可能指向随后只读交互验收，明确记录新增模型和参数更新均为0，不能把它冒充新训练。新版入口七个代码单元已在真实Python3.11/CUDA IPython中通过续跑验收，并直接显示PNG。

本轮没有C/D合格前置方案，入口及控制器不提供未注册C/D训练。未来新的独立运行如达到门槛，须先登记后续阶段代码和配置，不能静默启动未实现/未登记实验。

训练日志中的梯度观测为每个Epoch首批Train的真实训练模式梯度；全局梯度范围和裁剪比例覆盖所有批次。`cnn_off`及`lstm_only`是同一已训练F2的临时推理干预，仍有TFT主体，不是新的普通LSTM或独立训练消融。

所有新结果仅保存在本目录；旧代码、权重和实验结果全部只读。训练前工程检查及失败证据见preflight和每次执行记录。论文候选图片为真实CSV生成的300dpi PNG/PDF，不能拿未完成阶段的空白或假想结果代替。

图片验收实际通过26组分析图及15组训练曲线。`FIGURE_INDEX.csv`/`TRAINING_FIGURE_INDEX.csv`列明图片、数据和脚本；`ARTIFACT_SHA256.csv`、`FINAL_INTEGRITY.json`、`HISTORY_PROTECTION.json`为最终交付完整性和历史文件保护凭据。P95/P99尾部误差保留原float32路径，并另报float64精度差，主指标、样本身份和预测严格核对。

代码职责：`frozen_f2.py`为独立原生F2/预处理，`residual_fusion.py`只覆盖融合，`train.py`执行真实新训练和只读预测，`stage_a.py`/`stage_b.py`按冻结预算执行，`verify_stage.py`独立核验，`report_a.py`/`report_b.py`/`final_reports.py`由真实CSV生成图表。没有import会自动训练的旧交互程序。
