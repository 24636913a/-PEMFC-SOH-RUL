# RealHyFC原始秒级300→120

本次交付性质：**代码重构、显式备份、归档、静态审查**。没有训练、推理、Python测试、数据全量解析或Holdout性能评价。LSTM/F2指标均待用户运行。旧研究数值未改写。

正式入口在 `../REAL_Paper_File/F2主程序/F2_Train_Interactive.py`。使用VS Code Run All Cells，仅运行一个新初始化模型，自动保存到本目录 `runs/<唯一ID>/`。不运行旧实验或其他Phase。

## 归档与恢复

修改前main：`9ba5daafab3ebce4c9095ad4f1e6411d62d346fb`，工作区干净。旧主程序SHA256：`3dbdc602eb2229a54ee7d5eaa841a6dd94798e8227dbfc6f60db1e78f66419de`。

`backups/v1_20261022_9ba5daaf/code/`：64个代码、配置、说明及Notebook副本；`CODE_BACKUP_SHA256.csv`记录对应哈希。另备份三份外部路径适配文件的原始版本至 `external_code/`。

`archives/F2_legacy_9ba5daaf/`：旧正式目录全部342个实体文件，共329089728字节。包括旧入口/Notebook、241个训练产物、历史审计/验收/测试和1个pyc缓存；不删除任何证据。完整目录整体归档保留内部结构，未重新复制大型权重。逐文件分类A–F、Git追踪/LFS状态、大小与SHA见 `INVENTORY_BEFORE.csv`。本次旧入口都属于历史B类，新版入口才是当前A类。

三份现有外部研究代码只更改归档文件的路径字符串：`F2_LSTM_DualPath_Comparison/core.py`、`F2_Performance_Improvement/frozen_f2.py`、`F2_Rolling_Horizon_Study/scripts/common.py`。指向原封不动的归档源码和结果，原有源码SHA断言保留。**这些适配未运行验收**。根目录 `RealHyFC_F2_Main.py` 和既有Raw1s阶段源码不动，新运行代码也不依赖它们。

旧归档内部的历史入口/审计脚本含有相对 `__file__` 推导和原绝对路径，故不应直接在归档位置启动；其字节与历史身份保持原样，而不是伪称已适配。需要重放原实验时，用户在独立历史checkout检出上述修改前commit，把该归档或backup/code按清单恢复到原 `REAL_Paper_File/F2主程序`，并恢复所需LFS载荷。不要覆盖当前新版目录。这样保留原始路径布局与哈希。历史报告中的旧绝对路径是历史记录，不修改；可按目录迁移表定位现存证据。新运行只写runs。

## 固定协议（在用户运行前声明）

- 唯一原始权威：已有Raw1s审计列出的17份Loadcycle CSV，清单副本 `reports/LEGACY_RAW_SOURCE_AUTHORITY.json`。本次未重读860万行；用户执行时重新核验每份原始文件大小、SHA256，再逐块读取。旧清单的滚动/分区规则不复用，新协议由下述代码独立生成。
- 目标：十个有效单体通道均值V，U6诊断独立保存，不推断其故障原因。原始24项候选来自旧Raw1s实现，不含Utot。
- 同步观测假设：原始一行传感器视为同一离散时刻。连续规则：DateTime严格递增，相邻950–1050ms，OpHrs2不回退（1e-7h浮点容限），文件边界强制断开；缺失候选仪器打断窗口。不按电压大小或误差删样本。DateTime倒序/重复直接报错，不静默排序。
- 所有候选窗口支撑420条记录，整个输入及目标必须在分区内；分区行为前70%、中10%、后20%，确切数量/行号/时钟/索引SHA由用户运行准备阶段在训练前冻结。所有记录保留，非法窗口有统计原因和底层segment/valid/source-row数组可复核。
- 原24候选中30点std/ptp、一步电流/状态变化，在每个300条输入内部重新计算；起始std/ptp/差分为0（其定义是只有1条观测的统计量/无可观测前驱），不伪造原始观测，不跨窗口取额外29行。无全局EWMA、无旧半小时Top10索引。
- Spearman多步定义：300条输入的逐特征均值，与未来120条真实平均电压的均值计算Train-only秩相关。最多10000个Train合法起点按固定均匀序号抽取，数量与方法预定、与预测误差无关。六项必选为电压均值及原生门控五项，其余四项按绝对rho排序；同分按候选顺序；常量不选，非必选完全秩重复不选。必选门控中电流/电流密度统计可能冗余，明确保留以维持原生五项语义，不能宣称完全独立信息。若不足10项报错，不自动改变规则。
- scaler仅以上述Train样本的300条历史拟合，目标scaler仅其Train未来标签；两模型预处理算法一致。F2/LSTM都是24槽位，其中选中10项实际非零；F2门控按名称映射，必选字段不会被筛掉。固定同seed、同采样设置才构成受控对照。
- 主目标实际V，训练MSE使用Train标准化后的V。DeltaV仅数据实现预留，正式入口拒绝启用，不自动做第二组对照。R²按还原后的实际V计算。
- 主RMSE/MAE为全部合法Validation起点×120步等权查询，单位mV。Validation始终stride1。重叠查询不独立。Holdout只记录边界和合法数，无训练选择、无指标。默认全量Train stride1，可在首次运行前明确减少训练起点，但不能依据误差选择。
- F2借鉴已有Raw1s原生适配：原生VSN/GRN/LSTM encoder/decoder、因果多尺度CNN、五项门控、注意力与输出层；无未来仪器输入，decoder仅相对步坐标。LSTM单层64+LayerNorm+线性120步，内部Dropout=0。模型容量不匹配，不宣称全结构变量控制。原生类静态复制到正式模块，避免执行任何旧交互入口。
- 共享AdamW lr=0.0009605449687795866、weight_decay=6.95942232671908e-06、MSE、clip1；F2 hidden64/continuous32/head4/layer1/dropout.1，沿用Raw1s参数，不做Optuna。microbatch16累计2；epoch450、patience40。普通CUDA可有数值差异，不承诺逐位一致。OOM报错留现场，不自动改batch或模型再跑。
- 每次运行唯一目录、源代码快照、配置、边界与特征身份、初始/训练后状态哈希、optimizer更新次数、最佳Epoch、完整Epoch日志。只重载本次新训练最佳权重进行Validation复算，非历史回放。

## 待运行模型表

| 模型 | RMSE (mV) | MAE (mV) | R² |
|---|---|---|---|
| LSTM | 待运行 | 待运行 | 待运行 |
| F2 | 待运行 | 待运行 | 待运行 |

大型权重、缓存、预测数组不自动加入Git。报告JSON/CSV/PNG/PDF和短日志可在用户检查后提交。默认不存全部海量预测，使用流式指标并保存固定首个Validation窗口的逐点预测与完整时间身份；完整预测可由本次checkpoint、缓存和索引复算，但需用户自行运行。不要称本次已通过模型运行验收。
