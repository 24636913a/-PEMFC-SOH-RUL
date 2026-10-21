# RealHyFC 原始约1秒：F2训练预算与事件损失检验

本研究只检验用户已经明确指定的方案：扩大合法训练起点和训练预算后，原生F2是否改善；再用一次固定1:3事件加权损失检验收益与普通区间代价。并不预设F2胜过LSTM或RMSE低于10mV。

唯一数据是原17份RealHyFC Loadcycle CSV。历史 `RealHyFC_Raw1s_Dynamic_Study`、旧模型代码、权重与预测全部只读。程序不涉及IEEE 2014，不构造SOH或RUL标签。新产物只写本目录。

先读00训练协议和01覆盖分析，再读05总结；02–04保留逐模型、事件与误差细节。配置权威为 `outputs/frozen_protocol.json`，不得按结果改写。运行次数、真实Epoch和更新步数见 `outputs/final_execution_summary.json` 与各run日志，数字以最终存在且COMPLETE的实例为准。

新增唯一VS Code交互入口为 `RealHyFC_Raw1s_F2_Interactive.py`，默认只读。`scripts/`是明确选择后调用的批处理程序，不会因导入自动训练。

约1Hz原始采样、逐秒预测目标、每60秒的预测起点是不同概念。本研究不声称完成每秒在线更新评价。120秒主指标保留所有起点×提前量查询；同一目标秒的多次预测不算独立实验重复。辅助去重统计固定保留最新起点的真实预测，不做预测平均。

代码与报告由AI辅助实现；所有数值结论以原始记录、真实训练日志、逐查询输出及独立核验为依据。文献是方法依据，不保证当前方案有效。所有结论限于Seed202与冻结预算。

## 阅读和复现

1. 用VS Code打开本目录唯一交互入口，保持 `MODE="READ_ONLY"` 可逐单元读取冻结起点、状态、指标和报告。显式 `LOAD_MODEL` 默认CPU加载，不触发训练。
2. 原始17个CSV保留在仓库的Git LFS位置，不复制到新目录。运行时需要历史目录已校验的8个只读memmap缓存；新目录和ZIP本身不是脱离原仓库即可训练的独立数据包。若缓存缺失，应先根据旧研究脚本在独立位置重建并验证SHA，不得在历史目录上覆盖重建。
3. 环境与版本见 `outputs/environment.json`；正式训练使用PyTorch2.6.0+cu124、RTX3070 Laptop单GPU串行执行。下列命令均需在本目录运行，使用原环境Python；不会修改历史文件。

```powershell
& 'D:\ruanjian\python3.11.9\python.exe' -B .\RealHyFC_Raw1s_F2_Interactive.py
& 'D:\ruanjian\python3.11.9\python.exe' -B .\scripts\verify_optimization.py --stage final
& 'D:\ruanjian\python3.11.9\python.exe' -B .\scripts\verify_diagnostic_outputs.py
```

`--stage final`可复核已保存、身份SHA完全相符的独立GPU重放证据，不自动训练。首次重新产生GPU重放证据必须先确认无训练进程，再显式使用 `--allow-serial-gpu-replay`；这只是推理，不增加训练实例。`train.py --run ID`是训练命令，不是阅读结果所需步骤。交互入口另要求显式RUN_ID和第二次训练确认；已完成run复用，未完成/失败run拒绝覆盖，B2仍必须通过B1门槛。

## 复现边界与失败记录

A2独立CPU重载曾未通过预先设定的1e-5V容差。完整探针最大差未序列化，保留null；固定前64个起点另测最大差1.4508604783e-5V，不能把此子集值冒充完整最大差。未放宽容差、改预测或重训。后续在原GPU环境用字节一致输入、batch16、相同CuDNN开关，按更严格1e-7V对全部60秒及600秒网格独立重放。逐模型真实结果见 `outputs/review_checkpoint_gpu_reloads.json`，CPU失败见 `outputs/review_checkpoint_cpu_probe.json`。没有证明CPU跨后端逐值一致，原因未唯一确定。

`outputs/final_acceptance.json`是独立数值核验，`outputs/independent_diagnostic_acceptance.json`另复算尾部、episode、工况、逐提前量和先兆表，`outputs/review_checkpoint3.json`记录最终报告审阅及作者角色边界。图像统一位于 `figures/`，每项含300dpi PNG、PDF和绘图CSV；逐目标原始毫秒时间戳在各run的NPZ。文件清单SHA见 `outputs/artifact_manifest.json`。
