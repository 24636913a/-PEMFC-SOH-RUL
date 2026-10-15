# 动态F2固定种子筛选：Development Validation

只训练冻结Phase H的动态48→24 F2，原14槽位/Top10 mask、Train-only scaler、EWMAα0.3、MSE/AdamW及全部超参数不变。完整配置和原代码SHA在configs/frozen_seed_screening.json，20seed顺序固定，不追加。没有SOH/RUL/PINN或新F0/F1训练，也不访问Test或对Assessment作新推断。

第一次控制器在Seed101首批日志写入时失败，零前向、零反向、零优化步骤。原失败、原worker与原配置均保留，不重跑101。不可变LOGGER_REPAIR_AMENDMENT记录修复日志dict重复键，并从原列表未启动Seed103继续。原8小时截止时间不重置。不要删除ledger、覆盖权重或重复启动既有runID。

实际运行状态见outputs/execution_ledger/PROGRESS_ENGINEERING_REPAIR.json；原PROGRESS.json是保留的首次故障现场。正常结束后自动执行9项测试、历史权重保护、独立逐点指标复算和保存权重同GPU重放、6组真实图表、中文报告。独立publisher等待控制器恢复临时防睡眠并退出，再封存、仅提交本阶段并普通推送main。网络失败保留本地提交，不修改代理配置、不强推。

完整排名F2_SEED_RANKING.csv、Top5、全池统计和FINAL_SUMMARY均在outputs。Top5/最佳是同一Validation筛选后的候选成绩，不能冒充随机初始化一般性能或公平结构优越性。R²沿用Phase H的ΔV口径。电压图使用原始起点V加ΔV（只恢复一次），各曲线仅对应一个起点的24步，不拼接重叠窗口。

复核（Python3.11.9、Torch2.6.0+cu124、pytorch-forecasting1.8.0及原项目依赖；CUDA3070）：

```powershell
$env:PYTHONUTF8='1'
D:\ruanjian\python3.11.9\python.exe -B -m unittest discover -s tests -v
D:\ruanjian\python3.11.9\python.exe -B scripts/audit_seed_results.py
D:\ruanjian\python3.11.9\python.exe -B scripts/protect_history.py
D:\ruanjian\python3.11.9\python.exe -B scripts/audit_delivery.py --verify-local-journals
```

训练中不得重放GPU独立审计争用设备。复算不执行optimizer；原始N1不是复算必需，合法冻结输入张量已保存。生成正式图表会重写本阶段图表字节，因此SHA核验应先针对交付原件。

正式checkpoint、预测、epoch日志计划随Git提交。完整batch_events.jsonl仅保留本地，不声称已备份；具体路径/大小/SHA/best epoch见LOCAL_ARTIFACT_INVENTORY.csv。提交与推送状态见仅本地outputs/GIT_RECEIPT.local.json；成功GitHub推送是已跟踪文件的备份证据，未跟踪完整batch日志仍未备份。
