# 第二轮F2优质模型实例筛选

仅改变随机种子；201～250在训练前一次冻结。每个种子从头初始化，单GPU串行执行。A级为Validation pooled RMSE≤6.30mV；同时满足12h终点RMSE≤10mV和最大单点绝对误差≤50mV方为合格。找到3个新合格实例立即停止启动后续种子；未达到则受50候选与12小时原始墙钟预算限制。为完整450epoch最慢训练及收尾留3420秒才启动下一次；不增加训练预算，不删除不利目标。

预检按read_csv(nrows=3007)、bin_end_ophrs2_h<1819.997738独立重建3005条Development记录；1380训练窗口、205验证窗口及4920个预测目标、全部张量和指纹必须与第一轮严格一致。训练只读取新的合法冻结张量。没有Test或Assessment推断。

训练计算继承第一轮train_seed_v3，逐字符差异核验：只迁移输出目录/身份校验、将已有有界原子替换重试用于checkpoint保存、在训练结束后增加指标。日志重复epoch修复保留。所有失败及partial权重保留，不重训。源代码、配置和历史文件有SHA256保护。

本轮门槛在阅读第一轮结果后提出，并于201～250训练前固定；属于进一步探索性Validation筛选，不是独立确认性实验。找到了优质实例不等于总体稳定性或结构优势获得独立证明。

全部有效新种子的均值也仅描述截至预设停止时已完成的固定种子序列；按“成功3个即停止”的规则得到的样本，不能自动视为无偏的随机种子总体估计。

training_time_s沿用第一轮GPU拟合循环的墙钟计时，包含验证与日志开销，不是CUDA Event的纯内核时间。运行中的GPU资源观察如有记录，保存在outputs/execution_ledger/RESOURCE_OBSERVATIONS.jsonl；资源占用波动不作为修改训练协议或重新运行种子的理由。

Historical 42/103/193使用历史保存权重只读GPU重放并核对逐点真值；不修改历史结果。主指标和R²仍为原始ΔV口径。绝对电压仅以原始起点电压+ΔV恢复一次；图中每条曲线只包含同一起点的24步，无重叠拼接。6个候选是筛选后的实例，均值不能作为模型无偏总体性能。

执行：Python3.11.9 / Torch2.6.0+cu124 / PF1.8.0 / RTX3070 Laptop；所有命令加-B并设置PYTHONUTF8=1、PYTHONDONTWRITEBYTECODE=1。

沿用Phase H普通CUDA协议，不启用已知不支持相关插值反向算子的严格确定性模式；不声称跨训练逐位确定性。独立复算核验保存权重的同GPU推理与真实预测缓存是否逐点一致。

```powershell
python -B scripts/prepare_round2.py  # 一次性，完成后禁止重做
python -B -m unittest discover -s tests -v
python -B scripts/run_round2.py
python -B scripts/monitor_progress.py --once
python -B scripts/audit_round2.py    # 仅训练结束后；只读GPU重放
python -B scripts/audit_delivery.py --verify-local-journals
```

断点续做先核对进程和ledger；不得重启STARTED/FAILED或已有run_state的种子。--resume仅接受已解除现场问题且仍在原12小时预算内的未运行种子，不重置计时。超过预算的后续继续需单独登记新的执行预算，不可偷偷重复启动。

若因watchdog或执行错误中断，finalize先保存INTERRUPTION_REVIEW记录，不把中断自动发布为完成或反复提交partial结果。续做前必须保留原ledger及日志，核验GPU、配置、已完成结果和partial权重；只继续NOT_STARTED，不改变原始起点、训练规则或筛选阈值。

最终自动生成六组300dpi PNG/PDF、六份中文报告、完整候选状态、逐点预测/24步误差、独立审计与SHA。仅本阶段提交main并普通推送，失败保留本地提交及错误，不修改代理和电源计划。完整逐批journal仅本地，其余是否备份以GIT_RECEIPT.local.json实际推送核验为准。结束后停止。
