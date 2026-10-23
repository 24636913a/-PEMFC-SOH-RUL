# Phase P2 — Final Gate Ablation

仅明确授权的G0/G1/G2 × Seed42/52/62九次新训练。P1保留工程预检查身份；P2单独授权。不增加模型、物理损失或搜索，不访问Assessment/Test，不加载历史权重，不生成SOH/RUL结论。

仓库根目录执行一次：

```powershell
python -B RENew-PEMFC-2300h-Pytorch/Phase_P2_Final_Gate_Ablation_Training/scripts/run_p2_training.py --freeze
python -B RENew-PEMFC-2300h-Pytorch/Phase_P2_Final_Gate_Ablation_Training/scripts/run_p2_training.py --run
```

禁止重置ledger或失败重跑。源码、协议、数据、scaler、初始参数/RNG、采样顺序、数值、checkpoint重载及预测缓存任一校验失败都会停止后续组并保留现场。每组子进程串行，顺序42:G0/G1/G2；52:G1/G2/G0；62:G2/G0/G1。

只使用nrows3007→年龄过滤3005的Development，再构建Train[0,2103)/Validation[2103,2404)的1380/205窗口。复用原H数据处理、P1 fresh共享初始化和E→C batch32/no_grad评价；保持450epoch/pat40/标准化RMSE早停。

checkpoint及详细batch journal本地保留在`outputs/`，Git忽略大型文件但从不删除唯一副本。最终本地manifest记录路径、SHA、bytes与真实备份状态（无独立备份，不伪称有备份）。完整epoch日志、采样序列、metrics、predictions与审计提交Git。

图表基于真实九次结果，采用scientific-figure-making风格、全部seed散点、sample SD和300dpi PNG/PDF；不把合成guard-test数值当研究结果。完整预定门槛和技术失败独立记录，新模型不自动替换冻结F2。完成或硬停止后停止。
