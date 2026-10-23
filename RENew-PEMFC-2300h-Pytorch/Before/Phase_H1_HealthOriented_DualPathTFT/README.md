# H1健康导向事件级双路径TFT：完成

正式判定：H1_HEALTH_TFT_NO_GAIN。先读08_通俗中文总结.md。27个登记实验全部完成，54个真实GPU拟合pass（内选epoch及独立全Train重拟合），15次CPU Ridge拟合；不再重训已完成runID。

固定141个原P1S middle50标签，目标rawID+1。完整周期确认后预测单事件原始V，输入48×14，保留原native TFT/CNN/门控/注意力，单个全零事件查询输出[B,1]。原架构、checkpoint、Ridge和M2只读，无Test数值访问或高频原始重读。

预训练协议及工程验证已先提交77f60bd5c2de9b00099f2e7937cb1019636e91e7；configs/FROZEN_PROTOCOL.json及其SHA不变。正式权重在outputs/checkpoints，各run日志/选择依据在outputs/training_logs。ignored outputs/runtime保存本机恢复状态，不是新增独立实验；正式inner/final权重均交付。

复核（本Phase目录，Python3.11 + torch2.6.0+cu124/pytorch-forecasting1.8.0/numpy/pandas/scipy/sklearn/matplotlib/PyYAML）：

```powershell
$env:PYTHONUTF8='1'
D:\ruanjian\python3.11.9\python.exe -B -m unittest discover -s tests -v
D:\ruanjian\python3.11.9\python.exe -B scripts/verify_independent.py
D:\ruanjian\python3.11.9\python.exe -B scripts/seal_artifacts.py --verify
```

独立程序不优化模型；重放Train-only处理、闭式Ridge、27权重及全部指标。CPU跨设备精度失败保留，以同GPU严格逐位重放进一步核验；需要可用CUDA设备，不把CPU误差阈值放宽掩盖问题。先验证，再按scripts/render_figures.py从CSV重画。readiness是历史asof仿真，不是声称2025年真实部署。

一次性流程prepare_data→engineering→freeze_protocol→预训练Git提交→train→baselines→analyze_results→reference_qualification→verify_independent→result_diagnostics→render_figures→write_reports。现有完成标志阻止重复拟合；不要在原目录删除标志重跑。

最终质量门记录见outputs/FINAL_QUALITY_GATE.json（18项测试），历史1011个已跟踪权重与基准Git字节身份逐一核对，清单为outputs/HISTORICAL_CHECKPOINT_PROTECTION.csv。outputs/ARTIFACT_SHA256.csv与SEAL_RECEIPT.json记录完整交付字节身份；outputs/FINAL_CODE_SHA256.csv覆盖本阶段全部Python代码。图表的真实CSV和绘图脚本对应关系见data_contracts/FIGURE_SOURCE_INDEX.csv，视觉核验记录见outputs/FIGURE_VISUAL_QA.json。重画图或重写核验记录会改变SHA，复核时先使用--verify检查交付原件。
