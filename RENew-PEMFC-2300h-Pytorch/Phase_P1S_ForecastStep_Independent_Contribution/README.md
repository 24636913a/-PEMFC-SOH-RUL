# P1S预测步骤独立贡献

仅RealHyFC Development派生缓存，零训练、零新校准、零Test/GPU。冻结协议、源SHA、13份精确输入快照、18条预测、6条误差分解及全部测试代码保存于本目录。

从研究主目录复核（不会重新训练或改写历史）：

```powershell
$env:PYTHONUTF8='1'
D:\ruanjian\python3.11.9\python.exe -B Phase_P1S_ForecastStep_Independent_Contribution/scripts/verify_independent.py
D:\ruanjian\python3.11.9\python.exe -B -m unittest discover -s Phase_P1S_ForecastStep_Independent_Contribution/tests -v
```

run_audit.py freeze/run为一次性运行入口；已存在COMPLETE时禁止再run。需要完整独立复现时使用干净仓库副本的新阶段目录，复制脚本、删除副本生成物后freeze/run，不删除原成果。绘图render_paper_figures.py只读真实CSV，可重绘。它也可以原样在导出的论文包scripts目录运行。

关键文件：SOH_METRICS、PAIRWISE_COMPARISONS、SEQUENTIAL_CONTRIBUTION、ERROR_DECOMPOSITION、H3_VS_H1_ERROR_CANCELLATION、ORIGIN_MEASUREMENT_TIME_AND_IDENTITY_AUDIT。完整缓存汇总是pooled，仅供背景；历史正式三fold等权指标不变。

SHA256见inputs/SOURCE_SHA256.csv和outputs/ARTIFACT_SHA256.csv；独立高精度报告见INDEPENDENT_VERIFICATION。最终研究判断见reports/00。论文成果只导出到REAL_Paper_File下独立子目录，不替换旧文件。
