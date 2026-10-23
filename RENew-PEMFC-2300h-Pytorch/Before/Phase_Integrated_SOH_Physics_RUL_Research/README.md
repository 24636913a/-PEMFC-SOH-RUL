# RealHyFC一体化研究

仅在本目录写入；原历史只读。Python3.11，numpy/pandas/matplotlib；正式新拟合只包括低自由度过去仿射校准与描述性轨迹，不需要torch或GPU。

已完成输出存在时不得再次run。复核命令从研究主目录运行：

```powershell
$env:PYTHONUTF8='1'
D:\ruanjian\python3.11.9\python.exe -B Phase_Integrated_SOH_Physics_RUL_Research\scripts\verify_results.py a
D:\ruanjian\python3.11.9\python.exe -B -m unittest discover -s Phase_Integrated_SOH_Physics_RUL_Research/tests -v
```

一次性复现需将本目录完整复制到另一干净副本，删除副本生成物（保留代码），先运行run_stage_a.py freeze，再run；此说明不授权删除原始成果。原协议、源SHA、模型系数与全部事件结果均保存。最终的b/c、final复算命令随交付脚本保存。

最终复核：verify_results.py final；一次性B/C计算run_stage_bc.py freeze后run；绘图render_results.py a/bc，报告write_reports.py a/final。A/B/C完成标记阻止重复计算；未来复核只用verify/tests。
