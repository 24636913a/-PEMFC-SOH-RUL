# 双路径TFT电压型SOH指标可行性

实际结论B：回顾性条件电压保持率有统计共同支持；BoL/在线/参考SOH资格均未成立。

先读06_下一步决策.md及02_同工况匹配资格分析.md。全部结果仅Development，原阶段与权重只读。

现有结果复算（无需读取原混合数据或训练）：

```powershell
$env:PYTHONUTF8='1'
D:\ruanjian\python3.11.9\python.exe -B scripts/verify_independent.py
D:\ruanjian\python3.11.9\python.exe -B -m unittest discover -s tests -v
```

工作目录为本Phase。Python3.11，numpy/pandas/SciPy/PyYAML/matplotlib；不需要torch。prepare.py→freeze.py→analyze.py→diagnostics.py是一次性数据读取与分析，完成标志阻止覆盖；不要在现有成果上重复运行。原始重现只能在独立干净副本按顺序执行。verify_independent.py从原冻结源与本阶段CSV独立复算，不导入分析函数。render_figures.py仅用CSV绘图；write_reports.py从实际CSV/JSON生成中文报告；quality_gate.py保留每次测试/验证失败证据，seal.py封存SHA和代码/图片索引。

主规则SHA见configs/FROZEN_PROTOCOL.sha256；来源Git blob和SHA见data_contracts/SOURCE_SHA256.csv；全成果SHA见outputs/ARTIFACT_SHA256.csv。图源/代码见data_contracts/FIGURE_SOURCE_INDEX.csv。无Test数值读取、无新模型推理训练、无GPU。
