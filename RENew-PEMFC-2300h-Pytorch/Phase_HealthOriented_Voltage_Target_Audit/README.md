# 健康导向电压预测目标资格审查

实际推荐C：保留动态双路径TFT主目标和冻结Ridge96辅助；不是宣称固定电压不可预测，而是当前不足以直接替换主任务并开展健康TFT训练。

先读07_通俗中文总结.md和06_推荐方案与最小训练协议.md。代码只在本阶段写入，无模型训练/推理、Optuna、Test数值访问或GPU。

一次性执行顺序：freeze.py→extract_development.py→audit_targets.py→supplement.py。完成标志禁止重复原始提取/审查；原始复现须在独立干净副本执行，不能删除现有结果。提取读各原始源认证的Development前缀，30分钟数据另用3007身份→3005数值规则。

现有结果复核（本Phase为工作目录）：

```powershell
$env:PYTHONUTF8='1'
D:\ruanjian\python3.11.9\python.exe -B -m unittest discover -s tests -v
D:\ruanjian\python3.11.9\python.exe -B scripts/verify_independent.py
```

Python3.11 + numpy/pandas/PyYAML/matplotlib。独立程序核对全部新清单/稳定门槛/窗口标签数量、3796个历史片段和141个冻结周期，并以csv+datetime+statistics单独抽查三种主要电流状态首/中/末共9个真实片段。不是声称对10680个片段全部重新读原始数值。prefix SHA追溯14份合法源，旧代码与checkpoint的SHA保持。

失败的原始包含75%边界版本及错误记录保存在outputs/failures，不进入正式分析。原始有效表为RAW_FIXED_STATE_DWELL_MEASUREMENTS.csv，QC子集不能覆盖/替换冻结PAA周期标签。所有figure均来自CSV，另有推荐路线SVG。
