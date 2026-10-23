# RealHyFC长期P1S轨迹与一体化可行性

结果：真实观测、样本身份、逐点预测430条、逐折/尾部/回升指标、新H6/H12参数、原生188点复现、条件/物理/TFT/RUL资格均保存。禁止重跑已完成拟合。

从研究主目录复核，不训练：
```powershell
$env:PYTHONUTF8='1'
python -B Phase_P1S_LongTerm_Trajectory_Forecasting/scripts/verify_independent.py
python -B -m unittest discover -s Phase_P1S_LongTerm_Trajectory_Forecasting/tests -v
```

一次性实验入口run_experiment.py freeze/run；本次原执行器在H12不足时中断，随后resume_and_finalize.py在冻结规则下恢复并完成。完整经过见reports/05，失败不是仍在运行。reproduce.py提供安全的完整编排：已完成则只verify；干净副本则运行、仅识别已知资格中断后resume，再qualify/verify/reports/figures，不发布论文或推送Git。

依赖Python3.11、numpy、pandas、scikit-learn、matplotlib，CPU即可。原历史路径只读白名单；未读Raw/混合N1/Test。没有torch、没有新增SOH/物理/RUL网络。所有文件SHA与来源清单保存，模型系数就是实际CPU权重。

候选图scripts/render_figures.py可独立从outputs重绘。经核验后导出的论文包使用相同脚本从source_csv重绘，边界见CLAIM_BOUNDARIES。正式包只新增独立子目录，不替换既有文件。
