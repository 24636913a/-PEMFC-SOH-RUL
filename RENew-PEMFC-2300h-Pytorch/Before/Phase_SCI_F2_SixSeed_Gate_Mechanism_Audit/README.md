# 六个筛选F2实例的门控机制审计

基准提交：`dae9331ec660b3e91e23c2064d9c547b26b46393`。协议SHA256：`81f3c990bc5ee4aeca2fb706da746a0c8961af63b34999c715ed6f841525e409`。

六个实例按原始F2的Validation成绩筛选。本轮是筛选后机制诊断，不能代表一般随机初始化或独立Test。每个实例205个起点、4920个查询、251个去重目标时刻；重叠窗口不构成独立统计样本。不读取Assessment/Test，不训练、不更新任何参数。

本阶段仅前向推理；原始权重、历史输出、J及P2结论只读。先查outputs/audits再续做，禁止重复冻结协议。

运行（项目环境Python3.11.9 / Torch2.6.0+cu124 / PF1.8.0 / 原RTX3070 CUDA协议）：

```powershell
python -B scripts/prepare.py  # 已完成；不能重新冻结
python -B -m unittest discover -s tests -v
python -B scripts/run_inference.py  # 完成的Seed会严格检查后跳过
python -B scripts/analyze.py
python -B scripts/plot_figures.py
python -B scripts/verify.py
python -B scripts/finalize_figures.py  # presentation-only QA
python -B scripts/finalize_reports.py # dependency-free reports + evidence interpretation
python -B scripts/seal.py
```

所有命令设置PYTHONUTF8=1和PYTHONDONTWRITEBYTECODE=1。输入为历史nrows=3007→过滤3005行协议构建并已冻结的合法Train/Validation张量；本轮不打开原始CSV、不重新构造样本。SHA协议与历史保护表位于configs及outputs/audits。临时hook每次移除；每个干预后正常预测逐点恢复。独立verify不调用主指标函数，以标量运算重算全部CSV并检查键/真值/时间/门控分布。完整图源数据及索引见outputs/FIGURE_INDEX.csv。最终SHA256清单见outputs/DELIVERY_MANIFEST.json。

初始冻结源码保持不变。后加的两个finalize脚本仅修复报告可选依赖缺失及绘图排版/缺口连线，均不运行模型或改变计算规则；身份由最终交付SHA清单锁定。完整错误说明见outputs/audits/REPORTING_AMENDMENT.json。图01不跨缺失起点连线；图片仅显示合法观测。图06/07/08每个panel为单一起点24步，标题中的205为每Seed总体起点数。

## 最终核验与复现说明

17项测试通过；78组独立GPU重放共383760点严格一致；624项标量指标复算最大差异8.881784197001252e-16；1099个历史checkpoint及共1646个历史文件SHA未变。以上证据分别见outputs/audits/FINAL_TESTS.json、INDEPENDENT_VERIFICATION.json及HISTORY_BEFORE.csv。

报告用finalize_reports.py生成，保留原冻结报告代码及依赖错误记录；图表用plot_figures.py再运行finalize_figures.py。最终运行python -B scripts/delivery_qa.py后才运行seal.py。封存后重新绘图会改变PDF元数据，因此应先核验原件SHA，再在副本中复现图形。未创建任何训练任务或后续自动执行。
