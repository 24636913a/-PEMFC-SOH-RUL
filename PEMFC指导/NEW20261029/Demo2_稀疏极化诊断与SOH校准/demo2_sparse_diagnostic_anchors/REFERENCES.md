# 科学背景与数据来源核对

本包的 CSV 均由随附代码生成，属于模拟数据。本包没有下载、加工或重分发下面的原始实验数据。文献用于说明问题背景与建模边界，不为本包的数值结果提供实验背书。

## 1. 可逆恢复与长期损失

Gazdzicki, P., Mitzel, J., Garcia Sanchez, D., Schulze, M., & Friedrich, K. A. (2016). *Evaluation of reversible and irreversible degradation rates of polymer electrolyte membrane fuel cells tested in automotive conditions*. Journal of Power Sources, 327, 86–95.

- DOI：https://doi.org/10.1016/j.jpowsour.2016.07.049
- 作者机构公开记录与全文：https://elib.dlr.de/105216/
- 用途：支持区分可逆与不可逆电压损失、按明确的恢复协议解释恢复后电压，以及恢复后随运行时间变化的可逆损失模型。
- 边界：不能据此断言任何停机重启都会完全恢复，也不能把所有观测电压上升唯一归因于某一个机理。

Zhang, Q., Schulze, M., Gazdzicki, P., & Friedrich, K. A. (2021). *Quantification of effects of performance recovery procedures for polymer electrolyte membrane fuel cells*. Journal of Power Sources. DOI: 10.1016/j.jpowsour.2021.230467.

- DOI：https://doi.org/10.1016/j.jpowsour.2021.230467
- 作者机构记录：https://elib.dlr.de/144088/
- 用途：不同恢复程序需要分别量化恢复和未恢复的损失；比较电压应交代电流密度、条件及恢复基准。

## 2. FC1 原始数据的可核验信息

Pahon, E., Péra, M.-C., Bouquain, D., & Hissel, D. (2022). *Impact of current ripples on the durability of proton exchange membrane fuel cells based on two ageing datasets*. Data in Brief, 45, 108601.

- 原文：https://pmc.ncbi.nlm.nih.gov/articles/PMC9679439/
- DOI：https://doi.org/10.1016/j.dib.2022.108601
- 数据 DOI：https://doi.org/10.25666/DATAUBFC-2021-07-19
- 原文说明：FC1 为 5 片电堆，在恒电流、名义电流密度 0.7 A/cm² 条件下试验；恒电流与恒功率不是同一工况。
- 原文 Table 3 列出的 FC1 极化曲线与 EIS 诊断时刻为 0、48、185、348、515、658、823、991 h，共 8 个会话。
- 本包所用的 809 h 恢复事件、采样频率及各数值轨迹是演示设定，不是从这个原始数据逐条复现的事件。

## 3. RealHyFC 名称与 10 片电堆

Fensterle, E., et al. (2026). *Spatially Resolved Current Density and Electrochemical Impedance Spectroscopy in PEM Fuel Cells*. Zenodo dataset, version 1.

- 数据记录：https://zenodo.org/records/19068126
- 数据 DOI：https://doi.org/10.5281/zenodo.19068126
- 相关论文：https://doi.org/10.1002/fuce.70057
- 记录确认：相关 RealHyFC 项目测量使用 10 片 HyFaB 短堆，包含极化曲线、工况敏感性测试及 EIS 等。
- 本包没有把该记录等同于用户手头的特定全寿命文件，也没有从此记录得出“FC1 老化一定快 3 倍”。3 倍速率、动态载荷轨迹和诊断日程均是用户场景启发的模拟设定。

## 4. 工具参考

- NumPy 线性最小二乘：https://numpy.org/doc/stable/reference/generated/numpy.linalg.lstsq.html
- SciPy 带边界线性最小二乘：https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.lsq_linear.html
- 代码实际采用哪些方法以 `run_demo.py` 和本包的模块为准；本包不是文献算法的逐行复现。

资料核对日期：2026-10-29。
