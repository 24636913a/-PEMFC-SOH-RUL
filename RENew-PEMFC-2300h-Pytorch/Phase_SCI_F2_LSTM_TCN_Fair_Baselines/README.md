# SCI：LSTM与TCN最小公平基线

状态：`SCI_BASELINE_6_RUNS_COMPLETED`。本阶段6/6正式GPU训练和完整重载审计完成，无重跑、无额外Seed。除授权3007行前缀读取后筛选外，不构造/使用Test或Assessment窗口、目标或评价。历史论文正文/权重只读。

入口：[五模型比较](reports/03_五模型整体性能比较.md)、[工况分析](reports/05_不同历史工况模型比较.md)、[论文影响](reports/08_对SCI论文的影响及下一步建议.md)。表格与真实预测在`tables/`和`outputs/predictions/`；6组英文300dpi PNG/矢量PDF在`figures/`。

RealHyFC数据归属Fensterle等，数据记录[10.5281/zenodo.20715542](https://doi.org/10.5281/zenodo.20715542)，官方已保存元数据为CC BY4.0；原数据论文[10.1016/j.ijhydene.2026.156060](https://doi.org/10.1016/j.ijhydene.2026.156060)。本目录的数据仅为已授权Development缓存/派生预测，不重新发布第三方全文或新的晚期目标。

冻结源：`configs/FROZEN_BASELINE_PROTOCOL.yaml`及.sha256，`outputs/audits/PREFLIGHT_REPORT.json`；模型类逐字继承N2前三个相关class，AST一致。TCN对称padding，不冒称历史内严格因果。两基线参数较少，学习率来自TFT历史配置，没有为其调优。

## 运行与复算

Python3.11.9、torch2.6.0+cu124、pytorch_forecasting1.8.0，RTX3070 Laptop。设置PYTHONUTF8=1/PYTHONDONTWRITEBYTECODE=1并用`python -B`。

- `scripts/preflight.py`：已执行一次性严格前缀读取/身份核验/协议冻结，禁止重新冻结。
- `scripts/run_baselines.py`：已完成6次；只允许跳过已完成Run，失败/中断不自动重训；不得重新运行训练来取得不同结果。
- `scripts/audit_run.py --model LSTM --seed 42`：仅重载最佳checkpoint和现有合法Validation输入、独立复算，不训练。其余5组同理。
- `scripts/compare_models.py`：15设置/135工况设置的公平表、1200标量独立指标核对。
- `scripts/audit_execution.py`：正式6次预算、每轮44个batch和同Seed采样顺序与H的一致性核验。
- `scripts/plot_baselines.py`、`scripts/write_reports.py`：从真实CSV生成图与8份报告。
- `python -B -m unittest discover -s tests -v`：工程和交付检查。
- `scripts/finalize_baselines.py --verify-seal`：最终文件SHA核验；无参数终检仅用于本阶段提交前，勿覆盖原审计。

早停使用Validation属于原Development规则，并未获得独立Test资格。门控/优秀Seed历史结论不因本轮结果自动修改。Git发布身份以提交和被忽略的`outputs/GIT_RECEIPT.local.json`回执为准。没有删除历史数据或上传第三方文献PDF。
