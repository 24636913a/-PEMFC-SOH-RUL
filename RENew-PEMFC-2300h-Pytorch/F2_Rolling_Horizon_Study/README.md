# RealHyFC动态电压：Seed202，48→6/12/24

本目录是探索性单种子研究。先阅读 `reports/05_最终通俗总结.md`，实际结果见 `outputs/ACTIVE_RUN.json` 指向运行中的 `FINAL_RESULT_TABLE.csv`。历史模型来自指定20261019T055734批次，不能与其他批次的最佳种子混用。

## VS Code正式入口

打开 `F2_Rolling_Horizon_Interactive.py`，选择项目Python3.11/CUDA环境，使用`# %%`的Run Cell或Run All Cells。

- `MODE='show'`：默认，仅展示本轮已保存的真实结果和PNG。
- `MODE='analyze'`：只读从逐点CSV重新计算指标，不训练、不改写结果。
- `MODE='train'`：创建全新时间戳目录，先历史复算，再执行最多6次真实Seed202训练，显示Epoch和新Loss图；全部best冻结后统一后留评价并独立验收。新图保存到该新运行的presentation目录，不覆盖本次交付图。不要反复运行以挑选有利CUDA轨迹。

只支持Seed202。H6/12/24对应3/6/12小时，历史48步。没有SOH、RUL、长期代理、Top11或更多种子。训练模式不热启动旧权重。show/analyze显示均明确为历史记录或只读结果，不冒充实时训练。

## 六次新训练

|任务|模型|
|---|---|
|48→6|原生F2、单层hidden64 LSTM|
|48→12|原生F2、单层hidden64 LSTM|
|48→24|本轮原生F2 C0、本轮输出残差头F2 C1|

历史Seed202 H24只读复算，不计入新训练。普通CUDA不保证同种子训练逐位相等；不重跑直到获得好成绩。全部实例的初始化、梯度、优化器步数、随机批次顺序、最佳/停止Epoch和权重SHA独立保存。

## 结果与复现索引

- `configs/FROZEN_PROTOCOL.json`：训练前冻结的目标、时间分区、H、预算、超参数和C1数学结构。
- `scripts/historical_audit.py`：P1指定旧批次的前6/12/24步、固定提前量、尾部和身份核验；不会训练旧模型。
- `scripts/data.py`：H专属合法窗口、Train-only scaler、因果EWMA、Top10、原始ΔV标签及恢复电压。
- `scripts/native_models.py`：完整原生F2类；只改必要的预测长度接口；不import旧训练入口。
- `scripts/models.py`：普通LSTM及C1输出头，保留原生TFT和双路径。
- `scripts/train.py`：真实GPU训练，单GPU串行；完成的实例不重复训练，partial异常保留等待工程核验。
- `scripts/evaluate.py`：全部6个best冻结后的一次后留推理；H全部起点与H24共同起点分别保存，禁止重复正式评价。
- `scripts/verify_independent.py`：独立重加载6个checkpoint、18份预测回放、指标/时间/窗口/Scaler/梯度与Early Stopping；`--history`检查所有旧文件SHA。
- `scripts/render_figures.py`：真实CSV绘制300dpi PNG/PDF。连续时间电压图严格固定末步，断点断线。
- `scripts/write_reports.py`：生成六份中文报告和最终表格，不重训。
- `tests/test_frozen_task.py`：结构、因果输入、目标、Scaler、共同起点和合成梯度检查。合成反向测试不更新optimizer，不计为真实数据正式训练实例。

每个运行 `models/<model>_H<h>/` 保存initial.pt、best.pt、逐epoch日志、参数变化、Train/Validation/后留逐点预测、运行身份。失败时另存原始traceback及partial.pt，不静默补分。完整指标保留全部异常点。主R²针对原始ΔV；绝对电压散点仅作显示。

## 主要图像

|图|分析用途|
|---|---|
|F1|历史H24截断和固定提前量误差|
|F2|3/6/12h的全部起点及共同起点RMSE|
|F3|相同527起点的逐提前量曲线|
|F4|预先固定起点/历史主导事件的完整预测轨迹|
|F5|固定末步的真实/预测电压随时间变化|
|F6|全部预测散点，坐标一致，不删异常|
|F7|Train阈值定义的工况误差分布，未来状态仅事后诊断|
|F8|6次本轮真实Loss记录|
|F9|同轮C0/C1配对结果|
|F10|真实历史工况门控|
|F11|历史路径潜变量幅值与输出头实际补充幅度|
|F12|尾部SSE贡献与目标时间等权统计|

图像保留PNG/PDF及原始绘图CSV。潜变量幅值不是电压精度的因果贡献；C1 head-off是同权重推理反事实，不是独立训练C0或F1。

## 验收命令

从本目录用项目Python执行：

```powershell
python -B -m unittest discover -s tests -v
python -B scripts/verify_independent.py
python -B scripts/verify_independent.py --history
```

复绘与只读重算不启动训练。不要对已完成运行再次调用正式后留评价；需要核验请使用verify_independent。若要新的训练，使用交互入口train模式创建独立运行。

最后20%已经被多次历史研究查看，Seed202也按该区间表现选中。本轮是探索性单种子对照，不能称独立盲测或随机种子总体性能。不同H的全部合法样本和Train拟合Scaler不同，额外共同起点指标帮助比较，但不能把所有差异都归因于预测长度。本轮没有H24新LSTM，因此该跨度与LSTM的同轮胜负记NA。

所有旧目录、checkpoint、数据和研究结果保留。本轮不自动启动下一阶段。
