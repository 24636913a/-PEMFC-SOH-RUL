# F2主程序

本文件夹保存当前六个优秀种子的F2完整主程序副本，以及配套权重映射。

在此文件夹打开终端运行：

```bash
python -B RealHyFC_F2_Main.py --seed 42
python -B RealHyFC_F2_Main.py --seed all
```

仅支持42、103、193、202、210、212，默认42。all输出六个种子等权平均RMSE、MAE、R²，不进行预测集成。

为适配当前位置，副本仅调整研究根目录定位和本地种子映射文件定位；所有模型类、处理函数、评价和绘图逻辑与研究根目录原文件完全一致。原主文件和历史成果均保留。

数据、权重、冻结配置和预测缓存仍从原研究目录读取，不重复复制大型文件。因此请保留该文件夹在当前仓库中的位置，不能脱离仓库独立运行。输出仍保存到研究根目录的RealHyFC_F2_Main_Outputs，每次运行新建目录，不覆盖旧结果。本次没有训练或调参。

## VS Code交互窗口

1. 在VS Code打开本仓库，启用Python和Jupyter扩展，选择原项目解释器 `D:\ruanjian\python3.11.9\python.exe`。该环境已具备Python 3.11、CUDA、IPython及ipykernel。
2. 打开 `F2_VSCode_Interactive.py`。在第一段代码顶部设置 `SEED = 202` 或 `SEED = "all"`；也支持42、103、193、210、212。交互入口不读取Jupyter的命令行连接参数。
3. 点击每个 `# %%` 上方的 **Run Cell** 顺序运行，或者通过命令面板运行 **Jupyter: Run Current File in Python Interactive Window** / **Run All Cells**。第一单元必须先运行；不要只运行末尾的预测单元。
4. 表格、阶段进度和图像直接出现在Python Interactive Window。程序保留Agg后端，使用 `IPython.display.Image` 显示实际保存的PNG，不依赖弹出窗口。
5. 也可直接打开同名 `F2_VSCode_Interactive.ipynb`，选择同一解释器对应的内核，在第一代码单元改种子后点击 **Run All**。Notebook为 `.py` 各代码单元的等价副本。

交互展示按真实顺序包括：3843条原始分箱电压和四区间合法窗口、Train-only scaler统计、Train Spearman排名及冻结Top10、首个合法Train窗口的因果EWMA alpha=0.3、当前实际采用的超参数、两类历史搜索、各种子的历史训练曲线、冻结权重真实GPU推理、预测曲线和散点图，最后给出RMSE/MAE/R²。

基础TFT历史Optuna图读取三个原SQLite数据库，采用只读连接；纵轴是各自历史inner-stop的标准化RMSE。F2后续六次调参图读取独立阶段CSV，纵轴是Train内inner-TUNE的mV RMSE。两者目标和单位不同，不放在同一排名中。当前六个优秀checkpoint仍使用原继承配置，后来的F2调参未整体改善RMSE，未替换这些权重。

训练图只读取真实历史Epoch日志，明确标注“historical / no new training”。进度对应实际读取、资格核验、加载、推理和保存，均不模拟训练或Optuna。

每次从第一单元重新Run All会创建全新的 `RealHyFC_F2_Main_Outputs/<UTC时间>_F2_interactive_seed*/`。数据展示图位于 `presentation/`；每种子保留原有三张PNG、预测CSV、模型副本和预处理参数。指标仅汇总Validation，全部4920个24步查询/种子；时间曲线固定第24步，不拼接重叠窗口。重复中间单元可能遇到已有输出保护；需要完整重跑时请从第一单元开始新建本次运行。

## 真实显示验收

`interactive_acceptance/Seed202.executed.ipynb` 和 `SixSeeds.executed.ipynb` 保存了原Python 3.11/CUDA环境中实际执行的所有单元及图像，可直接在VS Code打开查看。前者有8个、后者有23个真实 `image/png` 显示输出。`ACCEPTANCE.json`记录逐点与原CLI完全一致、历史文件未变及原CLI三行指标回归结果。

本次显示验收是在真实IPython内核中完成，未声称已经通过自动点击验收VS Code图形界面。VS Code Interactive Window和Notebook使用同一IPython显示协议。验收不是截图模拟，已执行Notebook内保存的是内核真正发出的PNG数据。

静态及已保存显示记录测试：

```bash
python -B -m unittest discover -s tests -p test_interactive_contract.py
```

`tests/run_interactive_acceptance.py`为本次真实内核验收脚本（202和all，再检查原CLI202），不会训练或搜索。它会保留完整执行记录并拒绝覆盖已有验收文件夹；已有验收结果可直接查看和运行上述轻量测试。
