# 正式入口：F2_Train_Interactive.py

新版正式运行入口只有 `F2_Train_Interactive.py`。它内嵌原始F2的必要网络类，能够独立处理数据、从头训练和预测，不导入旧主程序，不读取旧Seed Assets、旧checkpoint、旧预测或旧Epoch日志作为本次结果。

## VS Code运行

1. 用VS Code打开仓库，选择原项目Python 3.11/CUDA解释器（本机为 `D:\ruanjian\python3.11.9\python.exe`），启用Python和Jupyter扩展。
2. 打开 `F2_Train_Interactive.py`，在首单元设置：

```python
SEED = 202
FEATURE_MODE = "compare"
MAX_EPOCHS = 450
PATIENCE = 40
```

3. 点击 **Run All Cells**，或按顺序逐个运行 `# %%` 单元。每次从第一单元开始完整运行，都会新建实验目录、重新初始化模型和AdamW优化器、真实执行反向传播与参数更新。
4. `SEED`只支持42、103、193、202、210、212和`"all"`；`FEATURE_MODE`只支持`"top10"`、`"top11"`、`"compare"`。all+compare是12次串行独立训练，不能把它当成快速历史回放。
5. 不要在仍在训练的运行中反复执行训练单元。需要重跑时，从第一单元开始创建全新运行；旧运行和partial结果均保留，不进行热启动或恢复训练。

依赖为现有环境中的PyTorch、pytorch-forecasting、NumPy、Pandas、SciPy、scikit-learn、matplotlib、IPython；不需要旧主程序。数据仍读取研究根目录 `Phase_N1_Data_Foundation/outputs/causal_feature_tables/causal_features_30min.csv`，因此保留当前目录与数据资产的位置关系。

## 新协议与准确边界

完整源文件固定核验3843行，按记录时间顺序进行一次70:10:20划分，取整数边界floor(0.7N)、floor(0.8N)，不做随机划分、不做交叉验证。

| 区间 | 零基行号，左闭右开 | 记录数 | 实际运行小时范围 | 合法起点 | 24步查询 | 唯一目标时间 |
|---|---|---:|---|---:|---:|---:|
| Train | [0,2690) | 2690 | 73.0—1622.0 h | 1750 | 42000 | 2026 |
| Validation | [2690,3074) | 384 | 1622.5—1854.0 h | 218 | 5232 | 264 |
| 时间后留评价 | [3074,3843) | 769 | 1854.5—2257.0 h | 527 | 12648 | 619 |

表中是实际运行小时，不将分箱行号直接当成小时，也不强行把名义“2300h”数据集末点写成2300运行小时。各区间准确日历边界和每个合法窗口身份保存为CSV。

保持48步历史、24步未来、0.5h分辨率、原14通道接口和原始ΔV目标。预测起点位于所属分区，未来24个目标全部位于对应分区；历史可读取起点前已经发生的更早分区观测。无效分箱、segment变化、运行时间断点和非有限值会使窗口被排除，排除原因完整保存。

分区与每个48步历史内部的时序不打乱。训练mini-batch仅在Train合法窗口内使用固定种子的重排，与原F2训练方式一致；Top10/Top11对应种子的初始化和共同Epoch内的采样顺序均核验相同。Validation与后留预测按原窗口顺序执行。

标准化只拟合Train的原始合法历史窗口；只对历史电压做每窗口因果EWMA alpha=0.3，再应用同一个Train scaler和特征掩码。标签与恢复绝对电压所用起点电压都不进行EWMA。

该后20%区间已受历史研究反馈影响，属于新的探索性时间后留评价，不能称为从未接触的独立盲测集。新协议结果与旧协议分别保存，不放入同条件比较表。

## Top10/Top11与固定训练配置

Top10固定为`[0,3,13,11,4,9,6,7,10,8]`，Top11只追加平均电流索引5。14候选特征的Spearman排名只使用Train，作为本次检查和展示，不依据Validation或后留成绩重新选择掩码。

实际未选的标准化通道置零，不只是改名称。两组的其他13个输入通道完全相同，网络结构和参数量均为原始F2的332436个可训练参数。

| 项目 | 冻结值 |
|---|---|
| 学习率 | 0.0009605449687795866 |
| Dropout | 0.1 |
| weight_decay | 0.00000695942232671908 |
| batch_size | 32 |
| hidden_size / hidden_continuous_size | 64 / 32 |
| 优化器 / 损失 | AdamW / 标准化ΔV的MSE |
| gradient clipping | 1.0 |
| 最大Epoch / Early Stopping patience | 450 / 40 |

配置继承原F2训练设置，不运行新Optuna，也不显示历史Fold搜索图。Validation全部5232个查询的标准化RMSE用于选取最佳Epoch，耐心计数连续无改善的Epoch；每组可自然早停，不能把“完整训练”误写成每组都必须跑满450轮。

当前一次Run All请求的全部模型训练完成并固定最佳权重后，才计算后留指标。all的六种子结果作各自RMSE、MAE、R²的等权平均，不加权、不集成、不挑最优种子。六个种子来自历史筛选名单，新平均值也不能当作随机初始化总体的无偏表现。

## 实时显示及输出

训练时逐Epoch打印本次Train Loss、Validation Loss、最佳Epoch和耐心计数。损失图在首轮、每10轮和训练结束时刷新，图像由实际IPython显示/更新消息发出，并保存300dpi PNG。图不是旧日志回放。

每次输出到本文件夹下 `F2_Training_Outputs/<UTC时间>_seed<种子>_<特征模式>/`，包括：

- `PROTOCOL_BEFORE_TRAINING.json`：首次更新前冻结的配置、源数据/代码SHA、边界和描述性分组规则。
- `PARTITIONS_AND_WINDOWS.csv`、`windows/`、`EXCLUDED_WINDOWS.csv`：完整数据身份与合法窗口。
- `PREPROCESSOR.json`、Train scaler、Spearman表和掩码实际应用审计。
- `seed*/top*/best_model.pt`：只来自本次训练的最佳权重，含本次run_id、最佳Epoch、步数和初始化身份。
- `training_history.csv`、`TRAINING_LEDGER.json`：真实反向传播、优化器更新、梯度和权重变化证据。
- `training_history.png`、`prediction_vs_true.png`、`scatter_true_vs_pred.png`、`scatter_colored_by_load.png`及对应绘图CSV。
- `predictions_TRAIN/VALIDATION/TEST.csv`、逐种子三分区指标、后留均值和`TOP11_MINUS_TOP10.csv`。

默认展示后留评价。三项指标保持原始ΔV定义：RMSE=sqrt(mean(error²))、MAE=mean(abs(error))、R²=1−SSE/SST，单位V。不是将绝对电压散点的R²冒充ΔV的R²。

时间曲线固定取每个合法起点的第24步，断点断线，不拼接重叠预测。散点使用全部12648个查询，不删除极端点、不人为修改预测值。原始起点电压加预测ΔV恢复绝对电压。

着色采用目标时刻真实电流密度，仅用于事后诊断，绝不作为未来decoder输入。四个电压分组仅由Train真实电压固定的4组描述性聚类定义，不是新的预测模型，不证明恰好四种物理状态。分组负载统计按619个唯一后留目标时间计算，避免把重叠查询当作独立事件；同时记录零电流占比，区分负载混合与模型预测错误。

## 本轮验收与旧文件

`fresh_training_acceptance/`保存Seed202两组完整训练通过后，再启动all十二组的实际执行证据、独立checkpoint重放和旧文件保护记录。已执行Notebook只用于查看验收输出；若重新运行其单元，也会重新训练，并不是历史回放入口。

科学合同测试：`python -B -m unittest discover -s tests -p test_fresh_training_contract.py`。正式执行记录不覆盖；汇总脚本`tests/summarize_fresh_training.py`仅在完整验收PASS后生成真实结果报告。

旧主程序、旧交互脚本及Notebook、原Seed Assets、旧测试、权重、预测和历史结果全部保留。暂不删除、不覆盖、不移动。后续仅建议将旧入口标记为“历史冻结回放”；如要移动归档，须另列具体文件和依赖清单并取得用户确认。
