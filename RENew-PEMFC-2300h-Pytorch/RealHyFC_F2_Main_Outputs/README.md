# F2单文件主程序整合验收

唯一新增Python源码为研究根目录的`RealHyFC_F2_Main.py`，按指定13节排列。基准main为`a42eb782ceeca98320d1da322d4c75cf8746637e`。本目录仅保存新主程序的输出、图片与验收证据，没有新增模型结构、模型搜索或对比模型程序。

## 运行

在`RENew-PEMFC-2300h-Pytorch`目录执行：

```powershell
python -B RealHyFC_F2_Main.py --mode evaluate --seed 42
python -B RealHyFC_F2_Main.py --mode evaluate --seed all
python -B RealHyFC_F2_Main.py --mode train --seed 42
```

默认`evaluate`、Seed42、CUDA。`all`只用于评价42/52/62，不自动启动多个训练。训练模式从头初始化新实例，使用独立UTC时间戳目录和`NEW_best.pt`，不覆盖历史checkpoint。运行环境严格复用历史H：Python3.11.9、torch2.6.0+cu124、pytorch_forecasting1.8.0、同一CUDA/cuDNN及GPU，环境不一致会报错，不以宽松容差冒充逐点复现。

本次实际运行`--mode evaluate --seed all`。没有运行训练命令或Optuna。新增训练函数已完成数据、MSE、AdamW、梯度裁剪、Validation early stopping、checkpoint保存及重新加载的完整实现；只做静态调用链验收，没有宣称新训练的动态测试成绩。

## 结构及参数来源

原始CanonicalStrictHistoryTFT、私有左填充卷积、CausalMultiScaleHistoryBranch、DualPathStrictHistoryTFT、RegimeFusionGate和RegimeAwareDualPathTFT均完整嵌入第5节，六个类AST与原源码严格相同。继续使用第三方pytorch-forecasting的原生VSN、GRN、LSTM encoder/decoder、Attention及输出头，门控位置不变。用户入口只构建F2，参数量332436。

第6节读取`Phase_TFT_Hyperparameter_Optimization_Restart/outputs/T1R_SELECTED_CONFIGURATIONS.json`中的Fold 2、trial13：

| 参数 | 冻结值 |
|---|---:|
| learning_rate | 0.0009605449687795866 |
| weight_decay | 6.95942232671908e-06 |
| dropout | 0.1 |
| hidden_size | 64 |
| hidden_continuous_size | 32 |
| batch_size | 32 |

这些是历史基础TFT的Optuna结果，由F2继承使用，不是F2专属搜索。最大450 epoch、patience40、clip1、AdamW/MSE从历史H冻结协议单独读取并验证；历史Optuna的最佳epoch不替代正式early-stopping协议。Assessment及后段不参与参数选择。

主程序不导入、exec或动态加载任何旧项目Python文件。需要读取的只包括数据、JSON/YAML、NPZ、CSV和checkpoint等只读资产。原模型源码路径及SHA只作为嵌入的来源记录，不在运行时读取旧Python文件。

## 实际复现

最终源码结果目录：`20261017T151243_601888Z_evaluate_seedall`。初次软件核验目录`20261017T150141_719559Z_evaluate_seedall`亦保留；提交前只去除了主文件末尾多余空行，两版整个模块AST相同，12组预测CSV字节完全一致。详见SOURCE_FORMAT_ONLY_AUDIT.json；这两次软件核验不是独立随机种子实验。

| Seed | Validation RMSE（mV） | 起点数 | 目标数 | 与历史预测差异 |
|---|---:|---:|---:|---:|
| 42 | 6.265169 | 205 | 4920 | 0 V，逐点严格相等 |
| 52 | 7.034632 | 205 | 4920 | 0 V，逐点严格相等 |
| 62 | 8.436399 | 205 | 4920 | 0 V，逐点严格相等 |

完整3843行的Train/Validation/Assessment/后段分别1380/205/378/548个合法起点，全部窗口、原始输入、标签、预处理和四分区预测与冻结全时域结果严格一致。Train-only原始窗口scaler、历史Train Spearman排名及固定Top10、α=0.3因果EWMA均在文件内完整实现。未来decoder仅用确定的相对时间，绝对电压用未经EWMA的起点电压恢复。

独立隔离及历史保护报告位于保留的初次核验目录；最终目录具有当前源码的完整三种子Evaluate验收。

## 验收证据

- `INTEGRATION_ACCEPTANCE.json`：三个模型完整加载、Validation及四区间逐点一致、状态未变、训练未执行。
- `INDEPENDENT_SINGLE_FILE_TEST.json`：隔离副本只含单个Python主文件，项目Python模块导入数0；原checkpoint固定批次前向严格相等；优化器构造及backward拦截计数均0。36个指标独立复核最大数值差异7.105427357601002e-15，仅为求和精度差异。
- `FROZEN_RUN_CONFIGURATION.json`：源码、来源参数、环境、scaler及模型类来源SHA。
- `F2_PARTITION_METRICS.csv`、各seed的`predictions/`：12组完整分区预测和RMSE、MAE、R²。R²口径为ΔV，与历史H相同。
- 各seed的`figures/`：历史训练/验证MSE曲线、四区间固定起点电压图和单个24步预测曲线，300dpi PNG、PDF及绘图CSV。训练曲线明确来自历史日志，本轮没有新训练。
- `HISTORICAL_PROTECTION_AUDIT.json`：历史文件、源码、权重及输出保护核验。

所有曲线都是单个预测起点的24步输出，不拼接重叠预测窗口。保留全体不利样本和后段负面结果。Assessment及后段始终是探索性评价，不恢复Phase L的独立Test资格，不新增PINN、SOH或RUL模块。
