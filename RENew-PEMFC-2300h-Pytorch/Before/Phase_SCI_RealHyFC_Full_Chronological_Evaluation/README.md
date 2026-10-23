# RealHyFC完整时间序列时间分区预测评价

基准提交：`c54ed2bb3343b8e090565d8f0dab5cfa0950c9e3`。完整文件实际为3843条、214个字段。15个冻结checkpoint，五模型各42/52/62，未训练、未选种子、未调参。原始来源与代码身份见configs及outputs/audits。

这是新授权的探索性全时域评价。Train为样本内回放，Validation已参与训练早停与历史研究；历史Assessment和后段均有历史反馈影响。任何跨区间汇总不具备独立Test身份。Phase L的`PHASE_L_CONFIRMATORY_TEST_NOT_ELIGIBLE`保持不变。

## 结果入口

- tables/TIME_PARTITIONS.csv、WINDOW_COUNTS.csv：完整分区和有效数量。
- tables/FIVE_MODEL_PARTITION_RMSE.csv：五模型四区间主结果。
- tables/FIVE_MODEL_ENDPOINT_RMSE.csv：12小时终点。
- outputs/metrics/PER_MODEL_SEED_PARTITION.csv：60组逐种子正式描述指标。
- outputs/predictions/：60个完整逐点CSV，原始ΔV/绝对电压和双时间身份。
- outputs/audits/：完整数据结构、合法/排除窗口、15个checkpoint清单、独立复算、历史保护和产物SHA。
- figures/：7套300dpi PNG/PDF及绘图CSV。
- reports/：六份中文结果与限制。

## 可重复运行

环境沿用H：Python3.11.9、torch2.6.0+cu124、pytorch_forecasting1.8.0、CUDA GPU。仓库内历史源码和checkpoint只读。项目根目录下依次执行：

```powershell
$env:PYTHONUTF8="1"
$env:PYTHONDONTWRITEBYTECODE="1"
$pipeline="RENew-PEMFC-2300h-Pytorch/Phase_SCI_RealHyFC_Full_Chronological_Evaluation/scripts/run_pipeline.py"
# 首次执行顺序（prepare/freeze已完成会拒绝覆盖，不重新冻结）：
python -B $pipeline prepare
python -B $pipeline preflight
python -B $pipeline validation
python -B $pipeline freeze
python -B $pipeline infer
python -B $pipeline audit
python -B $pipeline figures
python -B $pipeline reports
python -B "RENew-PEMFC-2300h-Pytorch/Phase_SCI_RealHyFC_Full_Chronological_Evaluation/scripts/polish_figure_layout.py"
python -B $pipeline finalize
```

现有成果核验：运行audit、figures、reports、finalize；infer会核验已有完成receipt并跳过，无需重做GPU推理。Validation阶段是只读权重推理复现而不是训练，不调用历史一次性评价入口。源码、配置、数据或checkpoint身份改变会报错并留存FAILURE日志，不能悄悄改协议。若更换根路径，保持历史各Phase相对布局即可；SHA不依赖根目录。

## 四区间平均RMSE（mV）

| model | TRAIN | VALIDATION | HISTORICAL_ASSESSMENT | LATE_LIFE |
| --- | --- | --- | --- | --- |
| F0 | 9.324077 | 9.067563 | 21.745493 | 43.924052 |
| F1 | 8.506738 | 9.155598 | 22.892878 | 45.556509 |
| F2 | 6.078403 | 7.245400 | 12.197052 | 48.720862 |
| LSTM | 2.817140 | 3.507430 | 7.921892 | 35.530573 |
| TCN | 26.927149 | 19.989031 | 35.811418 | 40.618992 |

## 论文限制

这是冻结模型的探索性全时域评价，包含样本内Train、复用Validation以及已受历史研究反馈影响的Assessment/后段。不得将全时域或后段RMSE写成独立Test RMSE。

图形目视QA发现原图01标注重叠、图07图例遮挡低电压，独立polish_figure_layout.py仅调整排版。原冻结源码及协议未改，绘图CSV SHA完全不变。重绘时按上述顺序在reports后执行排版脚本；事后极端工况诊断可单独运行diagnose_extreme_errors.py。
