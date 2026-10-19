# 真实LSTM与完整F2对照

以六种子等权平均RMSE衡量，本轮普通LSTM更低。F2减LSTM的平均RMSE差为19.335064 mV（相对LSTM为+77.70%）。F2的RMSE在0/6种子更低，MAE在0/6种子更低。MAE和R²按各自结果解释，不能只用一个有利指标概括全部性能。

| Seed | LSTM RMSE(mV) | F2 RMSE(mV) | LSTM MAE(mV) | F2 MAE(mV) | LSTM R²(ΔV) | F2 R²(ΔV) |
|---|---:|---:|---:|---:|---:|---:|
| 42 | 25.060866 | 41.455588 | 5.804616 | 11.224501 | 0.983298812 | 0.954299456 |
| 103 | 24.926502 | 44.309040 | 5.547002 | 11.600653 | 0.983477419 | 0.947791659 |
| 193 | 24.767142 | 48.398668 | 4.810536 | 13.547583 | 0.983688007 | 0.937709470 |
| 202 | 24.594026 | 43.510554 | 5.005282 | 7.994754 | 0.983915244 | 0.949656380 |
| 210 | 25.301401 | 46.263481 | 6.686179 | 16.154000 | 0.982976676 | 0.943084332 |
| 212 | 24.662724 | 41.385717 | 4.963313 | 14.295771 | 0.983825260 | 0.954453377 |

| 模型 | RMSE均值±样本SD(mV) | MAE均值±样本SD(mV) | R²均值±样本SD |
|---|---:|---:|---:|
| F2 | 44.220508 ± 2.750398 | 12.469543 ± 2.842581 | 0.947832446 ± 0.006542271 |
| LSTM | 24.885444 ± 0.266084 | 5.469488 ± 0.707753 | 0.983530237 ± 0.000352969 |

均值是6个逐种子指标的等权平均，不是预测集成或拼接预测后的RMSE；样本SD使用ddof=1。配对差值全部列于tables/PAIRED_DIFFERENCES.csv，符号为F2减LSTM。未做12648次独立实验假设下的统计检验。

## 执行与工程验收

先完整训练Seed202两模型，pilot仅用Validation检查真实更新、样本身份和新checkpoint重放；随后12个正式实例全部重新初始化，没有复用pilot权重。共14次新训练：2次pilot+12次正式，pilot不混入正式均值。全部正式权重冻结后才计算本轮新后留成绩；P1历史缓存只用于之前的误差诊断。

本轮自然产生了Seed202的pilot与formal两次独立训练，可用其Validation结果检查重复波动（没有给pilot计算后留成绩，也不从两次中择优）：

```text
model cohort  best_epoch  stop_epoch  RMSE_mV   MAE_mV       R2
 LSTM  pilot         350         390 6.053422 3.772425 0.999050
 LSTM formal         350         390 6.053422 3.772425 0.999050
   F2  pilot         159         199 7.592943 4.657728 0.998506
   F2 formal         269         309 4.736469 3.290087 0.999418
```

逐项首次分歧轮次（NA表示共同Epoch内未观察到差异）：

```text
model  seed  common_epochs  initialization_equal  first_optimizer_update_equal  first_different_epoch_train_standardized_MSE  first_different_epoch_validation_standardized_MSE  first_different_epoch_gradient_norm_min  first_different_epoch_gradient_norm_max  first_different_epoch_sampler_order_sha256  first_different_epoch_state_sha256  first_different_epoch_rng
 LSTM   202            390                  True                          True                                           NaN                                                NaN                                      NaN                                      NaN                                         NaN                                 NaN                        NaN
   F2   202            199                  True                          True                                           6.0                                                6.0                                      7.0                                      6.0                                         NaN                                 6.0                        NaN
```

F2 Seed202的两次新训练初始化相同，从第6个Epoch末首次观察到权重分歧；pilot/formal最佳Epoch为159/269，Validation RMSE为7.592943/4.736469 mV。采样顺序和RNG状态是否相同详见轨迹表。不能把一次更好结果替换进正式均值，也不能仅凭此锁定唯一CUDA算子。

该两次记录仅是工程诊断，不是预先设计的多次重复总体性能估计。相同模型/种子的初始化SHA及采样顺序另行严格核对；普通CUDA训练仍不承诺逐位一致。

所有模型共用同一组1750/218/527合法窗口、原始ΔV标签、Train-only scaler及固定Top10掩码。相同种子的共同Epoch采样顺序逐项一致，48步历史和24个目标全部严格对齐。原14通道中未选的4通道真实置零。

完整分区均值（mV；R²目标为原始ΔV）：

```text
                    RMSE_mV     MAE_mV        R2
model partition                                 
F2    TEST        44.220508  12.469543  0.947832
      TRAIN       15.196772   7.537515  0.992154
      VALIDATION  18.100564   8.730057  0.990013
LSTM  TEST        24.885444   5.469488  0.983530
      TRAIN        4.059753   2.859110  0.999544
      VALIDATION   6.947833   4.405408  0.998706
```

尾部与辅助口径（逐种子后再平均）：

```text
          P95_mV      P99_mV  max_abs_mV  top1pct_SSE_fraction  P1_two_event_four_targets_SSE_fraction  target_time_equal_RMSE_mV  fixed_step24_RMSE_mV
model                                                                                                                                                  
F2     23.570900  193.284997  546.226151              0.878472                                0.510296                  43.873558             43.314117
LSTM    9.722927   31.425184  319.784551              0.964704                                0.003025                  24.687776             27.292527
```

## 最差查询所在完整窗口的时序诊断

图12同时展示Seed202两模型各自最差查询所在的24步窗口，并在每个相同起点叠加两个模型。图09的固定Step24曲线用于工况背景，不能代替全部步长的最大误差诊断。以下是各模型最差查询附近±1步内，预测局部最小值相对真实局部最小值的位置差；负号表示提前，正号表示延后：

```text
model  seed  origin_row  worst_target_row  true_local_min_step  predicted_local_min_step  predicted_minus_true_min_lead_h
 LSTM   202        3100              3119                   19                        20                              0.5
   F2   202        3104              3112                    9                         8                             -0.5
```

这些例子可见半小时状态低谷误判。输入、标签与时间身份已严格核验，因此不能把它们直接说成CSV时序错位；仅两个事后案例也不足以宣称所有预测都有系统性偏移。没有按真实曲线移动预测、修正输出或重新计算较好指标。完整逐点绘图数据见figures/P3/12_paired_worst_query_windows_data.csv。

## 科学主张边界

LSTM是单层hidden64、最后隐藏状态接linear24，没有Attention/Transformer/TCN/外部Dropout；单层内部dropout=0。F2保留完整TFT、LSTM/CNN双路径与历史工况门控，原Dropout=.1。两者参数22040与332436，容量约差15.08倍。相同训练政策不等于控制了容量、正则化、输入归纳偏置或各模型专属调参预算。

本轮只比较完整F2与普通LSTM的总体表现，不能把差值全部归因于双路径或门控。F2沿用历史基础TFT选出的配置，LSTM使用同一配置，不宣称已完成LSTM专属最优超参搜索。六个种子来自历史F2筛选名单，当前平均不代表无偏随机初始化总体。

普通CUDA配置保留，同种子逐位重复性不保证；不能因一次复跑更好就替换正式实例。最后20%已受历史研究反馈影响，所有后留成绩均为探索性时间后留结果，不能写成独立盲测、外部泛化验证或健康/RUL验证。

每次训练均保留BEFORE_TRAINING、逐Epoch训练日志、最佳checkpoint、初始/首步/末步权重SHA、更新数与随机状态摘要。训练曲线是本次真实日志；Train显示批次平均MSE，Validation显示全查询MSE，正式Train/Validation分区指标另按同一全量口径计算。
