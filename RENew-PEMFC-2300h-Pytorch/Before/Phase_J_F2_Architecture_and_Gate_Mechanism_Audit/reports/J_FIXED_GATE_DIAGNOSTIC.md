# 唯一固定门控反事实诊断

Base: `427d4dc402f083a5b9e094b12981c5e231df8100`. Protocol SHA256: `0244f05fd777ccd615fda7eb9533d129323dc27fbb58212e091130011d0d9b24`. Code identity SHA256: `67b0d3588f75db56d435ddc5277d38324708f660ac10cc1850a5908d771612c8`.

Validation only, 205 origins/seed, 615 gates; all 4920 queries per seed retained. New training/Assessment inference/Test/seed selection: NO.

|seed|learned_RMSE_mV|fixed0p5_RMSE_mV|learned_minus_fixed_RMSE_mV|prediction_difference_RMSE_mV|mean_absolute_prediction_difference_mV|P95_absolute_prediction_difference_mV|max_absolute_prediction_difference_mV|different_prediction_fraction|
|---|---|---|---|---|---|---|---|---|
|42|6.2651692|6.25299001|0.0121791857|0.376512564|0.169763545|0.766920391|5.35704917|0.999390244|
|52|7.03463242|7.03779817|-0.00316575814|0.0618864724|0.0233735755|0.116996467|1.1193566|0.878658537|
|62|8.43639894|8.45146908|-0.0150701424|0.444471401|0.227474875|0.926875323|4.39702906|0.999390244|

差值固定为 learned−fixed，负值表示当前checkpoint学习门控的RMSE更低。预测差异统计衡量数值依赖。全部指标由持久化CSV重新读取，并用独立标量计算核验（48项，最大差异3.55e-15）。原H完整Validation指标保持，未删除目标、重加权选择模型或挑Seed。

仅临时forward输出hook返回 `torch.full_like(out,0.5)`，finally移除；三个模型恢复后的全205预测与正常缓存np.array_equal。没有改变文件权重、state_dict、buffer、requires_grad或梯度。每项操作及独立进程PID见execution/state审计。

三Seed等权均值：学习门控7.245400185 mV，固定门控7.247419090 mV，learned−fixed约−0.002018905 mV。Seed42学习门控略差，Seed52/62略好；方向混合、幅度很小，不能据此声称一致预测精度增益。预测值差异则在三个Seed均存在，支持数值依赖。没有新增显著性检验。

固定.5的F2不是独立训练F1。其余参数已与学习门控共同训练，推理时替换改变训练条件一致性。因此结果最多支持当前checkpoint对门控的局部预测依赖，不是严格控制训练过程的结构消融，更不是物理因果机制识别。
