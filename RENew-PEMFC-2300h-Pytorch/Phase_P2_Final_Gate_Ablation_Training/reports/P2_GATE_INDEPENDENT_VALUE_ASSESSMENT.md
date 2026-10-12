# 门控独立价值审查

主对比G2−G1，辅助G2−G0。

本次未证明历史工况门控相对全局学习比例和固定融合具有达到全部预设门槛的稳定独立增益；不得以平均或个别Seed优势升级该主张。

```json
{
  "G0": {
    "mean_RMSE_gain_mV": -0.7563862244069472,
    "required_gain_mV": 0.5,
    "mean_gain_PASS": false,
    "paired_RMSE_wins": 1,
    "endpoint_each_seed_PASS": false,
    "target_time_equal_each_seed_PASS": false,
    "mean_MAE_PASS": false,
    "mean_P99_PASS": true
  },
  "G1": {
    "mean_RMSE_gain_mV": -0.7934980935982878,
    "required_gain_mV": 0.5,
    "mean_gain_PASS": false,
    "paired_RMSE_wins": 1,
    "endpoint_each_seed_PASS": false,
    "target_time_equal_each_seed_PASS": false,
    "mean_MAE_PASS": true,
    "mean_P99_PASS": true
  }
}
```

不能把G1/G2相同57参数解释为相同函数有效自由度。G1全局可学习比例，G2历史条件依赖；本次是受控经验比较，不是某工况变量电化学因果识别。
