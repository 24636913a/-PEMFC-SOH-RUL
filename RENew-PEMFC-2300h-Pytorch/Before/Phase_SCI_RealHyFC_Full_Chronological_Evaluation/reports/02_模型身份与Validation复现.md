# RealHyFC完整时间序列时间分区预测评价

基准提交：`c54ed2bb3343b8e090565d8f0dab5cfa0950c9e3`。完整文件实际为3843条、214个字段。15个冻结checkpoint，五模型各42/52/62，未训练、未选种子、未调参。原始来源与代码身份见configs及outputs/audits。

这是新授权的探索性全时域评价。Train为样本内回放，Validation已参与训练早停与历史研究；历史Assessment和后段均有历史反馈影响。任何跨区间汇总不具备独立Test身份。Phase L的`PHASE_L_CONFIRMATORY_TEST_NOT_ELIGIBLE`保持不变。

## 身份核验

CHECKPOINT_INVENTORY.csv逐一与历史正式checkpoint SHA核对，九个H模型和六个最小公平基线。未使用六个经Validation筛选的优秀F2种子。严格加载state_dict、参数数目及历史协议SHA。

15个模型全部205个Validation起点、4920个ΔV预测与历史CSV逐点严格相等，最大差异0 V，不使用浮点容差掩盖身份问题。基线checkpoint中的完整标准化预测也一致。权重及BatchNorm状态在前向推理前后SHA不变。

先冻结窗口，再执行工程测试及Validation复现，随后冻结完整代码、输入、模型、指标和案例选择规则，才执行Train/Assessment/后段推理。GPU eval/no_grad、batch32沿用历史正常运行环境。

14槽位、固定Top10、窗口内电压因果EWMA α=0.3；原Train-only scaler从冻结JSON读取，没有用后段重新拟合。原始ΔV输出仅恢复一次尺度，再加未经EWMA的起点电压。R²基于ΔV，与历史H相同；不将其混称绝对电压R²。TCN在已知历史内部使用对称padding，此事实不变。
