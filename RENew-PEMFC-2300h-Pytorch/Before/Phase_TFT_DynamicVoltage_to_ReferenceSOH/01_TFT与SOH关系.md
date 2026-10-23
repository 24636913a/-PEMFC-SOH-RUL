# 输出和参考定义
 原F2用过去24h、48×14槽位，经固定Top10置零及历史EWMA，预测未来24个0.5h窗的ΔV。恢复动态电压使用未滤波起点电压 Vhat=Vorigin+ΔVhat，缓存ΔV单位已核验为V，不能重复乘尺度。
 参考SOH定义 Vpol(j*=1.0)/0.6989533305168152，P1S定义 VP1S/0.71383023635。两种分母及测量条件不同。禁止直接动态V除极化BoL。
 低频动态电压变化也混有负载、温湿压、恢复和边界效应。K/HA保留FORECAST_ONLY_REPRESENTATION、HEALTH_EVIDENCE_NOT_IDENTIFIABLE：共同工况支持及年龄效应可辨识未成立。这项负面证据没有改写。
 已有96h健康门控HEALTH_CONTEXT_NO_CLEAR_GAIN保持；本轮没有重启门控、改变双路径或生成连续SOH标签。
