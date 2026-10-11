# 预测表征不等于健康表征

保留E19/E20的正式状态：
`FORECAST_ONLY_REPRESENTATION`
`HEALTH_EVIDENCE_NOT_IDENTIFIABLE`。

`PROJECT_VERIFIED_EVIDENCE`：HA基于1992个Development cache origins；没有positive-current stratum通过joint support和1:1matching。24h/48h condition-adjusted point estimates分别约−0.006207/−0.007029 V/1000h，但设计矩阵condition number无限，moving residual bootstrap成功率0%、CI不可估，因此不能作为可识别年龄关联证据。lag1 revision P95≈106.8mV是回顾性边界修订，不是online roughness；matched recovery0不能用来证明没有恢复。

EMD缓存端点已恢复到V、slope为V/h，单位正确并不授予健康含义。低频包含负载/温湿度/压力变化、停机恢复、采样边界和算法端点效应；平滑或负斜率本身不能分离不可逆损伤。严格因果输入只是防未来信息泄漏，不保证工况调整的可辨识性。

Phase DF的`EMD_SPECIFIC_ADVANCE`回答预测输入有没有价值，HA回答endpoint有没有HI资格；两者不同，不修改DF、不撤销其已冻结结果。没有匹配支持的失败应解释为**健康证据不可辨识**，不是“EMD完全没有任何退化信息”。

`RESEARCH_HYPOTHESIS`：reference-condition映射、足够condition overlap及独立锚点可能提供健康证据；尚未执行、不预设结果。
`INSUFFICIENT_EVIDENCE`：不可逆健康状态、退化电化学机制、SOH、EOL、RUL。不得重搜EMD、以强制单调趋势补缺证据，或将HA迁移为F2健康资格。
