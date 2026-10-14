# P1S预测到参考SOH的接口

本实验预测raw P1S健康代理。无周期级真实极化SOH目标，所以不报告真实SOH预测MAE，也不把retention直接改名SOH。

提供causal_soh_bridge接口：输入模型预测的未来R及至少3个当时已可知、原冻结映射的参考锚点；实际result_release_datetime必须有证据且≤origin_datetime，才按原AHC非负affine数学形成a*Rhat+b。未知发布时间或不足锚点返回不可用；不使用全六锚点后验B2，不用未来真实R。

目前已有资料没有精确参考发布时间，因此本轮实际auxiliary status全为NO_CERTIFIED_ANCHOR_RELEASE_TIME，SOH数字、真实目标、MAE均留空。接口合成测试验证数学和可获得性检查，不能冒充真实SOH成功。它不参加M0/M96/M192精度排名。

以后“短期代理→参考SOH→97%条件阈值”必须逐项验证时间、映射和参考测量。此轮不执行97%外推，不重跑250～350h实验，不定义真实EOL。
