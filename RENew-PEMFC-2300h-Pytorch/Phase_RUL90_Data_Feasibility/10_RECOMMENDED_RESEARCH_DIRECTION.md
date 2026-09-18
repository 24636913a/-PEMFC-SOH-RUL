# 推荐研究方向

唯一推荐主任务：**固定名义工况下的稀疏 reference-condition SOH estimation 与长期健康电压退化预测**，并把 0.5 h / 12 h TimeXer 作为动态电压支路或与慢 health-state 相联系的辅助支路。

不推荐将现有 12 h `DeltaV` 直接循环外推生成数千小时 RUL，也不推荐将本数据写成已具有真实 90% EOL 监督标签的 RUL 数据集。TimeXer 可以继续使用，但预测目标应从单独的未来动态 `DeltaV` 扩展为：过去运行工况 → 当前/未来合法 `V_ref` 或 sparse SOH anchor，并严格区分未来工况未知与 protocol-known scenario。核心创新应是可验证的 fast dynamic response 与 slow reference-condition health state 的因果对齐，而不是堆叠新的网络模块。
