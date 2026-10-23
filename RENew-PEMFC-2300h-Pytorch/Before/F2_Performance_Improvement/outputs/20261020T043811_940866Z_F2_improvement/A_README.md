# 阶段A导航

先读01_阶段A_训练配置研究.md；完整数值在tables/A_VALIDATION_PER_SEED.csv及A_VALIDATION_MEAN_SD.csv，固定判据与结果在PROTOCOL.json及A_RESULT.json。

figures/A：01验证集三指标、02九次真实训练曲线、03门控、04路径尺度、05训练梯度、06推理反事实、07逐步误差、08固定Step24绝对电压、09全部Validation查询散点、10门控分布。全部PNG300dpi/PDF/绘图CSV。

models/A每个实例独立保存本次最佳checkpoint、日志、Train/Validation预测及门控路径、验证反事实、身份审计。historical只读诊断均为Train/Validation；没有读取旧后留预测文件。
