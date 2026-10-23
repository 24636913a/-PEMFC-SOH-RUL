# 辅助目标泄漏防线

CSHSE事故说明provenance操作也属于访问范围：即使不参与模型，混合文件whole hash仍越界。本轮明确只读F2 Train2103记录，hash仅静态白名单，限定source fingerprint与whole-source SHA不同。

辅助目标隔离必须覆盖raw cells、exported average、mean/min/max/std/range、voltage-derived统计、EWMA后继、邻接平滑与重复缓存。生产no_voltage_view用显式9非电压列而非黑名单；真实Train电压列与synthetic EWMA/raw/exported值扰动均不改变view。来源依赖皆为I/j/operation_point/count，不以V构造。

时间/源范围仍需独立恢复实际label dwell并证明窗口先于/不含label测量；本轮没有恢复raw身份，完整实际隔离资格未通过。不得根据低loss推断隔离成功。未来control采用同view，主F2不改。本阶段不创建任何auxiliary model。
