# 实现、测试和信息边界

中断前已完成14项测试，`outputs/SYNTHETIC_TEST_RESULTS.json`记录2026-10-14 01:11（北京时间），returncode=0、无真实数据加载。恢复后保留原记录，没有重复运行或覆盖。冻结manifest包含这些原脚本和测试SHA。

|要求|实际证据|结论|
|---|---|---|
|前向模型与G2一致|直接调用原gate_ablation_models的G2，无新网络wrapper；原类合成前向及状态不变检查|通过|
|lambda=0等于MSE|数值与梯度逐元素相同；真实G2参数梯度逐项比较|通过|
|单位恢复正确|使用raw起点加s×标准化DeltaV，测试排除输入mean|通过|
|区间内不惩罚|合成区间内部loss和梯度都为0|通过|
|越界梯度|人工上下越界均有限、非零、方向正确|通过，仅合成|
|真实Train非零作用|3 seed×3角色、每次33120预测值|失败，全部为0|
|未来输入隔离|predict_delta仅历史输入；decoder前14变量为0；推理未增加未来工况|通过|
|参数/输出身份|332436可训练参数，[B,24]输出|通过|

真实资格检查复现1380合法Train窗口、冻结Top10 mask、历史电压EWMA(0.3)、原scaler；与P3的原始/预处理x、y、起点、目标行、scaler及Development对象指纹逐一一致。三个种子初始化权重、训练初始RNG和第一轮采样顺序匹配。六个checkpoint文件SHA、模型/seed/协议元数据核验通过。

仅执行初始化Train模式带dropout前向，以及best/last的Train eval前向；没有optimizer、反向更新、新Validation推理。约17秒使用GPU作只读前向审计，不能称GPU正式训练。旧最佳checkpoint历史上由Validation选出，本轮不利用其成绩寻找物理公式；本次全Train的零活性证据已足以停止。

## 读写边界

数据reader严格`nrows=3007`后过滤`bin_end_ophrs2_h < 1819.997738`得到3005行，再先切Train `[0,2103)`才构造窗口。允许的有界Development对象读取包含Validation段，但没有为资格判断构造或评价Validation窗口。没有读取全混合文件计算SHA。

生产脚本使用审计hook限制仓库路径、禁止历史写入、禁止N1全量/3005替代读取、禁止其它历史CSV；合成防线测试覆盖拒绝、指纹后缀不变/合法前缀变化、源哈希篡改拒绝。hook是本进程工程防线，不是整个操作系统的安全沙箱；报告读取另有来源清单。

完整记录：[Train身份](outputs/TRAIN_DATA_AUTHORITY.json)、[运行访问日志](outputs/RUNTIME_DATA_ACCESS_AUDIT.json)、[冻结来源](outputs/SOURCE_AND_CODE_MANIFEST.json)、[控制台](outputs/QUALIFICATION_CONSOLE_20261014.log)。原历史数据、模型、协议和checkpoint未修改。

## 字节码缓存拒绝与交付验证修复

运行日志保留9次历史__pycache__/*.pyc读取拒绝；Python随后读取已allowlist并核验SHA的.py源文件，无历史写入。交付验证器最初把任何DENIED都视为失败，人工检查确认均为上述缓存后，仅对此类已认证源对应的只读.pyc拒绝设审计例外；未改生产防线或重跑资格审计。
