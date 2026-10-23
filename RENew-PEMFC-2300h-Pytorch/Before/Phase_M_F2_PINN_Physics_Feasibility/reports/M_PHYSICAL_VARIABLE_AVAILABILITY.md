# 物理变量可用性：14 槽位不等于全部 N1 变量

依据 H 冻结协议、E 预处理、N1 字段字典/availability、N1 构建代码和配置。没有读取 N1 时序表。下表的“当前可用”指 bin 完成后历史值按字段契约可用；不是重新证明传感器质量。未来实际值全部未知。Top10 外槽位在标准化后置零，不能通过 inverse scaler 恢复被屏蔽的真实值。

|序号|Raw F2 槽位|来源、单位、bin 聚合|Top10|物理用途及限制|
|---:|---|---|---|---|
|0|cell_voltage_mean|有效十单体 U1–U11 排除 U6；V；逐 raw 时刻跨 cell 均值再时间均值|是|预测目标；EWMA 后编码值不是未滤波物理观测。|
|1|cell_voltage_std|同上；V；逐时刻跨 cell population std 的时间均值|否|空间离散度，不是 bin 内时间标准差。|
|2|cell_voltage_range|同上；V；逐时刻跨 cell max−min 的时间均值|否|不等于 bin 全时段峰峰值。|
|3|cell_voltage_min|同上；V；bin 内单体最小值的最小值|是|极端单体，不是参考条件平均 cell 电压。|
|4|cell_voltage_max|同上；V；bin 内单体最大值的最大值|是|同上；不能代入平均 cell 极化方程。|
|5|actual_current_A_mean|I.S.act [A]；A；时间均值|否|源库有，冻结 F2 实际不传入；不可假称平均 I 与平均 j 同时进入网络。|
|6|actual_current_A_std|I.S.act；A；时间 population std，ddof=0|是|负载波动；不是电化学电阻。|
|7|actual_current_A_peak_to_peak|I.S.act；A；max−min|是|极值不能由均值/方差推导。|
|8|actual_current_density_A_cm2_mean|j.S.act [A/cm2]；A/cm²；时间均值|是|历史 j；未来 j 不已知。源字段以 N1 静态定义为准。|
|9|actual_current_density_A_cm2_std|同上；A/cm²；population std|是|同尺度历史波动，不能还原瞬时 j 分布。|
|10|actual_current_density_A_cm2_peak_to_peak|同上；A/cm²；max−min|是|不足以定义稳态 dwell。|
|11|load_transition_count|相邻有限 I.S.act 的绝对差 ≥5 A 的次数；count|是|阈值化事件，不是寿命损伤量；仅 bin 内相邻对。|
|12|operation_point_transitions|operation_point 相邻有限编码变化次数；count|否|工况协议事件；编码不作为物理量求导。|
|13|zero_current_fraction|raw 样本中 \|I.S.act\|≤0.10 A 的比例；无量纲|是|零电流不等于零电压；不是未来开停机 schedule。|

冻结选中顺序 `[0,3,13,11,4,9,6,7,10,8]`；14 个槽位顺序不变。字典中 derived 字段有缺单位/online 标记，以上 count/ratio 解释来自构建代码，不静默修改历史字典。

## N1 存在但不是 F2 输入的字段

|字段族|静态来源例|单位/聚合|可用性与物理缺口|
|---|---|---|---|
|入口/出口气体温度、冷却温度|T.Si.A、T.So.A、T.Si.C、T.So.C、T.Si.CL、T.So.CL|°C；mean/std/min/max/range|历史 actual；不是直接测得催化层局部 T。物理式用 K，需加 273.15。|
|气体/冷却入口出口压力|p.Si.A、p.So.A、p.Si.C、p.So.C|kPag；上述统计|表压需当地大气压转换绝压；总压不是反应界面 H2/O2 分压。|
|入口 RH 与露点|RH.Si.A、RH.Si.C、DPT.Si.A、DPT.Si.C|% / °C；range 族|入口 RH 不等于膜含水率；需 T、饱和蒸气压与传质映射。|
|H2/N2/air/total gas flow|F.Si.H.A、F.Si.N.A、F.Si.Air.C、F.Si.A.total、F.Si.C.total|nl/min；mean/std|标准体积流量需确认参考 T/P、混合组分/循环/排放；不能直接当电极消耗流量。|
|冷却流量|F.Si.CL|l/min；mean/std|不是反应物流；无法单独闭合热平衡。|
|计量比|Stoic.S.A、Stoic.S.C|无量纲；mean/std|actual/set 分开；不能反向假设已知界面组成。|

set / controller / actual 按 N1 FEATURE_SPECS 分开。存在 set 字段也不证明起点已知未来控制计划；更不保证实际量等于设定量。N1 元数据可定位潜在 Train 补证来源，不能暗中扩充 F2 input。

## 参数观测分级

有效面积 A：历史 G-STOP 数据支持的约 280 cm²，是独立 Train 估计，不在本轮重新估计。j0、α、ASR、j_lim、膜含水状态、气体界面分压：当前白名单证据没有可直接驱动 F2 的联合辨识结果。EIS/极化文件存在不是参数已辨识的证明；本轮不打开它们。

运行时 I、j、电压统计的单位定义可靠到 schema/历史 audit 层级；其样本完整性、校准偏差和共同支持需未来明确授权才可验证。
