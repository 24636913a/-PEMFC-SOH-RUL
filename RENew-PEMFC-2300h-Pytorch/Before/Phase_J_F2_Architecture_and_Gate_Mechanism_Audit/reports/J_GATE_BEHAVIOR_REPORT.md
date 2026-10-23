# 工况门控行为

Base: `427d4dc402f083a5b9e094b12981c5e231df8100`. Protocol SHA256: `0244f05fd777ccd615fda7eb9533d129323dc27fbb58212e091130011d0d9b24`. Code identity SHA256: `67b0d3588f75db56d435ddc5277d38324708f660ac10cc1850a5908d771612c8`.

Validation only, 205 origins/seed, 615 gates; all 4920 queries per seed retained. New training/Assessment inference/Test/seed selection: NO.

门控证据状态：`PHASE_J_GATE_DYNAMIC_EFFECT_UNRESOLVED`。

|seed|n|mean|median|SD|min|max|P5|P25|P75|P95|IQR|low_lt_0p1_fraction|middle_0p4_to_0p6_fraction|high_gt_0p9_fraction|temporal_range|mean_absolute_deviation_from_0p5|
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
|42|205|0.42757698|0.428729981|0.0439657024|0.351626664|0.506279409|0.358451927|0.391606361|0.462881446|0.496276927|0.0712750852|0|0.624390244|0|0.154652745|0.0725621718|
|52|205|0.48876248|0.488812208|0.00458767126|0.47961992|0.496499777|0.480817723|0.485837698|0.492600322|0.495664173|0.00676262379|0|1|0|0.0168798566|0.0112375198|
|62|205|0.45618234|0.454176784|0.0316802096|0.394615918|0.512564719|0.401892143|0.431204379|0.481781512|0.506371593|0.0505771339|0|0.956097561|0|0.1179488|0.0451538916|

SD采用ddof=1。低门控g<.1，中间.4≤g≤.6，高门控g>.9；诊断性dynamic定义SD>.001且range>.01，均在结果产生前冻结，属于PROJECT_ANALYSIS_DESIGN，不用于调参/筛选。g为LSTM权重，1−g为Conv权重；一个origin仅一个g，广播到48×64历史表示，不为24个horizon分别生成门控。

|seed_a|seed_b|mean_absolute_gate_difference|gate_difference_RMSE|paired_gate_Spearman|
|---|---|---|---|---|
|42|52|0.0616194382|0.072741843|0.991244028|
|42|62|0.0286053607|0.0319366158|0.984255687|
|52|62|0.0352293301|0.0423917695|0.985272527|

三个Seed都满足描述性dynamic标准。Seed52基本接近0.5且振幅小，Seed42/62有更明显变化；平均g均低于0.5，仅表示温和偏向Conv，不是始终选择某个分支。三对排序Spearman均超过0.984，变化次序相似，但绝对水平及振幅有差别。全部低/高门控比例为0，没有接近0/1的饱和。中间区间比例依次为62.44%、100%、95.61%。只报告配对分布差异，不选择最佳Seed或把不同权重函数视为同一物理状态。低中高区间不覆盖全部[0,1]，三项比例不必加和为1。

|seed|context|Spearman_rho|scope|
|---|---|---|---|
|42|load_transition_count|-0.843564448|descriptive mathematical input dependence; overlapping windows; no p-value|
|42|zero_current_fraction|0.034258774|descriptive mathematical input dependence; overlapping windows; no p-value|
|42|actual_current_density_A_cm2_std|-0.96613921|descriptive mathematical input dependence; overlapping windows; no p-value|
|42|actual_current_A_std|-0.965866195|descriptive mathematical input dependence; overlapping windows; no p-value|
|42|actual_current_density_A_cm2_mean|-0.515561839|descriptive mathematical input dependence; overlapping windows; no p-value|
|52|load_transition_count|-0.79337447|descriptive mathematical input dependence; overlapping windows; no p-value|
|52|zero_current_fraction|0.0217587345|descriptive mathematical input dependence; overlapping windows; no p-value|
|52|actual_current_density_A_cm2_std|-0.987506791|descriptive mathematical input dependence; overlapping windows; no p-value|
|52|actual_current_A_std|-0.987290886|descriptive mathematical input dependence; overlapping windows; no p-value|
|52|actual_current_density_A_cm2_mean|-0.464041454|descriptive mathematical input dependence; overlapping windows; no p-value|
|62|load_transition_count|-0.843834028|descriptive mathematical input dependence; overlapping windows; no p-value|
|62|zero_current_fraction|-0.105591339|descriptive mathematical input dependence; overlapping windows; no p-value|
|62|actual_current_density_A_cm2_std|-0.977394102|descriptive mathematical input dependence; overlapping windows; no p-value|
|62|actual_current_A_std|-0.977128052|descriptive mathematical input dependence; overlapping windows; no p-value|
|62|actual_current_density_A_cm2_mean|-0.383081445|descriptive mathematical input dependence; overlapping windows; no p-value|

门控与历史电流/电流密度波动呈强负秩相关（约−0.966至−0.988），与负载转换次数也负相关；与零电流比例的关系弱且跨Seed方向不一致。这些相关关系说明数学输入依赖，不证明不可逆退化或工况的物理因果作用。标准化工况均值从masked 48步输入被动采集，使用保存的scaler反变换，并独立与未滤波Raw工况窗口均值校验。五项工况不接受电压EWMA。原始单位与标准化值分列，未混淆。

Train1380个合法历史窗口确定每变量1/3、2/3分位阈值，冻结后应用Validation；退化/重复分位记录degenerate，空组不强制补样本。完整分层见 J_TRAIN_DEFINED_STRATA.csv。重叠窗口不是独立统计样本，相关/分组均描述性，不报告独立性p值或电化学机制。

图注：随年龄曲线的连线仅连接已观测的合法origin；跨缺失区间的线段不代表该区间有观测、插值或真实平台。不得根据连接线推断未观测时段的门控状态。
