# O-07 — 既有实验可信度与投稿材料缺口

**本报告所有性能数字直接整理历史保存表，不读取预测CSV/NPZ，不重算指标，不画新结果图，不加载权重。** 基准 `9723ed75910594155b4c0ae4888d9bbe8519ec11`。本轮只执行A类整理；完整性能再现不是本阶段产出。

## 评价身份与边界

H/J/I2是**单个Development chronological blocked划分中的三Seed**，不是此前TimesNet三expanding folds。Train rows `[0,2103)`到1310h；Validation `[2103,2404)`到1460.5h；Assessment `[2404,3005)`到1819.5h； Development界限1819.997738h。行区间右端排除。合法窗口数量Train1380、Validation205、Assessment378为历史authority记录。历史上下文可早于评价区间起点，但24个目标全部在评价区间，同segment、0.5h连续、合法/finite。O没有重新构造窗口。

Formal RMSE为逐 `(origin,horizon)` 查询等权；模型汇总是三Seed RMSE算术均值及sample SD。**不是三折等权，也不是把三Seed预测混在一起计算RMSE。** I2保存的pooled值只作描述；target-time-equal先每目标平均SSE，再等目标平均开根号，不是平均多个预测做ensemble。

身份/配置：H frozen protocol、L frozen model manifest、J identity、I2 manifest；数值：H `outputs/metrics/H_VALIDATION_METRICS.csv`、`H_MULTISEED_STATISTICS.csv`；J fixed_gate表；I2 `outputs/metrics/I2_ASSESSMENT_METRICS.csv`、`I2_MULTISEED_SUMMARY.csv`；分horizon为I2 `outputs/diagnostics/I2_ALL_HORIZON_METRICS.csv`。均相对研究根目录。源码/manifest当前哈希在O静态来源清单，权重本轮未读取。

## Validation：九正式模型，mV

|模型|Seed|RMSE|MAE|
|---|---:|---:|---:|
|F0|42|8.307161|6.413513|
|F0|52|10.381544|7.029733|
|F0|62|8.513985|3.316270|
|F1|42|9.212743|4.365416|
|F1|52|8.005451|4.861956|
|F1|62|10.248600|5.726064|
|F2|42|6.265169|4.704244|
|F2|52|7.034632|5.033394|
|F2|62|8.436399|3.085123|

每模型205origins/4920预测点。历史摘要RMSE mean/sampleSD：F0 **9.067563/1.142630**；F1 **9.155598/1.122666**；F2 **7.245400/1.100853**。H重复A/B不加入三Seed均值。F2 RMSE逐seed优于F0/F1，但Seed62对F0的余量很小；F2 MAE对F1不是3/3优。F1 RMSE仅1/3 seed优于F0，不能“CNN独立有效”。

### 已保存的F2 Validation扩展诊断

J `outputs/fixed_gate_diagnostic/J_VALIDATION_METRICS.csv`，取learned模式：

|Seed|P95|P99|12h endpoint RMSE|target-time-equal RMSE|
|---:|---:|---:|---:|---:|
|42|12.961445|19.655886|7.146145|6.657435|
|52|14.114391|22.450021|8.555617|8.107005|
|62|7.729978|12.061571|29.817395|27.053894|

Seed62 query RMSE8.436399并不能代表端点29.817395和目标等权27.053894。该run最大误差307.984904mV（历史保存值），须讨论尾部/重叠权重，不删除异常目标后重报。F0/F1同身份的Validation P95/P99/端点/target等权未在本轮许可保存表完整找到：**不能用早期Phase F不同权重成绩补齐**。

## Assessment：已复用Development区，非确认性Test

每run378origins/9072queries/447独立target rows。I2保存值：

|模型/Seed|RMSE|MAE|P99|12h endpoint RMSE|target等权RMSE|
|---|---:|---:|---:|---:|---:|
|F0/42|18.912760|8.031377|28.352991|9.128234|19.201285|
|F0/52|19.255238|8.500392|39.493916|29.881898|19.259790|
|F0/62|27.068482|5.790569|21.272327|16.344224|26.270804|
|F1/42|18.469522|6.152009|23.333484|6.466830|18.616104|
|F1/52|38.157043|9.076820|123.376205|44.975976|35.743811|
|F1/62|12.052070|6.668921|28.167425|16.043866|11.619888|
|F2/42|17.661515|6.151469|23.361588|7.787619|17.943480|
|F2/52|13.764316|6.217432|27.573671|9.117415|13.631900|
|F2/62|5.165326|3.072110|14.308786|4.302552|5.319125|

|模型|等Seed RMSE均值|sample SD|pooled query RMSE（仅描述）|等Seed endpoint RMSE均值|
|---|---:|---:|---:|---:|
|F0|21.745493|4.613023|22.069280|18.451452|
|F1|22.892878|13.603013|25.444944|22.495557|
|F2|12.197052|6.393819|13.267330|7.069195|

F2 Assessment均值优但Seed波动不小，不能说seed不稳定已解决。I2已有216行horizon表（9×24，每步378预测）：可直接整理，不在O重算或新画图。F2 Seed42 h1/h12/h24为17.119257/21.800472/7.787619；Seed52为17.008872/8.506225/9.117415；Seed62为4.841384/3.117647/4.302552。不是每个horizon都同幅改善，也不将这些值当独立泛化证明。

历史I2 target SSE集中：F2三Seed最大目标share .451410/.240558/.239438，top5 share .846976/.613489/.375371；最大目标位置row2952、1793.5h、multiplicity19仅引用保存诊断坐标，O未读取该目标V。重叠窗口不能当独立样本显著性检验。RMSE和SSE集中度、MAE/P99不同排序可共存，必须一起呈现。

## gate独立收益与实验历史

J learned/fixed .5，RMSE mV：

|Seed|learned|fixed|learned−fixed|
|---:|---:|---:|---:|
|42|6.265169|6.252990|+.012179|
|52|7.034632|7.037798|−.003166|
|62|8.436399|8.451469|−.015070|

历史均值learned7.245400185、fixed7.247419090，差−.002018905。learned gate均值/SD分别.427577/.043966、.488762/.004588、.456182/.031680；尤其Seed52几乎固定.5。不能把五种工况相关统计改称电化学因果机制。

fixed对照是同一F2 checkpoint inference perturbation，**不是从头训练固定融合的等价反事实**，也不是原F1 weights。J首次wrapper有 `BLOCKED_REPORT_INTERFACE_ERROR_PRESERVED`，后来cache-only审计 `CACHE_ONLY_INTEGRITY_COMPLETED`；独立gate状态仍未解决，保留失败历史。H严格CUDA bitwise不支持，正常runtime重复性阈值通过；不称严格逐位可复现。

## 投稿证据清单与补救分类

A=整理保存证据；B=只读重算/可视化；C=新Development实验；D=当前缺独立确认数据无法解决。**本轮只做A；B/C/D不执行。**

|项|可用状态|补救|现有证据/风险|
|---|---|---|---|
|九权重身份/配置|VERIFIED_AVAILABLE|A|L/J/I2 manifest；O不重新hash权重|
|预处理schema/Top10/EWMA/scaler|VERIFIED_AVAILABLE|A|F源码、H配置/h_common、E scaler；静态+历史审计|
|窗口边界/完整origin数/真值keys一致|AVAILABLE_BUT_INCOMPLETE|A；若要本轮重新证明须B|H/J/I2历史审计声称exact匹配；O未读逐点缓存，不能升级为新验证|
|九Validation RMSE/MAE/R²|VERIFIED_AVAILABLE|A|H表；不选最佳seed|
|九Validation尾部/endpoint/target等权|AVAILABLE_BUT_INCOMPLETE|B|J仅F2齐；不得不同weights拼表|
|九Validation24horizon曲线/target SSE|MISSING|B|许可保存表缺，需另授权只读缓存方案|
|九Assessment扩展/horizon/SSE|VERIFIED_AVAILABLE|A|I2表，非独立Test|
|固定gate推理对照|VERIFIED_AVAILABLE|A|J，不能推导独立训练增益|
|gate独立训练/同预算容量对照|MISSING|C|预注册等容量/多seed/无Assessment调参；当前未执行、不自动启动|
|已有冻结结果图|AVAILABLE_BUT_INCOMPLETE|A|文件名/历史报告可定位；未逐图数据重新认证，不生成新图或掩盖缺口|
|同数据公开强基线公平比较|AVAILABLE_BUT_INCOMPLETE|C或B取决于是否有同协议缓存|早期TimesNet等任务/边界/版本不同；不能直接拼排名|
|task-level独立Test/跨stack泛化|MISSING|D|L明确资格否决；同已暴露区域重新分割不能修复|
|物理机理/SOH/RUL实证|NOT_APPLICABLE|不作为F2论文补救项|N NO-GO，K未建立真实SOH/RUL，不为投稿新增虚假贡献|

本轮整理 `evidence/O_HISTORICAL_FIGURE_INDEX.csv`，记录44个H/J/I2已有图文件名（含PNG/PDF/SVG格式重复）：H重复性/三Seed/均值SD/配对/epoch/cost，J架构与gate诊断，I2 Assessment/horizon/targetSSE/endpoint/P99。只核对名称存在，不读取图像重新提取数值，也不等于44项独立实验；使用时须保持对应H权重身份及Development标签。

本轮提供的“VERIFIED_AVAILABLE”指静态保存证据存在且身份可定位，**不是当前性能重计算确认**。总体 `EVIDENCE_INCOMPLETE`：可以写透明的条件性实证初稿，但不能以独立验证/模块机制强主张直接投稿。要求确认性结论时存在关键验证缺口；此事实不撤销已存Development正面结果。
