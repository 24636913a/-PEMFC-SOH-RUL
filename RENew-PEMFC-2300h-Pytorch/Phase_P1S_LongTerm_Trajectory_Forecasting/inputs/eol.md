# EOL与删失资格

当前结论：**EOL_NOT_ESTABLISHED**；没有可验证的完整failure-time RUL标签。本轮不搜索Test EOL、不读取late-life电压或极化值。

## 原资料与证据范围

`SOURCE_DOCUMENTED_FACT`（通过E21历史原资料摘录；不是本轮独立重读原PDF）：研究目标超过2000 operating hours，约200 cycles后结束以做完整characterization，支持研究性EoT。500mV lower-voltage-limit的文字确实存在，具体摘录关联60/50kPaG、1.8A/cm²的条件特定测量限制。不能说“原文从未提供限值”；但该限值不等于已验证durability failure timestamp，不自动等于j*=1.0的normalized SOH阈值。堆平均、单体下限、极化点和工况特定限值也不是同一量。

E21的OFFICIAL_EOL_DEFINITION=NOT_FOUND具体指未发现官方90%reference定义、结构化failure-time标签或“记录终止即failure”的证据。不能把有限范围的not-found扩张成所有原资料永远无失效概念。

|检查|本轮判断|
|---|---|
|预定义可引用失效准则|有运行/测量下限摘录，但缺可执行、同参考条件的正式failure定义|
|已验证EOL事件及时间|未建立；本轮不访问Test验证|
|2300h记录终点|EoT不能替代EOL；不构造2300−t|
|90%/95%阈值|本轮不设定、不计算何时跨越|
|删失|仅条件性适用：先定义failure criterion并证明观察期间未发生该failure，才将终止当right-censoring|
|独立轨迹|10个cell通道、数千重叠窗口、7个锚点不等于多条独立stack lifetimes|

`METHOD_LITERATURE_SUPPORT`：L6解释未失败单位观察结束后的right-censoring；其数学框架不证明本数据已满足具体failure定义或非信息性删失假设。

因此可写“该研究终止记录缺完整失效结局，具有需要按准则审定的潜在右删失属性”，不能写“已获得真实RUL标签”或在无准则时直接拟合带精确EOL的生存数据。删失解释也不能从单条未失败轨迹识别可靠寿命分布。

`INSUFFICIENT_EVIDENCE`：failure-mode与参考条件阈值的对应、终止原因核验、failure时间/区间、独立寿命数及其条件差异。未来需要可追溯run-to-failure或充分定义的多轨迹censored数据；本轮不引入任何人为阈值。
