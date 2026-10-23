# O1 — Validation身份核验

基准 `7553fcf3ad5f2b10bcf1f590ce90d19d46468bb4`；Stage0的协议SHA与允许来源/hash记录见protocol目录。九文件来自H正式Validation，role=MULTISEED_PRIMARY；Repeat-A/B未读取。

身份 `VALIDATION_CACHE_IDENTITY_PASS`。九文件实际SHA逐一匹配H保存的prediction_file_sha256；H protocol SHA匹配冻结文件。每run205origins、4920queries、24horizon各205。整数key无重复、每origin完整24步，target=origin+h且在[2103,2404)，预测/真值finite。九模型按显式键连接检查共同key集合与真值精确相同，不删除、补齐、自动去重或修改原文件。origin不连续沿用缓存宇宙，不重新构造raw segment。

模型/seed/epoch/参数量/权重路径通过H/L/J metadata对应；权重SHA仅对历史manifest，不读取权重bytes。H static audit窗口规则作为segment合法性的历史依据，O1未接触raw/N1重新验证每个segment。所有模型完整缓存通过之后才算指标，不按误差筛样本。

复算使用round_trip float64及冻结fsum/linear quantile。正式RMSE/MAE与H逐run、F2七项扩展与J learned一致，绝对容差1e-10mV，实际差异完整保留于 `outputs/O1_HISTORICAL_METRIC_DIFFERENCES.csv`。J已保存的4920 keys/truth/pred exact审计通过，但O1不重新读取J逐点预测；此为历史身份+本轮指标一致性，不冒充新的J逐点重推理。

唯一目标时刻数各run=251，只是去重索引数量，不代表统计独立。所有缓存查询保留；没有新预测、模型加载、训练、Assessment逐点或Test访问。完整各文件证据见 `outputs/O1_IDENTITY_AND_RECOMPUTATION_AUDIT.json`。
