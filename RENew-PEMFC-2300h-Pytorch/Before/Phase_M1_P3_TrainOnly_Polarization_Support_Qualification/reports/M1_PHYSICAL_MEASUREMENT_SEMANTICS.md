# 物理测量语义

依据：N1 dictionary/config/build code、G-STOP PHYSICAL_RELATION_AUDIT、PAA protocol/selection code、PAV PAA_CLAIM_BOUNDARY、CSHSE incident、Phase M identifiability。具体源hash见来源台账。

|来源|量与状态|本轮可证/不能互换的原因|
|---|---|---|
|durability bin|有效10cells动态运行30min mean V；cell_std是瞬时跨cell离散度时间mean|不是120s稳定dwell；cellstd不是时间噪声；没有raw时间顺序/扫描方向证据|
|PAA stable P1S|固定221||221.200，raw run>=120s，middle50% dwell voltage median，再interval median|固定单负载token不能提供低/高j曲线；本轮不执行更宽Development读器、不复用全表数值|
|nominal polarization|characterization curve、per-cell V与actual j|不能混用运行bin的负载均值；温湿压/hold/扫描/配置需独立确认，本轮未读取真实曲线|
|reference-condition voltage|条件标准化诊断健康anchor|不是动态bin，也不是已验证连续SOH|

j.S.act单位A/cm²、I.S.act单位A；历史G-STOP支持约280cm²比例，但这不是极化关系证据。durability stack10cell，U6候选异常已排除；BoL短stack配置不能无核查拼接。

传感器语义：T.Si/So.A/C是入口/出口温度，不是反应界面温度；p.Si/So是kPag表压，不是反应气体界面分压；RH为入口湿度，不等于膜含水；F单位nl/min，stoichiometry为无量纲。材料/界面慢状态、恢复/滞后不由这些bin mean闭合。

未证实：实际raw dwell长度/稳定化、极化扫描方向/hold、Air-Air或其他恢复干预的Train逐事件身份、膜水与材料状态可比。`QUASI_STEADY_STATE_UNVERIFIABLE`。运行load transitions==0仅表示不存在相邻>=5A跳变，不能证明不存在慢漂移或热/水滞后。
