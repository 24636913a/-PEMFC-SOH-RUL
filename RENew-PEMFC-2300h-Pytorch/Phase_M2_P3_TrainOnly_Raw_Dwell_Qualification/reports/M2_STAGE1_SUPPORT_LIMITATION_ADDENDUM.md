# Stage1证据子域支持补记

依据本轮已经生成的dwell manifest复核，没有重新读取raw或调整规则。

calibration：3946保持片段、3727全保持positive-current、8个relative quiet。
evidence：3326保持片段、3139全保持positive-current、**0个relative quiet**。
跨校准/证据边界：1保持片段，单列，不参与两个域支持。

positive-current设定负载水平为[14.0, 84.0, 156.8, 221.2, 313.6]A，共5个；实际j按0.001A/cm²舍入后为[0.048, 0.298, 0.299, 0.558, 0.789, 1.121]，共6个数值。0.298与0.299是量化后的actual读数，不能据此声称额外独立实验负载。原summary的7个rounded actual j值还包含非positive hold，不作为认证levels。

因此除了state/measurement语义未知，也直接缺少冻结证据子域的稳定候选支持。8个校准quiet不能取代证据子域，未将caliper或P50改宽。当前状态继续M2_STATE_COMPARABILITY_UNRESOLVED，同时登记RAW_DWELL_EVIDENCE_SUPPORT_INSUFFICIENT；不是已经拥有可训练辅助pairs。

实际source header/plateau记录补入final cutoff manifest，原identity后、parse前snapshot保留。只做元数据收口，不改协议、不覆盖原运行summary/门槛/阈值记录。
