# 传感器质量与物理域

阳极入口原字段`RH.Si.A  [%]`(注意两空格)，dictionary标为actual，不是独立的controller字段`RH.Si.A.CV [%]`。单位%。合法raw中超出[0,100]的记录313495，最大212.7523651%。全部保留、flag PENDING，未clip100或静默删除。

在1933个完整raw样本数匹配的Train bins上，raw RH重新mean与既有N1 anode RH mean最大绝对差2.84217094304e-14个百分点。只能核查解析/聚合一致性，不能证明仪器校准有效。source raw已经存在异常高RH，不能将它归因于新聚合步骤。详情见M2_RH_RAW_VS_TRAIN_BIN_REAGGREGATION.csv。

原因仍未确认：sensor量程/校准、derived RH算法、过饱和/恢复状态或其他测量解释均无独立文档足以定案。读取了合法范围的controller RH/露点/T列用于保留schema，但没有用其相关性冒充传感器证书。入口RH不等于膜水或电极界面活动度。

每源温度/压力/flow/stoichiometry的finite/missing/min/max统计见runtime audit。pressure为kPag表压，不直接转成界面分压；flow为nl/min，stoic无量纲。负flow或非正stoic仅flag pending。未借物理单位同名将所有状态默认可比。

`M2_PHYSICAL_MEASUREMENT_QUALITY_UNRESOLVED`为并存限制；未声称所有RH<=100 dwell均已校准，也未根据RHflag补配一套更有利极化样本。
