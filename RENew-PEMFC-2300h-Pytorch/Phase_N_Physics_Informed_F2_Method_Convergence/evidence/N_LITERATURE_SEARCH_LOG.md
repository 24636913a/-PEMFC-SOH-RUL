# N 文献核查记录

日期2026-10-12；来源策略：出版商/原作者/政府手册优先，只用一手内容支持技术判断。二手检索结果仅定位DOI，不补造方法。无真实项目数据下载，未查阅含RealHyFC Test值的原始文档。

## 查询与收口

|检索组|查询示例|收口|
|---|---|---|
|改进TFT|PEMFC improved temporal fusion transformer；Future prediction on the remaining useful life …|确认Chou/Wang DOI123019，BiLSTM改造；排除电池131114与PatchTST156084误认|
|Ko|Physics-informed neural network for long-term prognostics；125318|一手摘要/引言明确age/current、负载复制、膜/催化剂；四作者publisher可见，二手五作者冲突不采纳|
|Crossformer|A hybrid framework…Crossformer physical constraints；239941|一手摘要/片段确认conv/DSW/TSA/三阶段及两PHM数据；第三dynamic身份仍UNKNOWN|
|未来KD|A future-guided knowledge distillation framework；128423；teacher n steps|一手引言确认teacher前移、推理移除；CRediT He/Xu/Zhang/Li/Kai；精确loss/OOF未取得|
|半经验UQ2026|Credible long-term degradation…112945；Aging Prognosis…Semiempirical ECM|PINNCP一手片段确认半经验/参数NN/conformal；IEEE3661158只有二手可见，不用于技术结论|
|基础理论|DOE Fuel Cell Handbook；Learning using privileged information Vapnik Vashist|政府PDF和原论文概念页；项目数学反例独立推导，不冒充论文实验|

## 一手来源身份

1. Chou, J.-H.; Wang, F.-K. Renewable Energy247(2025)123019。正式期刊。出版页：[S0960148125006810](https://www.sciencedirect.com/science/article/pii/S0960148125006810)。注意网上另有122895误引，依据此出版页DOI123019。
2. Ko, T.; Kim, D.; Park, J.; Lee, S.H. Applied Energy382(2025)125318。正式期刊。[出版页](https://www.sciencedirect.com/science/article/pii/S0306261925000480)。未以二手作者扩展修改历史ledger。
3. Wei, L.; Gao, S.; Chen, Z.; Wang, Y.; Sun, Z. Journal of Power Sources677(2026)239941。[出版页](https://www.sciencedirect.com/science/article/pii/S0378775326006919)CRediT可见；正文全作者顺序不以CRediT推断。本轮比旧M多核实贡献名单，不改M。
4. He, J.; Xu, T.; Zhang, K.; Li, Y.; Kai, Y. Applied Energy424，128423，编期Dec2026。[出版页](https://www.sciencedirect.com/science/article/pii/S0306261926010779)CRediT显示这些名字；最终署名顺序不以CRediT替代。期刊在线可检索，不是仅arXiv；首次online日期/版本标签UNKNOWN。
5. Liu, M.; Yang, Y.; Cheng, J.; Cui, J.; Zheng, X. Reliability Engineering & System Safety276，112945，编期Dec2026。[出版页](https://www.sciencedirect.com/science/article/abs/pii/S0951832026007556)。期刊在线记录，不是仅预印本；首次online日期/版本标签UNKNOWN。
6. EG&G/DOE NETL, Fuel Cell Handbook7th ed.(2004)，[政府PDF](https://www.netl.doe.gov/sites/default/files/netl-file/FCHandbook7.pdf)。理论参考，不提供RealHyFC参数。
7. Vapnik, V.; Vashist, A. Neural Networks22(5–6)(2009)544–557，DOI10.1016/j.neunet.2009.06.042，[出版页](https://www.sciencedirect.com/science/article/pii/S0893608009001130)。LUPI概念参考，不证明本项目蒸馏收益。

## 可见范围与访问限制

ScienceDirect直接open四个主要新条目返回403；同出版商search索引提供摘要/引言/section snippets，足以定位方法轮廓，不等于完整全文。没有取得不受限制的完整方法时不猜逐式、未披露的teacher未来输入、消融数量或显著性。未来December编期不能推定已完成纸刊发行或精确online日期。

每篇在主矩阵只给最少必要方法概述，不复制长段原文、不保存完整publisher网页。未取得全文的缺项阻止积极准入/首创宣称，而保守否决主要由项目M/M1/M2独立证据和数学反例支撑，不依赖猜测论文错漏。
