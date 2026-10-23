# 一手文献核实台账

检索日期：2026-10-11。采用 deep-research 的来源分级/反证原则；范围是定点文献与事实核查，不是完整系统综述或虚构的多代理审稿。技术主张仅使用作者、期刊出版页和 DOE 文档；聚合站仅作定位，未用其方法描述作为依据。

检索轴：PEMFC Nernst/Tafel/ohmic/concentration、long-term PINN/current-density availability、Crossformer physical constraints、steady-state model dynamic forecasting、historical auxiliary physics。出版社直接页面部分返回 403/内部错误；出版页的可索引摘要/章节片段仍可核实。**全文未完整取得的研究，不声称完成全文公式、参数或消融审计。** 未下载任何项目数据，没有用 Web/GitHub 重新请求 N1。

|ID|真实来源、DOI 与访问范围|可核实结论|尚未核实/不可移植之处|
|---|---|---|---|
|L1|EG&G/US DOE NETL，Fuel Cell Handbook，7th ed.，Nov 2004。[官方 PDF](https://www.netl.doe.gov/sites/default/files/netl-file/FCHandbook7.pdf)。档案 DOI 10.2172/834188（[政府文档馆藏](https://digital.library.unt.edu/ark:/67531/metadc785768/)）；本轮可读 PDF §§2.1–2.6、§3 相关段落。|Nernst、Tafel、欧姆、浓差、膜/水管理的理论参照。§2.5 的式2-20/22及相邻浓差讨论支持损失分解。|不是预测论文；无 RealHyFC 参数、数据划分或消融。理论存在不闭合当前未来信息集。|
|L2|Taehwan Ko、Dukyong Kim、Jaewoong Park、Seung Hwan Lee，Applied Energy 382(2025)125318。[出版页](https://www.sciencedirect.com/science/article/pii/S0306261925000480)，[DOI](https://doi.org/10.1016/j.apenergy.2025.125318)，[作者实验室记录](https://smlab.hanyang.ac.kr/publication/paper)。出版页索引摘要/有限片段，非完整全文。|标题为 Physics-informed neural network for long-term prognostics of proton exchange membrane fuel cells。输入包含 operating time/current density；摘要明确将训练期 current density 复制到预测期。涉及膜/催化剂退化，公开 PHM challenge 数据在出版片段出现。|这是有负载复制策略的长期预测，不是 F2 的严格 future-unknown 48→24任务。全部数据身份、方程系数、参数辨识、消融协议尚不能完整核实；不借其 RUL 或性能主张授权本项目。|
|L3|Journal of Power Sources 677(2026)239941，A hybrid framework for PEMFC degradation prediction integrating Crossformer and physical constraints。[出版页](https://www.sciencedirect.com/science/article/pii/S0378775326006919)，[DOI](https://doi.org/10.1016/j.jpowsour.2026.239941)，[作者机构记录](https://faculty.hfut.edu.cn/sunzhendong/zh_CN/zdylm/539958/list/index.htm)。摘要、引言及有限片段可索引；完整方法未取得。|卷号/文章号/DOI 核实；三阶段训练、卷积/Crossformer/物理约束及三种负载类别在摘要中明确。摘要宣称模块消融贡献。作者 Lin Wei 在出版页 CRediT 片段核实。|完整作者序列、三个数据集精确身份、逐式残差、未来工况来源、参数辨识及消融数值控制未充分核实。不能推定它证明“未来真实工况不可用时 Nernst 可直接计算”，也不据标题预设本项目首次。|
|L4|Qiang Liu、Weihong Zang、Wentao Zhang、Yang Zhang、Yuqi Tong、Yanbiao Feng，Energies18(10)(2025)2665。[期刊文章](https://www.mdpi.com/1996-1073/18/10/2665)，[DOI](https://doi.org/10.3390/en18102665)。正文索引包括 §3.3、式38、结论与数据声明。|半经验模型以 grey wolf optimizer 识别参数；1000h动态记录，前250h训练；当前预测步电流 demand 是已知输入，T/P等采用 nominal 条件；历史10步与模型参考电压共同输入。有同架构去模型信息的对照；文章称数据公开。|与 F2 future-unknown 协议不同：已知当前 demand 与名义条件是额外假设。公开声明不等于已验证数据下载/完整处理链；物理模型的全部系数与辨识置信度本轮未复算。|
|L5|Wenchao Zhu、Bowen Xu、Bingxin Guo、Changjun Xie、Rui Xiong，eTransportation(2026)100615。[出版预览](https://www.sciencedirect.com/science/article/abs/pii/S2590116826000731)，[DOI](https://doi.org/10.1016/j.etran.2026.100615)。摘要/方法片段，不是全文。|PhySR-PINN；train voltage segment 中符号搜索双时间常数退化先验，按负载条件组织；明确长期 prognostics。|完整公式、future-load 假设、数据身份与独立物理消融未完全核实；拟合指数并命名可逆/不可逆，不直接证明本项目状态可辨识。|
|L6|Maziar Raissi、Paris Perdikaris、George Em Karniadakis。[作者预印本 Part I](https://arxiv.org/abs/1711.10561)，2017；相关正式框架 JCP378(2019)686–707，[DOI](https://doi.org/10.1016/j.jcp.2018.10.045)。本轮核实作者摘要，未声称全文逐式验收。|PINN 的基本论点是对可微网络施加可计算的控制方程约束。|PDE 案例不是 RealHyFC 物理证据；不能仅因网络名含 PINN 就归类严格机理约束。|

## 文献贡献与项目证据不混同

L2–L5 证明相关研究真实存在、采用了不同的机理先验和信息假设；没有一项证明现有 F2 仅历史 Top10 信息能闭合未来电化学残差。尤其 L2 的复制负载、L4 的已知 demand，都不能暗中转写成未来实际负载。对 L3 的摘要消融主张，仅记录为 AUTHOR_REPORTED，不写成独立复现。所有文献预测指标均未进入项目结果表。

完整方法不可核实的事项是后续文献补证项，不是以推测填空。若需精确比较参数辨识与逐式损失，必须取得合法全文，再单独审核；本轮科学结论依赖信息集与项目历史证据，不依赖这些未核实细节。
