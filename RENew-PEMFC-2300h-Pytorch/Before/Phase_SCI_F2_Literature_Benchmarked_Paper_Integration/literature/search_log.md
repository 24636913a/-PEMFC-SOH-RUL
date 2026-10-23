# 官方文献身份、公开访问与数据来源核验日志

实查日期：2026-10-17（研究负责人所在时区Asia/Shanghai）。本次是指定九篇PEMFC论文的定向对标，加查原TFT与RealHyFC原始数据来源，不是系统综述，不代表覆盖全领域，不支持“全球首次”声明。

本日志负责文献身份与可访问性；方法、图表、实验比较的具体页码由本阶段全文证据索引另行负责。Crossref元数据核验不等于读过全文。没有公开全文链接不等于文献没有相关实验。

## 执行过程

1. 逐条以用户给出的九个DOI检索，优先出版社ScienceDirect。分批检索，每批最多4个查询；检索词包含精确DOI或完整题名。另查原TFT DOI以及RealHyFC durability/dataset/Zenodo。
2. 读取仓库Phase O的 `evidence/O_PDF_IDENTITY_AND_SOURCE_LEDGER.csv` 与相关报告，只用于确定已有原PDF位置与历史核验身份。Phase O的P编号与本轮编号不同，连接时必须使用DOI，不能按旧编号连接。
3. 对11个DOI调用官方 `https://api.crossref.org/works/{encoded_DOI}`，保存原始响应字节到 `outputs/evidence_index/CROSSREF_*.json`。题名、全作者、期刊、卷、期、页码/文章号、DOI均来自这些原始响应。
4. 官方元数据与出版社检索结果相互核对。部分ScienceDirect直接打开返回HTTP403；记录该访问限制，没有绕过付费访问或将检索摘要冒称全文。出版社索引仍可展示题名、作者、卷期及日期。
5. 原TFT另外核对作者arXiv记录；RealHyFC核对官方Zenodo论文及数据两个记录及双向关联，保存两个官方API元数据JSON。没有打开、下载或解析任一原始CSV、压缩包、Assessment或Test数值。
6. 运行 `python -B scripts/verify_metadata.py` 离线核验11条DOI记录和两个Zenodo记录的字段及SHA256，结果PASS。`--fetch`可重新获取官方元数据；远端元数据可能更新，重新获取会产生新的来源SHA，这与历史结果修改无关。

## 九篇指定文献身份核对

|本轮ID|作者（完整名单见JSON）|期刊、正式卷页身份|官方来源|
|---|---|---|---|
|P1|Khaled Benaggoune; Meiling Yue; Samir Jemei; Noureddine Zerhouni|Applied Energy 313 (2022), 118835|[出版社](https://www.sciencedirect.com/science/article/abs/pii/S0306261922002756)、[DOI](https://doi.org/10.1016/j.apenergy.2022.118835)|
|P2|Li Hongwei; Qiao Binxin; Hou Zhicheng; Liu Junnan; Yang Yue; Lu Guolong|International Journal of Hydrogen Energy 48(66) (2023), 25958–25971|[出版社](https://www.sciencedirect.com/science/article/abs/pii/S0360319923014611)、[DOI](https://doi.org/10.1016/j.ijhydene.2023.03.316)|
|P3|Chongyang Tu; Fen Zhou; Mu Pan|International Journal of Hydrogen Energy 74 (2024), 414–422|[出版社](https://www.sciencedirect.com/science/article/pii/S0360319924022948)、[DOI](https://doi.org/10.1016/j.ijhydene.2024.06.095)|
|P4|Jia-Hong Chou; Fu-Kwun Wang|Renewable Energy 247 (2025), 123019|[出版社](https://www.sciencedirect.com/science/article/pii/S0960148125006810)、[DOI](https://doi.org/10.1016/j.renene.2025.123019)|
|P5|Yurdagül Benteşen Yakut|International Journal of Hydrogen Energy 144 (2025), 652–664|[出版社](https://www.sciencedirect.com/science/article/pii/S0360319925008948)、[DOI](https://doi.org/10.1016/j.ijhydene.2025.02.317)|
|P6|Lin Wei; Shangrui Gao; Zonghai Chen; Yujie Wang; Zhendong Sun|Journal of Power Sources 677 (2026), 239941|[出版社](https://www.sciencedirect.com/science/article/pii/S0378775326006919)、[DOI](https://doi.org/10.1016/j.jpowsour.2026.239941)|
|A1|Jian Jiang; Dongsheng Du; Lin Su; Yifan Du|International Journal of Hydrogen Energy 207 (2026), 153481|[出版社](https://www.sciencedirect.com/science/article/abs/pii/S0360319926001187)、[DOI](https://doi.org/10.1016/j.ijhydene.2026.153481)|
|A2|Fengyan Yi; Xing Shu; Jiaming Zhou; Jinming Zhang; Chunxiao Feng; Hongtao Gong; Caizhi Zhang; Wenhao Yu|International Journal of Hydrogen Energy 111 (2025), 228–237|[出版社](https://www.sciencedirect.com/science/article/pii/S0360319925008808)、[DOI](https://doi.org/10.1016/j.ijhydene.2025.02.302)|
|A3|Taehwan Ko; Dukyong Kim; Jaewoong Park; Seung Hwan Lee|Applied Energy 382 (2025), 125318|[出版社](https://www.sciencedirect.com/science/article/abs/pii/S0306261925000480)、[DOI](https://doi.org/10.1016/j.apenergy.2025.125318)|

九个DOI与用户给出的题名、期刊和年份均相符。以下是需要保留的细节：

- Ko论文官方Crossref、原PDF及作者机构记录列出4位作者。部分二手网站额外插入Junghwan Park，不能据此增加第五位作者。
- Yakut论文是单作者；不得笼统改写成Li等或LSTM团队论文。
- Li论文作者姓名按出版社及Crossref的实际存储保存，不凭中文姓名习惯擅自交换given/family字段。
- 全部11条Crossref记录都未提供 `published-online` 日期，故JSON保留null。不能把DOI注册日期、搜索索引的“发表于几个月前”或论文接收日期当作online-first时间。JSON单独保存出版社显示的卷期日期；例如Ko卷期为2025-03-15、Yi为2025-03-20，不能互换为在线日期。

## 全文访问等级

九篇目标文献中，本地参考论文目录能定位到7篇PDF：Benaggoune、Li、Chou、Wei、Jiang、Yi、Ko；本轮全文内容角色应分别核对身份与实际阅读范围。Tu和Yakut尚未发现对应本地PDF，当前使用官方出版社页面/摘要/可见章节片段，缺失内容保留UNKNOWN，不推断其未做某项实验。

本元数据核验没有重新通读这7篇全文，不越过内容审查角色的阅读声明。原TFT另有[作者预印本](https://arxiv.org/abs/1912.09363)，初次提交2019-12-19、v3修订2020-09-27，与正式期刊2021年应分别记录；[正式出版社页面](https://www.sciencedirect.com/science/article/pii/S0169207021000637)标明Open access和Creative Commons。作者为Bryan Lim、Sercan Ö. Arık、Nicolas Loeff、Tomas Pfister。

## RealHyFC数据来源与可引用位置

原始数据论文为Fensterle等，*Spatially resolved electrochemical characterization of a PEM fuel cell stack during durability testing*，International Journal of Hydrogen Energy 250 (2026), 156060，[DOI](https://doi.org/10.1016/j.ijhydene.2026.156060)。

已只读本地该论文p1–3和p4的§2.3方法部分（在§3 Results前停止），未读取任何数据文件或本轮后期目标：

本地原PDF位于仓库 `数据/《耐久性测试期间PEM燃料电池堆的空间解析电化学表征》的基础数据/IJHE_2026_Fensterle_Spatially resolved electrochemical characterization of a PEM fuel cell stack during durability testing.pdf`，SHA256为 `c66c0ffa31627637dd2e6842df976731e833bb230b30e0c967d73bcd99fafad1`。该记录身份为 `SELECTED_PAGES`，不是全文复读。

|数据章节论断|可追溯位置|准确边界|
|---|---|---|
|RealHyFC项目、generic HyFaB硬件、活性面积280 cm²|原论文p2 §2.1 Test-setup|实验装置身份，不能与其他数据集混同|
|耐久性试验为10-cell短堆|p2 §2.1第一段|独立BoL敏感性测试使用8-cell fresh stack；不能把它当作同一个耐久性电堆的重复寿命轨迹|
|重载相关动态负载循环|p3 §2.3和Fig2|标称约12h循环，3次短停加1次长停；不能据此假设所有本项目实际周期精确等长|
|完整试验超过2000 operating hours|p1摘要；p4 §2.3方法末段|§2.3说明200cycles后结束；不是达到真实EOL的声明，也不能把完整试验时长直接当作Development范围|
|公开数据包含load cycle及表征资料|Zenodo数据记录Description第2–3段|只核对元数据；本稿仅使用已冻结Development接口|
|公开许可|原论文p1版权行；两个Zenodo API `metadata.license.id`|CC BY 4.0；不是CC0|

官方[论文记录20742676](https://zenodo.org/records/20742676)的DOI仍是上述期刊DOI，标注出版日期2026-06-16；官方[数据记录20715542](https://zenodo.org/records/20715542)的DOI为10.5281/zenodo.20715542，出版日期2026-06-29。两者网页均显示v1，API版本index=0且is_last=true，parent分别是20742675和20715541。论文与数据互为isSupplementedBy/isSourceOf关联。期刊卷期日期2026-07-13与上述存储库时间是不同日期概念。

同一项目另有[19068126](https://zenodo.org/records/19068126)，属于BoL空间测量及Fuel Cells前作，不能代替本研究2300h耐久性数据来源。项目grant DOI10.3030/101111904是项目身份，也不能代替数据DOI。

## 可复现核验入口

在本阶段目录运行：

```powershell
python -B scripts/verify_metadata.py
```

这只读取本阶段保存的官方元数据JSON，不访问原始实验数据，不加载模型。联网刷新命令为 `python -B scripts/verify_metadata.py --fetch`，仅请求Crossref及Zenodo元数据API，不跟随附件下载链接。
