# N — Train-only 特权信息与时间隔离契约

状态：`DESIGN_ONLY_NOT_EXECUTABLE`。当前不读Train时序、不建实际交叉拟合清单，不开启教师或F2训练。

## 1. 四个信息集合

|阶段|可用信息|禁止|
|---|---|---|
|教师参数/teacher scaler拟合|合法Train、已发生的认证测量，严格前置隔离块|Validation/Assessment/Test，接收软目标块的电压/诊断标签|
|为学生生成Train软目标|接收块历史X，及**额外注册**的该Train目标时段actual Z|同目标Y作teacher输入、同目标诊断状态反演、越Train界的Z|
|学生训练|原历史X、普通Train ΔV监督、合格OOF软目标|未来actual Z直接注入学生、teacher latent追加为推理输入|
|学生推理/Validation评分|原F2历史48步、raw origin锚点、确定性h索引|未来j/T/RH/P/flow/stoic、预测目标电压、teacher或诊断输入|

训练特权信息不同于推理已知协变量，[LUPI原论文](https://www.sciencedirect.com/science/article/pii/S0893608009001130)提供概念先例。本项目不把“已保存在Train文件”改名为“预测起点可观测”。只作为预注册训练监督，必须给 F2-Control 同样Z。POT历史禁止future-actual配对不改；若以后获批，这是新的独立实验信息契约。

## 2. 教师OOF必须采用时间块隔离

不能teacher在全部Train Y上拟合后，预测同一批Y再声称独立物理验证。概念契约为：在原Train内部按时间形成先行校准块与后继接收块；每次teacher参数、teacher scaler、材料状态初始化和任何选择，只用先行块。

接收块每个query的依赖集合包括history48、origin锚点、target24、用于物理测量的raw dwell、极化/EIS scan和共享诊断身份。teacher拟合依赖与接收依赖必须无交集，teacher完整target必须结束在接收历史开始前。均在原Train `[0,2103)` 内；不跨segment/gap。

若仅半小时row依赖且两侧均48→24，则先行最后origin oA、接收最早origin oB满足 `oA+24 < oB−47`，即至少 `oB−oA≥72`。这不是固定36h物理稳定化证明；若raw dwell/scan依赖更长，按完整依赖继续purge。不能用一个horizon长度embargo代替全部依赖核验。

首次校准段没有诚实teacher软目标时，可只用data loss，但需要将同一eligibility/权重安排用于非物理对照；不得从未来块回填teacher。若purge后无足够认证多负载支持，停止A，不通过共享dwell来补数。

本轮只检验symbolic依赖集合生产校验函数，未生成真实OOF count或soft labels。未来具体块清单、支持最低数量、物理teacher参数结构与λ尚未获得数据资格，必须先由独立科学证据冻结；这份文档不是隐含运行许可。

## 3. 为什么privileged不产生未来oracle

令教师输出T(X,Z)，推理学生仅g(X)。平方loss的最优投影是 `E[T(X,Z)|X]`，不是某次真实未来Z对应的T。若teacher只是理想条件均值 `E[Y|X,Z]`，再对Z边缘化得到 `E[Y|X]`；可能有有限样本学习价值，但不消除未观测负载造成的条件不确定性。相同过去两条未来负载得到不同电压的合成反例已经纳入测试。

禁止将训练特权信息收益写成物理识别，禁止部署时偷偷增加未来Z predictor来修补；那会改变模型与任务，N不设计/执行。

## 4. Source firewall

N脚本白名单只允许小型既有报告/config/scaler metadata与源码；不读取N1 causal表、raw、电化学观测、预测缓存或checkpoint字节。历史Test访问仅依据L报告文本理解，不请求实际target值。

任何未来真实Train补证需另行明确授权与exact source身份/截断；不能调用旧全文件hash函数读取mixed CSV。M2跨1310h的raw来源仍未认证，不因N有教师提案就放开。M2的全Train来源历史哈希是其真实记录，不能改写成过去没做过hash。

所有新增产物仅写N目录。审计脚本不import历史runner；任何越白名单读取调用均fail-closed。`N_STATIC_AND_SYNTHETIC_AUDIT.json` 的访问陈述限定本轮/本入口，不声称操作系统层面的历史全程取证。
