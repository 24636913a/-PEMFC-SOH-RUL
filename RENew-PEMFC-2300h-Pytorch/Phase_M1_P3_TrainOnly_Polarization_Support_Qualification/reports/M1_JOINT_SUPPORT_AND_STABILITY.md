# 联合支持与稳定域

真实F2 Train：2103 bins，11个原始temporal segments；14工况同时有限的bin数2103。按冻结必要条件剩余0个bin，校准/证据子域分别0/0。逐步分母和排除数见M1_PAIR_ELIGIBILITY_SUMMARY.csv。

**这些都是bin可用性，不是独立稳定dwell数量或联合匹配pair数量。** 每变量完整率/范围见M1_CONDITION_SUPPORT_REPORT.csv，仅是边际数据审计，不能把同时有限或range overlap叫joint matching支持。

逐步实际计数：2103 → coverage合格2094 → valid合格2090 → positive-current且zero fraction为0的1654 → 无load/operation-point转移0。最后这一步排除全部1654个剩余bins；不是工况字段缺失造成筛空，因为全部2103行的14个工况变量均有限。这表明**当前30min整bin视角无符合冻结转移条件的候选**，不能把其中的动态负载mean当稳态极化点。没有据此推断1s raw内部不存在稳定dwell。

校准行[0,1051)、证据[1051,2103)事先按时间隔离；无随机pair split。拟议caliper/最小Δj/时间距离/非共享dwell规则已在support protocol冻结，未因电压结果修改。它们为PROJECT_ANALYSIS_DESIGN，不是仪器精度或通用物理容差；还需原始稳定与状态证据才能执行配对。

未运行真实matching，因此neighbor-state/cross-aging/intervention/recovery pair比例、材料状态、扫描/hold一致性都为NOT_IDENTIFIABLE而非0。没有把早期低j/晚期高j拼成曲线。真正独立稳定dwell与pair数未知；0 certified pairs只指本轮尚未认证，不指数据里没有合法pair。
