from common import *
def table(d):
 return '| '+' | '.join(d.columns)+' |\n| '+' | '.join(['---']*len(d.columns))+' |\n'+'\n'.join('| '+' | '.join(str(v) for v in r)+' |' for r in d.itertuples(index=False,name=None))
def run():
 config();assert local('outputs/INDEPENDENT_VERIFICATION.json')['PASS'];counts=local('outputs/TRAJECTORY_COUNTS.json');primary=pd.read_csv(P/'outputs/PRIMARY_METHOD_COMPARISON.csv');folds=pd.read_csv(P/'outputs/FOLD_METRICS.csv');lead=pd.read_csv(P/'outputs/ACTUAL_LEAD_AUDIT.csv');cond=pd.read_csv(P/'outputs/MEASURED_CONDITION_RANGES.csv');native=pd.read_csv(P/'outputs/NATIVE_H1_H3_REPLAY.csv')
 md('reports/00_实际研究结果与论文意义.md',f'''# RealHyFC较长健康跨度的实际结果

本轮完成真实实验、失败资格证据和独立数值复核。结论：已有证据支持短周期P1S代理预测；没有证明比保持法更可靠的长周期轨迹，仍不足以支撑完整的物理SOH/RUL一体化主张。

真实middle-50观测147个：完整周期141，其中10个构造参考、131个可因果用于预测；部分周期6个仅描述。物理目录另有5个完整周期没有现成P1S缓存，不补造、不插值。因果131点实际测量结束从181.32838至1712.78运行小时；观察到35次相邻记录回升，包含跨缺口记录，不代表已识别不可逆损伤或可逆机理。

{table(primary)}

H1/H3原生结果完整复现，没有重训：Ridge三fold等权MAE改善10.641466%/10.420448%。局部96h趋势外推在这两任务反而更差。新P1S-only资格规则最终得到与原生相同的H1/H3验证键；新协议身份仍单独登记，不改写历史。

H6新实验：保持法0.547429、趋势0.716365、Ridge0.594674 mV-equivalent；Ridge平均差8.63%，第一折明显恶化，第二折仅1样本，第三折小幅改善。尾部有所下降也不能掩盖平均和跨折稳定性不足。

H12：A/B验证数11/0/7，Ridge为0/0/7。首折Train4，inner-fit0，禁止拟合；第二折无合法验证样本，禁止凑数；第三折Train15/inner-fit4/inner-selection3，Ridge MAE3.518118，保持法0.585254、趋势0.709704。不存在正式三折均值。

最长跨度约114–115运行小时；部分停机间隔使日历时间更长。它不是数百/数千小时剩余寿命。所有三种方法未来lead都未知，只有趋势采用过去周期节奏估计未来时间。

物理：P1S固定j，但完整轨迹前4个因果点温度低、RH高；实际训练标签的T/RH范围较集中，仍无法独立辨识退化、恢复和欧姆参数。没有认证ASR，真实物理loss/参数梯度NA，不训练物理网络。TFT健康增量没有合法Train对，参考SOH七点和原三个事件结论保持不变，RUL没有真实EOL验证。

因此可保留“动态预测分支＋短周期健康代理的探索性证据”；当前不能写“多时间尺度融合、有效软物理约束和真实RUL已统一验证”。''')
 md('reports/01_真实轨迹与冻结时间协议.md',f'''# 真实观测、资格与信息边界

{table(pd.DataFrame([{'item':k,'value':v} for k,v in counts.items()]))}

完整周期原始中间50%P1S算子保持不变，参考电压0.71383023635为前10完整周期电压中位数，直到保守dwell-end171.82091h可获得。前10不是早期因果目标。每条重建actual stable start/end仅131点有证据；精确stable-end日历未保存，保持NA。记录measurement_end_datetime是保守dwell-end日历，不伪造秒级观测时间。6个partial用较少dwell聚合，身份不等于完整周期，不混入预测标签。

主轨迹131点无缺失目标/已保存工况；有物理周期缺口和5个完整但无P1S值的周期，全部记身份和原因。Development截止1819.997738h；不读取原始文件、混合N1或Test。

{table(cond[cond.scope.eq('causal131')])}

不因低温/高湿点或回升结果不好事后删观测。正式模型标签的Train条件另存USED_TRAIN_TARGET_CONDITION_RANGES，常见T约74.96–75.36°C、RH约39.87–40.55%；整体131点中的4个早期低温/高湿点仍在轨迹/因果历史中。条件诊断不能等同物理因果识别。

三个expanding fold边界沿用原协议。目标测量start、保守可用end均落入相应Train/Validation范围；训练标签在预测起点前已可用；内层fit标签在最早inner-selection起点前可用。所有96h历史至少5点且跨度67.2h，不降低标准。

新H6/H12独立登记，五特征和alpha[0.0001,0.01,1,100]固定，Train内部最后15%按时间选参，标准化只用对应fit。H1/H3只读冻结系数。常规局部趋势使用实际历史小时斜率和过去相邻raw cycle的中位耗时，不读未来真实lead/目标日期/工况。

原生188预测与新方法逐点保存；H1/H3新P1S-only资格集最终与历史验证键相同。任何空折不计算正式均值；跨四跨度共同起点数5/0/4，也不能形成正式三折结论。''')
 md('reports/02_逐折尾部恢复与实际跨度.md',f'''# 误差及稳定性

单位均为P1S mV-equivalent，等于保持率误差乘0.71383023635×1000；不是动态电压RMSE或参考SOH百分点。主MAE为三fold等权；mean-fold RMSE/P95/max是各折数值平均，不能冒充全点RMSE/P95。全逐折值如下：

{table(folds[folds.scope.eq('own_horizon_cohort')][['H','fold','method','n','MAE_mVeq','RMSE_mVeq','P95_mVeq','max_mVeq']])}

{table(lead)}

跨跨度自身样本不同，不能仅靠均值大小归因于跨度；COMMON_ORIGIN_PREDICTIONS精确按fold/origin raw ID匹配，5/0/4个起点单独展示，不隐藏第二折空缺。H12首折11个A/B点仍真实评价；不为比较Ridge而隐藏它们，也不将其均值与Ridge第三折7点混比。

观测未来R增加只作事后评价分层，不进模型、不等同已认证恢复。合并样本描述性MAE：H1回升14点，保持0.079234、Ridge0.088397；H3回升11点，保持0.262200、Ridge0.274837；H6回升13点，保持0.254859、Ridge0.299261。Ridge平均短期收益没有保证回升阶段更准。H12没有回升验证标签，不能画虚构回升成绩。

完整STRATIFIED_METRICS逐折分层（空层NA）及POOLED_STRATUM_DIAGNOSTICS保留。未来目标是否落在Train条件范围之外同样只作事后诊断，不能把未来工况当已知输入或据此筛选好看样本。''')
 md('reports/03_物理与TFT_SOH_RUL资格.md','''# 有用轨迹不等于已建立物理SOH或RUL

P1S电流密度固定0.789229989A/cm²。多数正式目标T/RH较集中，支持状态内的性能代理比较，但不证明所有可逆因素已消除。真实Train中的候选欧姆关系V=Eeff−j*ASR，单位V、A/cm²、Ωcm²。Eeff含可逆电位、活化和传质影响，不能令其全部为零。已有观测包括电压、固定j、anode T/RH/gauge压力、flow、stoichiometry；缺独立损失分解、认证ASR与全部反应物分压等。

每一真实Train设计矩阵[1,−j]rank1，2参数不可辨识：Eeff+j*c、ASR+c保持电压。数学导数−j非零不等于已有有效物理训练梯度。ASR无认证标签，物理loss/对训练参数梯度、量级比和匹配控制训练成绩均NA，不能写0或编造曲线。相同容量普通线性回归也有同一秩亏，实际A/B物理训练未获资格。慢状态＋恢复分解的旧nullspace和普通平滑等价结论继续有效，不重命名PINN。

复用65EIS谱/3620点/33组旧电路拟合，无任何重拟合。模型R0不是认证HFR/ASR，同堆/通道等条件未认证，不把其串联为退化电阻。

F2最佳检查点的参数选择需要截至2025-05-26 17:29:26.005的Validation标签。各健康fold Train起点的可用F2参数样本数都是0；虽然末折部分Validation起点在此之后，现有缓存仍无post-selection预测，且动态平均V与固定P1S目标不等价。因此没有公平Train/Validation健康增量实验，未推理或重训TFT。

参考SOH仍是Vpol(j=1)/0.6989533305168152，只有七稀疏参考。131点是另一P1S代理，不是131个真实参考SOH。原M2_NC和三事件贡献/误差抵消结论不变，本轮零新校准、零事件优化。更长代理预测没有建立稳定增益，不能把它直接接成可靠长期SOH。

真实EOL尚未建立，旧条件特定500mV限值不是参考SOH失效时间；记录结束不等于失败，亦未认证右删失。本轮不设新阈值、不训练RUL、不扩展旧条件寿命场景。需要可追溯失效/删失准则、长期轨迹与未来参考验证后，才有真实RUL准确率。''')
 matrix=[('TFT短尺度动态预测','EXPLORATORY','历史结果保留，本轮未新增健康增量'),('H1/H3 P1S未来代理','EXPLORATORY','固定系数收益原样复现；回升阶段仍有弱项'),('H6可靠长期优势','NOT_SUPPORTED','新实验MAE均值恶化8.63%，fold2n1'),('H12完整三fold健康预测','NOT_EVALUABLE','fold1内层不足、fold2验证空；fold3 Ridge明显差'),('工况独立不可逆健康状态','NOT_EVALUABLE','fixedj与集中T/RH不足以辨识损失/恢复'),('有效软物理训练','NOT_EVALUABLE','无认证ASR/独立损失，实际loss/gradient NA'),('TFT健康融合增益','NOT_EVALUABLE','无参数时间合法Train对'),('新增真实SOH预测','NOT_EVALUABLE','未改参考或三事件结论；无新映射'),('真实RUL','NOT_EVALUABLE','无参考一致的真实EOL/认证删失')]
 cs('outputs/PAPER_EVIDENCE_MATRIX.csv',pd.DataFrame(matrix,columns=['claim','status','evidence']))
 md('reports/04_论文主张与下一步边界.md',table(pd.DataFrame(matrix,columns=['claim','status','evidence']))+'''

可形成透明的单RealHyFC Development短周期代理预测研究。不能把工作量、多个Phase或较长跨度的少量胜折写成物理SOH/RUL一体化网络验证。最长12周期约114–115运行小时，且评价不完整；不能外推成寿命可靠性。

本轮停止。没有第二数据集、backbone、D模型或事后改特征/alpha网格，没有新的物理GPU或RUL训练。''')
 md('reports/05_中断日志与数值恢复说明.md','''# 一次执行中断的完整记录

原冻结执行器run_experiment.py在三个H6折各完成5次sklearn Ridge计算后，于H12fold1的inner-fit不足断言中止。原日志CPU_RUN_LOG.txt、冻结协议/源代码SHA和中断时ledger副本全部保留，没有重新冻结原实验。执行器原本应把此资格失败写为NA，不能为此扩大样本。

H6参数当时尚在内存，未落盘；已保存的各折USED_TRAIN/VALIDATION表保留。resume_and_finalize.py在执行前登记独立代码SHA，用相同Train、固定alpha网格和普通Ridge正规方程恢复3个H6模型及12个inner系数，共15次线性方程求解，没有再次调用H6 sklearn.fit。不能将恢复文件称为异常前权重的逐字节备份；其依据是相同冻结统计目标，独立KKT方程和完整预测复算通过。

H12fold1 inner-fit0所以不拟合；fold2无Validation所以不拟合；fold3满足门槛，执行一次固定网格4个inner+1个final sklearn.fit。H12fold3 alpha0.0001由已冻结Train规则选择，恶化结果完整保留。

总计原H6 sklearn调用15、H6数值恢复15次线性求解、H12fold3 sklearn调用5。H1/H3 refit0、GPU0。恢复不是新超参数搜索或结果优化；不更改时间折、特征、资格和停止规则。今后只执行独立verify/tests，不能重新run已有实验。''')
 md('paper_candidates/CLAIM_BOUNDARIES.md','''# 已核验论文候选的主张边界

本包是单RealHyFC Development探索性分析。147个真实middle50值中141完整（10参考构造＋131因果）、6部分仅描述；5个其他完整周期无缓存，不补造。没有连续插值SOH。

H1/H3固定模型收益复现。H6新实验整体MAE不及保持，fold2仅1样本；H12没有正式三fold均值，首折Ridge不可拟合、第二折无验证，第三折Ridge明显恶化。不能选择性只报胜折或把缺失成绩画成0。跨四跨度共同起点5/0/4，不构成三fold对比。回升分层是事后描述，不是恢复机理认证。

H6模型按保存Train作冻结正规方程数值恢复，日志和恢复身份完整披露，不声称异常前权重逐字节复刻。更长代理预测不等于更长参考SOH或RUL验证。TFT健康增量、有效电化学loss、真实EOL/RUL均未获证据；原三个SOH事件结论不变。

所有图只来自真实CSV。空折/不可训练均NA，不升级为独立确认性证据。''')
 md('README.md','''# RealHyFC长期P1S轨迹与一体化可行性

结果：真实观测、样本身份、逐点预测430条、逐折/尾部/回升指标、新H6/H12参数、原生188点复现、条件/物理/TFT/RUL资格均保存。禁止重跑已完成拟合。

从研究主目录复核，不训练：
```powershell
$env:PYTHONUTF8='1'
python -B Phase_P1S_LongTerm_Trajectory_Forecasting/scripts/verify_independent.py
python -B -m unittest discover -s Phase_P1S_LongTerm_Trajectory_Forecasting/tests -v
```

一次性实验入口run_experiment.py freeze/run；本次原执行器在H12不足时中断，随后resume_and_finalize.py在冻结规则下恢复并完成。完整经过见reports/05，失败不是仍在运行。reproduce.py提供安全的完整编排：已完成则只verify；干净副本则运行、仅识别已知资格中断后resume，再qualify/verify/reports/figures，不发布论文或推送Git。

依赖Python3.11、numpy、pandas、scikit-learn、matplotlib，CPU即可。原历史路径只读白名单；未读Raw/混合N1/Test。没有torch、没有新增SOH/物理/RUL网络。所有文件SHA与来源清单保存，模型系数就是实际CPU权重。

候选图scripts/render_figures.py可独立从outputs重绘。经核验后导出的论文包使用相同脚本从source_csv重绘，边界见CLAIM_BOUNDARIES。正式包只新增独立子目录，不替换既有文件。
''')
 print('Reports and explicit publication boundaries written')
if __name__=='__main__':firewall();run()
