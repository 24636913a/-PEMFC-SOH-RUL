from common import *
def table(d):
 def v(x):return f'{x:.9f}' if isinstance(x,(float,np.floating)) else str(x)
 return '\n'.join(['| '+' | '.join(d.columns)+' |','| '+' | '.join(['---']*len(d.columns))+' |']+['| '+' | '.join(v(x).replace('|',' / ') for x in row)+' |' for row in d.itertuples(index=False,name=None)])
def run():
 cfg=config();dec=jl('outputs/RESEARCH_DECISION.json');summ=pd.read_csv(P/'outputs/THREE_SEED_MEAN_AND_SD.csv');act=pd.read_csv(P/'outputs/REAL_PHYSICS_ACTIVITY.csv');ev=pd.read_csv(P/'outputs/EVENT_SEED_METRICS.csv');valid=pd.read_csv(P/'data_contracts/ALL_LEGAL_SAMPLES.csv');pairs=pd.read_csv(P/'outputs/PAIRED_SEED_AB_COMPARISON.csv');orig=pd.read_csv(P/'data_contracts/B_EVENT_ORIGIN_IDENTITY.csv');ind=jl('outputs/INDEPENDENT_VERIFICATION.json')
 eventsummary=ev.groupby(['task','checkpoint','arm'])[['MAE_mV','RMSE_mV','P95_mV','max_mV','violation_rate']].mean().reset_index();cs('outputs/EVENT_THREE_SEED_MEANS.csv',eventsummary)
 bactive=act[act.arm.eq('B_PHYSICS')];maxgrad=float(bactive.physics_grad_max.max());maxratio=float(bactive.weighted_grad_ratio_max.max());active=int(bactive.active_epochs.sum());last=int(bactive.positive_loss_last_epoch.max())
 md('reports/00_实验结论_通俗版.md',f'''# 独立弱物理分支：真实A/B实验结论
明确决定：`{dec['integration_decision']}`。本轮已经建模、训练并完成消融；不是只有规划，也不是数据完全不可评价。两项任务均为`PHYSICS_DESCRIPTIVE_ONLY_NO_INCREMENT`。

1. 成功建立了独立电流条件化模型：3输入、2层各32SiLU、1217参数；连续改变j可得到V(j)。原双路径TFT和H1/H2/H3保持冻结。两项任务使用同一结构但分别训练，context含义不同，不能混用权重。
2. 物理损失实际计算并参与训练；真实参数梯度经独立有限差分核验。12个物理组共30000epoch中，{active}个epoch同时有非零损失和有效梯度，集中在最初4–8epoch；不是整个训练从未激活。
3. 检查点等权、再对三seed平均的电压误差如下，单位mV（不是P1S的mV-equivalent）：

{table(summ[['task','arm','MAE_mV_mean','RMSE_mV_mean','MAE_mV_seed_sd','violation_rate_mean']])}

任务A的MAE下降{dec['task_decisions'][0]['MAE_gain_percent']:.9f}%，任务B下降{dec['task_decisions'][1]['MAE_gain_percent']:.9f}%；远小于种子波动，未达到冻结的2%及违例改善等成功判据。A的平均RMSE还略增。不是统计显著性结果。
4. 最终普通/物理模型在全部评价采样网格上违例率都为0；没有额外违例改善。有限网格0不能证明连续域处处满足不等式。普通模型自己也在早期学会下降形状。
5. A是已测检查点内留出平台重建（58Train/43Val，7事件），B是下一计划检查点的名义协议条件回放（3时间折各13Val）；B只用历史P1S和计划电流/方向。B按测量完成可用假设进行，不是认证发布时刻的严格在线SOH预测。
6. 本轮提供“真实物理规律＋合法独立模型＋有效梯度＋公平消融但无增量”的可重复证据。可以作为论文物理资格/负面消融结果，不能写成物理约束已提升预测的创新成功。
7. 当前不优先接入TFT或SOH：没有明确独立增益，且辅助教师的时间交叉拟合、未知未来工况、参考j*=1目标/扫描算子匹配和独立SOH验证仍缺资格。没有启动任何融合。
8. 具体失败类别是**约束短暂激活，但普通模型已满足形状且预测无清楚增量**。不是数学导数不可定义，不是全部训练约束未激活，也不是所有数据不足以评价。严格在线未来任务因发布时间未知仍不可认证，和B条件回放结果分开。

24个RunID各执行一次，每次2500epoch，无Validation选epoch、无lambda搜索、无重训补种子、无GPU、无Test、无RUL训练。独立NumPy复算{ind['predictions_checked']}条逐点预测及{ind['evaluation_derivatives_checked']}个导数，通过后才允许导出论文候选。''')
 md('reports/01_协议_数据_时间资格.md',f'''# 固定数据与任务
复用此前已独立核验的7次Development极化缓存。154个middle50候选中105个通过原门槛，402个四窗口合格差商无正斜率违例。本轮不重解析原始极化文件，也不读取2000/2300h或1500h_afterAirAirStart新曲线。未更改面积280cm²、active10均值、原窗口/门槛或参考SOH。

模型输入j为计划/指令电流除以面积，单位A/cm²，而不是未来实测j。实际平台j与计划j的差≤0.005已检查；命令电流域内的连续探针不是新增测量标签。升/降方向显式输入。实际T/露点/压力/化学计量比只用于预定稳定资格与B参考协议支持审查，不作为未来输入；这是近稳态名义协议条件模型，不是消除了全部环境扰动的普适电压模型。

任务A固定Train电流A：{cfg['A_training_setpoints_A']}；留出{cfg['A_holdout_setpoints_A']}。按完整平台隔离，其他窗口/逐秒行未用作训练目标。每个检查点/方向的context是已在Train中的j=0.36平台实测V；重建起点为该次测量完成。43留出平台均被同事件同方向Train包围，4条未包围候选被剔除。不能把这种同事件已测context当成提前已知健康状态。

任务B使用首次在前一极化结束后且目标事件前可获得、T74–76°C/RH39–41%的合法P1S保持率作为起点状态。按原始缓存日历逐项匹配，未使用目标事件电压/未来P1S/未来真实工况/实际futurelead。未来目标电流设定点和扫描方向来自固定测量计划，不将未来实际测量j输入模型。

{table(orig[['checkpoint','previous_checkpoint','origin_calendar','context_source_id','context','calendar_lead_h_AUDIT_ONLY']])}

训练事件扩展为300/600/850→1150，加入1150→1500，再加入1500→1750；Train40/50/63、Val13/13/13。各折Train标签结束早于验证起点。共同电流域由对应Train事件/方向确定；按匹配设定点的历史条件范围加冻结容差筛查目标参考支持，未来条件数值仅用于评价资格，不进入输入或选参。所有剔除原因保留。

极化导出OpHrs不等于耐久OpHrs2，本轮不补造准确耐久时刻；B只报告真实日历提前量，差异很大。模型不使用目标事件名义小时或实际目标日期预测。

**时间身份边界：**B是基于测量完成后原始资料可处理的回顾性条件回放，所有标签/状态均有前后顺序；原文件发布时间/现场处理日志未知，无法认证严格发布时刻在线任务，该严格解释为`FUTURE_POLARIZATION_FORECAST_NOT_EVALUABLE`。不能把条件B回放成绩升级为严格在线或真实SOH提前预测。

7曲线、3未来事件属于一条寿命；492模型预测行不是492独立寿命。所有Development参考已被前期多轮探索使用，本轮为探索性消融。''')
 md('reports/02_模型与物理损失.md',f'''# 模型与量纲
fθ输入[j_design,direction,context]；Linear3→32、SiLU、Linear32→32、SiLU、Linear32→1，总参数1217。A/B架构、输入、数据、初始化、seed42/52/62、全批Adam学习率0.001、2500步与预算一致；不设earlystop，固定末epoch，不用留出标签选择。各任务/fold的X和V标准化均只在Train上拟合。

任务A context为当前已测曲线的训练锚点电压V；任务B为过去起点P1S无量纲保持率。两任务单独训练，绝不能把A的context用于预测尚未测量的未来事件。其他工况在近稳态名义参考协议中作为适用条件，不凭空生成湿度或未来真实T。

Ldata=mean[(Vpred−Vtrue)/sV]²。令xj=(j−μj)/sj，yn=(V−μV)/sV，dV/dj=(sV/sj)∂yn/∂xj，单位V/(A/cm²)=Ω·cm²。

Lphys=mean[relu((sj/sV)dV/dj)]²=mean[relu(∂yn/∂xj)]²，是显式无量纲化形式；原量纲Lphys_raw单位(Ω·cm²)²也逐epoch保存。总损失普通组Ldata，物理组Ldata+1.0×Lphys，λ在任何训练分数之前冻结，未搜索。

对输入j使用create_graph=True，损失梯度传到MLP参数。context、方向与名义协议固定，只约束同状态下跨电流形状，绝不约束健康随时间单调。输出偏置对电流导数数学上无作用，相应梯度为结构性零；其他参数可产生真实非零梯度，不能拿合成非零代替真实训练证据。

每个训练事件/方向在Train限定电流域用17个均匀无标签探针；没有访问验证电压，探针不计真实样本。普通组在相同探针上仅记录诊断，不施加该梯度。最终101点网格用于评价，不用于选择权重。

成功判据在协议冻结：至少2seed违例率或正斜率RMS减少≥50%且基线非零；平均MAE/RMSE不恶化、MAE改善≥2%且至少2seed胜，无事件平均MAE恶化>20%。任何条件不满足即不能宣布promising；不是统计显著性阈值。实际两任务违例改善条件均未满足。''')
 md('reports/03_AB结果与逐事件.md',f'''# 真实预测结果
主指标是先对每个检查点计算MAE，再检查点等权、三seed等权。RMSE/P95/最大值的主汇总也为事件均值；它们不等于合并样本后的RMSE/P95/全局最大，后者另外保存在`POOLED_HOLDOUT_METRICS.csv`。

{table(summ)}

逐事件三seed平均如下；每seed及方向、当前电流域、所有不利项分别见CSV。

{table(eventsummary)}

种子配对的实际MAE差异：

{table(pairs)}

任务A有7事件43留出平台，任务B有3时间折39验证平台。模型输出都是真实平台电压目标，B有时间外推假设，但无认证在线发布时间。不混合A和B成绩，不用A重建误差代替未来SOH误差。未报告假设阈值RUL指标。''')
 md('reports/04_物理梯度与约束激活.md',f'''# 真实作用与不增益
12个物理组、共30000步，{active}步（{active/30000*100:.6f}%）同时有Lphys>1e−16与梯度L2>1e−12。各Run仅前4–8步激活，此后真实物理损失为0，不是NA。A任务物理组共16步，B任务共54步；最晚为epoch{last}。

真实最大物理参数梯度L2={maxgrad:.9g}，加权物理/数据梯度最大比={maxratio:.9g}。初始化真实Train、训练过程逐epoch、最终评价导数均保存。普通模型也只需早期数步就达到网格单调。

独立NumPy解析SiLU前向及导数复现全部权重预测，最大V差{ind['NumPy_prediction_max_abs_difference_V']:.3g}，斜率最大差{ind['NumPy_slope_max_abs_difference_ohm_cm2']:.3g}Ω·cm²；真实Train参数方向有限差分最大差{ind['real_parameter_gradient_fd_max_difference']:.3g}。

最终A/B模型在每个评价网格的违例率、正斜率RMS均为真实0。这既不证明连续域全局单调，也不提供物理组额外优势。MAE变化约0.007%/0.012%，只有微伏以下量级，不构成清楚可重复的物理增益。

因此`PHYSICS_ACTIVATED_DURING_TRAINING`与`PHYSICS_DESCRIPTIVE_ONLY_NO_INCREMENT`同时成立；**不能错误标成整轮PHYSICS_CONSTRAINT_INACTIVE**。本轮失败不是导数或训练不可计算，不是隐藏无数据问题，而是已学会形状的普通模型使该弱约束几乎冗余。不会增大λ、改损失或改采样域救结果。''')
 md('reports/05_TFT_SOH接口与论文边界.md','''# 保留核心双路径TFT；当前不进入融合
现有F2保持原14槽位/48步历史、变量选择网络、原生LSTM路径、因果多尺度卷积路径、历史工况门控及后续静态增强/注意力/24步DeltaV输出。4个原F2 checkpoint做SHA保护，没有载入训练或修改。H1/H2/H3、M2_NC和既有SOH/RUL输出同样只读。

P1训练期辅助监督：不能拿本轮全Development重建模型给早期TFTTrain回填教师。需在接收块前已拟合的模型或时间交叉拟合，并给无物理组同样信息、同容量对照；推理不依赖未来真实工况。当前分支未显示增益，不启动。

P2参考SOH：A需要同事件训练锚点，不能作为未来状态；B使用过去P1S较合法，但仅名义协议条件回放，部分升载训练支持止于j0.9，不能外推j1冒充有测量覆盖。其扫描平台均值与冻结SOH的j1参考观测算子仍需一致性核验；不改变SOH_ref=Vpol(j1)/0.6989533305168152。禁止用同次参考标签构建预测再独立验证，主要对照仍为冻结M2_NC。发布时间未知及样本稀疏问题未解决。

现阶段没有可以直接进入TFT或SOH融合训练的接口。下一轮也不应仅因为网络已经训练成功就启动融合。可保留这次负面物理消融和独立模型代码，以实际证据说明单调约束没有额外收益。

短期P1S不是SOH真值；7事件不是7寿命；曲线插值不是未来SOH；违例少不等于RUL准。没有真实EOL或合格删失，不训练RUL。最后决定RETAIN_DESCRIPTIVE_PHYSICS_ONLY。''')
 md('reports/06_复现与保护核验.md',f'''# 复现、技术记录与保护
基础Git为{cfg['base_commit']}；冻结协议SHA256为{sha(P/'configs/FROZEN_PROTOCOL.json')}。源清单同时保护历史数据身份、H系列合同及4个TFT权重SHA；只读防火墙禁止其他仓库文件读取和所有历史写入。

在正式冻结和任何训练之前，准备程序遇到pandas只读array的赋值错误，改为显式copy修复；当时没有协议/模型训练。随后一次测试因准备未完成而缺配置/探针报错，原失败日志保留在PRE_FREEZE_INCOMPLETE_DATA_TEST_LOG.txt。完成准备后10项合成/自动微分/数据合同测试全过，再启动真实训练。它们不是为改善验证结果而重跑的模型。

24个实际RunID各2500epoch。每500epoch保存同Run恢复点，已完成Run不会重训；本次未触发模型重启/续训。所有终态权重、优化器恢复点、逐epoch日志和不利成绩保留，不只保存最好seed。

独立核验通过{ind['runs_checked']}模型、{ind['predictions_checked']}预测、{ind['evaluation_derivatives_checked']}导数，未执行优化器或重新训练；使用NumPy独立前向/解析导数和有限差分，并核对原平台标签、真实P1S输入身份、Train-only scaler、相同初始状态与相同步数。所有源SHA和冻结核心代码均未改变。

复现环境Python/PyTorch/NumPy等版本另见CPU_ENVIRONMENT.json。先查看README，已完成时只运行核验/绘图；不要删完成标记重复训练。图片均由真实CSV与绘图代码生成，FIGURE_SOURCE_INDEX.csv关联来源、协议和Git基准。导出论文前还需人工检查PNG/PDF以及最终hash封存。''')
 md('README.md','''# Independent weak-physics polarization branch
本轮真实24次CPU A/B训练；单一RealHyFC Development，非新backbone搜索。结果：短暂激活，无清楚物理增量，不启动TFT/SOH融合。

首次执行顺序（已有完整输出时不再运行prepare/train）：
```
python -B scripts/prepare.py
python -B tests/test_autograd.py
python -B scripts/train.py
python -B scripts/analyze.py
python -B scripts/verify_independent.py
python -B scripts/render_figures.py
python -B scripts/write_reports.py
```
依赖Python3.11、PyTorch2.6、NumPy、pandas、matplotlib；CPU、float64、单线程。正式协议冻结后不得改网格、模型、lambda或预算。已完成RunID直接跳过；技术中断仅从同Run检查点继续。

TaskA重建context是同次测量Train锚点电压，TaskB是过去起点P1S保持率；分别训练不可混用。B有测量完成可用假设，不认证严格在线发布。H系列mV-equivalent与本轮真实电压mV不能混称。

论文子目录是存档镜像，脚本来源路径按本研究Phase定位；复现请回到原Phase目录，勿在镜像内启动训练。历史全部只读，所有原始Test文件均不可读。本轮结束后停止。
''');config();print('REPORTS_COMPLETED')
if __name__=='__main__':firewall();run()
