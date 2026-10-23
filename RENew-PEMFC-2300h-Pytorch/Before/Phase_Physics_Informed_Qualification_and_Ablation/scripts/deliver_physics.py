"""Analysis-only literature ledger, independent verification and final reports."""
import json,subprocess,sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from run_qualification import P,ROOT,sha,js,cs,md

def verify():
 cfg=json.loads((P/'configs/FROZEN_PROTOCOL.json').read_text());am=json.loads((P/'configs/CPU_PROBE_DEPENDENCY_AMENDMENT.json').read_text())
 sources=dict(cfg['source_sha256']);sources.update(am['static_dependency_sha256'])
 assert all(sha(ROOT/f)==h for f,h in sources.items())
 assert all(sha(P/'scripts'/f)==h for f,h in cfg['execution_code_sha256'].items())
 assert sha(P/'scripts/resume_cpu_probe.py')==am['resume_code_sha256']
 b=pd.read_csv(P/'inputs/TRAIN_BIN_CONDITIONS.csv');assert len(b)==2103 and b.bin_end_ophrs2_h.max()==1310
 domains=pd.read_csv(P/'outputs/REAL_TRAIN_PHYSICS_DOMAIN_AND_REJECTIONS.csv')
 assert len(domains)==4206 and not domains.independently_certified_domain.any() and domains.physics_residual_dimensionless.isna().all()
 inherited=pd.read_csv(P/'inputs/INHERITED_RAW_DWELL_DOMAIN_COUNTS.csv');assert inherited.holds.sum()==7273 and inherited.relative_quiet.sum()==8
 old=json.loads((ROOT/'Phase_DDP_SoftPhysics_Overnight_Experiment/outputs/TRAIN_PHYSICS_ACTIVITY.json').read_text())
 assert len(old)==9 and sum(r['n_forecasts'] for r in old)==298080 and all(r['physics_loss_dimensionless']==0 and r['violations']==0 for r in old)
 reviewed=[f'Phase_DDP_SoftPhysics_Overnight_Experiment/{name}' for name in ['00_MORNING_SUMMARY_CN.md','02_IMPLEMENTATION_AND_TESTS.md','03_TRAINING_RESULTS.md','04_ABLATION_AND_COMPARISON.md','07_FINAL_RESEARCH_DECISION.md','outputs/QUALIFICATION_DECISION.json']]
 js('outputs/HISTORICAL_FAILURES_RECHECK.json',dict(old_nonnegative_checks=9,old_forecast_values=298080,old_violations=0,
  old_zero_loss_reconfirmed=True,no_old_candidate_retrained=True,reviewed_report_sha256={f:sha(ROOT/f) for f in reviewed}))
 x=json.loads((P/'configs/SYNTHETIC_FIXTURE_NOT_REAL_PARAMETERS.json').read_text());m=json.loads((P/'outputs/SYNTHETIC_EQUATION_AND_GRADIENT_RESULTS.json').read_text())
 E=x['E0_V']+8.31446261815324*x['T_K']/(2*96485.33212)*np.log(x['a_H2']*np.sqrt(x['a_O2'])/x['a_H2O'])
 assert abs(E-m['nernst_fixture_E_V'])<1e-15
 scale=.13680318714138442;u=np.array([0.,4.]);r=np.maximum((.7+scale*u-E)/scale,0)
 assert abs(np.mean(r*r)-m['upper_loss'])<1e-14 and np.allclose(2*r/2,m['upper_output_gradient'],atol=1e-14,rtol=0)
 j=np.array([.2,.5,.8,1.,1.2,1.5]);teacher=x['E0_V']-x['b_V']*(np.log(j)-x['log_j0'])-x['ASR_ohm_cm2']*j+x['B_V']*np.log1p(-j/x['jlim_A_cm2'])
 assert abs(np.mean(((.7-teacher)/scale)**2)-m['amplitude_loss'])<1e-14
 assert np.allclose(2*(.7-teacher)/scale/6,m['amplitude_output_gradient'],atol=1e-14,rtol=0)
 sens=pd.read_csv(P/'outputs/SYNTHETIC_PARAMETER_SENSITIVITY.csv');assert np.linalg.matrix_rank(sens[['E','b','log_j0','ASR','B','jlim']].to_numpy())==5
 d=json.loads((P/'outputs/DECISION.json').read_text());assert d['formal_GPU_runs']==0 and d['actual_physics_loss'] is None and d['optimizer_steps']==0
 result=subprocess.run([sys.executable,'-B','-m','unittest','discover','-s',str(P/'tests'),'-v'],capture_output=True,text=True,encoding='utf-8')
 md('outputs/TEST_LOG.txt',result.stdout+result.stderr);assert result.returncode==0
 js('outputs/INDEPENDENT_VERIFICATION.json',dict(status='PASS',tests=9,Train_bins=2103,candidate_domain_rows=4206,
  formulas_independently_recomputed_with_numpy=True,synthetic_parameter_rank=5,source_integrity=True,
  real_model_gradient_probe='Engineering scalar record; not independently replayed or claimed as physical performance',
  no_qualification_from_synthetic_only=True))

def render():
 data=json.loads((P/'outputs/REAL_TRAIN_DATA_QUALIFICATION.json').read_text());m=json.loads((P/'outputs/SYNTHETIC_EQUATION_AND_GRADIENT_RESULTS.json').read_text())
 probe=json.loads((P/'outputs/REAL_TRAIN_CPU_GRADIENT_CONTRIBUTION.json').read_text());cfg=json.loads((P/'configs/FROZEN_PROTOCOL.json').read_text())
 candidate=pd.DataFrame([
  dict(candidate='C1_STATE_DEPENDENT_NERNST_UPPER_ENVELOPE',law='Thermodynamic reversible potential bounds normal steady generating voltage at the same state',
   equation='E=E0(T)+RT/(2F)*ln(aH2*sqrt(aO2)/aH2O); L=mean(ReLU((Vhat-E)/sV)^2)',units='T K; activities dimensionless; E,Vhat,sV V',
   necessary='Interface T, reactant/product activities, standard state and water phase, normal operating domain; aggregated bound needs an observation operator',
   available='Historical inlet/outlet T and gauge total pressure/RH; no certified interface activities or target-time state',
   unknown_parameters='E0(T) thermodynamic reference and phase identity; no extra electrochemical kinetics required for bound',
   Train_identification='Not established; do not identify an upper bound from the desired data quantile',
   inference='Future actual state unknown; only a certified origin-known global envelope could avoid it, none established',
   model_gradient='Synthetic nonzero beyond state-specific envelope; real certified contribution absent',
   degeneracy='Constant sufficiently low voltage gives zero upper-only loss',
   recovery='Same-state thermodynamic ceiling need not prohibit recovery; cross-state fixed ceiling is not certified',
   qualified_rows=0,real_loss=np.nan,real_activation_fraction=np.nan,status='NOT_QUALIFIED'),
  dict(candidate='C2_QUANTITATIVE_ELECTROCHEMICAL_AMPLITUDE_RESIDUAL',law='Kinetic, ohmic and mass-transport losses under certified common quasi-steady state',
   equation='Vphys=E-b ln(j/j0)-ASR*j+B ln(1-j/jlim); L=mean(((Vhat-Vphys)/sV)^2)',units='j,j0,jlim A/cm2; b,B,E V; ASR ohm cm2; log arguments dimensionless',
   necessary='Certified reaction state, quasi-steady multi-load support, Tafel regime, 0<j<jlim; identifiable kinetic/resistance/transport parameters',
   available='Historical j and external conditions; no certified joint physical state/operator/parameter identity',
   unknown_parameters='E,j0,b or alpha,ASR,B,jlim; possible reduced effective intercept requires external validation',
   Train_identification='E/logj0 symmetry; narrow-current ohmic/concentration mixing; certified raw dwells0',
   inference='Direct future residual missing future state; Train-only privileged teacher potentially legal only with independent physics, purged crossfit and matched nonphysical control',
   model_gradient='Synthetic 2r/n to standardized DeltaV; real physical teacher not available',
   degeneracy='Free teacher/students can co-adapt; teacher=true target duplicates MSE; correlated parameters absorb residual',
   recovery='Fixed materials ageing law not certified; reversible hydration/thermal changes may change effective parameters',
   qualified_rows=0,real_loss=np.nan,real_activation_fraction=np.nan,status='NOT_QUALIFIED')])
 cs('outputs/CANDIDATE_TEN_QUESTION_MATRIX.csv',candidate)
 ledger=pd.DataFrame([
  dict(source='DOE/NETL Fuel Cell Handbook 7th edition (2004)',url='https://netl.doe.gov/sites/default/files/netl-file/FCHandbook7.pdf',
   location='PDF pp55-57 and64-70 / printed2-3 to2-6 and2-12 to2-18',verification='Official full PDF accessed; Nernst general equation and thermodynamic ceiling; activation/ohmic/transport sections inspected',
   claim='Equation forms and physical prerequisites only; no RealHyFC parameter values or training gains'),
  dict(source='Wei et al., Journal of Power Sources677 (2026)239941',url='https://doi.org/10.1016/j.jpowsour.2026.239941',
   location='Local original PDF page6 equations12-15',verification='Local full PDF SHA'+cfg['local_physics_pdf_sha256']+'; page6 text and rendered image checked',
   claim='Stack/per-cell distinction; kinetic/ohmic/concentration decomposition. Equation15 ageing assumptions NOT transferred to RealHyFC'),
  dict(source='Fensterle et al., RealHyFC source article',url='https://doi.org/10.1016/j.ijhydene.2026.156060',
   location='Publisher abstract, not full text (full-page request failed)',verification='Publisher search abstract retrieved; no detailed formula/nominal-table claim derived from inaccessible full text',
   claim='Pressure, stoichiometry and spatial state matter; cannot certify individual Train dwells'),
  dict(source='RealHyFC underlying dataset record',url='https://zenodo.org/records/20715542',location='Public dataset description only',
   verification='Description accessed; no raw data downloads or late-life numerical tables opened',claim='Nominal polarization and condition sensitivity data exist; availability does not establish cached event alignment')])
 cs('inputs/LITERATURE_EVIDENCE_LEDGER.csv',ledger)
 plt.rcParams.update({'font.family':'Arial','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'savefig.dpi':300})
 def save(fig,name):
  fig.savefig(P/'figures'/f'{name}.png',bbox_inches='tight');fig.savefig(P/'figures'/f'{name}.pdf',bbox_inches='tight');plt.close(fig)
 fig,axs=plt.subplots(1,2,figsize=(8.3,3.3))
 labels=['Train bins','Near j*=1.0','Physically certified'];nums=[2103,468,0]
 axs[0].bar(range(3),nums,color=['#4C78A8','#F58518','#777777']);axs[0].set_xticks(range(3),labels,fontsize=8);axs[0].set_ylabel('Bins');axs[0].set_ylim(0,2450)
 for i,n in enumerate(nums):axs[0].text(i,n+45,str(n),ha='center')
 labels=['Raw holds','Statistically quiet','Physically certified'];nums=[7273,8,0]
 axs[1].bar(range(3),nums,color=['#4C78A8','#F58518','#777777']);axs[1].set_xticks(range(3),labels,fontsize=8);axs[1].set_ylabel('Inherited Train dwell records');axs[1].set_ylim(0,8450)
 for i,n in enumerate(nums):axs[1].text(i,n+150,str(n),ha='center')
 fig.suptitle('Measured coverage does not establish physical state qualification');fig.tight_layout();save(fig,'physical_domain_coverage')
 fig,axs=plt.subplots(1,2,figsize=(8.1,3.1));s=m['fixture_jacobian_singular_values']
 axs[0].semilogy(range(1,7),s,'o-',color='#4C78A8');axs[0].set_xlabel('Singular value index');axs[0].set_ylabel('Synthetic Jacobian singular value');axs[0].set_title('6 parameters, rank5: exact ambiguity',fontsize=10)
 axs[1].bar(['Upper bound\nviolating fixture','Amplitude\nfixture'],[np.linalg.norm(m['upper_output_gradient']),np.linalg.norm(m['amplitude_output_gradient'])],color=['#4C78A8','#F58518']);axs[1].set_ylabel('Output-gradient L2 (synthetic)');axs[1].set_title('Mathematical gradients, no RealHyFC gain',fontsize=10)
 fig.tight_layout();save(fig,'synthetic_gradient_and_identifiability')
 links='[DOE/NETL手册](https://netl.doe.gov/sites/default/files/netl-file/FCHandbook7.pdf)、[Wei等原论文](https://doi.org/10.1016/j.jpowsour.2026.239941)'
 texts={
 '00_软物理研究总结.md':f'''# 实际结果
 最终状态 `PHYSICS_ROUTE_NOT_QUALIFIED`。实际深入审查两个不同候选：状态相关Nernst热力学上界；包含活化、欧姆和浓差损失的定量幅值残差。没有重新执行旧非负或极化符号训练。
 2103个Train半小时窗中468个平均电流接近1.0，物理认证样本0；沿用Raw审查的7273个保持片段，统计安静8个、证据子域安静0个、独立物理认证0个。本轮重新核对这些缓存及来源，没有重新解析Raw。旧9次非负检查共298080值、全部零惩罚也已独立重新计数，保留原失败结论。
 两候选合成前后向、解析/数值梯度通过；9项测试通过。实际8个Train起点、192输出的CPU梯度贡献检查已运行，无参数更新。由于物理域为空，合法新增物理梯度为0；真实物理残差、激活比例及损失比是NA，不能称为已计算损失为0。
 **未启动正式物理训练，也未启动小GPU探针。** 数据/参数/测量算子门槛在GPU之前失败，合成成功不能解除。预测改善及物理一致性改善均不可评价。
 ''',
 '01_物理文献与方程依据.md':f'''# 核实依据
 {links}支持本轮候选的条件方程。NETL的可逆电势取决于标准态、温度和反应物/产物活动度；电压损失分为活化、欧姆、传质。采用这些规律之前必须认证对应状态，不能套固定1.229V上限。
 本地Wei原PDF p6式12–14已文本提取并视觉核对，SHA见协议。stack关系需除ncell后作用于单体；n取氢氧反应的电子计量数2，无量纲。论文分压写法在本实现改用相对于标准态的无量纲活动度，避免有单位量直接取log。式15的参数指数老化是外部模型假设，本轮不移植为每时刻强制单调。
 NETL通用传质近似的单独推导有适用前提，不能把简化方程当作PEMFC所有动态状态的精确定律。Tafel只适用于相应动力学区域，不能外推到j=0、启动或异常供给。
 RealHyFC发布者摘要说明压强和计量比影响空间电化学性能，数据记录说明nominal极化和条件敏感性数据存在；它们不证明每个缓存窗已有准稳态资格。[项目原论文](https://doi.org/10.1016/j.ijhydene.2026.156060)、[数据记录](https://zenodo.org/records/20715542)。原论文全文页面请求失败，只使用可核查摘要，不声称重读其详细条件表。
 每个文献的读取程度、位置及支持范围见inputs/LITERATURE_EVIDENCE_LEDGER.csv。未下载Raw或打开新的Test数值。原文页面仅留本地忽略的runtime供核查，不将整页版权图纳入Git成果。
 ''',
 '02_候选损失与适用条件.md':'''# 两候选
 C1：E=E0(T)+RT/(2F)ln(aH2 sqrt(aO2)/aH2O)，候选L1=mean(ReLU((Vhat−E)/sV)²)。T为K，R为J/(mol K)，F为C/mol，活动度无量纲，E和sV为V。真实域需认证正常发电、反应相与界面状态。它是条件上界，不是缩窄数据电压分位数。过低常数输出仍给0上界损失，所以单独不充分。
 C2：Vphys=E−b ln(j/j0)−ASR j+B ln(1−j/jlim)，候选L2=mean(((Vhat−Vphys)/sV)²)。j/j0/jlim为A/cm²，ASR为Ω·cm²，b/B为V；Rc×面积对应ASR。需要参数身份、0<j<jlim、Tafel适用域、共同准稳态及反应状态。
 两项都按原Train输出尺度sV=0.13680318714138442V无量纲化。若资格通过才可注册Ltotal=Ldata+lambda Lphysics；本次正式lambda和GPU预算不实例化，预算0，无Validation搜索。
 不新增head或参数。C2若以后形成独立物理教师，可只在Train使用额外目标工况作监督，不必送入学生推理；但必须先证明物理状态和参数、按时间crossfit/purge，并设置同额外信息与容量的非物理对照。当前缺的是这些科学资格，不能仅靠训练接口隔离宣称教师成立。
 十项问题逐候选回答见CANDIDATE_TEN_QUESTION_MATRIX.csv；未给未来真实j/T/RH/P作预测输入。
 ''',
 '03_RealHyFC物理数据资格.md':'''# 真正Train检查
 读取阶段A已冻结Development快照的前2103行，沿用F2 [0,2103)边界、最后bin结束1310h；未用Validation选参数。保存Train统计、4206条两候选逐窗资格及剔除原因。
 468个窗j均值在[0.95,1.05]仅为描述，认证0。入口温度、表压、RH、气体流量和计量比有数值，反应界面活动度、膜水状态、相身份和物理观测算子缺失。表压需转绝压、总压需转反应分压、入口RH不等于膜水或界面活动度；不能无证据默认。
 11个Train窗阳极入口RH平均值>100%，未clip/删值。M2原Raw异常记录及N1重新聚合一致的结论保持，传感器/计算解释仍未知。本轮数字是窗数，旧313495是Raw记录数，二者不混用。
 Raw域统计独立复核7273/8/0/0（保持/统计安静/证据安静/认证）；未重新配对或降低稳定性门槛。保持片段不等于物理准稳态。完整原F2 Train Raw覆盖身份也仍有缺口。
 半小时非线性方程不封闭：合成fixture中mean(V(j))−V(mean(j))=11.290273mV；这是反例，不是实测RealHyFC偏差。认证稳态及聚合算子必须由独立测量证明。
 ''',
 '04_损失激活与梯度实验.md':f'''# 已运行的检查
 合成Nernst E={m['nernst_fixture_E_V']:.12f}V，仅测试夹具。上界输出梯度[0,{m['upper_output_gradient'][1]:.12f}]，与有限差分最大差{m['upper_finite_difference_max_error']:.3g}；幅值残差有限差分最大差{m['amplitude_fd_max_error']:.3g}，两项解析梯度也通过。
 六参数合成Jacobian秩5，最小奇异值{m['fixture_jacobian_singular_values'][-1]:.3g}。E'=E−b c、log(j0')=log(j0)+c给同一V，最大差{m['E_logj0_symmetry_max_difference_V']:.3g}V。这是结构反例，不是拟合了RealHyFC参数。
 真实CPU探针加载原H-F2 Seed42，使用8个合法Train起点，192输出。数据MSE={probe['data_MSE_dimensionless']:.12f}、数据参数梯度L2={probe['data_gradient_L2']:.12f}，无optimizer更新。认证域为空时工程门控贡献梯度0。真实物理损失与激活比例为NA，不是算出“残差0”或真实不激活。
 物理/数据量级比同为NA，无法计算就不进行lambda选择。没有GPU训练或有效物理梯度的实证。合成fixture与真实结果文件分开保存。
 CPU导入首次被只读访问保护阻断：漏登记原模型读取的静态特征YAML。在模型构建及前向之前停止；保留原冻结代码与失败记录，登记该文件精确SHA后只恢复CPU一步。已完成的数据和公式检查没有重复，科研规则未改。
 ''',
 '05_模型消融结果_若合格.md':'''# 未启动正式物理训练
 合格候选数0。未启动正式GPU消融，未启动小GPU试跑，未生成权重或不存在的性能表。
 物理一致性改善：NA。预测RMSE/MAE改善：NA。三Seed配对改善：NA。新增模型参数0；额外训练成本0。本轮只有CPU审查及无更新的梯度检查，成本日志不能称模型训练成本。
 未来仅在数据资格齐全后才冻结唯一候选、lambda、预算、停止规则，并采用原F2及42/52/62配对，无约束/物理约束及必要同信息非物理对照。不在本轮自动补训。
 ''',
 '06_最终物理创新可行性判断.md':'''# 判定与论文主张
 PHYSICS_ROUTE_NOT_QUALIFIED。C1停在界面状态、未来状态及观测算子资格；C2停在共同物理状态、参数辨识及独立教师资格。有效合成梯度不能补足这些条件。
 建议停止当前数据上强行接入Nernst/定量残差，以及旧非负电压、未认证极化斜率、每时刻全局SOH/P1S单调、普通数据教师改名PINN。并非否定电化学规律，而是当前任务信息不足。
 值得保留的条件方向是先做纯物理模型辨识，利用独立HFR/ASR诊断约束欧姆项，再在受控多负载/温湿压、明确扫描与稳定化实验上检验动力学和传质。之后才重新审查Train-only物理监督；本轮不执行新的辨识或实验。
 目前最有证据的论文主线：固定历史输入下原TFT与因果多尺度双路径的多步动态电压预测，已有H三Seed整模型Development信号；门控单独稳定增益未成立，96h健康门控失败，不能称其正式独立创新。另有96h P1S Ridge三折轻量短期代理预测信号，可独立报告。两者均为反复探索的Development证据，不宣称独立Test确认或真实SOH/RUL。
 当前这轮的事件与物理资格证据适合支撑适用性边界，不是新增准确率或PINN性能创新。
 ''',
 '07_下一阶段研究建议.md':'''# 缺口优先级
 先补同步参考测量：极化精确durability时钟、真实测量及结果发布、同条件参考事件；再补反应物绝对/局部分压或活动度、水相、膜水/热状态、RH校准及可核验稳定化/扫描/干预身份。
 优先用独立HFR/EIS约束ASR，并在受控Train多负载极化曲线中检验参数rank、区间预测和不确定性。不要用原动态训练电压同时充当物理定义和独立验证。
 SOH还需可验证的工况标准化、足够合法参考事件以及可逆恢复分离；RUL还需失效准则、真正失效或合法删失、长期健康外推验证。记录终止不是失效时刻，未开展97%阈值RUL。
 本轮两阶段完成后停止，不自动扩展窗口、模型、未来Test或RUL。阶段A/B分别独立提交与正常推送main，不force。
 '''}
 for name,text in texts.items():md(name,text)
 md('README.md','# 物理资格审查\n\n已执行两候选Train核查、合成方程/梯度、真实CPU无更新探针。原run在静态依赖访问前停止，使用冻结amendment的resume_cpu_probe完成。最终无合格候选，无GPU训练。复核入口scripts/deliver_physics.py verify；完整结果outputs/DECISION.json。')
 md('reports/README.md','阶段根目录00–07为最终报告。inputs保存Train来源与文献证据，outputs保存真实审查及合成结果，两者不混作物理性能成绩。')

def seal():
 assert json.loads((P/'outputs/INDEPENDENT_VERIFICATION.json').read_text())['status']=='PASS'
 js('outputs/FINAL_INTEGRITY.json',dict(status='PASS',tests=9,source_identity=True,real_Train_bins=2103,
  real_qualified_physical_domains=0,formal_GPU_runs=0,small_GPU_probes=0,optimizer_steps=0,
  figures_visually_checked=2,no_Test=True,negative_history_preserved=True))
 files={str(f.relative_to(P)).replace('\\','/'):sha(f) for f in P.rglob('*') if f.is_file() and 'runtime' not in f.relative_to(P).parts and f.name not in ['ARTIFACT_MANIFEST.json','GIT_RECEIPT.local.json']}
 js('outputs/ARTIFACT_MANIFEST.json',dict(files=files));print('SEALED',len(files),'files')
if __name__=='__main__':{'verify':verify,'render':render,'seal':seal}[sys.argv[1]]()
