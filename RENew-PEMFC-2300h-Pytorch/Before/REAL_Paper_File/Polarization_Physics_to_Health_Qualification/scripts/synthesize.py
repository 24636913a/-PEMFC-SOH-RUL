"""Interpret real outputs, keep descriptive curve fitting distinct from health training."""
from common import *
def markdown_table(d):
 def fmt(v):return f'{v:.6f}' if isinstance(v,(float,np.floating)) else str(v)
 return '\n'.join(['| '+' | '.join(map(str,d.columns))+' |','| '+' | '.join(['---']*len(d.columns))+' |']+['| '+' | '.join(fmt(v).replace('|',' / ') for v in r)+' |' for r in d.itertuples(index=False,name=None)])
def run():
 cfg=config();dw=pd.read_csv(P/'outputs/POLARIZATION_DWELLS.csv');s=pd.read_csv(P/'outputs/NUMERICAL_SLOPES.csv');h=pd.read_csv(P/'outputs/HYSTERESIS.csv');fit=pd.read_csv(P/'outputs/PARAMETER_IDENTIFIABILITY.csv');domain=pd.read_csv(P/'outputs/COMPARABLE_CURRENT_DOMAIN.csv')
 h2=pd.read_csv(P/'outputs/H2_FOLD_METRICS.csv');re=pd.read_csv(P/'outputs/FROZEN_H1_H3_REPLAY.csv');av=pd.read_csv(P/'outputs/AVAILABILITY_COUNTS.csv')
 rows=[]
 for r in re.to_dict('records'):rows.append({'H':int(r['horizon'][1:]),'fold':r['fold'],'method':'RIDGE96' if r['method']=='RIDGE_96H' else 'PERSISTENCE',**{k:r[k] for k in ['n','MAE_mVeq','RMSE_mVeq','P95_mVeq','max_mVeq']},'identity':'ORIGINAL_FROZEN_RESULTS'})
 for r in h2.to_dict('records'):rows.append({'H':2,**r,'identity':'NEW_PREREGISTERED_H2'})
 allm=pd.DataFrame(rows);cs('outputs/H1_H2_H3_FOLD_METRICS.csv',allm);eq=allm.groupby(['H','method'])[['MAE_mVeq','RMSE_mVeq','P95_mVeq','max_mVeq']].mean().reset_index();cs('outputs/H1_H2_H3_EQUAL_MEANS.csv',eq)
 compare=[]
 for H in [1,2,3]:
  a=eq[(eq.H==H)&eq.method.eq('PERSISTENCE')].iloc[0];b=eq[(eq.H==H)&eq.method.eq('RIDGE96')].iloc[0];p=allm[(allm.H==H)&allm.method.eq('PERSISTENCE')].set_index('fold');r=allm[(allm.H==H)&allm.method.eq('RIDGE96')].set_index('fold')
  compare.append({'H':H,'Persistence_MAE_mVeq':float(a.MAE_mVeq),'Ridge_MAE_mVeq':float(b.MAE_mVeq),'improvement_percent':float((a.MAE_mVeq-b.MAE_mVeq)/a.MAE_mVeq*100),'fold_MAE_wins':int((r.MAE_mVeq<p.MAE_mVeq).sum()),'max_error_worse_folds':'|'.join(r.index[r.max_mVeq>p.max_mVeq].tolist())})
 cs('outputs/SHORT_HORIZON_COMPARISON.csv',pd.DataFrame(compare))
 # Loss and gradient on the actual health predictor are NA: it has no current-input coordinate or cross-current outputs.
 gates=[]
 for candidate,diag,reason in [
 ('A_shape','PHYSICS_DESCRIPTIVE_ONLY','Real branch shape verified; dV/dj of scalar fixedj P1S output is undefined. Monotonicity in futurecycle is a different assertion contradicted by recovery.'),
 ('B_Rdiff','PHYSICS_DESCRIPTIVE_ONLY','Composite differential slope, branch/state dependent. No certifiedHFR/ASR or validated independent health information; stop as health-input candidate.'),
 ('C_semiempirical','PHYSICS_DESCRIPTIVE_ONLY','Only1branch passes local numerical criterion; no stable longitudinal physical parameter series; fittedpolcurve is not scalarhealth model.'),
 ('P_A_calibration','PHYSICS_INTERFACE_NOT_QUALIFIED','Release timesunknown; only3eligible referenceevents; no new generalization test or new calibration fit. FrozenM2_NC remains control.'),
 ('P_B_health_loss','PHYSICS_INTERFACE_NOT_QUALIFIED','No meaningful nontrivial crosscurrent loss on existing scalarfixedj prediction, no effective healthparameter gradient or samecapacity physical experiment.')]:
  gates.append({'candidate':candidate,'qualification':diag,'reason':reason,'real_health_loss':'NA','real_health_gradient':'NA','trained_physics_improvement':'NA','GPU_training_qualified':False,'ordinary_smoothing_equivalent':'NA_NOT_DEFINED' if candidate!='P_B_health_loss' else 'NO_VALID_LOSS_TO_COMPARE','same_information_capacity_control':'NOT_RUN_NO_QUALIFIED_PHYSICAL_LOSS'})
 cs('outputs/PHYSICS_QUALIFICATION_GATES.csv',pd.DataFrame(gates))
 evidence=[]
 for candidate,law,unit,observed,scope,identified,independence in [
  ('A_shape','dV/dj<=0','V/(A/cm2)=ohm cm2','measuredj,active10V,direction,T,dewpoint,pressure,stoichiometry','Only screenednearsteadypolarbranch, not dynamicvoltage or fixedjhealth','Measuredfinite differences; no separate parameter','Crosscurrentlaw differs from time smoothing; unavailable outputdimension in healthmodel'),
  ('B_Rdiff','Rdiff=-DeltaV/Deltaj','ohm cm2','same as A; no independently certifiedHFR','Screenedmatchedcurrentbranch, state and windowdependent','Composite slope numerically computed, mechanisms notseparable','No validated incrementalhealth information beyondVref; diagnosticcenteredshape is not predictiveincrement'),
  ('C_semiempirical','V=E-b*ln(j/1)-R*j+c*ln(1-j/L)','V; R ohm cm2; j,L A/cm2','positivecurrentstablepolarpoints; no independentcomponentlossmeasurements','Predeclaredpositivej domain only; no extrapolatedOCV/jlimcertification','1/5 fittedbranchpasseslocalcriterion; no full7eventparametertrajectory','Mechanisticinspired curvefamily, not ordinarypolynomial; neither establishedhealthloss nor PINN')]:
  evidence.append({'candidate':candidate,'physical_relation':law,'dimensions':unit,'observables':observed,'applicable_conditions':scope,'identifiability':identified,'independence_from_statistics':independence,'meaning_on_actual_fixedj_health_output':False,'origin_parameter_release_certified':False,'nontrivial_health_loss_computable':False,'health_model_gradient_effective':False,'health_loss_value':'NA','health_gradient_value':'NA','same_information_same_capacity_trained_control':'NA_NO_QUALIFIED_TREATMENT','future_target_used_as_input':False,'forces_health_time_monotonicity':False,'training_qualified':False})
 cs('outputs/CANDIDATE_EVIDENCE_MATRIX.csv',pd.DataFrame(evidence))
 # Compare actual measured conditions at identical setpoints, keeping branch identity.
 matched=[];main=dw[dw.window.eq('middle50')&dw.branch.isin(['descending','ascending'])]
 for (branch,current),g in main.groupby(['branch','set_current_A']):
  for col in [c for c in g if c.endswith('_mean')]:
   matched.append({'branch':branch,'set_current_A':current,'variable':col,'checkpoint_count':g.checkpoint.nunique(),'min_across_checkpoint_means':float(g[col].min()),'max_across_checkpoint_means':float(g[col].max()),'span_across_checkpoint_means':float(g[col].max()-g[col].min()),'actualRH_or_composition_certified':False,'not_independent_life_samples':True})
 cs('outputs/CONDITIONS_AT_MATCHED_SETPOINTS.csv',pd.DataFrame(matched))
 missing=[]
 for k in CHECKPOINTS:
  raw=pd.read_csv(P/f'outputs/raw_selected/{k}.csv',float_precision='round_trip')
  for col in raw:missing.append({'checkpoint':k,'channel':col,'n':len(raw),'missing':int(raw[col].isna().sum())})
 cs('outputs/RAW_SELECTED_CHANNEL_MISSINGNESS.csv',pd.DataFrame(missing))
 # Genuine scalar-output support diagnosis, never labeled as an observed physics-loss gradient.
 meas=read('measurement');m=meas[(meas.stable_window=='middle_50')&meas.cycle_interval_id.isin(read('identity').cycle_interval_id)]
 j=m.current_density.to_numpy();X=np.column_stack([np.ones(len(j)),-j]);rank=int(np.linalg.matrix_rank(X));js('outputs/FIXED_CURRENT_IDENTIFIABILITY.json',{'n':len(j),'j_min':float(j.min()),'j_max':float(j.max()),'E_minus_Rj_design_rank':rank,'parameters':2,'real_health_loss':None,'real_health_gradient':None,'meaning':'rank1 at onecurrent cannot separate Eeff andReff; crosscurrent coordinate absent'})
 # Real polar fitting has computable parameter derivatives; these are explicitly not H1/H3 gradients.
 diag=[]
 for q in sorted((P/'models').glob('*.json')):
  if q.name.startswith('H2_'):continue
  f=jl(q.relative_to(P).as_posix())
  if f['status']!='FITTED_DESCRIPTIVE':continue
  z=pd.read_csv(P/f"outputs/fits/{f['checkpoint']}_{f['branch']}_residuals.csv");x=z.j_A_cm2.to_numpy();p=f['parameters'];E,b,R,c,L=[p[n] for n in ['Eeff_V','b_V','Reff_ohm_cm2','c_V','jlim_A_cm2']]
  # perturb b by1mV for a nontrivial real-data curve-loss diagnostic, not optimization.
  residual=E-(b+.001)*np.log(x)-R*x+c*np.log1p(-x/L)-z.observed_V.to_numpy();grad=float(2*np.mean(residual*(-np.log(x))))
  diag.append({'checkpoint':f['checkpoint'],'branch':f['branch'],'diagnostic':'Realpolarcurve_MSE_at_b_plus_0.001V_NOT_health_training','MSE_V2':float(np.mean(residual**2)),'dMSE_db_V':grad,'health_gradient':'NA','parameter_identifiable_local':f['identifiable']})
 cs('outputs/POLAR_FIT_REAL_DATA_GRADIENT_DIAGNOSTIC.csv',pd.DataFrame(diag))
 # Compare original M2_NC past-anchor legality independently of physical publication uncertainty.
 cal=sj('calibration');cs('outputs/M2_NC_FROZEN_PARAMETERS.csv',pd.DataFrame([{'nominal_h':r['nominal_h'],'horizon':r['horizon'],'a':r['a'],'b':r['b'],'training_events':str(r['training_events']),'past_only':r['past_only'],'changed':False} for r in cal]))
 profile=pd.read_csv(P/'outputs/JLIM_PROFILE.csv');profile_summary=[]
 for (k,b),g in profile.groupby(['checkpoint','branch']):
  profile_summary.append({'checkpoint':k,'branch':b,'profile_RMSE_min_mV':float(g.RMSE_mV.min()),'profile_RMSE_max_mV':float(g.RMSE_mV.max()),'tested_jlim_min':float(g.fixed_jlim.min()),'tested_jlim_max':float(g.fixed_jlim.max()),'not_confidence_interval':True})
 cs('outputs/PROFILE_SENSITIVITY_SUMMARY.csv',pd.DataFrame(profile_summary))
 outcome={'overall':'PHYSICS_INTERFACE_NOT_QUALIFIED','polarization_diagnostics':'PHYSICS_DESCRIPTIVE_ONLY','PHYSICS_LOSS_TRAINING_QUALIFIED':False,'candidate_decisions':gates,
 'H2_threefold_evaluable':True,'short_horizon_results':compare,'H1_H3_refits':0,'new_H2_fits':15,'new_physics_health_training':0,'Test_read':False,'GPU_runs':0,'new_SOH_RUL_training':0,
 'three_SOHevents_optimized':False,'SOH_ref_changed':False,'EOL_available':False,'next':'Retain frozenH1 main,H3 auxiliary;H2 optional exploratory bridge. Do not trainphysicshealth loss on these outputs. Futurequalification needs matchedcrosscurrent observableoutputs, pastpublishedparamavailability, identifiablephysicalparams and moreindependenthealthreference support.'}
 js('outputs/RESEARCH_DECISION.json',outcome)
 stable=dw[dw.window.eq('middle50')&dw.stable_domain];fitted=fit[fit.status.eq('FITTED_DESCRIPTIVE')];maxsens=pd.read_csv(P/'outputs/SLOPE_WINDOW_SENSITIVITY.csv');maxrange=float((maxsens.slope_max-maxsens.slope_min).max())
 table=markdown_table(pd.DataFrame(compare))
 md('reports/01_研究结论_通俗版.md',f'''# 研究结论（探索性Development）
本轮没有找到可直接训练现有H1/H3健康预测器的合格物理损失，结论为`PHYSICS_INTERFACE_NOT_QUALIFIED`。真实极化诊断为`PHYSICS_DESCRIPTIVE_ONLY`，没有候选获得`PHYSICS_LOSS_TRAINING_QUALIFIED`。

H1主任务、H3辅助任务保持冻结。新增H2按预注册规则一次完成，验证样本16/5/26；没有重训H1/H3，也没有启动GPU。

{table}

H2三折MAE都改善，但fold_1最大误差从1.147497增至1.202287mV-equivalent；不能称所有误差均改善。H1/H3历史尾部及恢复阶段局限保持原结论。三个跨度各有自己的样本，不把上表跨跨度误差差异解释为严格同样本因果效应。

7次极化测量能描述电流—电压关系、滞后与综合斜率；不能自动转为固定电流单输出的跨电流梯度。只有3个可评价参考SOH事件、发布时间未知，也不足以证实新的物理增强SOH校准。没有真实EOL，不进行RUL精度训练。''')
 md('reports/02_真实极化曲线数据资格.md',f'''# 数据资格
精确读取7个白名单Development原始极化文件，共{sum(jl(f'outputs/dwells/{k}_metadata.json')['rows'] for k in CHECKPOINTS)}行；未读取2000h/2300h或混合Test文件。每行原始rowID、DateTime、导出OpHrs和CalHrs在`outputs/raw_selected`保留。来源SHA见`data_contracts/SOURCE_SHA256.csv`。

导出OpHrs不是耐久试验OpHrs2，实际极化OpHrs2未知，保持NA，不能用文件名或前驱P1S时间补造。1750h文件导出OpHrs超过2000并不改变其冻结Development事件身份。测量结束只是可用时间下界，结果发布时间未知。

10个有效单体通道U1/2/3/4/5/7/8/9/10/11均值用于本轮曲线诊断；U6缺失。不替换冻结参考SOH：Vpol(j*=1)/0.6989533305168152。保留导出平均单体电压，差异见资格表，不能假定原因。

实际T、露点、出口表压、化学计量比在每个窗口逐项保存mean/min/max，`CONDITIONS_AT_MATCHED_SETPOINTS.csv`按相同设定电流和方向比较跨事件均值及跨度；原始选定通道缺失数另表保存。没有实测RH字段，不能把露点直接称为湿度百分比或用名义RH证明条件相同。气体组成/流量未由本轮通道独立认证；平台筛查通过仅证明预设的局部稳定性要求，不是全部环境条件严格等价。7事件同一耐久轨迹，不是7条寿命。''')
 md('reports/03_极化曲线及物理特征分析.md',f'''# 曲线、扫描和局部斜率
按冻结协议识别稳定设定电流平台；调理与返回段单列。首次504A端点属于降载转折，不能沿用旧描述表把它当成完整升载支路。0A转折不参与对数模型。0.36–1.80A/cm²域的平台约5分钟，低电流约2分钟仅描述。

middle50域内154个平台中{len(stable)}通过筛查，49个平台因T跨度>2°C失败；这项预先冻结筛查没有因结果放宽。跨7事件共同稳定设定电流：降载{domain[domain.all7_stable&domain.branch.eq('descending')].set_current_A.tolist()}A；升载{domain[domain.all7_stable&domain.branch.eq('ascending')].set_current_A.tolist()}A。共同区间内未测设定点不视为额外观测。

四种窗口共有{len(s)}个合格相邻稳定点差商，正斜率违例{int(s.monotonic_violation.sum())}。Rdiff范围{s.Rdiff_ohm_cm2.min():.6f}–{s.Rdiff_ohm_cm2.max():.6f}Ω·cm²。相邻稳定点不一定是相邻全部原始设定点，低/高端身份全部列出。最大同点对窗口斜率跨度{maxrange:.6f}Ω·cm²。

升载减降载电压差范围{h.up_minus_down_mV.min():.6f}–{h.up_minus_down_mV.max():.6f}mV，说明顺序/状态不可忽略。时间SD传播只是相关时间序列的描述性波动尺度，窗口敏感性不是置信区间。

`J1_CENTERED_SHAPES.csv`在真实稳定点包围j=1时做诊断插值并去除电压平移，展示形状变化；它不是新SOH、独立寿命标签或物理特征的预测增量验证。Rdiff包含多种损失及状态扰动，不能称膜内阻、真实ASR或HFR。没有足够证据证明超出直接参考电压的独立健康增量，B在此停止。''')
 md('reports/04_物理参数可辨识性.md',f'''# 半经验候选C的辨识诊断
固定公式V=Eeff−b ln(j/jref)−Reff j+c ln(1−j/jlim)，jref=1A/cm²，jlim>实测最大j；所有三项损失保留。Eeff为有效截距而非经认证可逆电势，b,c单位V，Reff单位Ω·cm²，jlim单位A/cm²。简化Tafel表达只考察正电流域，不扩展到OCV；有效参数不得未经证明归因于不可逆老化。自由j0与E不可分，故未引入j0。

14个事件×方向组合中，{len(fitted)}个满足至少8稳定点并实际拟合；其余9个NA。每个3固定初值+6固定jlim剖面，边界、残差、Jacobian列归一化奇异值和参数跨度全部保存。RMSE范围{fitted.RMSE_mV.min():.6f}–{fitted.RMSE_mV.max():.6f}mV；低残差不代表机制被确认。

仅BoL升载通过本轮局部数值筛查（{int(fitted.identifiable.sum())}/{len(fitted)}）；其余4个存在Reff近下边界。通过者也只有一次测量支路，无法形成跨7事件、可在线转移的物理参数序列。归一化Jacobian条件数约{fitted.jacobian_normalized_condition.min():.1f}–{fitted.jacobian_normalized_condition.max():.1f}，表明参数相关性仍需谨慎；预设通过规则不是统计可信区间。

剖面jlim在1.85–10A/cm²固定网格变化，不能将未实测极限电流认证为真实传质极限；不更换公式救结果。实际极化MSE的非平凡参数梯度可在CSV复算，但它不是H1/H3预测器的物理损失或梯度。''')
 md('reports/05_软物理约束数学定义.md','''# 损失定义及适用域
A：在同一近稳态支路上，有限差分s=(V(j2)−V(j1))/(j2−j1)，可定义Lshape=mean[max(0,s)]²用于具有可比较跨电流输出的曲线模型。单位(Ω·cm²)²；不同扫描支路不混合。实测差商均为负，不代表动态电压必须处处单调。

B：Rdiff=−s单位V/(A/cm²)=Ω·cm²；它不等于HFR/膜ASR。若没有独立可观测物理阻力真值，不能对现有R标量定义“阻力残差教师”。

C：Lcurve=mean[(Vmodel(j;θ)−Vobserved(j))²]，单位V²；Jacobian为[1,−ln(j/jref),−j,ln(1−j/L),cj/{L(L−j)}]。本轮在真实极化点和b+0.001V扰动处输出可复算非零梯度，明确仅为曲线拟合诊断。

H1/H3输出为固定j的单一P1S保持率，没有j输入维度或其他电流输出；∂Vpred/∂j不能由∂Rpred/∂模型参数冒充。真实健康训练物理损失、梯度和改善均为NA，未以0代替。单电流Eeff−Reff*j的设计矩阵秩1、参数2，不可区分有效电势和有效阻力。

时间平滑或强制健康单调下降与上述跨电流规律不同，实测恢复也反对强制单调。不存在合格损失时，不执行一个名为物理、实际普通正则化的训练。''')
 md('reports/06_H1_H3与物理信息接口.md',f'''# 因果接口
对全部{len(pd.read_csv(P/'outputs/H1_H3_PARAMETER_AVAILABILITY.csv'))}条冻结Train/Validation合法记录，按origin实际可用日历时间逐条匹配7条极化测量结束时间，保留预测键与来源事件。训练历史及验证条件完成下界计数为1–6；验证每折范围如下：

{markdown_table(av[av.role.eq('validation')])}

这些是“若测量结束即可获知”的条件回放计数。因发布时间未知，具有认证发布时间的参数数目为0（表示未获认证，并不证明当时物理上不可获知）；不能把完成时间下界直接写成严格在线可用时间。

P-A维持过去信息M2_NC，原6条关联及a,b未变化。仅3个参考事件、同一寿命轨迹，无资格新增多参数映射并宣布泛化改善。上阶段H1起点P1S+校准MAE0.080190→Ridge+校准0.079730SOH百分点；H3为0.063077→0.032385，但存在代理误差与校准误差抵消；本轮不再优化这些事件。

P-B除了发布时间问题，更根本的是固定电流单输出不承载跨电流损失；即使将来补齐发布时间，也不会自动解决输出维度与辨识问题。TFT仍冻结，没有新增同目标同时间因果健康增量证据。''')
 md('reports/07_H2资格及可选实验.md',f'''# H2一次性预注册CPU实验
H2作为可选逐周期展示接口，先生成`H2_ELIGIBILITY.csv`及预拟合授权记录再计算模型分数。Train40/57/62、Validation16/5/26，innerfit33/47/51，innerselection6/9/10，各fold内层剔除1条标签过晚记录。外层/内层均使用保守测量可用时钟，边界及96h资格不变，原始cycle+2且中间周期完整。

5固定特征、alpha[0.0001,0.01,1,100]，预测DeltaR，scaler仅在对应训练子集拟合；共12inner+3final CPU拟合，逐alpha及逐fold保存，禁止重跑已完成的拟合。最终3fold均选alpha100，不因此缩小或扩大网格。

{markdown_table(h2)}

实际跨度按`H2_ACTUAL_LEADS.csv`逐fold输出，运行小时为19.04–19.41107h，日历小时为23.701944–338.851389h，不能因停机间隔混用两种时钟；没有将未来真实lead输入模型。三折等权改善{compare[1]['improvement_percent']:.6f}%，三折MAE均胜；fold1最大误差增加。固定结论为探索性Development支持，不独立确认、无统计显著性主张。

不同跨度不是完全相同样本，不能从H2均值优于或劣于H1/H3推断机制或最佳提前量。逐点预测和可复算完整参数均已保存。''')
 md('reports/08_训练资格与独立对照.md','''# 训练资格与停止决定
候选A、B、C：PHYSICS_DESCRIPTIVE_ONLY。P-A与P-B：PHYSICS_INTERFACE_NOT_QUALIFIED。无PHYSICS_LOSS_TRAINING_QUALIFIED。

逐项见PHYSICS_QUALIFICATION_GATES.csv。对真实P1S输出定义、起点之前可得、非平凡损失、有效模型梯度、区别于统计正则、同信息同容量对照等要求，没有完整通过的候选。不运行没有资格的训练和对照；数值成绩NA。

未来只有取得具有实测覆盖的跨电流输出与同条件数据、可识别参数及因果可用记录后，才可重新研究。若届时获得资格，必须另行冻结同信息/同容量无物理对照、损失尺度和成功判据，并由负责人批准；这不是本轮训练计划或授权已满足。H2成功不授予物理损失资格。

独立复算不调用拟合器，检查原始7文件选定通道、平台窗口、斜率、物理模型曲线/残差/Jacobian、H2岭回归正规方程及内外层隔离、原H1/H3缓存和全部来源SHA。测试通过及图表人工检查后才导出论文候选子目录。''')
 md('reports/09_对最终SOH_RUL论文的贡献.md','''# 对一体化论文的真实贡献
支持：冻结TFT动态电压模块；H1为主要短期P1S代理任务、H3为辅助；新增同协议H2探索性结果；真实极化形状、方向滞后、局部综合斜率及物理可辨识性的可复算诊断。

未支持：物理损失已经改善H1/H3；物理增强SOH泛化优势；TFT对健康预测的因果增量；极化参数变化即不可逆损伤；短期代理改善即真实SOH改善或RUL准确。描述性曲线诊断不能晋级为已训练物理增强健康模型。

参考SOH定义与原三事件正负结果保持不变。7事件来自同一耐久轨迹，三个合法回放事件也不是两种提前量×3的6个独立样本。极化排程回顾性、结果发布时间未知，仍不是严格在线SOH预测。

本轮真实测量不提供与冻结SOH定义一致、独立可验证的EOL事件；不设阈值造失效标签。未来RUL需真实失效/删失时间与可比终点定义、足够长且可验证的健康轨迹，以及更多独立寿命或合格外部验证。物理诊断提供条件和不可辨识性边界，本轮不能填补这些证据。

论文应以“动态预测+短期健康代理+物理资格与解释边界+稀疏参考SOH回放”描述已完成证据，不宣称已完成物理增强SOH/RUL一体化精度验证。''')
 md('README.md','''# Polarization physics → health qualification
探索性Development；物理诊断可用、健康物理训练接口未合格。原始Test未读取；原结果只读。

在本目录执行（Python3.11，numpy/pandas/scipy/scikit-learn/matplotlib）：
```
python -B scripts/run_qualification.py freeze  # 仅首次；已有冻结文件时禁止
python -B scripts/run_qualification.py run     # 仅未完成时；单文件/拟合检查点续做
python -B scripts/run_h2.py                   # 仅未完成时；一轮预注册CPU实验
python -B scripts/synthesize.py
python -B scripts/verify_independent.py
python -B -m unittest discover -s tests -v
python -B scripts/plot_figures.py
```
已有完整结果时，仅运行独立核验、测试或绘图；不要删除完成标志重新拟合。上游源数据以SHA256校验，所有读取由白名单防火墙约束。源Raw文件7个，导出的选定通道CSV保留原始行身份供核验；不是独立寿命样本。
REAL_Paper_File中的同名材料是存档镜像，复现应回到仓库研究主目录下的原Phase目录执行；脚本历史来源路径按原Phase定位，不能在论文镜像内直接启动实验。
''')
 config();print(pd.DataFrame(compare).to_string(index=False))
if __name__=='__main__':firewall();run()
