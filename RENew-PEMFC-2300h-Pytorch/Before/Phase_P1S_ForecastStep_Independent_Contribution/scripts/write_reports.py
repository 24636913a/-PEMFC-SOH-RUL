"""Claims limited to completed ablation; no new performance estimation."""
from common import *
def table(d):
 return '| '+' | '.join(d.columns)+' |\n| '+' | '.join(['---']*len(d.columns))+' |\n'+'\n'.join('| '+' | '.join(str(v) for v in r)+' |' for r in d.itertuples(index=False,name=None))
def run():
 config();assert load_local('outputs/INDEPENDENT_VERIFICATION.json')['PASS']
 met=csv_read(P/'outputs/SOH_METRICS.csv');co=csv_read(P/'outputs/SEQUENTIAL_CONTRIBUTION.csv');pa=csv_read(P/'outputs/PAIRWISE_COMPARISONS.csv');dc=csv_read(P/'outputs/ERROR_DECOMPOSITION.csv');hcompare=csv_read(P/'outputs/H3_VS_H1_ERROR_CANCELLATION.csv')
 gains=csv_read(P/'outputs/POINTWISE_WINS.csv');gains=gains[gains.horizon.eq('H3')&gains.candidate.eq(METHODS[2])&gains.baseline.eq(METHODS[1])].copy()
 gains['share_of_H3_total_AE_reduction_percent']=gains.absolute_error_reduction_SOH_pp/gains.absolute_error_reduction_SOH_pp.sum()*100
 cs('outputs/H3_INCREMENT_CONCENTRATION.csv',gains[['nominal_h','absolute_error_reduction_SOH_pp','share_of_H3_total_AE_reduction_percent']])
 mapped_actual_MAE=float(dc[dc.horizon.eq('H3')].calibration_at_actual_target_SOH_pp.abs().mean())
 js('outputs/TARGET_CALIBRATION_RESIDUAL_DIAGNOSTIC.json',{'MAE_SOH_pp':mapped_actual_MAE,'role':'Actual future target R through frozen map; diagnostic only, not a deployable fourth method','unique_events':3})
 decision={'status':'FORECAST_STEP_GAIN_IS_HORIZON_DEPENDENT_AND_CALIBRATION_ERROR_DEPENDENT',
  'H1':'NEGLIGIBLE_MEAN_INCREMENT_ONLY_1_OF_3_WINS',
  'H3':'LARGER_SOH_REPLAY_INCREMENT_3_OF_3_WINS_WITH_PROXY_ERROR_CANCELLATION',
  'proven_better_future_P1S_on_these3_SOH_events':False,
  'next_Ridge_forecast_candidate_to_retain':'H3 exploratory; H1 no clear independent SOH gain',
  'current_primary_control':'Measured-origin P1S + frozen past-only M2_NC calibration',
  'validated_primary_SOH_forecasting_method_established':False,
  'unique_reference_events':3,'strict_online_SOH':False,'no_statistical_significance_claim':True,
  'new_training':0,'no_Test':True,'no_GPU':True,'historical_P1S_forecasting_result_overturned':False}
 js('outputs/RESEARCH_DECISION.json',decision)
 md('reports/00_预测步骤独立贡献结论.md',f'''# P1S预测步骤独立贡献核验

本轮只回答Ridge预测步骤相对于起点实测P1S的增量；原96h Ridge、F2、M2_NC和所有历史结果只读，不重新训练或拟合。

{table(met)}

{table(co[['horizon','Ridge_forecast_step_MAE_reduction_SOH_pp','Ridge_step_relative_to_origin_percent','Ridge_step_share_of_total_MAE_reduction_percent']])}

H1：起点P1S+校准MAE0.080190，Ridge+同校准0.079730 SOH百分点，额外降低0.000460（0.573738%），1胜2负。近期P1S和校准已贡献0.200982，Ridge步骤只占从M0到Ridge总MAE降幅的0.228393%。不能把相对M0的71.64%改善都归给未来预测。

H3：起点P1S+校准0.063077，Ridge+同校准0.032385，额外降低0.030692（48.657746%），3胜0负。Ridge步骤占相对M0总MAE降幅的12.336674%，其余87.663326%已由近期P1S+校准对照取得。这个比例是按指定顺序分解的描述性MAE差，不是唯一的因果贡献/方差贡献。

关键限制：H3在三个所选事件上的P1S预测都比“起点实测值保持”更差，SOH却都更准。其误差与校准残差反向抵消。历史完整P1S三fold任务的预测收益没有被推翻，这个发现只针对本次三个参考事件的桥接解释。

建议：若在H1/H3中只保留一个后续Ridge预测候选，保留H3进行探索；H1不宜作为有独立SOH贡献的主方法。当前论文以起点P1S+冻结校准为必要主对照，H3不得升级为已验证、更准确预测健康退化的主方法。三个事件和误差抵消不足以支持稳定泛化或机理解释。

独立60位Decimal复算与用户初步四项近似值在六位小数一致；计算未使用这些预期值。没有预期数值作为成绩输入、训练、选参或结果覆盖。''')
 md('reports/01_方法时间与科研主张边界.md',f'''# 固定方法与合法时间

基准提交5141b79d7116d3e9721fe1bd626ab97412b3d485。固定1150/1500/1750三个事件及六个精确关联；按horizon/fold/origin raw cycle ID/target raw cycle ID以及字符串cycle身份匹配。各关联恰有Persistence/Ridge两条旧缓存。全188条P1S缓存和94条Ridge系数重放独立核验，所有源文件SHA256保留。

M0=上次参考SOH；M_origin=a*R_origin+b；M_Ridge=a*R_pred+b。a,b完全复用该事件/提前量已冻结的过去M2_NC参数，无新拟合。R_origin从独立实际测量身份表提取，并核对middle_50 P1S电压除以0.71383023635、原Persistence缓存、原Ridge特征、最近可用合法观测身份一致。未来实测R不进入这三个预测函数。

起点、真实P1S测量start/end OpHrs2、保守可用end、已记录日历、极化start/end、上次校准参考结束时间、Train/inner标签界限逐行保存。精确stable子窗口结束的日历时刻没有单独保存，保持NA；已有measurement_end_datetime是保守dwell-end可用日历，不伪造精确秒级时刻。

事件选择属于回顾性排程，极化结果发布时间未知，参考实际OpHrs2未知；仍是探索性顺序回放。原周期目录在线判别能力未认证。M0和校准都沿用上轮“已测完且假定前次结果已发布”的条件，不升级为严格在线预测。

同三个事件用于H1/H3，六起点不是六个独立参考结局。没有p值、bootstrap、显著性检验；微小差不能当统计显著或超越测量噪声，未估计参考测量噪声底。

{table(pa)}

完整缓存的补充P1S代理指标使用合并样本均值（H1 n50、H3 n44），仅为背景；历史正式主指标仍为三个fold等权均值，本轮独立复算逐fold并保存，不拿合并均值改写历史正式数值。''')
 md('reports/02_代理误差与SOH误差抵消.md',f'''# 为什么H3的代理更差，SOH反而更好

令Y为参考SOH、R*为该关联的未来真实P1S，只在事后诊断中有：

e_SOH=100*(a*R*+b−Y)+100*a*(R_pred−R*)。

第一项是冻结映射在实际目标P1S上的残差；它不是可部署的第四个方法，也不能用未来R*改进预测。第二项是代理预测误差经过斜率a后的影响。两项相反时会抵消，最终SOH更准确并不必然代表P1S预测更准确。映射残差本身还混合状态/工况差异及稀疏校准，不能断言已识别具体电化学原因。

{table(hcompare)}

1150：共同映射残差−0.117178pp；H1代理项+0.017726，合成−0.099452；H3代理项+0.113241，合成−0.003936。H3几乎完整抵消负映射残差，因此大幅受益，虽然代理AE从H1的0.101569升至0.648875 mV-equivalent。

1500：映射残差+0.028063；H1代理项−0.002694，H3−0.005558，H3更大反向误差带来更小最终误差。

1750：映射残差+0.115285；H1代理项−0.000916，H3−0.044570，H3再次靠反向项减小误差。

对同提前量的起点保持对照：H3三事件P1S代理AE也全部更差（均值0.151552→0.320568 mV-equivalent），而SOH三次改善。H1代理MAE0.039175→0.041419，同时SOH仅微降；H1在1500/1750即使代理更准，SOH仍稍差。这说明本次桥接成绩由代理误差与映射残差的符号组合驱动，不能直接解释为更准确预测未来健康。

诊断上，即使未来P1S完全预测正确，冻结校准残差的MAE仍为{mapped_actual_MAE:.6f}pp，反而大于H3 Ridge的0.032385。它只说明校准残差被错误预测部分抵消，不是给未来实测输入建立新方法。H3总AE增量中1150事件占{float(gains.loc[gains.nominal_h.eq(1150),'share_of_H3_total_AE_reduction_percent'].iloc[0]):.2f}%，收益集中在一个事件。

抵消量定义|e_cal|+|e_proxy|−|e_cal+e_proxy|，不是可辨识的噪声消除能力。将来校准残差变号、幅度改变，抵消可能消失或变为放大；本轮不为此修改参数或寻找新模型。''')
 md('paper_candidates/CLAIM_BOUNDARIES.md','''# 论文候选成果的固定主张

可写：相同三参考事件、相同起点和冻结校准下，H1的额外MAE收益0.000460 SOHpp/1胜2负；H3为0.030692 SOHpp/3胜0负。H3的代理预测误差在这些事件上恶化，但经仿射映射与校准残差反向抵消，SOH回放误差更小。

不可写：Ridge贡献了相对参考Persistence的全部SOH改善；H3在这些事件上更准确预测真实未来P1S；H3的较小SOH误差证明机制正确、稳定可迁移或统计显著；六次独立验证；严格在线SOH；TFT/物理/RUL本轮获得新训练结果。

保留H3作为后续Ridge预测候选，起点P1S+冻结校准必须保留为主对照；尚未确立已验证的SOH预测主方法。F3未来真实P1S仅供事后诊断，不构成预测输入。事件排程回顾性、结果发布时间未知。图表和CSV仅在独立核验、测试和人工图审通过后复制到正式目录的独立子目录。''')
 md('README.md','''# P1S预测步骤独立贡献

仅RealHyFC Development派生缓存，零训练、零新校准、零Test/GPU。冻结协议、源SHA、13份精确输入快照、18条预测、6条误差分解及全部测试代码保存于本目录。

从研究主目录复核（不会重新训练或改写历史）：

```powershell
$env:PYTHONUTF8='1'
D:\\ruanjian\\python3.11.9\\python.exe -B Phase_P1S_ForecastStep_Independent_Contribution/scripts/verify_independent.py
D:\\ruanjian\\python3.11.9\\python.exe -B -m unittest discover -s Phase_P1S_ForecastStep_Independent_Contribution/tests -v
```

run_audit.py freeze/run为一次性运行入口；已存在COMPLETE时禁止再run。需要完整独立复现时使用干净仓库副本的新阶段目录，复制脚本、删除副本生成物后freeze/run，不删除原成果。绘图render_paper_figures.py只读真实CSV，可重绘。它也可以原样在导出的论文包scripts目录运行。

关键文件：SOH_METRICS、PAIRWISE_COMPARISONS、SEQUENTIAL_CONTRIBUTION、ERROR_DECOMPOSITION、H3_VS_H1_ERROR_CANCELLATION、ORIGIN_MEASUREMENT_TIME_AND_IDENTITY_AUDIT。完整缓存汇总是pooled，仅供背景；历史正式三fold等权指标不变。

SHA256见inputs/SOURCE_SHA256.csv和outputs/ARTIFACT_SHA256.csv；独立高精度报告见INDEPENDENT_VERIFICATION。最终研究判断见reports/00。论文成果只导出到REAL_Paper_File下独立子目录，不替换旧文件。''')
 print('Research conclusions and claim boundaries written')
if __name__=='__main__':firewall();run()
