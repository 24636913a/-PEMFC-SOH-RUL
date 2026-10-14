"""Independent final verification, no old runners or model imports."""
import json,hashlib,subprocess,sys
import numpy as np
import pandas as pd
from audit_read_rule import P,ROOT,A,B,N1,sha,js,cs,text_file

def main():
 before=json.loads((P/'inputs/ORIGINAL_STAGE_SHA_BEFORE.json').read_text())
 after=json.loads((P/'outputs/ORIGINAL_STAGE_SHA_AFTER.json').read_text())
 assert before==after and all(sha(ROOT/f)==h for f,h in before.items())
 result=json.loads((P/'outputs/FINAL_AUDIT.json').read_text());assert result['status']=='PASS_EXACT'
 # This parses only the proven non-Test3005 records. Compare to the actual
 # prior full-column3007-then-filter serialized Development fingerprint.
 full=pd.read_csv(N1,nrows=3005)
 payload=full.to_csv(index=False,header=True,lineterminator='\n',float_format='%.17g').encode('utf-8')
 authority=ROOT/'Phase_DDP_SoftPhysics_Overnight_Experiment/outputs/TRAIN_DATA_AUTHORITY.json'
 health=ROOT/'Phase_TFT_P1S96_HealthAware_Fusion/outputs/CPU_QUALIFICATION.json'
 old_sha=json.loads(authority.read_text())['development_prefix_fingerprint_sha256']
 health_sha=json.loads(health.read_text())['prefix_fingerprint_sha256']
 actual=hashlib.sha256(payload).hexdigest();assert actual==old_sha==health_sha
 js('outputs/LEGACY_FULL_DEVELOPMENT_FINGERPRINT_VERIFICATION.json',dict(status='PASS',rows=3005,full_source_fields=len(full.columns),
  new_serialized_Development_sha256=actual,DDP_historical3007_filtered_sha256=old_sha,Health96_historical3007_filtered_sha256=health_sha,
  authority_sha256=sha(authority),health_qualification_sha256=sha(health),
  no_Test_features_parsed=True,not_a_whole_N1_hash=True,
  meaning='Exact historical serialized full214-field Development content reproduced; historical all-column3007 parser not rerun'))
 common=pd.read_csv(P/'inputs/HISTORICAL_RULE_DEVELOPMENT_COMMON_FIELDS.csv',float_precision='round_trip')
 saved=pd.read_csv(A/'inputs/DEVELOPMENT_BIN_IDENTITY_AND_CONDITIONS.csv',float_precision='round_trip')
 pd.testing.assert_frame_equal(common,saved,check_exact=True,check_dtype=True,check_like=False)
 field=pd.read_csv(P/'outputs/ALL_COMMON_FIELD_PRECISION.csv');assert field.exact_PASS.all() and field.unequal_values.sum()==0
 assert pd.read_csv(P/'outputs/ALL_MISMATCH_RECORDS.csv').empty and pd.read_csv(P/'outputs/PREDICTION_IDENTITY_MISMATCHES.csv').empty
 stats=pd.read_csv(P/'outputs/CONCLUSION_STATISTICS_RECHECK.csv');assert stats.exact_equal.all()
 log=subprocess.run([sys.executable,'-B','-m','unittest','discover','-s',str(P/'tests'),'-v'],capture_output=True,text=True,encoding='utf-8')
 text_file('outputs/TEST_LOG.txt',log.stdout+log.stderr);assert log.returncode==0
 js('outputs/INDEPENDENT_VERIFICATION.json',dict(status='PASS',tests=7,strict_dataframe_check=True,
  source_full214_field_legacy_fingerprint_exact=True,original_historical_files_verified=len(before),original_bytes_unchanged=True,
  no_model_or_SOH_experiment_executed=True))
 lines=['| 核对量 | 本轮重新计数 | 原结果 | 一致 |','| --- | ---: | ---: | --- |']
 for row in stats.itertuples():lines.append(f'| {row.statistic} | {row.recomputed} | {row.previous} | 是 |')
 text_file('00_独立一致性核验结论.md',f'''# 实际结论
 两种规则得到**完全相同的3005个Development样本**，源行号均为0–3004，顺序完全一致。35个共用字段、105175个单元全部严格一致，数值最大绝对差/相对差/spacing差均为0。运行小时按float64位模式比较，日历字符串及纳秒时间均严格一致；没有用粗略容差掩盖错位。
 本轮另以原正式3007读取保存的完整Development指纹作独立交叉核验：214个全字段内容的序列化SHA完全相同，为 `{actual}`。这同时排除了“读取列数或两条排除行改变了保留数据解析”的疑虑。
 41568条已有预测的行号、运行小时及日历身份无不一致；Train1380/Validation205的窗口身份重新核验。SOH重合数仍0，目标窗698、近参考电流148、P1S落窗33；物理Train2103、近参考电流468、RH均值>100窗11均不变。
 因此此前 `TFT_TO_SOH_INSUFFICIENT_SUPPORT` 和 `PHYSICS_ROUTE_NOT_QUALIFIED` 的结论不受此次读取口径差异影响，没有重新拟合、重训或改写任何旧结果。
 **读取范围补充：**原始前3007条中的源行3005为boundary_excluded、源行3006标记test；二者均不通过过滤。为遵守本轮不读取Test数值的要求，本审计先只解析3007行身份元数据，按正式阈值求行号；确认保留行恰好为前3005条后，只解析这些行的全字段。没有重新执行会解析Test特征的“全字段3007行读取”，而是证明其过滤输出等价，并比对历史实际输出指纹。
 源行3006的时间/分区元数据用于边界核验，电压、工况等数值未解析。旧“读取3007条全部属于Development”的表述不准确，需保留这一边界说明；没有证据该排除行进入最终Train/Validation或影响本轮科研判断。
 原两阶段冻结协议、源代码、SHA与所有纳入保护的输出共{len(before)}份保持字节不变。7项测试和独立复核通过，仅在本新审计目录写文件。
 ''')
 text_file('01_读取范围与精度证据.md','''# 可重复核验方法
 历史选择规则：前3007个逻辑记录，以bin_end_ophrs2_h<1819.997738筛选并重置行序。首先仅用时间/分区/有效性元数据确定mask，不假设“前3005条就是筛选结果”。mask得到0–3004；若不连续或选入Test，特征解析器直接拒绝。
 全字段解析使用历史pd.read_csv默认浮点转换；历史新阶段快照用round_trip读取恢复其17位序列化float64值。所有共用数值按位比较，NaN缺失位置需一致，正负零区分；文本原样比较，日期另按纳秒比较。源码没有身份差异容差。逐字段精度、数据类型及缺失模式见ALL_COMMON_FIELD_PRECISION.csv；全部不一致CSV为空，仍保留表头。
 N1只作有界读取及3005条合法原始前缀SHA，不对混合全文件计算SHA。完整Development序列化指纹不是N1全文件指纹。
 窗口身份依历史48/24、valid、temporal_segment_id、连续0.5h与14槽位有限性规则独立判断，不产生训练张量、不调用旧实验runner。源段ID和部分原14槽位未保存于35列快照，不能声称这些字段直接逐列比对；它们由同一保留源行及完整214列历史指纹作身份补证。window规则原登记1e-6只用于重现历史合法性条件，不用于快照身份比较。
 既有预测只读取键及时间列，重数698/148/33/0及Train468/11，未重新评价性能、重建SOH或计算物理损失。过去关于界面状态/参数可辨识的科学判断没有在本次重新实验。
 审计初次在组合日历比较掩码时遇到pandas只读数组错误，尚未完成结果。改为创建新的布尔数组后通过；初次新审计脚本及协议保留在configs/PRECHECK_ATTEMPT1_*，原研究阶段未改动。
 ''')
 text_file('02_历史结论影响核对.md','# 结论统计复核\n\n'+'\n'.join(lines)+'''\n
 所有统计保持一致。两个原研究决定沿用，不能把数据等价解释为SOH映射成立或物理候选合格。本次没有新增模型准确率。
 读取口径在操作层面存在区别：直接3005不解析后两行；字面全字段3007会先解析一条后来排除的Test标记记录。两者筛选后的Development数值一致，不等于原始读取范围相同。本轮不覆盖旧读取范围声明，单独记录这项差别。
 ''')
 text_file('README.md','''# Development读取口径独立核验
 执行scripts/audit_read_rule.py，再scripts/verify_and_report.py。首次audit已完成，拒绝重复；verify只作独立内容与统计复核。最终PASS_EXACT。
 3007行元数据筛选 + 仅保留3005行特征解析，避免重读被排除Test数值；与历史214列Development指纹及35列本轮快照同时严格一致。
 无torch、模型、GPU、SOH构造、旧实验runner或历史写入。报告00–02，所有差异清单和SHA记录位于inputs/outputs。
 ''')
 js('outputs/ARTIFACT_MANIFEST.json',dict(files={str(f.relative_to(P)).replace('\\','/'):sha(f) for f in P.rglob('*') if f.is_file() and f.name not in ['ARTIFACT_MANIFEST.json','GIT_RECEIPT.local.json']}))
 print('VERIFIED_AND_REPORTED;7 tests; historical214-field fingerprint exact')

if __name__=='__main__':main()
