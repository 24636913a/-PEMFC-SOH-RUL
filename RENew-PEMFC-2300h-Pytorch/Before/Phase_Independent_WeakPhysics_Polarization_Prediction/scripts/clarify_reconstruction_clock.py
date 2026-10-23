"""Additive time-identity clarification; no split, weight, prediction or protocol edits."""
import shutil
from common import *
def run():
 cfg=config();assert not (P/'outputs/TIME_CLARIFICATION_RECEIPT.json').exists(),'Existing clarification must not be overwritten'
 data=pd.read_csv(P/'data_contracts/A_SAMPLES.csv',float_precision='round_trip');meta=rd('metadata');asof=str(pd.to_datetime(meta.end_calendar).max());latest_train=str(pd.to_datetime(data[data.role.eq('train')].end_calendar).max());assert pd.Timestamp(latest_train)<=pd.Timestamp(asof)
 records=[]
 for r in data.itertuples():records.append({'sample_id':r.sample_id,'role':r.role,'checkpoint':r.checkpoint,'branch':r.branch,'recorded_origin_calendar_means_event_completion':r.origin_calendar,'global_reconstruction_asof_calendar':asof,'latest_training_platform_end_calendar':latest_train,'global_model_qualified_at_that_earlier_event_time':False,'scope':'OFFLINE_ALL7_CHECKPOINT_PLATFORM_HOLDOUT_NOT_EVENT_TIME_FORECAST'})
 cs('data_contracts/A_RECONSTRUCTION_MODEL_TIME_IDENTITY.csv',pd.DataFrame(records))
 js('configs/A_RECONSTRUCTION_TIME_CLARIFICATION.json',{'original_protocol_SHA256':sha(P/'configs/FROZEN_PROTOCOL.json'),'reason':'A pools training platforms from all7 measured checkpoints. Perrow origin_calendar records that eventcompletion, not globalmodelavailability. Globalmodel may only be interpreted after completion of all7measurements.','correct_scope':'Offline measuredcampaign currentplatformholdout reconstruction','global_reconstruction_asof':asof,'latest_train_platform_end':latest_train,'new_training':0,'changed_weights':0,'changed_splits':0,'changed_predictions':0,'changed_metrics':0,'B_chronology_changed':False,'not_new_preregistration':True})
 md('reports/00a_A重建时间身份补充_请先读.md',f'''# A重建时间身份补充（不改模型与结果）
任务A是全部7个已测检查点的全局平台留出重建。原样本中`origin_calendar`及报告01所述“该次测量完成”只标识该事件及其context测量结束；**不能理解为全局模型在每个早期事件结束时已经可用**。全局模型使用了其他较晚检查点的训练平台。

正确统一回顾性重建时点为全部7次资料完成后：{asof}，最新训练平台结束为{latest_train}。逐行另表保留原字段并给出真正适用的重建时点，不改原表或冻结协议。A不是时间外推，不是逐事件当时在线预测；其合法评价范围仍是相同已测数据集中留出真实电流平台的重建。

这是时间身份的补充澄清，不是新的模型、实验、样本划分、超参数或成绩。A/B使用的权重、预测、指标均未修改，负面消融判断不变。B仍是逐折过去标签完成后、目标事件前的P1S起点回放；其发布时间假设和严格在线限制仍保留。

原报告01中关于A起点的简写以本补充为准。任何稿件必须写“offline measured-campaign reconstruction”，不能用A成绩支持早期事件时刻的模型可用性或未来SOH预测。
''')
 dest=ROOT/'REAL_Paper_File/Independent_WeakPhysics_Polarization_Prediction';assert dest.exists();relatives=['data_contracts/A_RECONSTRUCTION_MODEL_TIME_IDENTITY.csv','configs/A_RECONSTRUCTION_TIME_CLARIFICATION.json','reports/00a_A重建时间身份补充_请先读.md','scripts/clarify_reconstruction_clock.py'];copies=[]
 for rel in relatives:
  target=dest/rel;assert not target.exists();target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(P/rel,target);assert sha(P/rel)==sha(target);copies.append({'source':rel,'copy':rel,'SHA256':sha(target)})
 target=dest/'00_READ_FIRST_A_TIME_CLARIFICATION.md';assert not target.exists();shutil.copy2(P/relatives[2],target);copies.append({'source':relatives[2],'copy':target.relative_to(dest).as_posix(),'SHA256':sha(target)})
 js('outputs/TIME_CLARIFICATION_RECEIPT.json',{'status':'PASS','source_files_added':[{'path':rel,'SHA256':sha(P/rel)} for rel in relatives],'paper_copies_added':copies,'original_phase_manifest_preserved':True,'original_paper_files_untouched':True,'training_and_evaluation_not_rerun':True})
 config();print('ADDITIVE_TIME_IDENTITY_CLARIFIED',asof,len(records))
if __name__=='__main__':run()
