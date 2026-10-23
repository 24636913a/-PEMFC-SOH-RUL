"""New verified exploratory evidence package only; exclusive writes, no old replacement."""
import csv
from common import *
DEST=ROOT/'REAL_Paper_File/P1S_LongTerm_Trajectory_Forecasting'
def export():
 config();assert local('outputs/INDEPENDENT_VERIFICATION.json')['PASS'];assert local('outputs/TEST_RESULTS.json')['PASS'];review=local('outputs/FIGURE_VISUAL_REVIEW.json');assert review['PASS'] and len(review['figures'])==7
 for name,rec in review['figures'].items():
  for suffix in ['png','pdf']:assert sha(P/f'figures/{name}.{suffix}')==rec[suffix+'_sha256']
 assert not DEST.exists(),'Do not overwrite formal evidence'
 mappings=[]
 for q in sorted((P/'figures').iterdir()):
  if q.suffix in ['.png','.pdf']:mappings.append((q,'figures/'+q.name))
 for name in ['REAL_P1S_OBSERVATIONS.csv','PREDICTIONS.csv','FOLD_METRICS.csv','THREE_FOLD_EQUAL_MEANS.csv','PRIMARY_METHOD_COMPARISON.csv','POOLED_STRATUM_DIAGNOSTICS.csv',
  'ACTUAL_LEAD_AUDIT.csv','COMMON_ORIGIN_PREDICTIONS.csv','COMMON_ORIGIN_COUNTS.csv','PARAMETER_TIME_AUDIT.csv','PAPER_EVIDENCE_MATRIX.csv','MISSING_RIDGE_SCORES_ARE_NA.csv',
  'PHYSICS_QUALIFICATION_MATRIX.csv','REAL_TRAIN_PHYSICS_IDENTIFIABILITY.csv','TFT_PARAMETER_TIME_AND_HEALTH_SUPPORT.csv','MEASURED_CONDITION_RANGES.csv','NATIVE_H1_H3_REPLAY.csv']:
  mappings.append((P/'outputs'/name,'source_csv/'+name))
 mappings.extend([(P/'scripts/render_figures.py','scripts/render_figures.py'),(P/'FIGURE_SOURCE_INDEX.csv','FIGURE_SOURCE_INDEX.csv'),
  (P/'paper_candidates/CLAIM_BOUNDARIES.md','CLAIM_BOUNDARIES.md'),(P/'reports/00_实际研究结果与论文意义.md','实际研究结果.md'),
  (P/'reports/05_中断日志与数值恢复说明.md','数值恢复披露.md'),(P/'models/NEW_H6_H12_RIDGE_PARAMETERS.json','models/NEW_H6_H12_RIDGE_PARAMETERS.json'),
  (P/'configs/FROZEN_PROTOCOL.json','verification/FROZEN_PROTOCOL.json'),(P/'configs/FROZEN_PROTOCOL.sha256','verification/FROZEN_PROTOCOL.sha256'),
  (P/'configs/RESUME_CODE_IDENTITY.json','verification/RESUME_CODE_IDENTITY.json'),(P/'outputs/INDEPENDENT_VERIFICATION.json','verification/INDEPENDENT_VERIFICATION.json'),
  (P/'outputs/TEST_RESULTS.json','verification/TEST_RESULTS.json'),(P/'outputs/FIGURE_VISUAL_REVIEW.json','verification/FIGURE_VISUAL_REVIEW.json'),
  (P/'inputs/SOURCE_SHA256.csv','verification/SOURCE_SHA256.csv')])
 payload=[(src,rel,src.read_bytes(),sha(src)) for src,rel in mappings];DEST.mkdir(parents=True,exist_ok=False);receipt=[]
 for src,rel,data,h in payload:
  q=(DEST/rel).resolve();assert q.is_relative_to(DEST.resolve());q.parent.mkdir(parents=True,exist_ok=True)
  with q.open('xb') as f:f.write(data)
  assert sha(q)==h;receipt.append({'source':src.relative_to(ROOT).as_posix(),'destination':q.relative_to(ROOT).as_posix(),'sha256':h,'bytes':len(data)})
 with (DEST/'.gitattributes').open('x',encoding='utf-8',newline='\n') as f:f.write('* -text whitespace=blank-at-eol,blank-at-eof,space-before-tab,cr-at-eol\n*.png binary\n*.pdf binary\n')
 with (DEST/'README.md').open('x',encoding='utf-8',newline='\n') as f:f.write('''# 已核验探索性论文证据：P1S较长周期轨迹

本包包含真实观测、逐点预测、指标、源码和明确负面/不可评价结果。H1/H3原生信号复现，H6未优于保持，H12不能完整三fold评价。没有物理SOH/RUL一体化成功主张。

在本包目录运行`python -B scripts/render_figures.py`，可从source_csv重绘全部7套PNG/PDF。完整实验代码、输入身份、测试和独立数值复算位于研究主目录Phase_P1S_LongTerm_Trajectory_Forecasting。

H6在原执行器资格中断后按相同保存Train作正规方程数值恢复，细节见数值恢复披露.md；不冒称异常前权重逐字节备份。H12空折/不可拟合成绩保持NA。回升分层是事后描述，P1S保持率不是极化参考SOH，mV-equivalent不是动态电压RMSE。请保留CLAIM_BOUNDARIES全部限定。
''')
 manifest=[{'path':q.relative_to(DEST).as_posix(),'bytes':q.stat().st_size,'sha256':sha(q)} for q in sorted(DEST.rglob('*')) if q.is_file()]
 with (DEST/'PACKAGE_SHA256.csv').open('x',encoding='utf-8',newline='') as f:
  w=csv.DictWriter(f,fieldnames=['path','bytes','sha256']);w.writeheader();w.writerows(manifest)
 cs('outputs/PAPER_COPY_SHA256.csv',pd.DataFrame(receipt));js('outputs/PAPER_EXPORT_RECEIPT.json',{'PASS':True,'scope':'Verified exploratory evidence and explicit non-evaluable results','destination':DEST.relative_to(ROOT).as_posix(),'copied_files':len(receipt),'old_files_overwritten':0})
 print('EXPORTED',len(receipt),'verified files, no existing paper file replaced')
if __name__=='__main__':export()
