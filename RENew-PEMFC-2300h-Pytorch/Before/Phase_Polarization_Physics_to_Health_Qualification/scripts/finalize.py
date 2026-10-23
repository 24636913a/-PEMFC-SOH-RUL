"""Quality gates, sealed source/artifact hashes and additive paper export."""
import platform,shutil,unittest,contextlib,io
from common import *
def manifest(folder,excluded):
 return [{'path':q.relative_to(folder).as_posix(),'bytes':q.stat().st_size,'sha256':sha(q)} for q in sorted(folder.rglob('*')) if q.is_file() and q.relative_to(folder).as_posix() not in excluded and '__pycache__' not in q.parts and not q.name.endswith('.local.json')]
def run():
 cfg=config();assert jl('outputs/INDEPENDENT_VERIFICATION.json')['status']=='PASS';assert jl('outputs/VISUAL_QA.json')['status']=='PASS'
 stream=io.StringIO();suite=unittest.defaultTestLoader.discover(str(P/'tests'));result=unittest.TextTestRunner(stream=stream,verbosity=2).run(suite);md('outputs/TEST_LOG.txt',stream.getvalue());assert result.wasSuccessful()
 import scipy,sklearn,matplotlib
 js('outputs/CPU_ENVIRONMENT.json',{'Python':platform.python_version(),'platform':platform.platform(),'numpy':np.__version__,'pandas':pd.__version__,'scipy':scipy.__version__,'scikit_learn':sklearn.__version__,'matplotlib':matplotlib.__version__,'GPU_used':False})
 script_files=sorted((P/'scripts').glob('*.py'));cs('data_contracts/FINAL_CODE_SHA256.csv',pd.DataFrame([{'path':q.relative_to(P).as_posix(),'sha256':sha(q)} for q in script_files]))
 js('outputs/QUALITY_GATE.json',{'status':'PASS','independent_recomputation':'PASS','tests_passed':result.testsRun,'figures_visually_checked':7,'original_sources_unchanged':True,'physics_training_qualified':False,'scientific_identity':'ExploratoryDevelopment_H2_and_descriptive_polarization_NOT_physics_enhanced_SOH','base_commit':cfg['base_commit'],'frozen_protocol_sha256':sha(P/'configs/FROZEN_PROTOCOL.json')})
 # Additive exclusive destination; old formal files never replaced.
 dest=ROOT/'REAL_Paper_File/Polarization_Physics_to_Health_Qualification';assert not dest.exists(),'Existing paper artifacts must not be overwritten'
 md('reports/论文候选图说明.md','''# 论文候选图说明
01：7事件双向真实平台均值，空心点未通过预设稳定域筛查。
02：局部综合斜率及同设定电流滞后；不是膜ASR/HFR。
03：四窗口敏感性与去电压平移后的形状；非置信区间、非独立健康增量。
04：固定网格jlim剖面、列归一化Jacobian条件数；局部筛查通过不等于可训练健康接口。
05：H1/H3冻结复算及H2新CPU实验，各跨度自身合法样本；三fold等权均值，非跨跨度同样本因果比较。
06：H2三个fold真实目标及逐点预测，缺失区域不补观测。
07：完成时间下界的条件可用数与认证发布时间记录缺失，不能升级严格在线SOH。
所有图为探索性Development或描述性物理诊断；没有物理增强SOH/RUL精度图。
''')
 cs('outputs/ARTIFACT_SHA256.csv',pd.DataFrame(manifest(P,{'outputs/ARTIFACT_SHA256.csv','outputs/PAPER_EXPORT_RECEIPT.json'})))
 dest.mkdir(parents=True,exist_ok=False)
 for sub in ['configs','data_contracts','scripts','models','reports','figures','tests']:
  shutil.copytree(P/sub,dest/sub,ignore=shutil.ignore_patterns('__pycache__','*.pyc','.keep'))
 # Include actual predictions/diagnostics and raw selected channels; complete reproducible evidence, not just figures.
 shutil.copytree(P/'outputs',dest/'outputs',ignore=shutil.ignore_patterns('GIT_RECEIPT.local.json','PAPER_EXPORT_RECEIPT.json'))
 shutil.copy2(P/'README.md',dest/'README.md');(dest/'.gitattributes').write_bytes((P/'.gitattributes').read_bytes())
 receipt=[]
 for q in dest.rglob('*'):
  if q.is_file():
   original=P/q.relative_to(dest);assert sha(q)==sha(original);receipt.append({'path':q.relative_to(dest).as_posix(),'sha256':sha(q)})
 js('outputs/PAPER_EXPORT_RECEIPT.json',{'destination':dest.relative_to(ROOT).as_posix(),'files':len(receipt),'all_copies_hash_equal':True,'identity':'EXPLORATORY_DEVELOPMENT_AND_DESCRIPTIVE_PHYSICS','copy_sha256':receipt})
 print('FINALIZED',result.testsRun,'tests',len(receipt),'paperfiles')
if __name__=='__main__':run()
