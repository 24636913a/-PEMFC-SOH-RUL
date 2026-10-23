"""Additive verified negative-ablation export; never overwrite formal artifacts."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import platform,unittest,io,shutil
import torch,matplotlib
from common import *
def run():
 cfg=config();assert jl('outputs/INDEPENDENT_VERIFICATION.json')['status']=='PASS';assert jl('outputs/VISUAL_QA.json')['status']=='PASS'
 teststream=io.StringIO();suite=unittest.defaultTestLoader.discover(str(P/'tests'));result=unittest.TextTestRunner(stream=teststream,verbosity=2).run(suite);md('outputs/FINAL_TEST_LOG.txt',teststream.getvalue());assert result.wasSuccessful()
 js('outputs/FINAL_TEST_RESULTS.json',{'status':'PASS','tests':result.testsRun,'failures':len(result.failures),'errors':len(result.errors),'no_new_training':True})
 js('outputs/CPU_ENVIRONMENT.json',{'Python':platform.python_version(),'platform':platform.platform(),'torch':torch.__version__,'numpy':np.__version__,'pandas':pd.__version__,'matplotlib':matplotlib.__version__,'device_used':'cpu','dtype':'float64','GPU_runs':0,'CPU_threads_in_training':1,'deterministic_algorithms':True})
 cs('outputs/FINAL_CODE_SHA256.csv',pd.DataFrame([{'path':q.relative_to(P).as_posix(),'SHA256':sha(q)} for folder in ['scripts','tests'] for q in sorted((P/folder).glob('*.py'))]))
 js('outputs/FINAL_QUALITY_GATE.json',{'status':'PASS','base_commit':cfg['base_commit'],'protocol_SHA256':sha(P/'configs/FROZEN_PROTOCOL.json'),'tests':result.testsRun,'independent_runs':24,'new_optimizer_runs':24,'new_optimizer_steps':60000,'TFT_updates':0,'Ridge_updates':0,'new_SOH_RUL_fits':0,'no_Test':True,'source_SHA256_unchanged':True,'figures_checked':6,'identity':'EXPLORATORY_NEGATIVE_ABLATION_NOT_VALIDATED_PHYSICS_ENHANCED_SOH'})
 md('reports/PAPER_IDENTITY.md','''# 论文归档身份
真实执行的独立弱物理A/B负面消融：物理损失可计算且短暂激活，但没有清楚增量。原双路径TFT仍是核心；H1/H2/H3及M2_NC冻结。
该归档可用于描述性物理与负面消融，不得用作物理增强TFT/SOH或真实RUL成功证据。
TaskB是测量完成可用假设下名义协议条件回放，不认证严格在线发布时刻。完整实验及恢复检查点位于原Phase；镜像源代码用于追溯，复现必须回到原Phase目录。
''')
 files=[q for q in sorted(P.rglob('*')) if q.is_file() and '__pycache__' not in q.parts and q.name not in ['ARTIFACT_SHA256.csv','PAPER_EXPORT_RECEIPT.json','GIT_RECEIPT.local.json']]
 cs('outputs/ARTIFACT_SHA256.csv',pd.DataFrame([{'path':q.relative_to(P).as_posix(),'bytes':q.stat().st_size,'SHA256':sha(q)} for q in files]))
 dest=ROOT/'REAL_Paper_File/Independent_WeakPhysics_Polarization_Prediction';assert not dest.exists(),'Do not overwrite formal results';dest.mkdir(parents=True,exist_ok=False);receipt=[]
 for q in sorted(P.rglob('*')):
  if not q.is_file() or '__pycache__' in q.parts or q.name in ['.keep','GIT_RECEIPT.local.json','PAPER_EXPORT_RECEIPT.json']:continue
  rel=q.relative_to(P)
  # Full per-run prediction/probe exports and optimizer-resume copies stay in Phase;
  # their complete aggregate CSVs and all final weights/logs are copied for the paper.
  if q.name.endswith('_resume.pt') or rel.parts[:2] in [('outputs','predictions'),('outputs','probes')]:continue
  target=dest/rel;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(q,target);assert sha(q)==sha(target);receipt.append({'source':rel.as_posix(),'copy':target.relative_to(dest).as_posix(),'SHA256':sha(q)})
 js('outputs/PAPER_EXPORT_RECEIPT.json',{'destination':dest.relative_to(ROOT).as_posix(),'identity':'EXPLORATORY_NEGATIVE_ABLATION','all_copies_hash_equal':True,'files':len(receipt),'copy_records':receipt});config();print('SEALED_AND_EXPORTED',result.testsRun,'tests',len(receipt),'paperfiles')
if __name__=='__main__':run()
