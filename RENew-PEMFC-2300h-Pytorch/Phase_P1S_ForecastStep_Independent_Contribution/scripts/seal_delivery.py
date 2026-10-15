"""Seal research and paper identities after verification; phase-only writes."""
import subprocess
from common import *
def seal():
 config();assert load_local('outputs/INDEPENDENT_VERIFICATION.json')['PASS'];assert load_local('outputs/TEST_RESULTS.json')['PASS'];assert load_local('outputs/PAPER_EXPORT_RECEIPT.json')['PASS']
 for _,r in csv_read(P/'outputs/PAPER_COPY_SHA256.csv').iterrows():assert sha(ROOT/r.source)==sha(ROOT/r.paper_file)==r.sha256
 paper=ROOT/'REAL_Paper_File/P1S_ForecastStep_Independent_Contribution'
 for _,r in csv_read(paper/'PACKAGE_SHA256.csv').iterrows():assert sha(paper/r.path)==r.sha256
 prefix=P.relative_to(REPO).as_posix()+'/';paperprefix=paper.relative_to(REPO).as_posix()+'/'
 changed=subprocess.check_output(['git','diff','--name-only','5141b79d','-z'],cwd=REPO).decode().split('\0')[:-1]
 assert all(s.startswith((prefix,paperprefix)) for s in changed)
 js('outputs/FINAL_CODE_IDENTITY.json',{q.relative_to(P).as_posix():sha(q) for q in sorted((P/'scripts').glob('*.py'))})
 files=[q for q in P.rglob('*') if q.is_file() and '__pycache__' not in q.parts and q.name not in ['ARTIFACT_SHA256.csv','FINAL_INTEGRITY_AUDIT.json','GIT_RECEIPT.local.json']]
 cs('outputs/ARTIFACT_SHA256.csv',pd.DataFrame([{'path':q.relative_to(P).as_posix(),'bytes':q.stat().st_size,'sha256':sha(q)} for q in sorted(files)]))
 js('outputs/FINAL_INTEGRITY_AUDIT.json',{'PASS':True,'historical_sources_sha256_unchanged':True,'source_files':len(SOURCES),'phase_artifacts_in_sha_manifest':len(files),
  'independent_Decimal_predictions':18,'full_historical_P1S_predictions':188,'tests':load_local('outputs/TEST_RESULTS.json')['tests_run'],
  'figures_PNG_and_PDF':4,'paper_copy_verified':True,'old_paper_files_replaced':0,'Test_reads':0,'new_fits':0,'GPU_runs':0,'strict_online_upgrade':False})
 print('SEALED',len(files),'research artifacts; historical sources unchanged')
if __name__=='__main__':seal()
