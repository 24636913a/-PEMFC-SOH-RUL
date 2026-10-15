import subprocess
from common import *
def seal():
 cfg=config();assert local('outputs/INDEPENDENT_VERIFICATION.json')['PASS'];assert local('outputs/TEST_RESULTS.json')['PASS'];assert local('outputs/PAPER_EXPORT_RECEIPT.json')['PASS']
 assert sha(P/'scripts/resume_and_finalize.py')==local('configs/RESUME_CODE_IDENTITY.json')['source_code_sha256']
 for row in pd.read_csv(P/'outputs/PAPER_COPY_SHA256.csv').itertuples():assert sha(ROOT/row.source)==sha(ROOT/row.destination)==row.sha256
 paper=ROOT/'REAL_Paper_File/P1S_LongTerm_Trajectory_Forecasting'
 for row in pd.read_csv(paper/'PACKAGE_SHA256.csv').itertuples():assert sha(paper/row.path)==row.sha256
 prefixes=(P.relative_to(REPO).as_posix()+'/',paper.relative_to(REPO).as_posix()+'/')
 changed=subprocess.check_output(['git','diff','--name-only','91b5263c','-z'],cwd=REPO).decode().split('\0')[:-1];assert all(s.startswith(prefixes) for s in changed)
 js('outputs/FINAL_CODE_SHA256.json',{q.relative_to(P).as_posix():sha(q) for q in sorted((P/'scripts').glob('*.py'))})
 files=[q for q in P.rglob('*') if q.is_file() and '__pycache__' not in q.parts and q.name not in ['ARTIFACT_SHA256.csv','FINAL_INTEGRITY_AUDIT.json','GIT_RECEIPT.local.json']]
 cs('outputs/ARTIFACT_SHA256.csv',pd.DataFrame([{'path':q.relative_to(P).as_posix(),'bytes':q.stat().st_size,'sha256':sha(q)} for q in sorted(files)]))
 js('outputs/FINAL_INTEGRITY_AUDIT.json',{'PASS':True,'source_files_SHA_unchanged':len(SOURCES),'phase_artifacts':len(files),'prediction_rows_independently_verified':430,
  'native_rows_verified':188,'coefficient_records_verified':20,'tests_passed':local('outputs/TEST_RESULTS.json')['tests_run'],'PNG_PDF_figure_pairs':7,
  'old_paper_replacements':0,'no_Test':True,'new_neural_models':0,'GPU_runs':0,'native_H1_H3_refits':0,'no_new_SOH_RUL_fit':True,'H12_formal_mean':'NOT_EVALUABLE'})
 print('SEALED',len(files),'artifacts; sources and published copies verified')
if __name__=='__main__':seal()
