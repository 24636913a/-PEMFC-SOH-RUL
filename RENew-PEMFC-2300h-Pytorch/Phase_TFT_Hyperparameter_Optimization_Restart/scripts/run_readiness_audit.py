"""Produce T1-R auditable readiness artifacts without executing any training."""
from __future__ import annotations
import hashlib, json, subprocess, sys, unittest
from pathlib import Path
from t1r_common import PHASE, PROJECT, load_protocol, safe_development
from run_t1r import validate_isolation

OLD=PROJECT/'Phase_TFT_Hyperparameter_Optimization'
ARCHIVE=Path('E:/沈农信电/T1_Original_Invalidated_Archive/T1_original_2026-10-09/optuna_fold_1.db')
def sha(p:Path)->str: return hashlib.sha256(p.read_bytes()).hexdigest()
def main()->None:
    cfg,protocol_sha=load_protocol(); _,data=safe_development(cfg); validate_isolation(cfg)
    suite=unittest.defaultTestLoader.discover(str(PHASE/'tests'),pattern='test_*.py'); result=unittest.TextTestRunner(stream=open(Path('NUL'),'w'),verbosity=0).run(suite)
    tests={'tests_run':result.testsRun,'failures':len(result.failures),'errors':len(result.errors),'passed':result.wasSuccessful(),'synthetic_only':True}
    (PHASE/'outputs').mkdir(exist_ok=True)
    (PHASE/'outputs'/'T1_DECISION_UNIT_TEST_RESULTS.json').write_text(json.dumps(tests,indent=2)+'\n',encoding='utf-8')
    (PHASE/'outputs'/'T1R_STATIC_TEST_RESULTS.json').write_text(json.dumps(tests,indent=2)+'\n',encoding='utf-8')
    original={'protocol_sha256':sha(OLD/'configs'/'PREREGISTERED_T1_PROTOCOL.yaml'),'runner_sha256':sha(OLD/'scripts'/'run_t1.py'),'database_sha256':sha(OLD/'outputs'/'optuna_fold_1.db'),'archive_sha256':sha(ARCHIVE),'archive_exists':ARCHIVE.is_file()}
    readiness={'status':'T1R_EXECUTION_CHAIN_STATIC_PASS' if tests['passed'] and original['database_sha256']==original['archive_sha256'] else 'T1R_EXECUTION_CHAIN_STATIC_FAIL','protocol_sha256':protocol_sha,'original_t1':original,'data_firewall':data,'new_trials_created':False,'model_training_performed':False,'test_accessed':False,'old_t1_runner_imported':False,'study_prefix':cfg['search']['study_name_prefix'],'database_path_template':cfg['search']['database_relative_path']}
    (PHASE/'outputs'/'T1R_EXECUTION_READINESS.json').write_text(json.dumps(readiness,indent=2)+'\n',encoding='utf-8')
    (PHASE/'outputs'/'T1R_FINAL_STATIC_READINESS.json').write_text(json.dumps(readiness,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(readiness))
    if readiness['status']!='T1R_EXECUTION_CHAIN_STATIC_PASS': raise SystemExit(1)
if __name__=='__main__': main()
