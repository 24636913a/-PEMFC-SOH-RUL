"""T1-R runner; training is token-gated and is not invoked during preparation."""
from __future__ import annotations
import argparse, hashlib
import numpy as np
import torch
from torch.utils.data import DataLoader,TensorDataset
from t1r_common import PHASE,T1RFailure,load_protocol,safe_development,load_t0_model
from t1r_decision import check_trial_budget
from t1r_optuna import assert_new_study_path

def make_loader(x:np.ndarray,y:np.ndarray,batch_size:int,seed:int)->DataLoader:
    """Create once per fit. Reusing this loader advances its generator like T0."""
    return DataLoader(TensorDataset(torch.from_numpy(x),torch.from_numpy(y)),batch_size=batch_size,shuffle=True,generator=torch.Generator().manual_seed(seed),num_workers=0,drop_last=False)
def validate_isolation(cfg:dict)->None:
    path=str(cfg['search']['database_relative_path'])
    if 'Phase_TFT_Hyperparameter_Optimization/' in path or 'T1R' not in cfg['search']['study_name_prefix']:
        raise T1RFailure('OLD_T1_STUDY_ISOLATION_FAILURE')
    if check_trial_budget(16,12,['COMPLETE']*12+['FAILED']*4)['pass'] is not True: raise T1RFailure('TRIAL_BUDGET_ENGINE_FAILURE')
    assert_new_study_path(PHASE / 'outputs' / 'optuna_T1R_fold_1.db', PHASE, PHASE.parent / 'Phase_TFT_Hyperparameter_Optimization')
def check_authorization(token:str)->None:
    # An absent authorization artifact is deliberately fail-closed in this phase.
    authorization=PHASE/'configs'/'T1R_EXECUTION_AUTHORIZATION.yaml'
    if not authorization.is_file(): raise T1RFailure('NO_SEPARATE_T1R_EXECUTION_AUTHORIZATION')
    if token!='EXECUTE_T1R_SEPARATE_TRAINING': raise T1RFailure('EXECUTION_TOKEN_INVALID')
def main()->None:
    parser=argparse.ArgumentParser(); parser.add_argument('--authorization',default='PREPARATION_ONLY'); args=parser.parse_args()
    cfg,protocol_sha=load_protocol(); validate_isolation(cfg); _,audit=safe_development(cfg); model,_=load_t0_model()
    if args.authorization=='EXECUTE_T1R_SEPARATE_TRAINING': check_authorization(args.authorization); raise T1RFailure('AUTHORIZATION_FILE_SCHEMA_NOT_YET_APPROVED')
    print({'status':'T1R_READY_FOR_SEPARATE_EXECUTION','protocol_sha256':protocol_sha,'data_audit':audit,'canonical_tft_loaded':model.__name__,'model_training_performed':False,'test_accessed':False})
if __name__=='__main__': main()
