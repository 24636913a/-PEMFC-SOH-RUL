"""T1-R firewall and isolated canonical-TFT loading; it never imports T1 runner."""
from __future__ import annotations
import hashlib, importlib.util, subprocess, sys
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd
import yaml

SCRIPT_DIR=Path(__file__).resolve().parent; PHASE=SCRIPT_DIR.parent; PROJECT=PHASE.parent; REPO=PROJECT.parent
T0=PROJECT/'Phase_TFT_Original_Baseline'; N1=PROJECT/'Phase_N1_Data_Foundation'/'outputs'/'causal_feature_tables'/'causal_features_30min.csv'
PROTOCOL=PHASE/'configs'/'PREREGISTERED_T1R_PROTOCOL.yaml'; PROTOCOL_SHA=PHASE/'configs'/'PREREGISTERED_T1R_PROTOCOL.sha256'
COLS=['bin_start_ophrs2_h','bin_end_ophrs2_h','last_datetime','temporal_segment_id','bin_valid_for_windows','cell_voltage_mean','cell_voltage_std','cell_voltage_range','cell_voltage_min','cell_voltage_max','actual_current_A_mean','actual_current_A_std','actual_current_A_peak_to_peak','actual_current_density_A_cm2_mean','actual_current_density_A_cm2_std','actual_current_density_A_cm2_peak_to_peak','load_transition_count','operation_point_transitions','zero_current_fraction']
class T1RFailure(RuntimeError): pass
def sha(path: Path)->str: return hashlib.sha256(path.read_bytes()).hexdigest()
def load_protocol()->tuple[dict[str,Any],str]:
    expected=PROTOCOL_SHA.read_text(encoding='utf-8').split()[0]; observed=sha(PROTOCOL)
    if observed!=expected: raise T1RFailure('T1R_PROTOCOL_HASH_MISMATCH')
    cfg=yaml.safe_load(PROTOCOL.read_text(encoding='utf-8'))
    if cfg.get('status')!='FROZEN_BEFORE_SEPARATE_EXECUTION': raise T1RFailure('T1R_PROTOCOL_NOT_FROZEN')
    if subprocess.run(['git','-C',str(REPO),'merge-base','--is-ancestor',cfg['base_commit'],'HEAD'],check=False).returncode: raise T1RFailure('T1R_BASE_NOT_ANCESTOR')
    return cfg,observed
def safe_development(cfg:dict[str,Any])->tuple[pd.DataFrame,dict[str,Any]]:
    prefix=pd.read_csv(N1,usecols=COLS,nrows=3007)
    if len(prefix)!=3007: raise T1RFailure('PREFIX_ROW_COUNT')
    dev=prefix.loc[prefix.bin_end_ophrs2_h.to_numpy(float)<1819.997738].copy().reset_index(drop=True)
    if len(dev)!=3005: raise T1RFailure('DEVELOPMENT_ROW_COUNT')
    return dev,{'nrows':3007,'development_rows':3005,'full_file_hash':False,'test_accessed':False}
def load_t0_model()->tuple[type,Any]:
    spec=importlib.util.spec_from_file_location('t1r_canonical_tft',T0/'models'/'canonical_tft.py')
    if spec is None or spec.loader is None: raise T1RFailure('T0_MODEL_IMPORT')
    module=importlib.util.module_from_spec(spec); sys.modules['t1r_canonical_tft']=module; spec.loader.exec_module(module)
    return module.CanonicalStrictHistoryTFT,module.count_parameters
