"""Phase-local writes; exact repository allowlist; bounded logical N1 access."""
from __future__ import annotations
import hashlib, json, os, sys
from pathlib import Path
from datetime import datetime,timezone
import numpy as np
import pandas as pd

PHASE=Path(__file__).resolve().parents[1]
PROJECT=PHASE.parent
REPO=PROJECT.parent
P3=PROJECT/"Phase_P3_Full450_Epoch_Sensitivity"
P1=PROJECT/"Phase_P1_Gate_Ablation_Engineering_Preflight"
F=PROJECT/"Phase_F_TFT_DualPath_RegimeFusion"
H=PROJECT/"Phase_H_Determinism_MultiSeed_Stability"
E=PROJECT/"Phase_TFT_FullFeature_CausalFilter_Comparison"
C=PROJECT/"Phase_TFT_SingleSplit_Feature_Filter_Ablation"
N1=PROJECT/"Phase_N1_Data_Foundation/outputs/causal_feature_tables/causal_features_30min.csv"
PROTOCOL=PHASE/"configs/DDP_SOFT_PHYSICS_FROZEN_PROTOCOL.json"
BASE="8997b8c881f206680235361b36f892f0edc2af62"
EVENTS=[];READS=[];_N1_OPEN=False;_INSTALLED=False
STATIC=[P3/"configs/P3_FROZEN_FULL450_PROTOCOL.json",
        P3/"configs/P3_FROZEN_FULL450_PROTOCOL.sha256",
        P3/"outputs/audits/P3_DATA_AUTHORITY.json",
        P1/"scripts/gate_ablation_models.py",
        E/"scripts/run_e_comparison.py",C/"scripts/run_c_ablation.py",
        E/"configs/PREREGISTERED_E_COMPARISON.yaml",
        E/"configs/PREREGISTERED_E_COMPARISON.sha256",
        E/"outputs/E_COMMON_SCALER.json",H/"scripts/h_common.py",
        PROJECT/"Phase_TFT_Original_Baseline/models/canonical_tft.py"]
STATIC+=sorted((F/"models").glob("*.py"))
for seed in (42,52,62):
    folder=P3/f"outputs/training_logs/P3-G2-Seed{seed}"
    STATIC+=[folder/"initial.json"]
    STATIC+=[P3/f"outputs/audits/sampler_authority/seed{seed}_epoch001.json"]
    for role in ("best","last450"):
        STATIC+=[folder/f"{role}_result.json",
                 P3/f"outputs/checkpoints/P3-G2-Seed{seed}_{role}.pt"]
ALLOW={str(p.resolve()).casefold() for p in STATIC}
ALLOW.update(str(p.resolve()).casefold() for p in (P3/"scripts").glob("*.py"))

def utc():return datetime.now(timezone.utc).isoformat()
def read_json(path):return json.loads(Path(path).read_text(encoding="utf-8"))
def write_json(path,value):
    path=Path(path).resolve()
    if not path.is_relative_to(PHASE):raise PermissionError("NEW_PHASE_OUTPUT_ONLY")
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+".writing")
    with tmp.open("w",encoding="utf-8",newline="\n") as handle:
        json.dump(value,handle,ensure_ascii=False,indent=2,allow_nan=False)
        handle.write("\n");handle.flush();os.fsync(handle.fileno())
    os.replace(tmp,path)

def permitted_open(path,writing=False):
    if isinstance(path,int):return
    p=Path(os.fsdecode(path)).resolve()
    if p==N1.resolve():
        if writing or not _N1_OPEN:raise PermissionError("N1_ONLY_IN_NROWS_3007_READER")
        return
    if p.is_relative_to(PHASE):return
    if p.is_relative_to(REPO):
        if writing or str(p).casefold() not in ALLOW:
            raise PermissionError("DDP_REPOSITORY_SOURCE_NOT_ALLOWLISTED")
    elif "数据" in p.parts or p.name.startswith(("Loadcycle_","Polcurve_")) or p.suffix.lower() in (".pt",".npz",".csv",".parquet"):
        raise PermissionError("DDP_EXTERNAL_REAL_DATA_DENIED")

def validate_read_csv_request(path,kwargs):
    p=Path(path).resolve()
    if p==N1.resolve():
        if kwargs.get("nrows")!=3007 or kwargs.get("chunksize") or kwargs.get("iterator"):
            raise PermissionError("N1_REQUIRES_NROWS_3007")
    elif not p.is_relative_to(PHASE):
        raise PermissionError("OTHER_HISTORICAL_CSV_NOT_ALLOWED")

def install_firewall():
    global _INSTALLED
    if _INSTALLED:return
    def hook(event,args):
        if event!="open" or isinstance(args[0],int):return
        path,mode,flags=args
        writing=bool((isinstance(mode,str) and any(c in mode for c in "wax+")) or
                     (isinstance(flags,int) and flags&(os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC|os.O_APPEND)))
        try:permitted_open(path,writing)
        except PermissionError:
            EVENTS.append({"path":os.fsdecode(path),"writing":writing,"action":"DENIED"});raise
        if str(REPO).casefold() in os.fsdecode(path).casefold():
            EVENTS.append({"path":os.fsdecode(path),"writing":writing,"action":"ALLOWED"})
    sys.addaudithook(hook)
    original=pd.read_csv
    def bounded(path,*args,**kwargs):
        global _N1_OPEN
        validate_read_csv_request(path,kwargs)
        previous=_N1_OPEN;_N1_OPEN=Path(path).resolve()==N1.resolve()
        try:
            value=original(path,*args,**kwargs)
            if _N1_OPEN:READS.append({"read_nrows":3007,"parsed_rows":len(value)})
            return value
        finally:_N1_OPEN=previous
    pd.read_csv=bounded;_INSTALLED=True

def file_hash(path):
    p=Path(path).resolve()
    if p==N1.resolve():raise PermissionError("FULL_N1_HASH_PROHIBITED")
    permitted_open(p)
    digest=hashlib.sha256()
    with p.open("rb") as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b""):digest.update(chunk)
    return digest.hexdigest()

def array_hash(value):
    value=np.ascontiguousarray(value)
    h=hashlib.sha256(str((value.shape,str(value.dtype))).encode())
    h.update(value.tobytes());return h.hexdigest()

def read_development_prefix(path=N1):
    # nrows and the filtering order are part of the production contract.
    frame=pd.read_csv(path,nrows=3007)
    if len(frame)!=3007:raise RuntimeError("PREFIX_COUNT_MISMATCH")
    dev=frame.loc[frame.bin_end_ophrs2_h<1819.997738].reset_index(drop=True)
    if len(dev)!=3005:raise RuntimeError("DEVELOPMENT_ROW_COUNT_MISMATCH")
    payload=dev.to_csv(index=False,header=True,lineterminator="\n",float_format="%.17g").encode("utf-8")
    return dev,hashlib.sha256(payload).hexdigest()

def load_legacy():
    # Frozen import only: no old runner or output-writing entry point called.
    sys.path.insert(0,str(P1/"scripts"))
    import gate_ablation_models as a
    sys.path.insert(0,str(E/"scripts"))
    import run_e_comparison as e
    return a,e

def load_protocol():
    cfg=read_json(PROTOCOL)
    digest=file_hash(PROTOCOL)
    if digest!=PROTOCOL.with_suffix(".sha256").read_text().split()[0]:
        raise RuntimeError("PROTOCOL_SHA_MISMATCH")
    verify_hash_manifest(cfg["input_and_code_sha256"])
    return cfg,digest

def verify_hash_manifest(records):
    for path,expected in records.items():
        if file_hash(Path(path))!=expected:
            raise RuntimeError("FROZEN_INPUT_OR_CODE_CHANGED:"+path)

def runtime_access_audit():
    return {"read_nrows":3007,"development_filter":"bin_end_ophrs2_h < 1819.997738",
            "development_rows":3005 if READS else None,"bounded_reads":READS,
            "full_N1_hash":False,"new_Test_access":False,"Assessment_windows_built":False,
            "Validation_windows_built_for_qualification":False,"new_polarization_or_health_target_read":False,
            "events":EVENTS}
