"""Stage-local runtime, identity checks and read-only repository firewall."""
import sys,os,json,hashlib,platform,inspect
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import yaml
from data_alignment import P,PROJECT,N1,IDENTITY,H_PROTOCOL,CATALOG,ah,json_write,csv_write
sys.path.insert(0,str(P/'models'))
from health_fusion import cohort,MODELS,forward_model,gate_model,state_hash,seed_all
REPO=PROJECT.parent
def file_sha(path):
 path=Path(path).resolve()
 if path==N1.resolve():raise RuntimeError('FULL_N1_HASH_PROHIBITED')
 return hashlib.sha256(path.read_bytes()).hexdigest()
def configure(h):
 p=h['normal_runtime'];torch.set_num_threads(p['CPU_threads']);torch.use_deterministic_algorithms(False,warn_only=False)
 torch.backends.cudnn.deterministic=p['cudnn_deterministic'];torch.backends.cudnn.benchmark=p['cudnn_benchmark']
 torch.backends.cuda.matmul.allow_tf32=p['TF32_matmul'];torch.backends.cudnn.allow_tf32=p['TF32_cudnn']
 import pytorch_forecasting as pf
 result=dict(python=platform.python_version(),torch=str(torch.__version__),pytorch_forecasting=pf.__version__,CUDA=torch.version.cuda,cuDNN=torch.backends.cudnn.version(),
  GPU=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,CPU_threads=torch.get_num_threads(),cudnn_deterministic=torch.backends.cudnn.deterministic,
  cudnn_benchmark=torch.backends.cudnn.benchmark,torch_deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),warn_only=torch.is_deterministic_algorithms_warn_only_enabled(),
  TF32_matmul=torch.backends.cuda.matmul.allow_tf32,TF32_cudnn=torch.backends.cudnn.allow_tf32,CUBLAS_WORKSPACE_CONFIG=os.environ.get('CUBLAS_WORKSPACE_CONFIG'))
 for k,v in p.items():
  if k!='claim' and result[k]!=v:raise RuntimeError('H_RUNTIME_MISMATCH:'+k)
 return result
def predict(model,x,health,device,batch=32,return_gate=False):
 model.eval();out=[];gates=[]
 with torch.no_grad():
  for start in range(0,len(x),batch):
   xb=torch.from_numpy(x[start:start+batch]).to(device);hb=torch.from_numpy(health[start:start+batch]).to(device)
   out.append(forward_model(model,xb,hb).cpu().numpy())
   if return_gate:gates.append(gate_model(model,xb,hb).cpu().numpy().ravel())
 return (np.concatenate(out),np.concatenate(gates)) if return_gate else np.concatenate(out)
def inputs():
 d=dict(np.load(P/'inputs/MATCHED_INPUTS.npz',allow_pickle=False));fp=json.loads((P/'inputs/INPUT_FINGERPRINTS.json').read_text())
 for k,a in d.items():assert ah(a)==fp[k],k
 return d,fp
def load_protocol():
 p=P/'configs/FROZEN_PROTOCOL.json';assert file_sha(p)==(P/'configs/FROZEN_PROTOCOL.sha256').read_text().strip()
 cfg=json.loads(p.read_text(encoding='utf-8'))
 for path,digest in cfg['source_sha256'].items():assert file_sha(PROJECT/path)==digest,path
 for path,digest in cfg['execution_code_sha256'].items():assert file_sha(P/path)==digest,path
 for path,digest in cfg['upstream_sha256'].items():assert file_sha(Path(path))==digest,path
 return cfg,file_sha(p)
def install_firewall():
 # N1 is never opened in workers: they consume bounded, audited stage-local arrays.
 # Repository data outside this stage are restricted to exact frozen inputs/code.
 allowed=set()
 cfgpath=P/'configs/FROZEN_PROTOCOL.json'
 if cfgpath.exists():
  cfg=json.loads(cfgpath.read_text(encoding='utf-8'));allowed={str((PROJECT/k).resolve()).lower() for k in cfg['source_sha256']}
 else:
  roots=['Phase_F_TFT_DualPath_RegimeFusion/models','Phase_TFT_Original_Baseline/models']
  allowed={str(x.resolve()).lower() for r in roots for x in (PROJECT/r).glob('*.py')}
 def hook(event,args):
  if event!='open' or not args or not isinstance(args[0],(str,bytes,os.PathLike)):return
  path=Path(os.fsdecode(args[0])).resolve();mode=args[1] if len(args)>1 else 'r';flags=args[2] if len(args)>2 else 0
  writing=(isinstance(mode,str) and any(c in mode for c in 'wax+')) or (isinstance(flags,int) and bool(flags&(os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC)))
  if path.is_relative_to(REPO) and not path.is_relative_to(P):
   if writing:raise PermissionError('HISTORICAL_WRITE_PROHIBITED:'+str(path))
   if str(path).lower() not in allowed:raise PermissionError('NON_ALLOWLISTED_REPOSITORY_READ:'+str(path))
 sys.addaudithook(hook)
