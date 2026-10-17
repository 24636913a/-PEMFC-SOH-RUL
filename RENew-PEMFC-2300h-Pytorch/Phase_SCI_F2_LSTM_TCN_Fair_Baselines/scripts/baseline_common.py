"""New-phase writes only; frozen H data interface, N2 model class copies."""
from pathlib import Path
import os,sys,json,hashlib,time,subprocess,traceback,math
import numpy as np
import pandas as pd
import torch,yaml
P=Path(__file__).resolve().parents[1];ROOT=P.parent;REPO=ROOT.parent
H=ROOT/'Phase_H_Determinism_MultiSeed_Stability';N2=ROOT/'Phase_N2_Model_Screening'
R2=ROOT/'Phase_SCI_F2_Seed_Screening_Round2';PAPER=ROOT/'Phase_SCI_F2_Literature_Benchmarked_Paper_Integration'
BASE='d41ae91689cc7ebe1d83fc4351d01122023e914e'
sys.path.insert(0,str(H/'scripts'));import h_common as h
sys.path.insert(0,str(P/'models'));from baseline_models import LSTMForecaster,TCNForecaster
RUNS=[('LSTM',42),('TCN',42),('LSTM',52),('LSTM',62),('TCN',52),('TCN',62)]
ARCH=dict(LSTM=dict(input_dim=14,horizon=24,hidden_dim=64,num_layers=2,dropout=.1),TCN=dict(input_dim=14,horizon=24,hidden_dim=48,num_layers=3,kernel_size=3,dropout=.1))

def sha(f):
    f=Path(f);assert f.name!='causal_features_30min.csv','FULL_MIXED_RAW_READ_FORBIDDEN'
    return hashlib.sha256(f.read_bytes()).hexdigest()

def path(name):
    f=(P/name).resolve();assert f.is_relative_to(P),'HISTORY_WRITE_FORBIDDEN';f.parent.mkdir(parents=True,exist_ok=True);return f

def replace(src,dst):
    dst=path(dst)
    for i,delay in enumerate([.02,.05,.1,.2,.5,1.,None]):
        try:os.replace(src,dst);return
        except PermissionError as exc:
            event('IO_RENAME_RETRY',path=dst.relative_to(P).as_posix(),attempt=i+1,error=repr(exc))
            if delay is None:raise
            time.sleep(delay)

def write(name,s):
    f=path(name);tmp=f.with_suffix(f.suffix+'.writing')
    with tmp.open('w',encoding='utf-8',newline='\n') as stream:stream.write(s);stream.flush();os.fsync(stream.fileno())
    replace(tmp,f)

def js(name,obj):write(name,json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def read(f):return json.loads(Path(f).read_text(encoding='utf-8'))
def rd(f):return pd.read_csv(f,float_precision='round_trip')
def cs(name,f):write(name,pd.DataFrame(f).to_csv(index=False,float_format='%.17g',lineterminator='\n'))
def event(kind,**kw):
    f=path('outputs/execution_ledger/events.jsonl')
    with f.open('a',encoding='utf-8',newline='\n') as stream:stream.write(json.dumps(dict(UTC=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),event=kind,**kw),ensure_ascii=False,allow_nan=False)+'\n');stream.flush();os.fsync(stream.fileno())

def cfg():
    f=P/'configs/FROZEN_BASELINE_PROTOCOL.yaml';assert sha(f)==f.with_suffix('.sha256').read_text().strip(),'PROTOCOL_CHANGED'
    c=yaml.safe_load(f.read_text(encoding='utf-8'));assert c['runs']==[dict(model=m,seed=s) for m,s in RUNS]
    for q,v in c['critical_sha256'].items():assert sha(P/q)==v,('EXECUTION_CODE_CHANGED',q)
    for q,v in c['data_sha256'].items():assert sha(P/q)==v,('DATA_CHANGED',q)
    for q,v in c['sources_sha256'].items():assert sha(REPO/q)==v,('SOURCE_CHANGED',q)
    return c

def data():return np.load(P/'outputs/audits/FROZEN_H_DATA.npz',allow_pickle=False)
def model(name,device):return {'LSTM':LSTMForecaster,'TCN':TCNForecaster}[name](**ARCH[name]).to(device)
def predict(m,x,device):
    m.eval();out=[]
    with torch.no_grad():
        for b in range(0,len(x),32):out.append(m(torch.from_numpy(x[b:b+32]).to(device)).cpu().numpy())
    return np.concatenate(out)

def metrics(f):
    y=f.y_true_delta_V.to_numpy('float64');e=(f.y_pred_delta_V.to_numpy('float64')-y)*1000;a=abs(e)
    z=e[f.horizon_step.to_numpy()==24];se=pd.DataFrame(dict(t=f.target_row,se=e*e)).groupby('t').se.mean()
    return dict(RMSE_mV=float(np.sqrt(np.mean(e*e))),MAE_mV=float(a.mean()),R2=float(1-np.sum((e/1000)**2)/np.sum((y-y.mean())**2)),endpoint_RMSE_mV=float(np.sqrt(np.mean(z*z))),target_time_equal_RMSE_mV=float(np.sqrt(se.mean())),P95_mV=float(np.quantile(a,.95)),P99_mV=float(np.quantile(a,.99)),max_abs_error_mV=float(a.max()),n=len(f),origins=int(f.origin_row.nunique()),unique_targets=int(f.target_row.nunique()))

def frame(name,seed,pred,d):
    f=pd.DataFrame(dict(model=name,seed=seed,origin_row=np.repeat(d['val_origins'],24),target_row=d['val_targets'].ravel(),horizon_step=np.tile(np.arange(1,25),205),origin_age_h=np.repeat(d['val_origin_age_h'],24),target_age_h=d['val_target_age_h'].ravel(),origin_raw_voltage_V=np.repeat(d['val_origin_raw_voltage_V'],24),y_true_delta_V=d['val_y'].ravel(),y_pred_delta_V=pred.ravel(),true_raw_target_voltage_V=d['val_target_raw_voltage_V'].ravel()))
    f['true_label_voltage_V']=f.origin_raw_voltage_V+f.y_true_delta_V;f['predicted_voltage_V']=f.origin_raw_voltage_V+f.y_pred_delta_V
    return f

def assert_keys(f):
    k=rd(P/'outputs/audits/VALIDATION_KEY_TRUTH.csv')
    assert len(f)==4920
    for col in k:assert np.array_equal(f[col],k[col]),('KEY_TRUTH_CHANGED',col)

def protect():
    f=rd(P/'outputs/audits/HISTORY_BEFORE.csv')
    for r in f.itertuples():assert sha(REPO/r.path)==r.sha256,('HISTORY_CHANGED',r.path)
    return len(f)
