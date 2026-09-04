# -*- coding: utf-8 -*-
"""FrequencyAwareTFT + Adaptive GRN Loop 20 验证"""
import sys, os, pickle, random, time
import numpy as np, pandas as pd
import torch, torch.nn as nn, torch.nn.functional as F, torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from scipy.signal import savgol_filter
from pathlib import Path
import matplotlib; matplotlib.use('Agg')

seed_value = 42
os.environ['PYTHONHASHSEED'] = str(seed_value)
random.seed(seed_value); np.random.seed(seed_value); torch.manual_seed(seed_value)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
USE_FC = 'FC2'; SAMPLING_INTERVAL = 0.5; MAX_LAG = 4; LAGS = [1, 2, 4]
N_LOOPS = 20

DATA_PATH = Path(r"E:\沈农信电\课题-预测质子交换膜燃料电池（PEMFC）的健康状态和剩余寿命\代码\Long-Term-PEMFC-DL-Prognostics-main\Long-Term-PEMFC-DL-Prognostics-main\Long-Term-PEMFC-DL-Prognostics-main\IEEE2014DataChallengeData")

# === 数据加载 ===
data_dict = {'FC1': {'ageing': []}, 'FC2': {'ageing': []}}
for dirpath, _, filenames in os.walk(DATA_PATH):
    for fname in filenames:
        if not fname.endswith('.csv'): continue
        df = pd.read_csv(os.path.join(dirpath, fname), encoding='ISO-8859-1')
        if 'FC1' in dirpath or 'FC1' in fname: data_dict['FC1']['ageing'].append(df)
        elif 'FC2' in dirpath or 'FC2' in fname: data_dict['FC2']['ageing'].append(df)

fc_dfs = [df for df in data_dict[USE_FC]["ageing"] if 'Time (h)' in df.columns]
age_df = pd.concat(fc_dfs, ignore_index=True).sort_values('Time (h)').reset_index(drop=True)
t_hours = age_df['Time (h)'].values
age_df['t_bin'] = (t_hours / SAMPLING_INTERVAL).astype('int32')
sampled_df = age_df.iloc[age_df.drop_duplicates(subset=['t_bin']).index].reset_index(drop=True)
if 't_bin' in sampled_df.columns: sampled_df = sampled_df.drop(columns=['t_bin'])
meas_cols = [c for c in sampled_df.columns if c != 'Time (h)']
s_df = sampled_df[meas_cols].copy().ffill().bfill().fillna(0)
meas_cols = [c for c in meas_cols if s_df[c].std() > 1e-10]
smoothed_df = pd.DataFrame({'Time (h)': sampled_df['Time (h)']})
for col in meas_cols:
    smoothed_df[col] = savgol_filter(s_df[col].values, 21, 2)
target_col = 'Utot (V)'; all_cols = list(meas_cols); target_idx = all_cols.index(target_col)
data_values = smoothed_df[all_cols].values.astype(np.float32); N = len(data_values)
print(f"FC2: {N}帧")

# L=1+lag
X_secs, y_secs = [], []
for lag in LAGS:
    lm = lag / 10.0
    X_secs.append(np.concatenate([data_values[MAX_LAG-lag:N-lag], np.full((N-MAX_LAG, 1), lm, dtype=np.float32)], axis=1))
    y_secs.append(data_values[MAX_LAG:N, target_idx])
X_raw = np.concatenate(X_secs, axis=0).astype(np.float32); y_raw = np.concatenate(y_secs, axis=0).astype(np.float32)
n_per_lag = N - MAX_LAG; n_total = len(X_raw)

# 衍生特征 + Freq特征
def _f(nm):
    for i, c in enumerate(all_cols):
        if nm in c: return i
    return 0
ui=_f('Utot'); ii=_f('I (A)'); twi=_f('TinWAT'); two=_f('ToutWAT'); pai=_f('PinAIR'); pao=_f('PoutAIR')
u1=_f('U1 '); u2=_f('U2 '); u3=_f('U3 '); u4=_f('U4 '); u5=_f('U5 '); ucl=[u1,u2,u3,u4,u5]
Xs=X_raw[:,:24]
du=np.zeros(n_total,dtype=np.float32)
for k in range(len(LAGS)):
    s=k*n_per_lag; e=s+n_per_lag; du[s:e]=np.diff(y_raw[s:e], prepend=y_raw[s])
v0_all=data_values[0,target_idx]; soh=y_raw/(v0_all+1e-8)
dtw=Xs[:,twi]-Xs[:,two]; pdv=Xs[:,pai]-Xs[:,pao]; ust=np.std(Xs[:,ucl],axis=1); umo=np.mean(Xs[:,ucl],axis=1)
roh=(Xs[:,ui]-umo)/(Xs[:,ii]+1e-6)
Xd=np.column_stack([du,soh,dtw,pdv,ust,roh]).astype(np.float32)
smooth_I = smoothed_df[[c for c in meas_cols if 'I (A)' in c][0]].values.astype(np.float32)
smooth_V = smoothed_df[target_col].values.astype(np.float32)
win=min(10,N//4); I_ripple=np.zeros(N,dtype=np.float32); V_fluct=np.zeros(N,dtype=np.float32)
for t in range(N):
    lo=max(0,t-win); I_ripple[t]=float(np.std(smooth_I[lo:t+1])/(np.mean(smooth_I[lo:t+1])+1e-8)); V_fluct[t]=float(np.std(smooth_V[lo:t+1]))
I_rip_lag=np.zeros(n_total,dtype=np.float32); V_fluc_lag=np.zeros(n_total,dtype=np.float32)
for k in range(len(LAGS)):
    s=k*n_per_lag; e=s+n_per_lag; I_rip_lag[s:e]=I_ripple[MAX_LAG:N]; V_fluc_lag[s:e]=V_fluct[MAX_LAG:N]
Xf=np.column_stack([I_rip_lag,V_fluc_lag]).astype(np.float32)
X_raw=np.concatenate([X_raw,Xd,Xf],axis=1); n_features=X_raw.shape[1]
print(f"特征: {n_features}维")

# 划分+归一化
tr_end=int(n_per_lag*0.5); vl_end=int(n_per_lag*0.6)
X_tr_p,X_vl_p,X_ts_p=[],[],[]; y_tr_p,y_vl_p,y_ts_p=[],[],[]
for k in range(len(LAGS)):
    s=k*n_per_lag
    X_tr_p.append(X_raw[s:s+tr_end]); X_vl_p.append(X_raw[s+tr_end:s+vl_end]); X_ts_p.append(X_raw[s+vl_end:s+n_per_lag])
    y_tr_p.append(y_raw[s:s+tr_end]); y_vl_p.append(y_raw[s+tr_end:s+vl_end]); y_ts_p.append(y_raw[s+vl_end:s+n_per_lag])
X_tr_r=np.concatenate(X_tr_p); X_vl_r=np.concatenate(X_vl_p); X_ts_r=np.concatenate(X_ts_p)
y_tr_r=np.concatenate(y_tr_p); y_vl_r=np.concatenate(y_vl_p); y_ts_r=np.concatenate(y_ts_p)
print(f"划分: 训练{len(X_tr_r)} 验证{len(X_vl_r)} 测试{len(X_ts_r)}")

sX=MinMaxScaler(feature_range=(0,1)); sy=MinMaxScaler(feature_range=(0,1))
X_tr_s=sX.fit_transform(X_tr_r); X_vl_s=sX.transform(X_vl_r); X_ts_s=sX.transform(X_ts_r)
y_tr_s=sy.fit_transform(y_tr_r.reshape(-1,1)).ravel(); y_vl_s=sy.transform(y_vl_r.reshape(-1,1)).ravel()
X_tr_t=torch.tensor(X_tr_s,dtype=torch.float32).reshape(-1,1,n_features)
X_vl_t=torch.tensor(X_vl_s,dtype=torch.float32).reshape(-1,1,n_features)
X_ts_t=torch.tensor(X_ts_s,dtype=torch.float32).reshape(-1,1,n_features)
y_tr_t=torch.tensor(y_tr_s,dtype=torch.float32).view(-1,1); y_vl_t=torch.tensor(y_vl_s,dtype=torch.float32).view(-1,1)

dV_tr=np.diff(y_tr_s,prepend=y_tr_s[0]); dV_vl=np.diff(y_vl_s,prepend=y_vl_s[0])
tr_ld=DataLoader(TensorDataset(X_tr_t,y_tr_t,torch.tensor(dV_tr,dtype=torch.float32).view(-1,1)),batch_size=64,shuffle=False)
vl_ld=DataLoader(TensorDataset(X_vl_t,y_vl_t,torch.tensor(dV_vl,dtype=torch.float32).view(-1,1)),batch_size=64,shuffle=False)
y_ts_s=sy.transform(y_ts_r.reshape(-1,1)).ravel(); dV_ts=np.diff(y_ts_s,prepend=y_ts_s[0])
X_ts_t2=torch.tensor(X_ts_s,dtype=torch.float32).reshape(-1,1,n_features)
y_ts_t2=torch.tensor(y_ts_s,dtype=torch.float32).view(-1,1)
ts_ld=DataLoader(TensorDataset(X_ts_t2,y_ts_t2,torch.tensor(dV_ts,dtype=torch.float32).view(-1,1)),batch_size=64,shuffle=False)

# === FrequencyAwareTFT + Adaptive GRN ===
class ARN(nn.Module):
    def __init__(self, d_in, d_hidden, d_out, dropout=0.1):
        super().__init__()
        self.lin1=nn.Linear(d_in,d_hidden); self.lin2=nn.Linear(d_hidden,d_out)
        self.gate_lin=nn.Linear(d_in,d_out); self.gate_mod=nn.Linear(1,d_out)
        self.proj=nn.Linear(d_in,d_out) if d_in!=d_out else nn.Identity()
        self.ln=nn.LayerNorm(d_out); self.dropout=nn.Dropout(dropout)
    def forward(self,x,dy=None):
        a=self.proj(x); z=F.elu(self.lin1(x)); z=self.lin2(z)
        g=torch.sigmoid(self.gate_lin(x))
        if dy is not None:
            mod=torch.tanh(self.gate_mod(dy.view(-1,1))); g=torch.sigmoid(torch.logit(g.clamp(1e-5,1-1e-5))+0.3*mod)
        return self.ln(a+self.dropout(g*z))

class FreqTFT(nn.Module):
    def __init__(self, input_size, hidden_size=96, nhead=4, d_ff=128, dropout=0.1):
        super().__init__()
        if hidden_size%nhead!=0: nhead=2
        self.hidden_size=hidden_size; self.freq_start=input_size-2
        self.feat_weights=nn.Parameter(torch.zeros(input_size)); self.feat_bias=nn.Parameter(torch.zeros(input_size))
        self.lstm=nn.LSTM(input_size,hidden_size,1,batch_first=True)
        self.grn=ARN(hidden_size,d_ff,hidden_size,dropout)
        self.attn=nn.MultiheadAttention(hidden_size,nhead,dropout=dropout,batch_first=True)
        self.attn_ln=nn.LayerNorm(hidden_size); self.do=nn.Dropout(dropout)
        self.fc_V=nn.Linear(hidden_size,1); self.fc_dV=nn.Linear(hidden_size,1)
    def get_dyn(self,x):
        return (x[:,0,self.freq_start]+x[:,0,self.freq_start+1])/2.0
    def forward(self,x):
        w=torch.sigmoid(self.feat_weights+self.feat_bias)
        x_w=x*w.unsqueeze(0).unsqueeze(0)
        if self.training: x_w=x_w+torch.randn_like(x_w)*0.01
        ds=self.get_dyn(x)
        lstm_out,_=self.lstm(x_w); lstm_out=self.do(lstm_out)
        grn_out=self.grn(lstm_out,ds)
        attn_out,_=self.attn(grn_out,grn_out,grn_out); attn_out=self.attn_ln(attn_out+grn_out)
        shared=attn_out[:,-1,:]
        return self.fc_V(shared),self.fc_dV(shared)

# 纯TFT基线 (无Adaptive GRN, 33维全用)
class PlainTFT(nn.Module):
    def __init__(self, input_size, hidden_size=96, nhead=4, d_ff=128, dropout=0.1):
        super().__init__()
        if hidden_size%nhead!=0: nhead=2
        self.feat_weights=nn.Parameter(torch.zeros(input_size)); self.feat_bias=nn.Parameter(torch.zeros(input_size))
        self.lstm=nn.LSTM(input_size,hidden_size,1,batch_first=True)
        self.grn_lin1=nn.Linear(hidden_size,d_ff); self.grn_lin2=nn.Linear(d_ff,hidden_size)
        self.grn_gate=nn.Linear(hidden_size,hidden_size); self.grn_proj=nn.Linear(hidden_size,hidden_size)
        self.grn_ln=nn.LayerNorm(hidden_size); self.grn_do=nn.Dropout(dropout)
        self.attn=nn.MultiheadAttention(hidden_size,nhead,dropout=dropout,batch_first=True)
        self.attn_ln=nn.LayerNorm(hidden_size); self.do=nn.Dropout(dropout)
        self.fc_V=nn.Linear(hidden_size,1); self.fc_dV=nn.Linear(hidden_size,1)
    def _grn(self,x):
        a=self.grn_proj(x); z=F.elu(self.grn_lin1(x)); z=self.grn_lin2(z)
        g=torch.sigmoid(self.grn_gate(x)); return self.grn_ln(a+self.grn_do(g*z))
    def forward(self,x):
        w=torch.sigmoid(self.feat_weights+self.feat_bias)
        x_w=x*w.unsqueeze(0).unsqueeze(0)
        if self.training: x_w=x_w+torch.randn_like(x_w)*0.01
        lstm_out,_=self.lstm(x_w); lstm_out=self.do(lstm_out)
        grn_out=self._grn(lstm_out)
        attn_out,_=self.attn(grn_out,grn_out,grn_out); attn_out=self.attn_ln(attn_out+grn_out)
        shared=attn_out[:,-1,:]
        return self.fc_V(shared),self.fc_dV(shared)

def train_one(model,tr_ld,vl_ld,epochs=200,patience=40,lr=3e-4):
    dev=device; model.to(dev); opt=optim.Adam(model.parameters(),lr=lr,weight_decay=1e-5)
    sch=optim.lr_scheduler.ReduceLROnPlateau(opt,mode='min',factor=0.5,patience=20)
    mse=nn.MSELoss()
    swa_start=40; swa_model=None; best_val=float('inf'); pc=0
    for ep in range(epochs):
        model.train(); tl=0.0
        for bx,by,bdV in tr_ld:
            bx,by,bdV=bx.to(dev),by.to(dev),bdV.to(dev); opt.zero_grad()
            Vp,dVp=model(bx)
            Lm=mse(Vp,by); LdV=nn.HuberLoss(delta=0.01)(dVp,bdV)
            df=Vp[1:]-Vp[:-1]; Lmo=torch.mean(torch.clamp(df,min=0)**2)
            if len(Vp)>=3: sd=Vp[2:]-2*Vp[1:-1]+Vp[:-2]; Lsm=torch.mean(sd**2)
            else: Lsm=torch.tensor(0.,device=dev)
            loss=Lm+0.3*LdV+0.1*Lmo+0.05*Lsm
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),0.5); opt.step(); tl+=Lm.item()
        atl=tl/len(tr_ld)
        model.eval(); vl=0.0
        with torch.no_grad():
            for bx,by,_ in vl_ld: bx,by=bx.to(dev),by.to(dev); Vp,_=model(bx); vl+=mse(Vp,by).item()
        avl=vl/len(vl_ld); sch.step(avl)
        if avl<best_val: best_val=avl; pc=0; torch.save(model.state_dict(),'_tmp.pth')
        else: pc+=1
        if pc>=patience: break
        if ep>=swa_start:
            if swa_model is None: swa_model={k:v.clone().cpu() for k,v in model.state_dict().items()}; sc=1
            else:
                for k in swa_model: swa_model[k]=swa_model[k]*(sc/(sc+1))+model.state_dict()[k].cpu()*(1/(sc+1))
                sc+=1
    model.load_state_dict(torch.load('_tmp.pth'))
    if swa_model is not None: model.load_state_dict({k:v.to(dev) for k,v in swa_model.items()})
    os.remove('_tmp.pth')
    return model

def eval_one(model,ts_ld):
    dev=device; model.eval(); pr,tr=[],[]
    with torch.no_grad():
        for bx,by,_ in ts_ld: bx,by=bx.to(dev),by.to(dev); Vp,_=model(bx); pr.append(Vp.cpu().numpy()); tr.append(by.cpu().numpy())
    ypn=np.concatenate(pr).ravel(); ytn=np.concatenate(tr).ravel()
    yp_real=sy.inverse_transform(ypn.reshape(-1,1)).ravel(); yt_real=sy.inverse_transform(ytn.reshape(-1,1)).ravel()
    return r2_score(yt_real,yp_real), yt_real, yp_real

# === Loop 20 (FreqTFT) ===
print(f"\n=== FreqTFT Loop {N_LOOPS} ===")
lr_freq=[]; t0=time.time()
for li in range(N_LOOPS):
    torch.manual_seed(seed_value+li*10)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed_value+li*10)
    m=FreqTFT(n_features,hidden_size=96,nhead=4,d_ff=128,dropout=0.1)
    m=train_one(m,tr_ld,vl_ld,epochs=200,patience=40,lr=3e-4)
    rl,yt,yp=eval_one(m,ts_ld)
    lr_freq.append((rl,yt,yp))
    print(f"  [{li+1}/{N_LOOPS}] R2={rl:.4f} ({time.time()-t0:.0f}s)")
freq_r2s=[r for r,_,_ in lr_freq]
print(f"FreqTFT: Min={min(freq_r2s):.4f} Mean={np.mean(freq_r2s):.4f} Max={max(freq_r2s):.4f}")

# === Loop 20 (PlainTFT on same 33-dim data) ===
print(f"\n=== PlainTFT Loop {N_LOOPS} ===")
lr_plain=[]; t0=time.time()
for li in range(N_LOOPS):
    torch.manual_seed(seed_value+li*10)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed_value+li*10)
    m=PlainTFT(n_features,hidden_size=96,nhead=4,d_ff=128,dropout=0.1)
    m=train_one(m,tr_ld,vl_ld,epochs=200,patience=40,lr=3e-4)
    rl,yt,yp=eval_one(m,ts_ld)
    lr_plain.append((rl,yt,yp))
    print(f"  [{li+1}/{N_LOOPS}] R2={rl:.4f} ({time.time()-t0:.0f}s)")
plain_r2s=[r for r,_,_ in lr_plain]
print(f"PlainTFT: Min={min(plain_r2s):.4f} Mean={np.mean(plain_r2s):.4f} Max={max(plain_r2s):.4f}")

print(f"\n=== 对比 ===")
print(f"FreqTFT (Adaptive GRN):     Min={min(freq_r2s):.4f} Mean={np.mean(freq_r2s):.4f} Max={max(freq_r2s):.4f}")
print(f"PlainTFT (33维,standard GRN): Min={min(plain_r2s):.4f} Mean={np.mean(plain_r2s):.4f} Max={max(plain_r2s):.4f}")
delta = np.mean(freq_r2s) - np.mean(plain_r2s)
print(f"差异: {delta:+.4f} (Freq/Adaptive - Plain)")
