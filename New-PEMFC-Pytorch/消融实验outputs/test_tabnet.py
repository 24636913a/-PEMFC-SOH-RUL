# -*- coding: utf-8 -*-
"""TabNet 快速验证: FC2, L=1+lag, 33维 (含Freq特征)"""
import sys, os, pickle, random
import numpy as np, pandas as pd
import torch, torch.nn as nn, torch.nn.functional as F, torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score, mean_squared_error
from scipy.signal import savgol_filter
from pathlib import Path
import matplotlib; matplotlib.use('Agg')

seed_value = 42
os.environ['PYTHONHASHSEED'] = str(seed_value)
random.seed(seed_value); np.random.seed(seed_value); torch.manual_seed(seed_value)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
USE_FC = 'FC2'; SAMPLING_INTERVAL = 0.5; MAX_LAG = 4; LAGS = [1, 2, 4]

DATA_PATH = Path(r"E:\沈农信电\课题-预测质子交换膜燃料电池（PEMFC）的健康状态和剩余寿命\代码\Long-Term-PEMFC-DL-Prognostics-main\Long-Term-PEMFC-DL-Prognostics-main\Long-Term-PEMFC-DL-Prognostics-main\IEEE2014DataChallengeData")

# === 数据加载 (同FT-Transformer) ===
data_dict = {'FC1': {'ageing': []}, 'FC2': {'ageing': []}}
for dirpath, _, filenames in os.walk(DATA_PATH):
    for fname in filenames:
        if not fname.endswith('.csv'): continue
        fpath = os.path.join(dirpath, fname)
        df = pd.read_csv(fpath, encoding='ISO-8859-1')
        if 'FC1' in dirpath or 'FC1' in fname: data_dict['FC1']['ageing'].append(df)
        elif 'FC2' in dirpath or 'FC2' in fname: data_dict['FC2']['ageing'].append(df)

fc_dfs = [df for df in data_dict[USE_FC]["ageing"] if 'Time (h)' in df.columns]
age_df = pd.concat(fc_dfs, ignore_index=True).sort_values('Time (h)').reset_index(drop=True)
t_hours = age_df['Time (h)'].values
age_df['t_bin'] = (t_hours / SAMPLING_INTERVAL).astype('int32')
sampled_idx = age_df.drop_duplicates(subset=['t_bin']).index
sampled_df = age_df.iloc[sampled_idx].reset_index(drop=True)
if 't_bin' in sampled_df.columns: sampled_df = sampled_df.drop(columns=['t_bin'])
meas_cols = [c for c in sampled_df.columns if c != 'Time (h)']
s_df = sampled_df[meas_cols].copy().ffill().bfill().fillna(0)
meas_cols = [c for c in meas_cols if s_df[c].std() > 1e-10]
smoothed_df = pd.DataFrame({'Time (h)': sampled_df['Time (h)']})
for col in meas_cols:
    smoothed_df[col] = savgol_filter(s_df[col].values, 21, 2)
target_col = 'Utot (V)'; all_cols = list(meas_cols)
target_idx = all_cols.index(target_col)
data_values = smoothed_df[all_cols].values.astype(np.float32); N = len(data_values)

# L=1+lag
X_secs, y_secs = [], []
for lag in LAGS:
    lm = lag / 10.0
    X_secs.append(np.concatenate([data_values[MAX_LAG-lag:N-lag],
                 np.full((N-MAX_LAG, 1), lm, dtype=np.float32)], axis=1))
    y_secs.append(data_values[MAX_LAG:N, target_idx])
X_raw = np.concatenate(X_secs, axis=0).astype(np.float32)
y_raw = np.concatenate(y_secs, axis=0).astype(np.float32)
n_per_lag = N - MAX_LAG; n_total = len(X_raw)

# 衍生+Freq (同FT-Transformer)
def _f(nm):
    for i, c in enumerate(all_cols):
        if nm in c: return i
    return 0
ui=_f('Utot'); ii=_f('I (A)'); twi=_f('TinWAT'); two=_f('ToutWAT')
pai=_f('PinAIR'); pao=_f('PoutAIR')
u1=_f('U1 '); u2=_f('U2 '); u3=_f('U3 '); u4=_f('U4 '); u5=_f('U5 ')
ucl=[u1,u2,u3,u4,u5]; Xs=X_raw[:,:24]
du=np.zeros(n_total,dtype=np.float32)
for k in range(len(LAGS)):
    s=k*n_per_lag; e=s+n_per_lag
    du[s:e]=np.diff(y_raw[s:e], prepend=y_raw[s])
v0_all=data_values[0,target_idx]; soh=y_raw/(v0_all+1e-8)
dtw=Xs[:,twi]-Xs[:,two]; pdv=Xs[:,pai]-Xs[:,pao]
ust=np.std(Xs[:,ucl],axis=1); umo=np.mean(Xs[:,ucl],axis=1)
roh=(Xs[:,ui]-umo)/(Xs[:,ii]+1e-6)
Xd=np.column_stack([du,soh,dtw,pdv,ust,roh]).astype(np.float32)
smooth_I = smoothed_df[[c for c in meas_cols if 'I (A)' in c or 'I(A' in c][0]].values.astype(np.float32)
smooth_V = smoothed_df[target_col].values.astype(np.float32)
win=min(10,N//4)
I_ripple=np.zeros(N,dtype=np.float32); V_fluct=np.zeros(N,dtype=np.float32)
for t in range(N):
    lo=max(0,t-win)
    I_ripple[t]=float(np.std(smooth_I[lo:t+1])/(np.mean(smooth_I[lo:t+1])+1e-8))
    V_fluct[t]=float(np.std(smooth_V[lo:t+1]))
I_rip_lag=np.zeros(n_total,dtype=np.float32); V_fluc_lag=np.zeros(n_total,dtype=np.float32)
for k in range(len(LAGS)):
    s=k*n_per_lag; e=s+n_per_lag
    I_rip_lag[s:e]=I_ripple[MAX_LAG:N]; V_fluc_lag[s:e]=V_fluct[MAX_LAG:N]
Xf=np.column_stack([I_rip_lag,V_fluc_lag]).astype(np.float32)
X_raw=np.concatenate([X_raw,Xd,Xf],axis=1)
n_features=X_raw.shape[1]

# 划分+归一化
tr_end=int(n_per_lag*0.5); vl_end=int(n_per_lag*0.6)
X_tr_p,X_vl_p,X_ts_p=[],[],[]
y_tr_p,y_vl_p,y_ts_p=[],[],[]
for k in range(len(LAGS)):
    s=k*n_per_lag
    X_tr_p.append(X_raw[s:s+tr_end]); X_vl_p.append(X_raw[s+tr_end:s+vl_end]); X_ts_p.append(X_raw[s+vl_end:s+n_per_lag])
    y_tr_p.append(y_raw[s:s+tr_end]); y_vl_p.append(y_raw[s+tr_end:s+vl_end]); y_ts_p.append(y_raw[s+vl_end:s+n_per_lag])
X_tr_r=np.concatenate(X_tr_p); X_vl_r=np.concatenate(X_vl_p); X_ts_r=np.concatenate(X_ts_p)
y_tr_r=np.concatenate(y_tr_p); y_vl_r=np.concatenate(y_vl_p); y_ts_r=np.concatenate(y_ts_p)
sX=MinMaxScaler(feature_range=(0,1)); sy=MinMaxScaler(feature_range=(0,1))
X_tr_s=sX.fit_transform(X_tr_r); X_vl_s=sX.transform(X_vl_r); X_ts_s=sX.transform(X_ts_r)
y_tr_s=sy.fit_transform(y_tr_r.reshape(-1,1)).ravel()
y_vl_s=sy.transform(y_vl_r.reshape(-1,1)).ravel()
y_ts_s=sy.transform(y_ts_r.reshape(-1,1)).ravel()
X_tr_t=torch.tensor(X_tr_s,dtype=torch.float32).unsqueeze(1)
X_vl_t=torch.tensor(X_vl_s,dtype=torch.float32).unsqueeze(1)
X_ts_t=torch.tensor(X_ts_s,dtype=torch.float32).unsqueeze(1)
y_tr_t=torch.tensor(y_tr_s,dtype=torch.float32).view(-1,1)
y_vl_t=torch.tensor(y_vl_s,dtype=torch.float32).view(-1,1)
y_ts_t=torch.tensor(y_ts_s,dtype=torch.float32).view(-1,1)
dV_tr=np.diff(y_tr_s,prepend=y_tr_s[0]); dV_vl=np.diff(y_vl_s,prepend=y_vl_s[0])
tr_ld=DataLoader(TensorDataset(X_tr_t,y_tr_t,torch.tensor(dV_tr,dtype=torch.float32).view(-1,1)),batch_size=64,shuffle=False)
vl_ld=DataLoader(TensorDataset(X_vl_t,y_vl_t,torch.tensor(dV_vl,dtype=torch.float32).view(-1,1)),batch_size=64,shuffle=False)

# === TabNet (简化版: Ghost BatchNorm + 稀疏特征选择) ===
class GLULayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.fc = nn.Linear(in_dim, out_dim * 2)
    def forward(self, x):
        x = self.fc(x)
        return x[:, :x.shape[-1]//2] * torch.sigmoid(x[:, x.shape[-1]//2:])

class FeatureBlock(nn.Module):
    def __init__(self, n_features, n_d, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(n_features, n_d)
        self.bn1 = nn.BatchNorm1d(n_d)
        self.glu = GLULayer(n_d, n_d)
        self.fc2 = nn.Linear(n_d, n_features)
        self.bn2 = nn.BatchNorm1d(n_features)
        self.do = nn.Dropout(dropout)
    def forward(self, x):
        h = self.do(self.bn1(self.fc1(x)))
        h = self.glu(h)
        h = self.do(self.fc2(h))
        return self.bn2(h + x)  # skip connection

class AttentiveMask(nn.Module):
    """简单版Feature Mask: 基于当前特征 → sparse selection"""
    def __init__(self, n_features):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(n_features, n_features * 2), nn.BatchNorm1d(n_features * 2))
    def forward(self, x, prior_scale):
        mask_raw = self.net(x).chunk(2, dim=-1)[0]
        mask = F.softmax(mask_raw * prior_scale, dim=-1)
        return mask  # (B, n_features)

class TabNetModel(nn.Module):
    def __init__(self, n_features, d_model=128, n_steps=3, dropout=0.1):
        super().__init__()
        self.n_steps = n_steps
        self.feat_blocks = nn.ModuleList([FeatureBlock(n_features, d_model, dropout) for _ in range(n_steps)])
        self.mask_nets = nn.ModuleList([AttentiveMask(n_features) for _ in range(n_steps - 1)])
        self.head_V = nn.Linear(n_features, 1)
        self.head_dV = nn.Linear(n_features, 1)
    def forward(self, x):
        x = x.squeeze(1)  # (B, F)
        prior = torch.ones_like(x)
        out = 0
        for step in range(self.n_steps):
            x = self.feat_blocks[step](x)
            if step < self.n_steps - 1:
                mask = self.mask_nets[step](x, prior)
                prior = prior * (1 - mask)  # 逐步减少已选特征的权重
                x = x * mask
            out = out + x
        out = out / self.n_steps
        return self.head_V(out), self.head_dV(out)

# === 训练 ===
model = TabNetModel(n_features, d_model=128, n_steps=3, dropout=0.1).to(device)
opt = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
sch = optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=0.5, patience=20)
mse = nn.MSELoss()
print(f"\nTabNet: {sum(p.numel() for p in model.parameters()):,} params")

best_val=float('inf'); pc=0
for ep in range(150):
    model.train(); tl=0.0
    for bx,by,_ in tr_ld:
        bx,by=bx.to(device),by.to(device)
        opt.zero_grad()
        Vp,dVp=model(bx)
        loss=mse(Vp,by)
        loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),0.5)
        opt.step(); tl+=loss.item()
    atl=tl/len(tr_ld)
    model.eval(); vl=0.0
    with torch.no_grad():
        for bx,by,_ in vl_ld:
            bx,by=bx.to(device),by.to(device)
            Vp,_=model(bx); vl+=mse(Vp,by).item()
    avl=vl/len(vl_ld); sch.step(avl)
    if avl<best_val: best_val=avl; pc=0; torch.save(model.state_dict(),'_tab_best.pth')
    else:
        pc+=1
        if pc>=30: print(f"Early stop ep{ep+1}"); break
    if (ep+1)%20==0: print(f"Ep{ep+1} Train:{atl:.6f} Val:{avl:.6f}")

model.load_state_dict(torch.load('_tab_best.pth'))
model.eval(); pr,tr=[],[]
with torch.no_grad():
    for bx,by in DataLoader(TensorDataset(X_ts_t,y_ts_t),batch_size=64,shuffle=False):
        bx,by=bx.to(device),by.to(device)
        Vp,_=model(bx); pr.append(Vp.cpu().numpy()); tr.append(by.cpu().numpy())
ypn=np.concatenate(pr).ravel(); ytn=np.concatenate(tr).ravel()
yp_real=sy.inverse_transform(ypn.reshape(-1,1)).ravel()
yt_real=sy.inverse_transform(ytn.reshape(-1,1)).ravel()
r2=r2_score(yt_real,yp_real); rmse=np.sqrt(mean_squared_error(yt_real,yp_real))
print(f"\nTabNet FC2: R2={r2:.4f} RMSE={rmse:.6f}V")
os.remove('_tab_best.pth')
