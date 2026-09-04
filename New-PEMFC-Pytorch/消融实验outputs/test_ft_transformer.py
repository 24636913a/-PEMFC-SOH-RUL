# -*- coding: utf-8 -*-
"""FT-Transformer 快速验证: FC2, L=1+lag, 33维 (含Freq特征)"""
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
USE_FC = 'FC1'; SAMPLING_INTERVAL = 1.0; MAX_LAG = 4; LAGS = [1, 2, 4]

DATA_PATH = Path(r"E:\沈农信电\课题-预测质子交换膜燃料电池（PEMFC）的健康状态和剩余寿命\代码\Long-Term-PEMFC-DL-Prognostics-main\Long-Term-PEMFC-DL-Prognostics-main\Long-Term-PEMFC-DL-Prognostics-main\IEEE2014DataChallengeData")

# === 数据加载 ===
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

# 降采样
t_hours = age_df['Time (h)'].values
age_df['t_bin'] = (t_hours / SAMPLING_INTERVAL).astype('int32')
sampled_idx = age_df.drop_duplicates(subset=['t_bin']).index
sampled_df = age_df.iloc[sampled_idx].reset_index(drop=True)
if 't_bin' in sampled_df.columns: sampled_df = sampled_df.drop(columns=['t_bin'])

# SG滤波
meas_cols = [c for c in sampled_df.columns if c != 'Time (h)']
s_df = sampled_df[meas_cols].copy().ffill().bfill().fillna(0)
meas_cols = [c for c in meas_cols if s_df[c].std() > 1e-10]
smoothed_df = pd.DataFrame({'Time (h)': sampled_df['Time (h)']})
for col in meas_cols:
    smoothed_df[col] = savgol_filter(s_df[col].values, 21, 2)

target_col = 'Utot (V)'; all_cols = list(meas_cols)
target_idx = all_cols.index(target_col)
data_values = smoothed_df[all_cols].values.astype(np.float32)
N = len(data_values)
print(f"FC1: {N}帧 (降采样{SAMPLING_INTERVAL}h)")

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

# 衍生特征 (6维)
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

# Freq特征 (+2维)
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
print(f"特征: {n_features}维 (24传感器+1lag+6衍生+2Freq)")

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
print(f"划分: 训练{len(X_tr_r)} 验证{len(X_vl_r)} 测试{len(X_ts_r)}")

sX=MinMaxScaler(feature_range=(0,1)); sy=MinMaxScaler(feature_range=(0,1))
X_tr_s=sX.fit_transform(X_tr_r); X_vl_s=sX.transform(X_vl_r); X_ts_s=sX.transform(X_ts_r)
y_tr_s=sy.fit_transform(y_tr_r.reshape(-1,1)).ravel()
y_vl_s=sy.transform(y_vl_r.reshape(-1,1)).ravel()
y_ts_s=sy.transform(y_ts_r.reshape(-1,1)).ravel()

X_tr_t=torch.tensor(X_tr_s,dtype=torch.float32).unsqueeze(1)  # (N,1,F)
X_vl_t=torch.tensor(X_vl_s,dtype=torch.float32).unsqueeze(1)
X_ts_t=torch.tensor(X_ts_s,dtype=torch.float32).unsqueeze(1)
y_tr_t=torch.tensor(y_tr_s,dtype=torch.float32).view(-1,1)
y_vl_t=torch.tensor(y_vl_s,dtype=torch.float32).view(-1,1)
y_ts_t=torch.tensor(y_ts_s,dtype=torch.float32).view(-1,1)
dV_tr=np.diff(y_tr_s,prepend=y_tr_s[0]); dV_vl=np.diff(y_vl_s,prepend=y_vl_s[0]); dV_ts=np.diff(y_ts_s,prepend=y_ts_s[0])

tr_ld=DataLoader(TensorDataset(X_tr_t,y_tr_t,torch.tensor(dV_tr,dtype=torch.float32).view(-1,1)),batch_size=64,shuffle=False)
vl_ld=DataLoader(TensorDataset(X_vl_t,y_vl_t,torch.tensor(dV_vl,dtype=torch.float32).view(-1,1)),batch_size=64,shuffle=False)
ts_ld=DataLoader(TensorDataset(X_ts_t,y_ts_t,torch.tensor(dV_ts,dtype=torch.float32).view(-1,1)),batch_size=64,shuffle=False)

# === FT-Transformer ===
class FTTransformer(nn.Module):
    def __init__(self, n_features, d_model=64, n_heads=4, n_layers=2, d_ff=128, dropout=0.1):
        super().__init__()
        self.token_embed = nn.Linear(1, d_model)  # 每个特征值 → token
        self.pos_embed = nn.Parameter(torch.randn(1, n_features, d_model)*0.02)
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model)*0.02)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads,
                         dim_feedforward=d_ff, dropout=dropout, batch_first=True,
                         activation='gelu')
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.ln = nn.LayerNorm(d_model)
        self.head_V = nn.Sequential(nn.Linear(d_model, d_model//2), nn.GELU(), nn.Linear(d_model//2, 1))
        self.head_dV = nn.Sequential(nn.Linear(d_model, d_model//2), nn.GELU(), nn.Linear(d_model//2, 1))
        self.dropout = nn.Dropout(dropout)
        self.n_features = n_features

    def forward(self, x):
        B, _, F = x.shape  # x: (B, 1, F)
        x = x.squeeze(1).unsqueeze(-1)  # (B, F, 1)
        x = self.token_embed(x)  # (B, F, d_model)
        x = x + self.pos_embed[:, :F, :].to(x.device)
        cls = self.cls_token.expand(B, -1, -1).to(x.device)  # (B, 1, d_model)
        x = torch.cat([cls, x], dim=1)  # (B, F+1, d_model)
        x = self.dropout(x)
        x = self.encoder(x)
        x = self.ln(x[:, 0, :])  # CLS token
        return self.head_V(x), self.head_dV(x)

# === 训练函数 ===
def train_one(model, tr_ld, vl_ld, epochs=200, patience=40):
    model.to(device)
    opt = optim.Adam(model.parameters(), lr=3e-4, weight_decay=1e-5)
    sch = optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=0.5, patience=20)
    mse = nn.MSELoss(); hub = nn.HuberLoss(delta=0.01)
    swa_start = 40; swa_model = None
    best_val = float('inf'); pc = 0
    for ep in range(epochs):
        model.train(); tl = 0.0
        for bx, by, bdV in tr_ld:
            bx, by, bdV = bx.to(device), by.to(device), bdV.to(device)
            opt.zero_grad()
            Vp, dVp = model(bx)
            Lm = mse(Vp, by); LdV = hub(dVp, bdV)
            df = Vp[1:]-Vp[:-1]; Lmo = torch.mean(torch.clamp(df, min=0)**2)
            if len(Vp) >= 3: sd = Vp[2:]-2*Vp[1:-1]+Vp[:-2]; Lsm = torch.mean(sd**2)
            else: Lsm = torch.tensor(0., device=device)
            loss = Lm + 0.3*LdV + 0.1*Lmo + 0.05*Lsm
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            opt.step(); tl += Lm.item()
        atl = tl/len(tr_ld)
        model.eval(); vl = 0.0
        with torch.no_grad():
            for bx, by, _ in vl_ld:
                bx, by = bx.to(device), by.to(device)
                Vp, _ = model(bx); vl += mse(Vp, by).item()
        avl = vl/len(vl_ld); sch.step(avl)
        if avl < best_val: best_val = avl; pc = 0; torch.save(model.state_dict(), '_ft_best.pth')
        else:
            pc += 1
            if pc >= patience: break
        if ep >= swa_start:
            if swa_model is None: swa_model = {k: v.clone().cpu() for k, v in model.state_dict().items()}; sc = 1
            else:
                for k in swa_model: swa_model[k] = swa_model[k]*(sc/(sc+1)) + model.state_dict()[k].cpu()*(1/(sc+1))
                sc += 1
    model.load_state_dict(torch.load('_ft_best.pth'))
    if swa_model is not None: model.load_state_dict({k: v.to(device) for k, v in swa_model.items()})
    os.remove('_ft_best.pth')
    return model

def eval_one(model, ts_ld):
    model.eval(); pr, tr = [], []
    with torch.no_grad():
        for bx, by, _ in ts_ld:
            bx, by = bx.to(device), by.to(device)
            Vp, _ = model(bx); pr.append(Vp.cpu().numpy()); tr.append(by.cpu().numpy())
    ypn = np.concatenate(pr).ravel(); ytn = np.concatenate(tr).ravel()
    yp_real = sy.inverse_transform(ypn.reshape(-1, 1)).ravel()
    yt_real = sy.inverse_transform(ytn.reshape(-1, 1)).ravel()
    return r2_score(yt_real, yp_real), yt_real, yp_real

# === Loop 50 ===
N_LOOPS = 50
print(f"\nFT-Transformer Loop {N_LOOPS}:")
lr_ft = []; t0 = time.time()
for li in range(N_LOOPS):
    torch.manual_seed(seed_value + li * 10)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed_value + li * 10)
    m = FTTransformer(n_features, d_model=64, n_heads=4, n_layers=2, dropout=0.1)
    m = train_one(m, tr_ld, vl_ld, epochs=200, patience=40)
    rl, yt, yp = eval_one(m, ts_ld)
    lr_ft.append((rl, yt, yp))
    print(f"  [{li+1}/{N_LOOPS}] R2={rl:.4f} ({time.time()-t0:.0f}s)")

ft_r2s = [r for r, _, _ in lr_ft]
total_time = time.time() - t0
print(f"\nFT-Transformer FC1 Loop {N_LOOPS}: Min={min(ft_r2s):.4f} Mean={np.mean(ft_r2s):.4f} Max={max(ft_r2s):.4f}")
print(f"总运行时间: {total_time:.1f}秒 ({total_time/60:.1f}分钟)")
with open('ft_transformer_fc1_loop50.pkl', 'wb') as f:
    pickle.dump({'r2s': ft_r2s, 'total_time_sec': total_time}, f)
