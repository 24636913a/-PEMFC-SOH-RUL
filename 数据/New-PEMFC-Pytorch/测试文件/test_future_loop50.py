# -*- coding: utf-8 -*-
"""
Chou & Wang 路线: L=10 + TFT + τ=1 未来预测, Loop 50 稳定性验证
任务: X[t-10:t] -> Utot[t+1]
"""
import sys, os, pickle, random, time, warnings
warnings.filterwarnings('ignore')
try: sys.stdout.reconfigure(encoding='utf-8')
except AttributeError: pass

import numpy as np, pandas as pd
import torch, torch.nn as nn, torch.nn.functional as F, torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score, mean_squared_error
from scipy.signal import savgol_filter
from pathlib import Path
import matplotlib; matplotlib.use('Agg')

seed_value = 42
random.seed(seed_value); np.random.seed(seed_value); torch.manual_seed(seed_value)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

DATA_PATH = Path(r"E:\沈农信电\课题-预测质子交换膜燃料电池（PEMFC）的健康状态和剩余寿命\代码\Long-Term-PEMFC-DL-Prognostics-main\Long-Term-PEMFC-DL-Prognostics-main\Long-Term-PEMFC-DL-Prognostics-main\IEEE2014DataChallengeData")

USE_FC = 'FC2'
SAMPLING_INTERVAL = 0.5
L = 10
TAU = 1
N_LOOPS = 50

# 数据加载
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
data_values = smoothed_df[all_cols].values.astype(np.float32)
N = len(data_values); n_feat = len(all_cols)
print(f"{USE_FC}: {N}帧, {n_feat}特征, L={L}, tau={TAU}")

# 构造序列
X_seq, y_seq = [], []
for i in range(N - L - TAU):
    X_seq.append(data_values[i:i+L])
    y_seq.append(data_values[i+L+TAU-1, target_idx])
X_raw = np.array(X_seq, dtype=np.float32); y_raw = np.array(y_seq, dtype=np.float32)
print(f"序列样本: {len(X_raw)}")

tr_end = int(len(X_raw)*0.8); vl_end = int(len(X_raw)*0.9)
sX = MinMaxScaler(); sy = MinMaxScaler()
X_tr = sX.fit_transform(X_raw[:tr_end].reshape(-1, n_feat)).reshape(-1, L, n_feat)
X_vl = sX.transform(X_raw[tr_end:vl_end].reshape(-1, n_feat)).reshape(-1, L, n_feat)
X_ts = sX.transform(X_raw[vl_end:].reshape(-1, n_feat)).reshape(-1, L, n_feat)
y_tr = sy.fit_transform(y_raw[:tr_end].reshape(-1,1)).ravel()
y_vl = sy.transform(y_raw[tr_end:vl_end].reshape(-1,1)).ravel()
y_ts = sy.transform(y_raw[vl_end:].reshape(-1,1)).ravel()

tr_ld = DataLoader(TensorDataset(torch.tensor(X_tr,dtype=torch.float32), torch.tensor(y_tr,dtype=torch.float32).view(-1,1)), batch_size=64, shuffle=False)
vl_ld = DataLoader(TensorDataset(torch.tensor(X_vl,dtype=torch.float32), torch.tensor(y_vl,dtype=torch.float32).view(-1,1)), batch_size=64, shuffle=False)
ts_ld = DataLoader(TensorDataset(torch.tensor(X_ts,dtype=torch.float32), torch.tensor(y_ts,dtype=torch.float32).view(-1,1)), batch_size=64, shuffle=False)

class TFTModel(nn.Module):
    def __init__(self, n_feat, hidden=96):
        super().__init__()
        self.feat_weights = nn.Parameter(torch.zeros(n_feat))
        self.feat_bias = nn.Parameter(torch.zeros(n_feat))
        self.lstm = nn.LSTM(n_feat, hidden, 1, batch_first=True)
        self.attn = nn.MultiheadAttention(hidden, 4, batch_first=True)
        self.ln = nn.LayerNorm(hidden)
        self.fc = nn.Linear(hidden, 1)
    def forward(self, x):
        w = torch.sigmoid(self.feat_weights + self.feat_bias)
        xw = x * w.unsqueeze(0).unsqueeze(0)
        lstm_out, _ = self.lstm(xw)
        attn_out, _ = self.attn(lstm_out, lstm_out, lstm_out)
        attn_out = self.ln(attn_out + lstm_out)
        return self.fc(attn_out[:, -1, :])

def train_one(model, epochs=150, lr=1e-3):
    model.to(device)
    opt = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    mse = nn.MSELoss()
    swa_start = 40; swa_model = None
    best_val = float('inf'); pc = 0
    for ep in range(epochs):
        model.train()
        for bx, by in tr_ld:
            bx, by = bx.to(device), by.to(device)
            opt.zero_grad(); loss = mse(model(bx), by); loss.backward(); opt.step()
        model.eval(); vl = 0.0
        with torch.no_grad():
            for bx, by in vl_ld:
                bx, by = bx.to(device), by.to(device)
                vl += mse(model(bx), by).item()
        avl = vl/len(vl_ld)
        if avl < best_val: best_val = avl; pc = 0; torch.save(model.state_dict(), '_tmp.pth')
        else:
            pc += 1
            if pc >= 40: break
        if ep >= swa_start:
            if swa_model is None: swa_model = {k:v.clone().cpu() for k,v in model.state_dict().items()}; sc=1
            else:
                for k in swa_model: swa_model[k]=swa_model[k]*(sc/(sc+1))+model.state_dict()[k].cpu()*(1/(sc+1)); sc+=1
    model.load_state_dict(torch.load('_tmp.pth')); os.remove('_tmp.pth')
    if swa_model is not None: model.load_state_dict({k:v.to(device) for k,v in swa_model.items()})
    return model

def eval_one(model):
    model.eval(); pr, tr = [], []
    with torch.no_grad():
        for bx, by in ts_ld:
            bx, by = bx.to(device), by.to(device)
            pr.append(model(bx).cpu().numpy()); tr.append(by.cpu().numpy())
    yp = np.concatenate(pr).ravel(); yt = np.concatenate(tr).ravel()
    return r2_score(yt, yp)

print(f"\n=== Loop {N_LOOPS} ===")
r2s = []; t0 = time.time()
for li in range(N_LOOPS):
    torch.manual_seed(seed_value + li*10)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed_value + li*10)
    m = TFTModel(n_feat)
    m = train_one(m)
    r2 = eval_one(m)
    r2s.append(r2)
    print(f"  [{li+1}/{N_LOOPS}] R2={r2:.4f} ({time.time()-t0:.0f}s)")

print(f"\n=== TFT L={L} tau={TAU} Loop {N_LOOPS} 结果 ===")
print(f"Min={min(r2s):.4f} Mean={np.mean(r2s):.4f} Max={max(r2s):.4f}")
print(f"总时间: {time.time()-t0:.0f}s ({(time.time()-t0)/60:.1f}分钟)")
with open(f'future_loop50_L{L}_tau{TAU}_{USE_FC}.pkl', 'wb') as f:
    pickle.dump({'r2s': r2s, 'total_time': time.time()-t0}, f)
