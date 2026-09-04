# -*- coding: utf-8 -*-
"""
复现 Chou & Wang 的"未来 τ 小时预测"任务
任务定义:
  输入: 过去 L 个时刻的传感器 + future step τ 标记
  输出: 未来 τ 时刻的 Utot 电压
  τ ∈ {1, 3, 5}

扫描 L ∈ {5, 10, 20, 50, 100}，用 LSTM/BiLSTM/TFT
"""
import sys, os, pickle, random, time, warnings
warnings.filterwarnings('ignore')
try: sys.stdout.reconfigure(encoding='utf-8')
except AttributeError: pass

import numpy as np, pandas as pd
import torch, torch.nn as nn, torch.nn.functional as F, torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from scipy.signal import savgol_filter
from pathlib import Path
import matplotlib; matplotlib.use('Agg')

seed_value = 42
random.seed(seed_value); np.random.seed(seed_value); torch.manual_seed(seed_value)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

DATA_PATH = Path(r"E:\沈农信电\课题-预测质子交换膜燃料电池（PEMFC）的健康状态和剩余寿命\代码\Long-Term-PEMFC-DL-Prognostics-main\Long-Term-PEMFC-DL-Prognostics-main\Long-Term-PEMFC-DL-Prognostics-main\IEEE2014DataChallengeData")

USE_FC = 'FC2'
SAMPLING_INTERVAL = 0.5
FUTURE_STEPS = [1, 3, 5]   # 预测未来 τ 小时
L_LIST = [5, 10, 20, 50, 100]  # 扫描的序列长度
MODELS = ['lstm', 'bilstm', 'tft']

# ============ 数据加载 ============
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
N = len(data_values)
n_features_orig = len(all_cols)
print(f"{USE_FC}: {N}帧, {n_features_orig}特征")

# ============ 模型 ============
class LSTMModel(nn.Module):
    def __init__(self, n_feat, hidden=96, n_layer=2):
        super().__init__()
        self.lstm = nn.LSTM(n_feat, hidden, n_layer, batch_first=True)
        self.fc = nn.Linear(hidden, 1)
    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

class BiLSTMModel(nn.Module):
    def __init__(self, n_feat, hidden=96, n_layer=2):
        super().__init__()
        self.lstm = nn.LSTM(n_feat, hidden, n_layer, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden*2, 1)
    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

class TFTModel(nn.Module):
    """简化 TFT: VS + LSTM + Attention"""
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

def make_model(name, n_feat):
    if name == 'lstm': return LSTMModel(n_feat)
    elif name == 'bilstm': return BiLSTMModel(n_feat)
    elif name == 'tft': return TFTModel(n_feat)

# ============ 训练评估 ============
def train_eval(model, tr_ld, vl_ld, ts_ld, epochs=150, lr=1e-3):
    model.to(device)
    opt = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    mse = nn.MSELoss()
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
            if pc >= 30: break
    model.load_state_dict(torch.load('_tmp.pth')); os.remove('_tmp.pth')
    model.eval(); pr, tr = [], []
    with torch.no_grad():
        for bx, by in ts_ld:
            bx, by = bx.to(device), by.to(device)
            pr.append(model(bx).cpu().numpy()); tr.append(by.cpu().numpy())
    yp = np.concatenate(pr).ravel(); yt = np.concatenate(tr).ravel()
    return r2_score(yt, yp)

# ============ 主扫描 ============
print(f"\n=== 扫描: L={L_LIST}, τ={FUTURE_STEPS}, 模型={MODELS} ===")
all_results = {}

for L in L_LIST:
    for tau in FUTURE_STEPS:
        # 构造: X[i:i+L] -> y[i+L+tau]
        max_offset = L + max(FUTURE_STEPS)
        X_seq, y_seq = [], []
        for i in range(N - max_offset):
            X_seq.append(data_values[i:i+L])
            y_seq.append(data_values[i+L+tau, target_idx])
        X_raw = np.array(X_seq, dtype=np.float32)
        y_raw = np.array(y_seq, dtype=np.float32)
        n_feat = X_raw.shape[2]

        # 划分
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

        for mname in MODELS:
            torch.manual_seed(seed_value)
            model = make_model(mname, n_feat)
            r2 = train_eval(model, tr_ld, vl_ld, ts_ld)
            key = f'L={L}, tau={tau}, {mname}'
            all_results[key] = r2
            print(f"  {key}: R2={r2:.4f}")

print("\n=== 汇总 ===")
for k, v in all_results.items():
    print(f"  {k}: R2={v:.4f}")

with open(f'future_forecast_scan_{USE_FC}.pkl', 'wb') as f:
    pickle.dump(all_results, f)
