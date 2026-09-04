# ================================================================
# 消融实验调参: 为每个模型找最优lr
# ================================================================
import sys, os, pickle, random
import numpy as np, pandas as pd
import torch, torch.nn as nn, torch.optim as optim
from pathlib import Path
from scipy.signal import savgol_filter
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score

seed_value = 42
random.seed(seed_value); np.random.seed(seed_value); torch.manual_seed(seed_value)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DATA_PATH = Path(r"E:\沈农信电\课题-预测质子交换膜燃料电池（PEMFC）的健康状态和剩余寿命\代码\Long-Term-PEMFC-DL-Prognostics-main\Long-Term-PEMFC-DL-Prognostics-main\Long-Term-PEMFC-DL-Prognostics-main\IEEE2014DataChallengeData")

# ================================================================
# 加载和预处理 (与主文件相同)
# ================================================================
csv_files = list(DATA_PATH.rglob("*.csv"))
data_dict = {"FC1": {"ageing": []}}
for f in csv_files:
    if "FC1" in f.name and "Ageing" in f.name:
        data_dict["FC1"]["ageing"].append(pd.read_csv(f, encoding="ISO-8859-1"))

age_df = pd.concat(data_dict["FC1"]["ageing"], ignore_index=True).sort_values('Time (h)').reset_index(drop=True)
sampled_idx = age_df.astype({'Time (h)': 'int32'}).drop_duplicates(subset=['Time (h)']).index
sampled_df = age_df.iloc[sampled_idx].reset_index(drop=True)

meas_cols = [c for c in sampled_df.columns if c != 'Time (h)']
s_df = sampled_df[meas_cols].ffill().bfill().fillna(0)
valid_cols = [c for c in meas_cols if s_df[c].std() > 1e-10]

smoothed = pd.DataFrame()
smoothed['Time (h)'] = sampled_df['Time (h)']
for col in valid_cols:
    smoothed[col] = savgol_filter(s_df[col].values, window_length=21, polyorder=2)

target_col = 'Utot (V)'
all_cols = [c for c in valid_cols]
target_idx = all_cols.index(target_col)
data_values = smoothed[all_cols].values.astype(np.float32)
N = len(data_values)
MAX_LAG = 4; LAGS = [1, 2, 4]
n_features = len(all_cols) + 1  # +1 for lag marker

X_sec, y_sec = [], []
for lag in LAGS:
    marker = lag / 10.0
    X_lag = data_values[MAX_LAG-lag : N-lag]
    y_lag = data_values[MAX_LAG : N, target_idx]
    lag_col = np.full((len(X_lag), 1), marker, dtype=np.float32)
    X_sec.append(np.concatenate([X_lag, lag_col], axis=1))
    y_sec.append(y_lag)
X_raw = np.concatenate(X_sec, axis=0); y_raw = np.concatenate(y_sec, axis=0)

n_per_lag = N - MAX_LAG
te = int(n_per_lag * 0.5); ve = int(n_per_lag * 0.6)
X_tr, X_va, X_ts = [], [], []; y_tr, y_va, y_ts = [], [], []
for k in range(len(LAGS)):
    s = k * n_per_lag
    X_tr.append(X_raw[s:s+te]); X_va.append(X_raw[s+te:s+ve]); X_ts.append(X_raw[s+ve:s+n_per_lag])
    y_tr.append(y_raw[s:s+te]); y_va.append(y_raw[s+te:s+ve]); y_ts.append(y_raw[s+ve:s+n_per_lag])
X_tr = np.concatenate(X_tr); X_va = np.concatenate(X_va); X_ts = np.concatenate(X_ts)
y_tr = np.concatenate(y_tr); y_va = np.concatenate(y_va); y_ts = np.concatenate(y_ts)

scX = MinMaxScaler((0,1)); scY = MinMaxScaler((0,1))
X_train = torch.tensor(scX.fit_transform(X_tr).reshape(-1,1,n_features), dtype=torch.float32)
y_train = torch.tensor(scY.fit_transform(y_tr.reshape(-1,1)).ravel(), dtype=torch.float32).view(-1,1)
X_val   = torch.tensor(scX.transform(X_va).reshape(-1,1,n_features), dtype=torch.float32)
y_val   = torch.tensor(scY.transform(y_va.reshape(-1,1)).ravel(), dtype=torch.float32).view(-1,1)
X_test  = torch.tensor(scX.transform(X_ts).reshape(-1,1,n_features), dtype=torch.float32)
y_test  = torch.tensor(scY.transform(y_ts.reshape(-1,1)).ravel(), dtype=torch.float32).view(-1,1)

# ================================================================
# 模型定义
# ================================================================
SEQUENCE_LENGTH = 1

class DLinearModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout=0.0):
        super().__init__()
        k = min(5, SEQUENCE_LENGTH); k = k-1 if k%2==0 else k
        self.decomp = nn.AvgPool1d(k, stride=1, padding=k//2)
        self.trend_linear = nn.Linear(SEQUENCE_LENGTH, output_size)
        self.residual_linear = nn.Linear(SEQUENCE_LENGTH, output_size)
    def forward(self, x):
        x_t = x.permute(0,2,1)
        trend = self.decomp(x_t); residual = x_t - trend
        return (self.trend_linear(trend) + self.residual_linear(residual)).mean(dim=1)

class LSTMBaseline(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout=0.0):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True,
                            dropout=dropout if num_layers > 1 else 0)
        self.norm = nn.LayerNorm(hidden_size)
        self.fc = nn.Linear(hidden_size, output_size)
    def forward(self, x):
        out, _ = self.lstm(x); out = out[:, -1, :]; out = self.norm(out)
        return self.fc(out)

import torch.nn.functional as F
class TSMixerBlock(nn.Module):
    def __init__(self, dim, hidden_dim, dropout):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim); self.fc2 = nn.Linear(hidden_dim, dim)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x):
        residual = x; x = F.relu(self.fc1(x)); x = self.dropout(x)
        return self.fc2(x) + residual

class TSMixerModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout=0.1):
        super().__init__()
        self.blocks = nn.ModuleList([TSMixerBlock(input_size, hidden_size, dropout) for _ in range(max(1,num_layers))])
        self.head = nn.Linear(input_size, output_size)
    def forward(self, x):
        if x.dim()==3: x = x.squeeze(1)
        for b in self.blocks: x = b(x)
        return self.head(x)

# ================================================================
# 训练 + 评估 (单次, 快速)
# ================================================================
def quick_train(model, lr, epochs=300, patience=60):
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=20)
    criterion = nn.MSELoss()
    best_val = float('inf'); best_state = None; cnt = 0
    for ep in range(epochs):
        model.train(); optimizer.zero_grad()
        loss = criterion(model(X_tr_gpu), y_tr_gpu)
        loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()
        model.eval()
        with torch.no_grad(): vl = criterion(model(X_va_gpu), y_va_gpu).item()
        scheduler.step(vl)
        if vl < best_val: best_val = vl; best_state = {k:v.clone() for k,v in model.state_dict().items()}; cnt = 0
        else: cnt += 1
        if cnt >= patience: break
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        pred = scY.inverse_transform(model(X_ts_gpu).cpu().numpy().reshape(-1,1)).ravel()
        true = scY.inverse_transform(y_test.numpy().reshape(-1,1)).ravel()
    return r2_score(true, pred)

X_tr_gpu = X_train.to(device); y_tr_gpu = y_train.to(device)
X_va_gpu = X_val.to(device); y_va_gpu = y_val.to(device)
X_ts_gpu = X_test.to(device)

# ================================================================
# 各模型调参
# ================================================================
models_to_tune = {
    'DLinear': (DLinearModel, {'input_size':25, 'hidden_size':64, 'num_layers':1, 'output_size':1}),
    'LSTM':    (LSTMBaseline, {'input_size':25, 'hidden_size':80, 'num_layers':1, 'output_size':1}),
    'TSMixer': (TSMixerModel, {'input_size':25, 'hidden_size':64, 'num_layers':2, 'output_size':1, 'dropout':0.1}),
}

lr_grid = [1e-4, 5e-4, 1e-3, 5e-3, 1e-2]

for name, (ModelClass, kwargs) in models_to_tune.items():
    print(f'\n=== {name} ===')
    best_lr, best_r2 = None, -999
    for lr in lr_grid:
        torch.manual_seed(42)
        m = ModelClass(**kwargs).to(device)
        r2 = quick_train(m, lr)
        print(f'  lr={lr:.0e}: R2={r2:.4f}')
        if r2 > best_r2: best_r2 = r2; best_lr = lr
    print(f'  >>> BEST: lr={best_lr:.0e}, R2={best_r2:.4f}')
