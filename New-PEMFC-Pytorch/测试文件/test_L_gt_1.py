# -*- coding: utf-8 -*-
"""
测试 L>1 主流手段在 IEEE2014 上是否可行
复现三类主流做法:
  1. 强滤波 (median filter + Savitzky-Golay)
  2. Bacon-Watts 拐点检测 (简化版: 找退化加速段)
  3. L>1 序列 + LSTM / BiLSTM / TCN

目标: 验证"主流 L>1 手段"能否在 FC1/FC2 上取得正 R²
"""
import sys, os, pickle, random, time, warnings
warnings.filterwarnings('ignore')
try: sys.stdout.reconfigure(encoding='utf-8')
except AttributeError: pass

import numpy as np, pandas as pd
import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from scipy.signal import savgol_filter, medfilt
from scipy.optimize import curve_fit
from pathlib import Path
import matplotlib; matplotlib.use('Agg')

seed_value = 42
random.seed(seed_value); np.random.seed(seed_value); torch.manual_seed(seed_value)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

DATA_PATH = Path(r"E:\沈农信电\课题-预测质子交换膜燃料电池（PEMFC）的健康状态和剩余寿命\代码\Long-Term-PEMFC-DL-Prognostics-main\Long-Term-PEMFC-DL-Prognostics-main\Long-Term-PEMFC-DL-Prognostics-main\IEEE2014DataChallengeData")

# ============================================================
# 配置
# ============================================================
USE_FC = 'FC2'          # FC2 纹波 / FC1 准静态
SAMPLING_INTERVAL = 0.5 # FC2:0.5h, FC1:1.0h
SEQ_LEN = 20            # L>1 序列长度 (0.5h*20=10h)
TEST_MODELS = ['lstm', 'bilstm', 'tcn']  # 测试的模型
USE_KNEE = True         # 是否用拐点检测挑退化加速段 (Chou&Wang做法)
USE_MEDFILT = True      # 是否用中值滤波 (RF-TCN做法)

# ============================================================
# 数据加载
# ============================================================
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

target_col = 'Utot (V)'
target_idx = list(meas_cols).index(target_col)

# ============================================================
# 强滤波: median + SG (RF-TCN 用 median window=5)
# ============================================================
smoothed_df = pd.DataFrame({'Time (h)': sampled_df['Time (h)']})
for col in meas_cols:
    v = s_df[col].values.astype(np.float64)
    if USE_MEDFILT:
        v = medfilt(v, kernel_size=5)  # 中值滤波去脉冲
    v = savgol_filter(v, 21, 2)        # SG平滑
    smoothed_df[col] = v

all_cols = list(meas_cols)
data_values = smoothed_df[all_cols].values.astype(np.float32)
N = len(data_values)
volt = data_values[:, target_idx]  # Utot 曲线
time_h = smoothed_df['Time (h)'].values

print(f"=== {USE_FC}: {N}帧, 滤波={'median+SG' if USE_MEDFILT else 'SG'} ===")

# ============================================================
# Bacon-Watts 拐点检测 (简化版)
# 真实 Bacon-Watts: 两条直线 + 平滑转折点拟合电压曲线
# 简化版: 用双线性拟合找 knee point
# ============================================================
def bacon_watts_knee(t, v):
    """双线性拟合找拐点, 返回 knee time"""
    def bilinear(t, t_knee, a, b1, b2):
        v_pred = np.where(t <= t_knee, a + b1*(t - t_knee), a + b2*(t - t_knee))
        return v_pred
    best_t, best_err = None, np.inf
    for t_knee in np.linspace(t[0]+10, t[-1]-10, 100):
        try:
            popt, _ = curve_fit(bilinear, t, v, p0=[t_knee, np.mean(v), -0.0001, -0.001], maxfev=5000)
            pred = bilinear(t, *popt)
            err = np.mean((pred - v)**2)
            if err < best_err:
                best_err = err; best_t = popt[0]
        except:
            continue
    return best_t if best_t else t[len(t)//2]

knee = bacon_watts_knee(time_h, volt) if USE_KNEE else None
print(f"拐点(knee point): {knee:.1f}h" if knee else "不使用拐点检测")

# 训练段选择: Chou&Wang 用 knee-onset 到 knee-point 之间的退化加速段
if USE_KNEE and knee:
    # 简化: 取 knee 前 50h 到 knee 后 50h 作为退化加速段
    mask = (time_h >= knee - 50) & (time_h <= knee + 50)
    if mask.sum() < SEQ_LEN * 3:
        mask = np.ones(N, dtype=bool)  # 样本太少就回退全数据
    seg_start, seg_end = time_h[mask][0], time_h[mask][-1]
    print(f"退化加速段: {seg_start:.1f}h ~ {seg_end:.1f}h ({mask.sum()}帧)")
else:
    mask = np.ones(N, dtype=bool)
    print(f"使用全数据段: 0 ~ {time_h[-1]:.1f}h")

# 取子序列
seg_data = data_values[mask]
seg_N = len(seg_data)

# ============================================================
# L>1 序列构造: X[i:i+L] -> y[i+L]
# ============================================================
def make_sequences(data, seq_len):
    X, y = [], []
    for i in range(len(data) - seq_len):
        X.append(data[i:i+seq_len])
        y.append(data[i+seq_len, target_idx])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)

X_raw, y_raw = make_sequences(seg_data, SEQ_LEN)
n_features = X_raw.shape[2]
print(f"序列构造: L={SEQ_LEN}, 样本数={len(X_raw)}, 特征维度={n_features}")

# 划分 80/10/10
tr_end = int(len(X_raw)*0.8); vl_end = int(len(X_raw)*0.9)
X_tr_r, X_vl_r, X_ts_r = X_raw[:tr_end], X_raw[tr_end:vl_end], X_raw[vl_end:]
y_tr_r, y_vl_r, y_ts_r = y_raw[:tr_end], y_raw[tr_end:vl_end], y_raw[vl_end:]

# 归一化
sX = MinMaxScaler(feature_range=(0,1)); sy = MinMaxScaler(feature_range=(0,1))
X_tr_s = sX.fit_transform(X_tr_r.reshape(-1, n_features)).reshape(-1, SEQ_LEN, n_features)
X_vl_s = sX.transform(X_vl_r.reshape(-1, n_features)).reshape(-1, SEQ_LEN, n_features)
X_ts_s = sX.transform(X_ts_r.reshape(-1, n_features)).reshape(-1, SEQ_LEN, n_features)
y_tr_s = sy.fit_transform(y_tr_r.reshape(-1,1)).ravel()
y_vl_s = sy.transform(y_vl_r.reshape(-1,1)).ravel()
y_ts_s = sy.transform(y_ts_r.reshape(-1,1)).ravel()

X_tr_t = torch.tensor(X_tr_s, dtype=torch.float32)
X_vl_t = torch.tensor(X_vl_s, dtype=torch.float32)
X_ts_t = torch.tensor(X_ts_s, dtype=torch.float32)
y_tr_t = torch.tensor(y_tr_s, dtype=torch.float32).view(-1,1)
y_vl_t = torch.tensor(y_vl_s, dtype=torch.float32).view(-1,1)
y_ts_t = torch.tensor(y_ts_s, dtype=torch.float32).view(-1,1)

tr_ld = DataLoader(TensorDataset(X_tr_t, y_tr_t), batch_size=64, shuffle=False)
vl_ld = DataLoader(TensorDataset(X_vl_t, y_vl_t), batch_size=64, shuffle=False)
ts_ld = DataLoader(TensorDataset(X_ts_t, y_ts_t), batch_size=64, shuffle=False)

# ============================================================
# 模型定义
# ============================================================
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

class TCNModel(nn.Module):
    """简化 TCN: 多层因果卷积"""
    def __init__(self, n_feat, hidden=64, n_layer=3):
        super().__init__()
        layers = []
        in_ch = n_feat
        for i in range(n_layer):
            layers.append(nn.Conv1d(in_ch, hidden, kernel_size=3, padding=2, dilation=2**i))
            layers.append(nn.ReLU())
            in_ch = hidden
        self.conv = nn.Sequential(*layers)
        self.fc = nn.Linear(hidden, 1)
    def forward(self, x):
        x = x.permute(0, 2, 1)  # (B, F, L)
        out = self.conv(x)
        out = out[:, :, -1]
        return self.fc(out)

def make_model(name, n_feat):
    if name == 'lstm': return LSTMModel(n_feat)
    elif name == 'bilstm': return BiLSTMModel(n_feat)
    elif name == 'tcn': return TCNModel(n_feat)

# ============================================================
# 训练 + 评估
# ============================================================
def train_eval(model, name, epochs=150, lr=1e-3):
    model.to(device)
    opt = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    mse = nn.MSELoss()
    best_val = float('inf'); pc = 0
    for ep in range(epochs):
        model.train()
        for bx, by in tr_ld:
            bx, by = bx.to(device), by.to(device)
            opt.zero_grad()
            loss = mse(model(bx), by)
            loss.backward(); opt.step()
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
    yp_r = sy.inverse_transform(yp.reshape(-1,1)).ravel()
    yt_r = sy.inverse_transform(yt.reshape(-1,1)).ravel()
    r2 = r2_score(yt_r, yp_r); rmse = np.sqrt(mean_squared_error(yt_r, yp_r))
    print(f"  [{name}] R2={r2:.4f} RMSE={rmse:.6f}V")
    return r2, rmse

# ============================================================
# 主流程
# ============================================================
print(f"\n=== 测试 L={SEQ_LEN} 序列 + {'拐点挑段' if USE_KNEE else '全数据'} + {'median+SG' if USE_MEDFILT else 'SG'} 滤波 ===")
results = {}
for mname in TEST_MODELS:
    model = make_model(mname, n_features)
    r2, rmse = train_eval(model, mname)
    results[mname] = {'r2': r2, 'rmse': rmse}

print(f"\n=== 汇总 ({USE_FC}, L={SEQ_LEN}) ===")
for k, v in results.items():
    print(f"  {k}: R2={v['r2']:.4f} RMSE={v['rmse']:.6f}V")

with open(f'l_gt_1_test_{USE_FC}_L{SEQ_LEN}.pkl', 'wb') as f:
    pickle.dump(results, f)
