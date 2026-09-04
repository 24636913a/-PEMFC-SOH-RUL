# -*- coding: utf-8 -*-
"""
强去噪 vs 轻去噪 vs 原始 —— 短窗口 L>1 能否被救活

回答核心问题:
  原始数据 L>1 失败, 到底是"没有时序信息"还是"信息被噪声遮住"?

去噪方案:
  A. 原始 (仅轻SG, 参考)
  B. 轻SG (window=21, 当前做法)
  C. 强去噪: 小波去噪 (db4, 软阈值) + SG
  D. 强去噪: SSA (奇异谱分析, 保留前几个主成分) + SG

窗口扫描: L = 5, 10, 20, 50
模型: LSTM (固定), FC2
"""
import sys, os, pickle, random, time, warnings
warnings.filterwarnings('ignore')
try: sys.stdout.reconfigure(encoding='utf-8')
except AttributeError: pass

import numpy as np, pandas as pd
import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score, mean_squared_error
from scipy.signal import savgol_filter
import pywt
from pathlib import Path
import matplotlib; matplotlib.use('Agg')

seed_value = 42
random.seed(seed_value); np.random.seed(seed_value); torch.manual_seed(seed_value)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

DATA_PATH = Path(r"E:\沈农信电\课题-预测质子交换膜燃料电池（PEMFC）的健康状态和剩余寿命\代码\Long-Term-PEMFC-DL-Prognostics-main\Long-Term-PEMFC-DL-Prognostics-main\Long-Term-PEMFC-DL-Prognostics-main\IEEE2014DataChallengeData")

USE_FC = 'FC1'
SAMPLING_INTERVAL = 1.0
L_LIST = [5, 10, 20, 50]
N_TRIALS = 3  # 每个配置跑3次取平均(快速)

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
target_col = 'Utot (V)'; target_idx = list(meas_cols).index(target_col)

# ============ 去噪方案 ============
def wavelet_denoise(signal, wavelet='db4', level=3):
    """小波软阈值去噪"""
    coeffs = pywt.wavedec(signal, wavelet, level=level)
    # 用全局阈值处理细节系数
    sigma = np.median(np.abs(coeffs[-1])) / 0.6745  # 噪声估计
    threshold = sigma * np.sqrt(2 * np.log(len(signal)))
    coeffs_denoised = [coeffs[0]]
    for c in coeffs[1:]:
        coeffs_denoised.append(pywt.threshold(c, threshold, mode='soft'))
    return pywt.waverec(coeffs_denoised, wavelet)[:len(signal)]

def ssa_denoise(signal, n_components=3):
    """奇异谱分析: 保留前 n_components 个主成分"""
    N = len(signal)
    L = max(2, N // 4)  # 窗口长度
    K = N - L + 1
    X = np.column_stack([signal[i:i+L] for i in range(K)])  # (L, K)
    U, s, Vt = np.linalg.svd(X, full_matrices=False)
    # 保留前 n_components
    X_recon = np.zeros_like(X)
    for i in range(min(n_components, len(s))):
        X_recon += s[i] * np.outer(U[:, i], Vt[i, :])
    # 对角平均还原一维序列
    recon = np.zeros(N)
    cnt = np.zeros(N)
    for i in range(L):
        for j in range(K):
            recon[i+j] += X_recon[i, j]
            cnt[i+j] += 1
    return recon / cnt

def apply_denoise(data_2d, method):
    """对每一列做去噪"""
    out = np.zeros_like(data_2d)
    for c in range(data_2d.shape[1]):
        sig = data_2d[:, c]
        if method == 'raw':
            out[:, c] = sig
        elif method == 'sg':
            out[:, c] = savgol_filter(sig, 21, 2)
        elif method == 'wavelet':
            out[:, c] = wavelet_denoise(sig)
            out[:, c] = savgol_filter(out[:, c], 21, 2)
        elif method == 'ssa':
            out[:, c] = ssa_denoise(sig)
            out[:, c] = savgol_filter(out[:, c], 21, 2)
    return out

raw_data = s_df[meas_cols].values.astype(np.float64)
print(f"{USE_FC}: {len(raw_data)}帧, {len(meas_cols)}特征")

METHODS = ['raw', 'sg', 'wavelet', 'ssa']
method_data = {m: apply_denoise(raw_data, m).astype(np.float32) for m in METHODS}

# ============ LSTM + 训练 ============
class LSTMModel(nn.Module):
    def __init__(self, n_feat, hidden=96):
        super().__init__()
        self.lstm = nn.LSTM(n_feat, hidden, 2, batch_first=True)
        self.fc = nn.Linear(hidden, 1)
    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

def run_single(data_2d, L):
    """用指定去噪后的数据 + 窗口L 训练一次LSTM, 返回R²"""
    N = len(data_2d)
    n_feat = data_2d.shape[1]
    X_seq, y_seq = [], []
    for i in range(N - L):
        X_seq.append(data_2d[i:i+L])
        y_seq.append(data_2d[i+L, target_idx])
    X_raw = np.array(X_seq, dtype=np.float32); y_raw = np.array(y_seq, dtype=np.float32)
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

    model = LSTMModel(n_feat).to(device)
    opt = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    mse = nn.MSELoss()
    best_val = float('inf'); pc = 0
    for ep in range(150):
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
print(f"\n=== 扫描: 去噪×窗口 ({USE_FC}) ===")
print(f"{'去噪':<10s} {'L':<5s} {'R²均值':<10s}")
results = {}
for mname in METHODS:
    for L in L_LIST:
        r2s = []
        for t in range(N_TRIALS):
            torch.manual_seed(seed_value + t*10)
            r2s.append(run_single(method_data[mname], L))
        mean_r2 = np.mean(r2s)
        results[f'{mname}_L{L}'] = mean_r2
        print(f"{mname:<10s} {L:<5d} {mean_r2:<10.4f}")

print("\n=== 汇总 ===")
with open(f'denoise_window_FC1.pkl', 'wb') as f:
    pickle.dump(results, f)
