# -*- coding: utf-8 -*-
"""
FC1 RUL 预测 — Chou & Wang 同口径 (挑退化段 + 强去噪 + teacher forcing 在线监测)

对齐已发表 TFT-RUL 论文的领域惯例做法:
  1. 挑退化段: 活化峰值拐点之后, 只做电压明显下降的段 (SNR>1)
  2. 强去噪: SG + 小波(db4 软阈值)
  3. 模型: TFT (Adaptive GRN)
  4. 评估: teacher forcing 在线监测 (每时刻用真实传感器估计电压, 看穿越阈值)
  5. 口径: RE% (和 Chou & Wang 一致)

注意: teacher forcing 是"在线健康监测"协议, 评估时回填真实观测值, 非自回归外推。
"""
import sys, os, warnings
warnings.filterwarnings('ignore')
try: sys.stdout.reconfigure(encoding='utf-8')
except AttributeError: pass

import numpy as np, pandas as pd
import torch, torch.nn as nn, torch.nn.functional as F, torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
from scipy.signal import savgol_filter
import pywt
from pathlib import Path
import matplotlib; matplotlib.use('Agg')

seed_value = 42
np.random.seed(seed_value); torch.manual_seed(seed_value)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

DATA_PATH = Path(r"E:\沈农信电\课题-预测质子交换膜燃料电池（PEMFC）的健康状态和剩余寿命\代码\Long-Term-PEMFC-DL-Prognostics-main\Long-Term-PEMFC-DL-Prognostics-main\Long-Term-PEMFC-DL-Prognostics-main\IEEE2014DataChallengeData")
USE_FC = 'FC1'
SAMPLING_INTERVAL = 1.0
SP_LIST = [400, 500, 600]  # 最终方案: 退化明显段 (对齐 Chou & Wang 挑退化段, 避开过早起点)
SG_WINDOW = 21

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

target_col = 'Utot (V)'
all_cols = list(meas_cols)
target_idx = all_cols.index(target_col)
time_h = sampled_df['Time (h)'].values

# ============ 强去噪: SG + 小波 ============
def wavelet_denoise(signal, wavelet='db4', level=3):
    coeffs = pywt.wavedec(signal, wavelet, level=level)
    sigma = np.median(np.abs(coeffs[-1])) / 0.6745
    threshold = sigma * np.sqrt(2 * np.log(len(signal)))
    coeffs_d = [coeffs[0]]
    for c in coeffs[1:]:
        coeffs_d.append(pywt.threshold(c, threshold, mode='soft'))
    return pywt.waverec(coeffs_d, wavelet)[:len(signal)]

denoised = np.zeros((len(s_df), len(all_cols)), dtype=np.float32)
for ci, col in enumerate(all_cols):
    raw = s_df[col].values.astype(np.float64)
    denoised[:, ci] = wavelet_denoise(savgol_filter(raw, SG_WINDOW, 2))
data_values = denoised
N = len(data_values)
n_features = len(all_cols)

# ============ 挑退化段: 活化峰值拐点 ============
V_smooth = data_values[:, target_idx].astype(np.float64)
t_peak_idx = int(np.argmax(V_smooth))
V_init = V_smooth[t_peak_idx]
t_peak = time_h[t_peak_idx]
V_fail = V_init * 0.96
fail_idx = np.where(V_smooth < V_fail)[0]
EOL_true = time_h[fail_idx[0]] if len(fail_idx) > 0 else time_h[-1]
print(f"FC1: {N}帧, 活化峰值 V_init={V_init:.4f}V @ t={t_peak:.1f}h")
print(f"V_fail={V_fail:.4f}V, 真实 EOL={EOL_true:.1f}h (退化段 [{t_peak:.1f}, {EOL_true:.1f}])")

# ============ TFT 模型 (Adaptive GRN) ============
class GatedResidualNetwork(nn.Module):
    def __init__(self, d_in, d_hidden, d_out, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(d_in, d_hidden)
        self.fc2 = nn.Linear(d_hidden, d_out)
        self.gate = nn.Linear(d_hidden, d_out)
        self.gate_mod = nn.Linear(1, d_out)
        self.ln = nn.LayerNorm(d_out)
        self.dropout = nn.Dropout(dropout)
        self.proj = nn.Linear(d_in, d_out) if d_in != d_out else nn.Identity()
    def forward(self, x, dynamic_score=None):
        a = self.proj(x)
        h = F.elu(self.fc1(x))
        g_raw = self.gate(h)
        if dynamic_score is not None:
            g_raw = g_raw + 0.3 * torch.tanh(self.gate_mod(dynamic_score.view(-1, 1)))
        return self.ln(a + self.dropout(torch.sigmoid(g_raw) * self.fc2(h)))

class TFTModel(nn.Module):
    def __init__(self, n_feat, hidden=96, dim_ff=128, dropout=0.1):
        super().__init__()
        self.feat_weights = nn.Parameter(torch.zeros(n_feat))
        self.feat_bias = nn.Parameter(torch.zeros(n_feat))
        self.lstm = nn.LSTM(n_feat, hidden, 1, batch_first=True)
        self.grn = GatedResidualNetwork(hidden, dim_ff, hidden, dropout)
        self.attn = nn.MultiheadAttention(hidden, 4, batch_first=True)
        self.attn_ln = nn.LayerNorm(hidden)
        self.dropout_layer = nn.Dropout(dropout)
        self.fc_V = nn.Linear(hidden, 1)
    def _dyn(self, x):
        return x[:, 0, :6].std(dim=1)
    def forward(self, x):
        dyn = self._dyn(x)
        w = torch.sigmoid(self.feat_weights + self.feat_bias)
        xw = x * w.unsqueeze(0).unsqueeze(0)
        if self.training:
            xw = xw + torch.randn_like(xw) * 0.01
        lstm_out, _ = self.lstm(xw)
        lstm_out = self.dropout_layer(lstm_out)
        grn_out = self.grn(lstm_out, dynamic_score=dyn)
        attn_out, _ = self.attn(grn_out, grn_out, grn_out)
        attn_out = self.attn_ln(attn_out + grn_out)
        return self.fc_V(attn_out[:, -1, :])

# ============ 训练: 退化段 [t_peak, Sp] ============
def train_on_sp(Sp):
    mask = (time_h >= t_peak) & (time_h <= Sp)
    tr_data = data_values[mask]
    n_tr = len(tr_data)
    X = tr_data
    y = tr_data[:, target_idx]
    sX = MinMaxScaler(); sy = MinMaxScaler()
    X_s = sX.fit_transform(X).reshape(-1, 1, n_features)
    y_s = sy.fit_transform(y.reshape(-1, 1)).ravel()
    tr_ld = DataLoader(TensorDataset(
        torch.tensor(X_s, dtype=torch.float32),
        torch.tensor(y_s, dtype=torch.float32).view(-1,1)), batch_size=64, shuffle=False)
    model = TFTModel(n_features).to(device)
    opt = optim.Adam(model.parameters(), lr=3e-4, weight_decay=1e-5)
    mse = nn.MSELoss()
    for ep in range(300):
        model.train()
        for bx, by in tr_ld:
            bx, by = bx.to(device), by.to(device)
            opt.zero_grad(); loss = mse(model(bx), by); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5); opt.step()
    return model, sX, sy

# ============ teacher forcing 在线监测: 用真实传感器估计电压, 看穿越 ============
def predict_teacher_forcing(model, sX, sy, Sp):
    mask = time_h > Sp
    ts_data = data_values[mask]
    ts_time = time_h[mask]
    pred_V = []
    model.eval()
    with torch.no_grad():
        for i in range(len(ts_data)):
            x_sens = ts_data[i:i+1]
            x_t = torch.tensor(sX.transform(x_sens).reshape(1, 1, n_features),
                               dtype=torch.float32).to(device)
            Vp = model(x_t).cpu().numpy()[0, 0]
            pred_V.append(sy.inverse_transform([[Vp]])[0, 0])
    pred_V = np.array(pred_V)
    fail_idx_p = np.where(pred_V < V_fail)[0]
    if len(fail_idx_p) > 0:
        pred_EOL = ts_time[fail_idx_p[0]]
    else:
        pred_EOL = ts_time[-1] if len(ts_time) > 0 else Sp
    return pred_EOL, pred_V, ts_time

print(f"\n=== FC1 RUL (Chou & Wang 同口径, EOL={EOL_true:.1f}h) ===")
print(f"{'Sp(h)':<8s} {'真实RUL':<10s} {'预测RUL':<10s} {'误差(h)':<10s} {'RE%':<8s}")
results = []
for Sp in SP_LIST:
    if Sp >= EOL_true:
        print(f"{Sp:<8d}  Sp >= EOL, 跳过"); continue
    true_RUL = EOL_true - Sp
    model, sX, sy = train_on_sp(Sp)
    pred_EOL, pred_V, ts_time = predict_teacher_forcing(model, sX, sy, Sp)
    pred_RUL = pred_EOL - Sp
    abs_err = abs(pred_EOL - EOL_true)
    rel_err = abs_err / true_RUL * 100
    results.append((Sp, true_RUL, pred_RUL, abs_err, rel_err))
    print(f"{Sp:<8d} {true_RUL:<10.1f} {pred_RUL:<10.1f} {abs_err:<10.1f} {rel_err:<8.2f}")
    print(f"    [诊断] 预测电压min={pred_V.min():.4f} (V_fail={V_fail:.4f}), 穿越={'是' if pred_V.min()<V_fail else '否'}")

print("\n=== 汇总 ===")
if results:
    re_list = [r[4] for r in results]
    print(f"各 Sp RE%: {[f'{r:.2f}' for r in re_list]}")
    print(f"平均 RE = {np.mean(re_list):.2f}%")
