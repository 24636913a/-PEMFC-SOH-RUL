# -*- coding: utf-8 -*-
"""
FC1 RUL 预测 — TFT 状态估计 + 物理外推 (无数据泄露)

两阶段混合框架:
  阶段1 数据驱动: TFT(Adaptive GRN) 融合多传感器, 估计历史电压/SOH 轨迹
  阶段2 物理驱动: 线性退化模型拟合 TFT 估计的轨迹, 外推到 96% 阈值

对比基线: SG 去噪电压 + 相同线性外推
关键: 外推只用历史(<=Sp), 不用未来 → 无数据泄露
"""
import sys, os, warnings, pickle, random, time
warnings.filterwarnings('ignore')
try: sys.stdout.reconfigure(encoding='utf-8')
except AttributeError: pass

import numpy as np, pandas as pd
import torch, torch.nn as nn, torch.nn.functional as F, torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
from scipy.signal import savgol_filter
from pathlib import Path
import matplotlib; matplotlib.use('Agg')

seed_value = 42
random.seed(seed_value); np.random.seed(seed_value); torch.manual_seed(seed_value)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

DATA_PATH = Path(r"E:\沈农信电\课题-预测质子交换膜燃料电池（PEMFC）的健康状态和剩余寿命\代码\Long-Term-PEMFC-DL-Prognostics-main\Long-Term-PEMFC-DL-Prognostics-main\Long-Term-PEMFC-DL-Prognostics-main\IEEE2014DataChallengeData")

USE_FC = 'FC1'
SAMPLING_INTERVAL = 1.0
MAX_LAG = 4
LAGS = [1, 2, 4]
SP_LIST = [300, 400, 500, 600]
SG_WINDOW = 21

# ============ 数据加载 ============
data_dict = {'FC1': {'ageing': []}}
for dirpath, _, filenames in os.walk(DATA_PATH):
    for fname in filenames:
        if not fname.endswith('.csv'): continue
        df = pd.read_csv(os.path.join(dirpath, fname), encoding='ISO-8859-1')
        if 'FC1' in dirpath or 'FC1' in fname: data_dict['FC1']['ageing'].append(df)

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
    smoothed_df[col] = savgol_filter(s_df[col].values, SG_WINDOW, 2)

target_col = 'Utot (V)'; all_cols = list(meas_cols); target_idx = all_cols.index(target_col)
data_values = smoothed_df[all_cols].values.astype(np.float32)
N = len(data_values); n_features_orig = len(all_cols)
time_h = smoothed_df['Time (h)'].values

# ============ EOL 定义: 活化峰值 ============
V_smooth = data_values[:, target_idx].astype(np.float64)
t_peak_idx = int(np.argmax(V_smooth))
V_init = V_smooth[t_peak_idx]
t_peak = time_h[t_peak_idx]
V_fail = V_init * 0.96
fail_idx = np.where(V_smooth < V_fail)[0]
EOL_true = time_h[fail_idx[0]] if len(fail_idx) > 0 else time_h[-1]
print(f"FC1: {N}帧, 活化峰值 V_init={V_init:.4f}V @ t={t_peak:.1f}h, V_fail={V_fail:.4f}V, EOL={EOL_true:.1f}h")

# 衍生特征列索引
def _f(nm):
    for i, c in enumerate(all_cols):
        if nm in c: return i
    return 0
ui=_f('Utot'); ii=_f('I (A)'); twi=_f('TinWAT'); two=_f('ToutWAT'); pai=_f('PinAIR'); pao=_f('PoutAIR')
u1=_f('U1 '); u2=_f('U2 '); u3=_f('U3 '); u4=_f('U4 '); u5=_f('U5 '); ucl=[u1,u2,u3,u4,u5]

# ============ TFT 模型 (Adaptive GRN, 无 MCLSTM cell 约束) ============
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
            mod = torch.tanh(self.gate_mod(dynamic_score.view(-1, 1)))
            g_raw = g_raw + 0.3 * mod
        g = torch.sigmoid(g_raw)
        return self.ln(a + self.dropout(g * self.fc2(h)))

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
        self.fc_dV = nn.Linear(hidden, 1)
    def _dynamic_score(self, x):
        return x[:, 0, :6].std(dim=1)
    def forward(self, x):
        dyn = self._dynamic_score(x)
        w = torch.sigmoid(self.feat_weights + self.feat_bias)
        xw = x * w.unsqueeze(0).unsqueeze(0)
        if self.training:
            xw = xw + torch.randn_like(xw) * 0.01
        lstm_out, _ = self.lstm(xw)
        lstm_out = self.dropout_layer(lstm_out)
        grn_out = self.grn(lstm_out, dynamic_score=dyn)
        attn_out, _ = self.attn(grn_out, grn_out, grn_out)
        attn_out = self.attn_ln(attn_out + grn_out)
        shared = attn_out[:, -1, :]
        return self.fc_V(shared)

def build_features(data_2d, y_target):
    n = len(data_2d)
    Xs = data_2d
    du = np.diff(y_target, prepend=y_target[0])
    soh = y_target / (V_init + 1e-8)
    dtw = Xs[:, twi] - Xs[:, two]
    pdv = Xs[:, pai] - Xs[:, pao]
    ust = np.std(Xs[:, ucl], axis=1)
    umo = np.mean(Xs[:, ucl], axis=1)
    roh = (Xs[:, ui] - umo) / (Xs[:, ii] + 1e-6)
    return np.column_stack([du, soh, dtw, pdv, ust, roh]).astype(np.float32)

# ============ 训练: [0, Sp] 状态估计 (L=1 + lag) ============
def train_on_sp(Sp):
    mask_tr = time_h <= Sp
    tr_data = data_values[mask_tr]
    n_tr = len(tr_data)
    X_secs, y_secs = [], []
    for lag in LAGS:
        lm = lag / 10.0
        X_secs.append(np.concatenate([
            tr_data[MAX_LAG-lag:n_tr-lag],
            np.full((n_tr-MAX_LAG, 1), lm, dtype=np.float32)], axis=1))
        y_secs.append(tr_data[MAX_LAG:n_tr, target_idx])
    X_raw = np.concatenate(X_secs, axis=0)
    y_raw = np.concatenate(y_secs, axis=0).astype(np.float32)
    Xd = build_features(X_raw[:, :n_features_orig], y_raw)
    X_raw = np.concatenate([X_raw, Xd], axis=1)
    n_feat = X_raw.shape[1]

    tr_end = int(len(X_raw)*0.9)
    sX = MinMaxScaler(); sy = MinMaxScaler()
    X_tr = sX.fit_transform(X_raw[:tr_end]).reshape(-1, 1, n_feat)
    X_vl = sX.transform(X_raw[tr_end:]).reshape(-1, 1, n_feat)
    y_tr = sy.fit_transform(y_raw[:tr_end].reshape(-1,1)).ravel()
    y_vl = sy.transform(y_raw[tr_end:].reshape(-1,1)).ravel()

    tr_ld = DataLoader(TensorDataset(torch.tensor(X_tr,dtype=torch.float32),
        torch.tensor(y_tr,dtype=torch.float32).view(-1,1)), batch_size=64, shuffle=False)
    vl_ld = DataLoader(TensorDataset(torch.tensor(X_vl,dtype=torch.float32),
        torch.tensor(y_vl,dtype=torch.float32).view(-1,1)), batch_size=64, shuffle=False)

    model = TFTModel(n_feat).to(device)
    opt = optim.Adam(model.parameters(), lr=3e-4, weight_decay=1e-5)
    mse = nn.MSELoss()
    best_val = float('inf'); pc = 0
    for ep in range(300):
        model.train()
        for bx, by in tr_ld:
            bx, by = bx.to(device), by.to(device)
            opt.zero_grad(); loss = mse(model(bx), by); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5); opt.step()
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
    model.load_state_dict(torch.load('_tmp.pth')); os.remove('_tmp.pth')
    return model, sX, sy, n_feat

# ============ TFT 估计历史电压 (无泄露: 只用 <=Sp 数据) ============
def tft_estimate(model, sX, sy, n_feat, t_start, t_end):
    mask = (time_h >= t_start) & (time_h <= t_end)
    seg_data = data_values[mask]
    seg_time = time_h[mask]
    pred = []
    model.eval()
    with torch.no_grad():
        for i in range(len(seg_data)):
            x_sens = seg_data[i:i+1]
            lm = np.array([[0.1]], dtype=np.float32)
            x_in = np.concatenate([x_sens, lm], axis=1)
            du_f = 0.0
            soh_f = x_sens[0, ui] / (V_init + 1e-8)
            dtw_f = x_sens[0, twi] - x_sens[0, two]
            pdv_f = x_sens[0, pai] - x_sens[0, pao]
            ust_f = float(np.std(x_sens[0, ucl]))
            umo_f = float(np.mean(x_sens[0, ucl]))
            roh_f = (x_sens[0, ui] - umo_f) / (x_sens[0, ii] + 1e-6)
            xd = np.array([[du_f, soh_f, dtw_f, pdv_f, ust_f, roh_f]], dtype=np.float32)
            x_in = np.concatenate([x_in, xd], axis=1)
            x_t = torch.tensor(sX.transform(x_in).reshape(1, 1, n_feat), dtype=torch.float32).to(device)
            Vp = model(x_t).cpu().numpy()[0, 0]
            pred.append(sy.inverse_transform([[Vp]])[0, 0])
    return seg_time, np.array(pred)

# ============ 线性外推 ============
def linear_extrapolate(t_seg, V_seg):
    p = np.polyfit(t_seg, V_seg, 1)
    return (V_fail - p[1]) / p[0]

print(f"\n=== FC1 RUL: TFT状态估计+物理外推 vs SG基线 (EOL={EOL_true:.1f}h) ===")
print(f"{'Sp(h)':<8s} {'真实RUL':<10s} {'SG-RE%':<10s} {'TFT-RE%':<10s} {'TFT预测RUL':<10s}")
results = []
for Sp in SP_LIST:
    if Sp >= EOL_true: continue
    true_RUL = EOL_true - Sp
    # SG 基线
    mask = (time_h >= t_peak) & (time_h <= Sp)
    t_seg = time_h[mask]; V_seg = V_smooth[mask]
    sg_eol = linear_extrapolate(t_seg, V_seg)
    sg_re = abs((sg_eol - Sp) - true_RUL) / true_RUL * 100
    # TFT 版
    model, sX, sy, n_feat = train_on_sp(Sp)
    t_est, V_est = tft_estimate(model, sX, sy, n_feat, t_peak, Sp)
    tft_eol = linear_extrapolate(t_est, V_est)
    tft_re = abs((tft_eol - Sp) - true_RUL) / true_RUL * 100
    results.append((Sp, true_RUL, sg_re, tft_re, tft_eol - Sp))
    print(f"{Sp:<8d} {true_RUL:<10.1f} {sg_re:<10.2f} {tft_re:<10.2f} {tft_eol-Sp:<10.1f}")

print("\n=== 汇总 ===")
sg_mean = np.mean([r[2] for r in results]); tft_mean = np.mean([r[3] for r in results])
print(f"SG 基线平均 RE = {sg_mean:.2f}%")
print(f"TFT+外推平均 RE = {tft_mean:.2f}%")
