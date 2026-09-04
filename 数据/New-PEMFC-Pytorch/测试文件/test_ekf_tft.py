# -*- coding: utf-8 -*-
"""
EKF + TFT 混合验证 — 最小版

EKF 用一阶退化模型估计隐藏状态 [电压 V, 退化率 a]:
  V_k = V_{k-1} - a_{k-1}·dt + w_v
  a_k = a_{k-1} + w_a              (退化率缓慢漂移)
  测量: z_k = V_k + 噪声

对比三种特征配置 (FC2 主任务, 单次训练):
  A. 基线 TFT (现有 31 维特征)
  B. + a_hat (EKF 估计退化率作为额外特征)
  C. V_hat 替换 (EKF 滤波电压替代原始电压 = EKF 去噪)
"""
import sys, os, warnings
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
np.random.seed(seed_value); torch.manual_seed(seed_value)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

DATA_PATH = Path(r"E:\沈农信电\课题-预测质子交换膜燃料电池（PEMFC）的健康状态和剩余寿命\代码\Long-Term-PEMFC-DL-Prognostics-main\Long-Term-PEMFC-DL-Prognostics-main\Long-Term-PEMFC-DL-Prognostics-main\IEEE2014DataChallengeData")
USE_FC = 'FC2'
SAMPLING_INTERVAL = 0.5
LAGS = [1, 2, 4]
MAX_LAG = 4

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
target_col = 'Utot (V)'
all_cols = list(meas_cols)
target_idx = all_cols.index(target_col)
data_values = smoothed_df[all_cols].values.astype(np.float32)
N = len(data_values)
n_features_orig = len(all_cols)

# 原始(未SG)电压用于 EKF 测量
V_meas = s_df[target_col].values.astype(np.float64)

# ============ EKF: 估计 [V, a] ============
def ekf(V_meas, dt, q_v=1e-6, q_a=1e-8, R=2.5e-5):
    n = len(V_meas)
    x = np.zeros((n, 2)); P = np.zeros((n, 2, 2))
    x[0] = [V_meas[0], 0.0]
    P[0] = np.diag([R, 1e-4])  # 初始不确定
    F = np.array([[1.0, -dt], [0.0, 1.0]])
    H = np.array([[1.0, 0.0]])
    Q = np.diag([q_v, q_a])
    I = np.eye(2)
    for k in range(1, n):
        # 预测
        x_pred = F @ x[k-1]
        P_pred = F @ P[k-1] @ F.T + Q
        # 更新
        S = H @ P_pred @ H.T + R
        K = P_pred @ H.T / S
        x[k] = x_pred + K.flatten() * (V_meas[k] - (H @ x_pred)[0])
        P[k] = (I - K @ H) @ P_pred
    return x  # x[:,0]=V_hat, x[:,1]=a_hat

dt = SAMPLING_INTERVAL
ekf_state = ekf(V_meas, dt)
V_hat = ekf_state[:, 0]
a_hat = ekf_state[:, 1]
print(f"EKF 完成: V_hat range [{V_hat.min():.4f}, {V_hat.max():.4f}], a_hat mean={a_hat.mean():.2e}")

# ============ lag 增强 (复用主代码) ============
X_sections, y_sections = [], []
for lag in LAGS:
    lag_marker = lag / 10.0
    X_lag = data_values[MAX_LAG-lag : N-lag]
    y_lag = data_values[MAX_LAG : N, target_idx]
    lag_col = np.full((len(X_lag), 1), lag_marker, dtype=np.float32)
    X_sections.append(np.concatenate([X_lag, lag_col], axis=1))
    y_sections.append(y_lag)
X_raw = np.concatenate(X_sections, axis=0)
y_raw = np.concatenate(y_sections, axis=0)
n_total = len(X_raw)
n_per_lag = N - MAX_LAG

# 衍生特征 (简化: 4 个干净特征, 不含 ΔUtot/SOH 以聚焦 EKF 增益)
def _f(nm):
    for i, c in enumerate(all_cols):
        if nm in c: return i
    return 0
tin_wat = _f('TinWAT'); tout_wat = _f('ToutWAT')
pin_air = _f('PinAIR'); pout_air = _f('PoutAIR')
u1=_f('U1 '); u2=_f('U2 '); u3=_f('U3 '); u4=_f('U4 '); u5=_f('U5 ')
ucl=[u1,u2,u3,u4,u5]
ii=_f('I (A)'); ui=_f('Utot')

def build_derived(X_sensor):
    dtw = X_sensor[:, tin_wat] - X_sensor[:, tout_wat]
    pdv = X_sensor[:, pin_air] - X_sensor[:, pout_air]
    ust = np.std(X_sensor[:, ucl], axis=1)
    umo = np.mean(X_sensor[:, ucl], axis=1)
    roh = (X_sensor[:, ui] - umo) / (X_sensor[:, ii] + 1e-6)
    return np.column_stack([dtw, pdv, ust, roh]).astype(np.float32)

# EKF 特征 (对齐 lag 增强后的样本)
a_hat_align = np.zeros(n_total, dtype=np.float32)
V_hat_align = np.zeros(n_total, dtype=np.float32)
for k, lag in enumerate(LAGS):
    sec = slice(k*n_per_lag, (k+1)*n_per_lag)
    a_hat_align[sec] = a_hat[MAX_LAG-lag : N-lag]   # 对齐 X 的传感器时间
    V_hat_align[sec] = V_hat[MAX_LAG-lag : N-lag]

def make_features(use_ekf_a=False, use_ekf_v=False):
    X_sensor = X_raw[:, :n_features_orig]
    Xd = build_derived(X_sensor)
    feats = [X_raw, Xd]
    if use_ekf_a:
        feats.append(a_hat_align.reshape(-1, 1))
    if use_ekf_v:
        # 用 V_hat 替换 X 里的 Utot 列
        X_mod = X_raw.copy()
        X_mod[:, ui] = V_hat_align
        feats = [X_mod, Xd]
    return np.concatenate(feats, axis=1)

# ============ TFT 模型 ============
class GatedResidualNetwork(nn.Module):
    def __init__(self, d_in, d_hidden, d_out, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(d_in, d_hidden); self.fc2 = nn.Linear(d_hidden, d_out)
        self.gate = nn.Linear(d_hidden, d_out); self.gate_mod = nn.Linear(1, d_out)
        self.ln = nn.LayerNorm(d_out); self.dropout = nn.Dropout(dropout)
        self.proj = nn.Linear(d_in, d_out) if d_in != d_out else nn.Identity()
    def forward(self, x, dyn=None):
        a = self.proj(x); h = F.elu(self.fc1(x)); g_raw = self.gate(h)
        if dyn is not None: g_raw = g_raw + 0.3*torch.tanh(self.gate_mod(dyn.view(-1,1)))
        return self.ln(a + self.dropout(torch.sigmoid(g_raw)*self.fc2(h)))

class TFT(nn.Module):
    def __init__(self, nf, hidden=96, dim_ff=128):
        super().__init__()
        self.fw = nn.Parameter(torch.zeros(nf)); self.fb = nn.Parameter(torch.zeros(nf))
        self.lstm = nn.LSTM(nf, hidden, 1, batch_first=True)
        self.grn = GatedResidualNetwork(hidden, dim_ff, hidden)
        self.attn = nn.MultiheadAttention(hidden, 4, batch_first=True)
        self.ln = nn.LayerNorm(hidden); self.drop = nn.Dropout(0.1)
        self.fc = nn.Linear(hidden, 1)
    def forward(self, x):
        dyn = x[:, 0, :6].std(dim=1)
        w = torch.sigmoid(self.fw + self.fb); xw = x * w.unsqueeze(0).unsqueeze(0)
        lstm_out, _ = self.lstm(xw); lstm_out = self.drop(lstm_out)
        g = self.grn(lstm_out, dyn=dyn); a, _ = self.attn(g, g, g)
        return self.fc(self.ln(a + g)[:, -1, :])

def run(name, use_a, use_v):
    torch.manual_seed(seed_value)
    X_all = make_features(use_a, use_v)
    nf = X_all.shape[1]
    tr_end = int(n_per_lag*0.5); vl_end = int(n_per_lag*0.6)
    parts = []
    for k in range(len(LAGS)):
        st = k*n_per_lag
        parts.append((X_all[st:st+tr_end], X_all[st+tr_end:st+vl_end], X_all[st+vl_end:st+n_per_lag],
                      y_raw[st:st+tr_end], y_raw[st+tr_end:st+vl_end], y_raw[st+vl_end:st+n_per_lag]))
    Xtr = np.concatenate([p[0] for p in parts]); Xvl = np.concatenate([p[1] for p in parts]); Xts = np.concatenate([p[2] for p in parts])
    ytr = np.concatenate([p[3] for p in parts]); yvl = np.concatenate([p[4] for p in parts]); yts = np.concatenate([p[5] for p in parts])
    sX = MinMaxScaler(); sy = MinMaxScaler()
    Xtr_s = sX.fit_transform(Xtr).reshape(-1,1,nf); Xvl_s = sX.transform(Xvl).reshape(-1,1,nf); Xts_s = sX.transform(Xts).reshape(-1,1,nf)
    ytr_s = sy.fit_transform(ytr.reshape(-1,1)).ravel(); yvl_s = sy.transform(yvl.reshape(-1,1)).ravel(); yts_s = sy.transform(yts.reshape(-1,1)).ravel()
    tr_ld = DataLoader(TensorDataset(torch.tensor(Xtr_s,dtype=torch.float32), torch.tensor(ytr_s,dtype=torch.float32).view(-1,1)), batch_size=64, shuffle=False)
    vl_ld = DataLoader(TensorDataset(torch.tensor(Xvl_s,dtype=torch.float32), torch.tensor(yvl_s,dtype=torch.float32).view(-1,1)), batch_size=64, shuffle=False)
    ts_ld = DataLoader(TensorDataset(torch.tensor(Xts_s,dtype=torch.float32), torch.tensor(yts_s,dtype=torch.float32).view(-1,1)), batch_size=64, shuffle=False)
    model = TFT(nf).to(device)
    opt = optim.Adam(model.parameters(), lr=3e-4, weight_decay=1e-5); mse = nn.MSELoss()
    best = float('inf'); pc = 0
    for ep in range(300):
        model.train()
        for bx, by in tr_ld:
            bx, by = bx.to(device), by.to(device)
            opt.zero_grad(); loss = mse(model(bx), by); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5); opt.step()
        model.eval(); vl = 0.0
        with torch.no_grad():
            for bx, by in vl_ld: bx=bx.to(device); vl += mse(model(bx), by.to(device)).item()
        if vl < best: best = vl; pc = 0; torch.save(model.state_dict(), '_tmp.pth')
        else:
            pc += 1
            if pc >= 40: break
    model.load_state_dict(torch.load('_tmp.pth')); os.remove('_tmp.pth')
    model.eval(); pr, tr = [], []
    with torch.no_grad():
        for bx, by in ts_ld: pr.append(model(bx.to(device)).cpu().numpy()); tr.append(by.numpy())
    yp = sy.inverse_transform(np.concatenate(pr)).ravel(); yt = sy.inverse_transform(np.concatenate(tr)).ravel()
    r2 = r2_score(yt, yp); rmse = np.sqrt(mean_squared_error(yt, yp))
    print(f"{name}: 特征数={nf}, R²={r2:.4f}, RMSE={rmse:.6f}V")
    return r2, rmse

print(f"\n=== EKF + TFT 验证 (FC2, 单次) ===")
rA, _ = run("A. 基线 TFT         ", False, False)
rB, _ = run("B. + a_hat(退化率)  ", True,  False)
rC, _ = run("C. V_hat 替换(去噪) ", False, True)
print("\n=== 汇总 ===")
print(f"A 基线 = {rA:.4f}")
print(f"B +退化率 = {rB:.4f}  (Δ = {rB-rA:+.4f})")
print(f"C 去噪电压 = {rC:.4f}  (Δ = {rC-rA:+.4f})")
