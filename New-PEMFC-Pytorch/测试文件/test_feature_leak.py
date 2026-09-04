# -*- coding: utf-8 -*-
"""
验证特征泄露对状态估计 R² 的影响

A/B 对比:
  A(泄露版): 6 衍生特征全含 (ΔUtot, SOH, ΔT_water, P_diff, U_std, R_ohm_est)
  B(干净版): 只含 4 个干净特征 (ΔT_water, P_diff, U_std, R_ohm_est), 去掉 ΔUtot+SOH

其余完全一致: 数据/模型/训练/seed。单次训练快速看 R² 差异。
"""
import sys, os, warnings
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
np.random.seed(seed_value); torch.manual_seed(seed_value)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

DATA_PATH = Path(r"E:\沈农信电\课题-预测质子交换膜燃料电池（PEMFC）的健康状态和剩余寿命\代码\Long-Term-PEMFC-DL-Prognostics-main\Long-Term-PEMFC-DL-Prognostics-main\Long-Term-PEMFC-DL-Prognostics-main\IEEE2014DataChallengeData")
USE_FC = 'FC2'
SAMPLING_INTERVAL = 0.5
MAX_LAG = 4
LAGS = [1, 2, 4]

# ============ 数据加载 (对齐主文件: Static 文件优先) ============
data_dict = {"FC1": {"ageing": []}, "FC2": {"ageing": []}}
csv_files = list(DATA_PATH.rglob("*.csv"))
for file in csv_files:
    filename = file.name
    if "Static" not in filename:
        continue
    try:
        df = pd.read_csv(file, encoding="ISO-8859-1")
    except Exception:
        continue
    if "FC1" in filename:
        stack = "FC1"
    elif "FC2" in filename:
        stack = "FC2"
    else:
        continue
    if "Ageing" in filename:
        data_dict[stack]["ageing"].append(df)

if len(data_dict["FC1"]["ageing"]) == 0 and len(data_dict["FC2"]["ageing"]) == 0:
    for file in csv_files:
        filename = file.name
        try:
            df = pd.read_csv(file, encoding="ISO-8859-1")
        except Exception:
            continue
        if "FC1" in filename: stack = "FC1"
        elif "FC2" in filename: stack = "FC2"
        else: continue
        if "Ageing" in filename:
            data_dict[stack]["ageing"].append(df)

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
n_features_orig = len(all_cols)
target_idx = all_cols.index(target_col)
data_values = smoothed_df[all_cols].values.astype(np.float32)
N = len(data_values)

# lag 增强
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

def _find_col(name_contains):
    for i, c in enumerate(all_cols):
        if name_contains in c:
            return i
    raise KeyError(f"找不到列: {name_contains}")

utot_idx_raw = _find_col('Utot')
i_idx_raw = _find_col('I (A)') if any('I (A)' in c for c in all_cols) else _find_col('I(A')
tin_wat_idx = _find_col('TinWAT')
tout_wat_idx = _find_col('ToutWAT')
pin_air_idx = _find_col('PinAIR')
pout_air_idx = _find_col('PoutAIR')
u1_idx = _find_col('U1 '); u2_idx = _find_col('U2 '); u3_idx = _find_col('U3 ')
u4_idx = _find_col('U4 '); u5_idx = _find_col('U5 ')
u_cols = [u1_idx, u2_idx, u3_idx, u4_idx, u5_idx]

def build_derived(with_leak):
    """返回衍生特征矩阵 (n_total, k)"""
    X_sensor = X_raw[:, :n_features_orig]
    feats = []
    if with_leak:
        # F1 ΔUtot (泄露): 目标电压差分
        delta_utot = np.zeros(n_total, dtype=np.float32)
        for k in range(len(LAGS)):
            ss, se = k * n_per_lag, (k+1) * n_per_lag
            section_y = y_raw[ss:se]
            delta_utot[ss:se] = np.diff(section_y, prepend=section_y[0])
        feats.append(delta_utot)
        # F2 SOH (泄露): 目标电压归一化
        train_end_global = int(n_per_lag * 0.5)
        v_init = np.percentile(y_raw[:train_end_global], 95)
        feats.append(y_raw / (v_init + 1e-8))
    # F3 ΔT_water (干净)
    feats.append(X_sensor[:, tin_wat_idx] - X_sensor[:, tout_wat_idx])
    # F4 P_diff (干净)
    feats.append(X_sensor[:, pin_air_idx] - X_sensor[:, pout_air_idx])
    # F5 U_std (干净)
    feats.append(np.std(X_sensor[:, u_cols], axis=1))
    # F6 R_ohm_est (干净)
    u_mean_5 = np.mean(X_sensor[:, u_cols], axis=1)
    feats.append((X_sensor[:, utot_idx_raw] - u_mean_5) / (X_sensor[:, i_idx_raw] + 1e-6))
    return np.column_stack([f.astype(np.float32) for f in feats])

# ============ TFT 模型 (Adaptive GRN + MCLSTM, 对齐主文件) ============
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
    def __init__(self, input_size, hidden_size=96, num_layers=1, dim_feedforward=128, dropout=0.1):
        super().__init__()
        self.feat_weights = nn.Parameter(torch.zeros(input_size))
        self.feat_bias = nn.Parameter(torch.zeros(input_size))
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.grn = GatedResidualNetwork(hidden_size, dim_feedforward, hidden_size, dropout)
        self.self_attn = nn.MultiheadAttention(hidden_size, 4, dropout=dropout, batch_first=True)
        self.attn_ln = nn.LayerNorm(hidden_size)
        self.dropout_layer = nn.Dropout(dropout)
        self.fc_V = nn.Linear(hidden_size, 1)
        self.fc_dV = nn.Linear(hidden_size, 1)
    def _compute_dynamic_score(self, x):
        return x[:, 0, :6].std(dim=1)
    def forward(self, x, return_cell_state=False):
        dyn = self._compute_dynamic_score(x)
        w = torch.sigmoid(self.feat_weights + self.feat_bias)
        xw = x * w.unsqueeze(0).unsqueeze(0)
        if self.training:
            xw = xw + torch.randn_like(xw) * 0.01
        lstm_out, (h_n, c_n) = self.lstm(xw)
        lstm_out = self.dropout_layer(lstm_out)
        grn_out = self.grn(lstm_out, dynamic_score=dyn)
        attn_out, _ = self.self_attn(grn_out, grn_out, grn_out)
        attn_out = self.attn_ln(attn_out + grn_out)
        shared = attn_out[:, -1, :]
        V_pred = self.fc_V(shared)
        dV_pred = self.fc_dV(shared)
        if return_cell_state:
            return V_pred, dV_pred, c_n
        return V_pred, dV_pred

# ============ 训练 + 评估 (单次) ============
def run_experiment(with_leak):
    torch.manual_seed(seed_value)
    Xd = build_derived(with_leak)
    X_all = np.concatenate([X_raw, Xd], axis=1)
    n_feat = X_all.shape[1]

    # 划分 50/10/40 (按 lag 区块)
    train_end_lag = int(n_per_lag * 0.5)
    val_end_lag = int(n_per_lag * 0.6)
    X_tr_p, X_vl_p, X_ts_p = [], [], []
    y_tr_p, y_vl_p, y_ts_p = [], [], []
    for k in range(len(LAGS)):
        st = k * n_per_lag
        X_tr_p.append(X_all[st:st+train_end_lag]); X_vl_p.append(X_all[st+train_end_lag:st+val_end_lag]); X_ts_p.append(X_all[st+val_end_lag:st+n_per_lag])
        y_tr_p.append(y_raw[st:st+train_end_lag]); y_vl_p.append(y_raw[st+train_end_lag:st+val_end_lag]); y_ts_p.append(y_raw[st+val_end_lag:st+n_per_lag])
    X_tr = np.concatenate(X_tr_p); X_vl = np.concatenate(X_vl_p); X_ts = np.concatenate(X_ts_p)
    y_tr = np.concatenate(y_tr_p); y_vl = np.concatenate(y_vl_p); y_ts = np.concatenate(y_ts_p)

    sX = MinMaxScaler(); sy = MinMaxScaler()
    X_tr_s = sX.fit_transform(X_tr).reshape(-1, 1, n_feat)
    X_vl_s = sX.transform(X_vl).reshape(-1, 1, n_feat)
    X_ts_s = sX.transform(X_ts).reshape(-1, 1, n_feat)
    y_tr_s = sy.fit_transform(y_tr.reshape(-1,1)).ravel()
    y_vl_s = sy.transform(y_vl.reshape(-1,1)).ravel()
    y_ts_s = sy.transform(y_ts.reshape(-1,1)).ravel()

    def _dV(y):
        return np.diff(y, prepend=y[0])
    y_tr_dV = torch.tensor(_dV(y_tr_s), dtype=torch.float32).view(-1,1)
    y_vl_dV = torch.tensor(_dV(y_vl_s), dtype=torch.float32).view(-1,1)

    tr_ld = DataLoader(TensorDataset(torch.tensor(X_tr_s, dtype=torch.float32), torch.tensor(y_tr_s, dtype=torch.float32).view(-1,1), y_tr_dV), batch_size=64, shuffle=False)
    vl_ld = DataLoader(TensorDataset(torch.tensor(X_vl_s, dtype=torch.float32), torch.tensor(y_vl_s, dtype=torch.float32).view(-1,1), y_vl_dV), batch_size=64, shuffle=False)
    ts_ld = DataLoader(TensorDataset(torch.tensor(X_ts_s, dtype=torch.float32), torch.tensor(y_ts_s, dtype=torch.float32).view(-1,1)), batch_size=64, shuffle=False)

    model = TFTModel(n_feat).to(device)
    opt = optim.Adam(model.parameters(), lr=3e-4, weight_decay=1e-5)
    mse = nn.MSELoss(); huber = nn.HuberLoss(delta=0.01)
    best_val = float('inf'); pc = 0
    for ep in range(300):
        model.train()
        for bx, by, bdV in tr_ld:
            bx, by, bdV = bx.to(device), by.to(device), bdV.to(device)
            opt.zero_grad()
            Vp, dVp, c_n = model(bx, return_cell_state=True)
            L_mse = mse(Vp, by)
            L_dV = huber(dVp, bdV)
            diff = Vp[1:] - Vp[:-1]
            L_mono = torch.mean(torch.clamp(diff, min=0)**2)
            if len(Vp) >= 3:
                L_smooth = torch.mean((Vp[2:] - 2*Vp[1:-1] + Vp[:-2])**2)
            else:
                L_smooth = torch.tensor(0., device=device)
            health = c_n[-1].mean(dim=1)
            L_cell = torch.mean(torch.clamp(health[1:]-health[:-1], min=0)**2)
            loss = L_mse + 0.3*L_dV + 0.1*L_mono + 0.05*L_smooth + 0.05*L_cell
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            opt.step()
        model.eval(); vl = 0.0
        with torch.no_grad():
            for bx, by, _ in vl_ld:
                bx, by = bx.to(device), by.to(device)
                Vp, _ = model(bx)
                vl += mse(Vp, by).item()
        avl = vl/len(vl_ld)
        if avl < best_val:
            best_val = avl; pc = 0; torch.save(model.state_dict(), '_tmp.pth')
        else:
            pc += 1
            if pc >= 40: break
    model.load_state_dict(torch.load('_tmp.pth')); os.remove('_tmp.pth')

    model.eval(); pr, tr = [], []
    with torch.no_grad():
        for bx, by in ts_ld:
            bx = bx.to(device)
            Vp, _ = model(bx)
            pr.append(Vp.cpu().numpy()); tr.append(by.numpy())
    yp = sy.inverse_transform(np.concatenate(pr)).ravel()
    yt = sy.inverse_transform(np.concatenate(tr)).ravel()
    r2 = r2_score(yt, yp)
    rmse = np.sqrt(mean_squared_error(yt, yp))
    mae = mean_absolute_error(yt, yp)
    return r2, rmse, mae, n_feat

print(f"FC2: {N}帧, 采样{0.5}h, lag={LAGS}")
print("="*60)
for name, wl in [("A(泄露版, 6衍生含ΔUtot+SOH)", True), ("B(干净版, 4衍生去ΔUtot+SOH)", False)]:
    r2, rmse, mae, nf = run_experiment(wl)
    print(f"{name}: 特征数={nf}, R²={r2:.4f}, RMSE={rmse:.6f}V, MAE={mae:.6f}V")
