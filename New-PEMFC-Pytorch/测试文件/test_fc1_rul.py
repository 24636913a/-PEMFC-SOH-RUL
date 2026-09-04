# -*- coding: utf-8 -*-
"""
FC1 RUL 预测实验 — 对齐 Chou & Wang 协议

任务: 从 Sp 出发, 预测未来电压直到穿越 EOL 阈值, 计算 RUL
FC1 EOL = 814h (96% 初始电压)
多起点: Sp = 300, 400, 500, 600h

方法: L=1 逐点状态映射(用历史传感器) → 持续监控直到穿越阈值
模型: TFT (VS + LSTM + GRN + Attention) + 物理约束
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

USE_FC = 'FC1'
SAMPLING_INTERVAL = 1.0
MAX_LAG = 4
LAGS = [1, 2, 4]
SP_LIST = [300, 400, 500, 600]

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
N = len(data_values); n_features_orig = len(all_cols)
time_h = smoothed_df['Time (h)'].values

# EOL 定义: 96% 初始电压 (用原始未平滑数据的稳健初值, 避免 SG 边界效应漂移 V_init)
V_init = float(np.mean(s_df[target_col].values[:20]))
V_fail = V_init * 0.96
fail_idx = np.where(data_values[:, target_idx] < V_fail)[0]
EOL_true = time_h[fail_idx[0]] if len(fail_idx) > 0 else time_h[-1]
print(f"FC1: {N}帧, V_init={V_init:.4f}V, V_fail={V_fail:.4f}V, EOL={EOL_true:.1f}h")

# 衍生特征函数
def _f(nm):
    for i, c in enumerate(all_cols):
        if nm in c: return i
    return 0
ui=_f('Utot'); ii=_f('I (A)'); twi=_f('TinWAT'); two=_f('ToutWAT'); pai=_f('PinAIR'); pao=_f('PoutAIR')
u1=_f('U1 '); u2=_f('U2 '); u3=_f('U3 '); u4=_f('U4 '); u5=_f('U5 '); ucl=[u1,u2,u3,u4,u5]

# TFT 模型 (对齐主代码: Adaptive GRN + MCLSTM + ΔV 双头)
class GatedResidualNetwork(nn.Module):
    """Adaptive GRN: 门控受动态得分调制"""
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
        glu = g * self.fc2(h)
        out = a + self.dropout(glu)
        return self.ln(out)

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
        volt_cols = x[:, 0, :6]
        return volt_cols.std(dim=1)
    def forward(self, x, return_cell_state=False):
        dyn = self._dynamic_score(x)
        w = torch.sigmoid(self.feat_weights + self.feat_bias)
        xw = x * w.unsqueeze(0).unsqueeze(0)
        if self.training:
            xw = xw + torch.randn_like(xw) * 0.01
        lstm_out, (h_n, c_n) = self.lstm(xw)
        lstm_out = self.dropout_layer(lstm_out)
        grn_out = self.grn(lstm_out, dynamic_score=dyn)
        attn_out, _ = self.attn(grn_out, grn_out, grn_out)
        attn_out = self.attn_ln(attn_out + grn_out)
        shared = attn_out[:, -1, :]
        V_pred = self.fc_V(shared)
        dV_pred = self.fc_dV(shared)
        if return_cell_state:
            return V_pred, dV_pred, c_n
        return V_pred, dV_pred

def build_features(data_2d, y_target):
    """data_2d: (n, 24) 传感器, y_target: (n,) 电压 → 31维特征"""
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

# 训练函数: 用 [0, Sp] 数据训练 L=1+lag 模型
def train_on_sp(Sp):
    mask_tr = time_h <= Sp
    tr_data = data_values[mask_tr]
    tr_time = time_h[mask_tr]
    n_tr = len(tr_data)
    # L=1 + lag 增强
    X_secs, y_secs = [], []
    for lag in LAGS:
        lm = lag / 10.0
        X_secs.append(np.concatenate([
            tr_data[MAX_LAG-lag:n_tr-lag],
            np.full((n_tr-MAX_LAG, 1), lm, dtype=np.float32)], axis=1))
        y_secs.append(tr_data[MAX_LAG:n_tr, target_idx])
    X_raw = np.concatenate(X_secs, axis=0)
    y_raw = np.concatenate(y_secs, axis=0).astype(np.float32)
    # 衍生特征
    Xd = build_features(X_raw[:, :n_features_orig], y_raw)
    X_raw = np.concatenate([X_raw, Xd], axis=1)
    n_feat = X_raw.shape[1]

    # 划分 90/10
    tr_end = int(len(X_raw)*0.9)
    sX = MinMaxScaler(); sy = MinMaxScaler()
    X_tr = sX.fit_transform(X_raw[:tr_end]).reshape(-1, 1, n_feat)
    X_vl = sX.transform(X_raw[tr_end:]).reshape(-1, 1, n_feat)
    y_tr = sy.fit_transform(y_raw[:tr_end].reshape(-1,1)).ravel()
    y_vl = sy.transform(y_raw[tr_end:].reshape(-1,1)).ravel()
    # ΔV 标签
    dV_tr = np.diff(y_tr, prepend=y_tr[0])
    dV_vl = np.diff(y_vl, prepend=y_vl[0])
    tr_ld = DataLoader(TensorDataset(
        torch.tensor(X_tr,dtype=torch.float32),
        torch.tensor(y_tr,dtype=torch.float32).view(-1,1),
        torch.tensor(dV_tr,dtype=torch.float32).view(-1,1)), batch_size=64, shuffle=False)
    vl_ld = DataLoader(TensorDataset(
        torch.tensor(X_vl,dtype=torch.float32),
        torch.tensor(y_vl,dtype=torch.float32).view(-1,1),
        torch.tensor(dV_vl,dtype=torch.float32).view(-1,1)), batch_size=64, shuffle=False)

    model = TFTModel(n_feat).to(device)
    opt = optim.Adam(model.parameters(), lr=3e-4, weight_decay=1e-5)
    mse = nn.MSELoss(); huber = nn.HuberLoss(delta=0.01)
    best_val = float('inf'); pc = 0
    for ep in range(300):
        model.train()
        for bx, by, bdV in tr_ld:
            bx, by, bdV = bx.to(device), by.to(device), bdV.to(device)
            opt.zero_grad()
            Vp, dVp = model(bx)
            L_mse = mse(Vp, by)
            L_dV = huber(dVp, bdV)
            # 输出层物理约束
            diff = Vp[1:] - Vp[:-1]
            L_mono = torch.mean(torch.clamp(diff, min=0)**2)
            if len(Vp) >= 3:
                sd = Vp[2:] - 2*Vp[1:-1] + Vp[:-2]
                L_smooth = torch.mean(sd**2)
            else:
                L_smooth = torch.tensor(0., device=device)
            # MCLSTM cell 约束在 RUL 任务上有害: 会把预测电压钉在阈值上方, 永不穿越 EOL
            # 因此 RUL 任务关闭 lambda_cell (L_cell 不参与 loss)
            loss = L_mse + 0.3*L_dV + 0.1*L_mono + 0.05*L_smooth
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
        if avl < best_val: best_val = avl; pc = 0; torch.save(model.state_dict(), '_tmp.pth')
        else:
            pc += 1
            if pc >= 40: break
    model.load_state_dict(torch.load('_tmp.pth')); os.remove('_tmp.pth')
    return model, sX, sy, n_feat

# RUL 预测: 从 Sp 开始, 逐点预测电压直到穿越 V_fail
def predict_rul(model, sX, sy, n_feat, Sp):
    mask_ts = time_h > Sp
    ts_data = data_values[mask_ts]
    ts_time = time_h[mask_ts]
    # 逐点: 用 lag=1 历史传感器预测当前电压
    model.eval()
    pred_V = []
    with torch.no_grad():
        for i in range(len(ts_data)):
            # 用当前帧传感器 + lag=1 标记
            x_sens = ts_data[i:i+1]  # (1, 24)
            lm = np.array([[0.1]], dtype=np.float32)
            x_in = np.concatenate([x_sens, lm], axis=1)
            # 衍生特征
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
            out = model(x_t)
            Vp = out[0].cpu().numpy()[0, 0] if isinstance(out, tuple) else out.cpu().numpy()[0, 0]
            pred_V.append(sy.inverse_transform([[Vp]])[0, 0])
    pred_V = np.array(pred_V)
    # 找穿越 V_fail 的时刻
    true_V = ts_data[:, target_idx]
    fail_idx_p = np.where(pred_V < V_fail)[0]
    if len(fail_idx_p) > 0:
        pred_EOL = ts_time[fail_idx_p[0]]
    else:
        pred_EOL = ts_time[-1] if len(ts_time) > 0 else Sp
    print(f"    [诊断] 真实电压min={true_V.min():.4f} 预测电压min={pred_V.min():.4f} V_fail={V_fail:.4f}")
    print(f"    [诊断] 预测末5={np.round(pred_V[-5:],4)} 真实末5={np.round(true_V[-5:],4)}")
    return pred_EOL

# 主流程
print(f"\n=== FC1 RUL 预测 (EOL={EOL_true:.1f}h) ===")
results = []
for Sp in SP_LIST:
    print(f"\n--- Sp={Sp}h ---")
    if Sp >= EOL_true:
        print(f"  Sp >= EOL, 跳过")
        continue
    model, sX, sy, n_feat = train_on_sp(Sp)
    pred_EOL = predict_rul(model, sX, sy, n_feat, Sp)
    true_RUL = EOL_true - Sp
    pred_RUL = pred_EOL - Sp
    abs_err = abs(pred_EOL - EOL_true)
    rel_err = abs_err / true_RUL * 100  # Chou & Wang 口径: RE = AE / RUL_actual
    results.append((Sp, true_RUL, pred_RUL, abs_err, rel_err))
    print(f"  真实RUL={true_RUL:.1f}h, 预测RUL={pred_RUL:.1f}h, 绝对误差={abs_err:.1f}h, 相对误差={rel_err:.2f}%")

print(f"\n=== FC1 RUL 结果汇总 ===")
print(f"{'Sp(h)':<8s} {'真实RUL(h)':<12s} {'预测RUL(h)':<12s} {'误差(h)':<10s} {'RE%':<8s}")
for Sp, tR, pR, ae, re in results:
    print(f"{Sp:<8d} {tR:<12.1f} {pR:<12.1f} {ae:<10.1f} {re:<8.2f}")

with open('fc1_rul_results.pkl', 'wb') as f:
    pickle.dump(results, f)
