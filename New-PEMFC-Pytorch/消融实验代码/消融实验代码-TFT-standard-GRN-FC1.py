# -*- coding: utf-8 -*-
"""
消融实验 — TFT 标准GRN (FC1)

消融: TFT 的 GRN 去掉自适应门控(gate_mod), 退化为标准 GRN。
"""
import sys, os, warnings, random
warnings.filterwarnings('ignore')
try: sys.stdout.reconfigure(encoding='utf-8')
except AttributeError: pass

import numpy as np, pandas as pd
import torch, torch.nn as nn, torch.nn.functional as F, torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score
from scipy.signal import savgol_filter
from pathlib import Path

seed_value = 42
random.seed(seed_value); np.random.seed(seed_value); torch.manual_seed(seed_value)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

DATA_PATH = Path(r"E:\沈农信电\课题-预测质子交换膜燃料电池（PEMFC）的健康状态和剩余寿命\代码\Long-Term-PEMFC-DL-Prognostics-main\Long-Term-PEMFC-DL-Prognostics-main\Long-Term-PEMFC-DL-Prognostics-main\IEEE2014DataChallengeData")
USE_FC = 'FC1'
SAMPLING_INTERVAL = 1.0
LAGS = [1, 2, 4]; MAX_LAG = 4
N_LOOPS = 50

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
n_features_orig = len(all_cols)
data_values = smoothed_df[all_cols].values.astype(np.float32)
N = len(data_values)

# ============ lag 增强 + 6 衍生特征 ============
X_sections, y_sections = [], []
for lag in LAGS:
    lm = lag/10.0
    X_lag = data_values[MAX_LAG-lag : N-lag]
    y_lag = data_values[MAX_LAG : N, target_idx]
    X_sections.append(np.concatenate([X_lag, np.full((len(X_lag),1), lm, dtype=np.float32)], axis=1))
    y_sections.append(y_lag)
X_raw = np.concatenate(X_sections, 0); y_raw = np.concatenate(y_sections, 0)
n_total = len(X_raw); n_per_lag = N - MAX_LAG
def _f(nm):
    for i,c in enumerate(all_cols):
        if nm in c: return i
    return 0
tin=_f('TinWAT'); tout=_f('ToutWAT'); pai=_f('PinAIR'); pao=_f('PoutAIR')
u1=_f('U1 '); u2=_f('U2 '); u3=_f('U3 '); u4=_f('U4 '); u5=_f('U5 '); ucl=[u1,u2,u3,u4,u5]
ii=_f('I (A)'); ui=_f('Utot')
X_sensor = X_raw[:, :n_features_orig]
delta_utot = np.zeros(n_total, dtype=np.float32)
for k in range(len(LAGS)):
    ss,se=k*n_per_lag,(k+1)*n_per_lag
    sec_y=y_raw[ss:se]; delta_utot[ss:se]=np.diff(sec_y, prepend=sec_y[0])
v_init=np.percentile(y_raw[:int(n_per_lag*0.5)],95); soh=y_raw/(v_init+1e-8)
dtw=X_sensor[:,tin]-X_sensor[:,tout]; pdv=X_sensor[:,pai]-X_sensor[:,pao]
ust=np.std(X_sensor[:,ucl],axis=1); umo=np.mean(X_sensor[:,ucl],axis=1); roh=(X_sensor[:,ui]-umo)/(X_sensor[:,ii]+1e-6)
Xd=np.column_stack([delta_utot,soh,dtw,pdv,ust,roh]).astype(np.float32)
X_all=np.concatenate([X_raw,Xd],axis=1)
n_features = X_all.shape[1]

tr_end=int(n_per_lag*0.5); vl_end=int(n_per_lag*0.6)
Xtr=[];Xvl=[];Xts=[];ytr=[];yvl=[];yts=[]
for k in range(len(LAGS)):
    st=k*n_per_lag
    Xtr.append(X_all[st:st+tr_end]); Xvl.append(X_all[st+tr_end:st+vl_end]); Xts.append(X_all[st+vl_end:st+n_per_lag])
    ytr.append(y_raw[st:st+tr_end]); yvl.append(y_raw[st+tr_end:st+vl_end]); yts.append(y_raw[st+vl_end:st+n_per_lag])
X_train = np.concatenate(Xtr); X_val = np.concatenate(Xvl); X_test = np.concatenate(Xts)
y_train = np.concatenate(ytr); y_val = np.concatenate(yvl); y_test = np.concatenate(yts)

sX = MinMaxScaler(); sy = MinMaxScaler()
X_train_s = sX.fit_transform(X_train).reshape(-1,1,n_features)
X_val_s = sX.transform(X_val).reshape(-1,1,n_features)
X_test_s = sX.transform(X_test).reshape(-1,1,n_features)
y_train_s = sy.fit_transform(y_train.reshape(-1,1)).ravel()
y_val_s = sy.transform(y_val.reshape(-1,1)).ravel()
y_test_s = sy.transform(y_test.reshape(-1,1)).ravel()

def _dV(y): return np.diff(y, prepend=y[0])
y_train_dV = torch.tensor(_dV(y_train_s), dtype=torch.float32).view(-1,1)
y_val_dV = torch.tensor(_dV(y_val_s), dtype=torch.float32).view(-1,1)
train_ld = DataLoader(TensorDataset(torch.tensor(X_train_s,dtype=torch.float32), torch.tensor(y_train_s,dtype=torch.float32).view(-1,1), y_train_dV), batch_size=64, shuffle=False)
val_ld = DataLoader(TensorDataset(torch.tensor(X_val_s,dtype=torch.float32), torch.tensor(y_val_s,dtype=torch.float32).view(-1,1), y_val_dV), batch_size=64, shuffle=False)
test_ld = DataLoader(TensorDataset(torch.tensor(X_test_s,dtype=torch.float32), torch.tensor(y_test_s,dtype=torch.float32).view(-1,1)), batch_size=64, shuffle=False)

# ============ 模型 ============
class StandardGRN(nn.Module):
    """标准 GRN(无自适应门控, 消融对照)"""
    def __init__(self, d_in, d_hidden, d_out, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(d_in, d_hidden)
        self.fc2 = nn.Linear(d_hidden, d_out)
        self.gate = nn.Linear(d_hidden, d_out)
        self.ln = nn.LayerNorm(d_out)
        self.dropout = nn.Dropout(dropout)
        self.proj = nn.Linear(d_in, d_out) if d_in != d_out else nn.Identity()
    def forward(self, x):
        a = self.proj(x)
        h = F.elu(self.fc1(x))
        g = torch.sigmoid(self.gate(h))
        return self.ln(a + self.dropout(g * self.fc2(h)))

class TFTStandardGRN(nn.Module):
    """标准 GRN 版 TFT(无自适应门控)"""
    def __init__(self, input_size, hidden_size=96, num_layers=1, output_size=1,
                 nhead=4, dim_feedforward=128, dropout=0.1):
        super().__init__()
        self.feat_weights = nn.Parameter(torch.zeros(input_size))
        self.feat_bias = nn.Parameter(torch.zeros(input_size))
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.grn = StandardGRN(hidden_size, dim_feedforward, hidden_size, dropout)
        self.self_attn = nn.MultiheadAttention(hidden_size, nhead, dropout=dropout, batch_first=True)
        self.attn_ln = nn.LayerNorm(hidden_size)
        self.dropout_layer = nn.Dropout(dropout)
        self.fc_V = nn.Linear(hidden_size, 1)
        self.fc_dV = nn.Linear(hidden_size, 1)
    def forward(self, x, return_cell_state=False):
        w = torch.sigmoid(self.feat_weights + self.feat_bias)
        x_weighted = x * w.unsqueeze(0).unsqueeze(0)
        if self.training:
            x_weighted = x_weighted + torch.randn_like(x_weighted) * 0.01
        lstm_out, (h_n, c_n) = self.lstm(x_weighted)
        lstm_out = self.dropout_layer(lstm_out)
        grn_out = self.grn(lstm_out)
        attn_out, _ = self.self_attn(grn_out, grn_out, grn_out)
        attn_out = self.attn_ln(attn_out + grn_out)
        shared = attn_out[:, -1, :]
        V_pred = self.fc_V(shared)
        dV_pred = self.fc_dV(shared)
        if return_cell_state:
            return V_pred, dV_pred, c_n
        return V_pred, dV_pred


# ============ 训练 ============
def train_one(seed):
    torch.manual_seed(seed)
    model = TFTStandardGRN(n_features).to(device)
    opt = optim.Adam(model.parameters(), lr=3e-4, weight_decay=1e-5)
    sched = optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=0.5, patience=20)
    mse = nn.MSELoss(); huber = nn.HuberLoss(delta=0.01)
    best = float('inf'); pc = 0
    best_state = None
    swa_state = None; swa_n = 0
    for ep in range(500):
        model.train()
        for bx,by,bdV in train_ld:
            bx,by,bdV = bx.to(device),by.to(device),bdV.to(device); opt.zero_grad()
            V, dV, c = model(bx, return_cell_state=True)
            L_mse = mse(V, by); L_dV = huber(dV, bdV)
            diff = V[1:]-V[:-1]; L_mono = torch.mean(torch.clamp(diff,min=0)**2)
            L_smooth = torch.mean((V[2:]-2*V[1:-1]+V[:-2])**2) if len(V)>=3 else torch.tensor(0.,device=device)
            health = c[-1].mean(dim=1); L_cell = torch.mean(torch.clamp(health[1:]-health[:-1],min=0)**2)
            loss = L_mse + 0.3*L_dV + 0.1*L_mono + 0.05*L_smooth + 0.05*L_cell
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),0.5); opt.step()
        model.eval(); vl=0.0
        with torch.no_grad():
            for bx,by,_ in val_ld:
                bx=bx.to(device); by=by.to(device)
                V, _, _ = model(bx, return_cell_state=True)
                vl+=mse(V, by).item()
        sched.step(vl)
        if vl<best: best=vl; pc=0; best_state = {k: v.clone() for k,v in model.state_dict().items()}
        else:
            pc+=1
            if pc>=100: break
        if ep>=40:
            if swa_state is None:
                swa_state = {k: v.clone() for k,v in model.state_dict().items()}
                swa_n = 1
            else:
                for k in swa_state:
                    swa_state[k] = swa_state[k]*(swa_n/(swa_n+1)) + model.state_dict()[k]*(1/(swa_n+1))
                swa_n += 1
    model.load_state_dict(swa_state if swa_state is not None else best_state)
    model.eval(); pr=[]; tr=[]
    with torch.no_grad():
        for bx,by in test_ld:
            V, _, _ = model(bx.to(device), return_cell_state=True)
            pr.append(V.cpu().numpy()); tr.append(by.numpy())
    yp=sy.inverse_transform(np.concatenate(pr)).ravel(); yt=sy.inverse_transform(np.concatenate(tr)).ravel()
    return r2_score(yt,yp)


print(f"=== TFT 标准GRN (FC1, Loop {N_LOOPS}) ===")
r2s = [train_one(seed_value+li*10) for li in range(N_LOOPS)]
r2s = np.array(r2s)
print(f"Min={r2s.min():.4f}  Mean={r2s.mean():.4f}  Max={r2s.max():.4f}")
print(f"\n本文主模型(Adaptive GRN+MCLSTM TFT) FC1 Loop50: Min=0.7786 Mean=0.9069 Max=0.9688")

# ============ 记录结果到消融实验outputs ============
_out_dir = Path(r"E:\沈农信电\课题-预测质子交换膜燃料电池（PEMFC）的健康状态和剩余寿命\代码\New-PEMFC-Pytorch\消融实验outputs")
_out_dir.mkdir(parents=True, exist_ok=True)
_summary = _out_dir / "消融结果汇总.txt"
_lines = []
if _summary.exists():
    for _line in _summary.read_text(encoding='utf-8').splitlines():
        _line = _line.rstrip()
        if not _line: continue
        if _line.endswith(']'):
            _pos = _line.rfind('[')
            _tail = _line[_pos:]
            if '倒数第' in _tail and _tail.endswith('次]'):
                _line = _line[:_pos].rstrip()
        _lines.append(_line)
for _i in range(len(_lines)):
    _lines[_i] = _lines[_i] + '  [倒数第' + str(len(_lines)-_i+1) + '次]'
_lines.append("[FC1] TFT-standard-GRN: Min={:.4f} Mean={:.4f} Max={:.4f}  [倒数第1次]".format(r2s.min(), r2s.mean(), r2s.max()))
_summary.write_text(chr(10).join(_lines) + chr(10), encoding='utf-8')
print("结果已记录到 消融实验outputs/消融结果汇总.txt (倒数第1次)")
