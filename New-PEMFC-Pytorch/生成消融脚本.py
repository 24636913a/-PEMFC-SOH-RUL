# -*- coding: utf-8 -*-
"""
批量生成 30 个消融实验脚本（统一口径）。

口径统一为:
  - 预处理: FC1(1.0h) / FC2(0.5h) + SG(21,2) + lag=[1,2,4] + 6 衍生特征 + 50/10/40 划分 + MinMaxScaler
  - 训练: Adam(3e-4, wd=1e-5), 300 epochs, early stop patience=40, in-memory best_state(无 _tmp.pth)
  - 评估: Loop 50, 每个 loop 不同 seed, 返回 R², 最后 Min/Mean/Max
  - 记录: 追加到 消融实验outputs/消融结果汇总.txt

模型清单(15 个 × FC1/FC2 = 30 文件):
  单头通用: DLinear / ResMLP / LSTM-Baseline / TSMixer / RF-TCN
  双头前人: ChouWang-BiLSTM-TFT
  双头 TFT : PINN-TFT-MT(主模型) / TFT-纯MSE / TFT-minus-GRN / TFT-minus-VS
             / TFT-standard-GRN / TFT-minus-dV / TFT-minus-MCLSTM / TFT-minus-mono / TFT-minus-smooth
"""
import os

BASE = r"E:\沈农信电\课题-预测质子交换膜燃料电池（PEMFC）的健康状态和剩余寿命\代码\New-PEMFC-Pytorch\消融实验代码"
DATA_PATH_STR = r"E:\沈农信电\课题-预测质子交换膜燃料电池（PEMFC）的健康状态和剩余寿命\代码\Long-Term-PEMFC-DL-Prognostics-main\Long-Term-PEMFC-DL-Prognostics-main\Long-Term-PEMFC-DL-Prognostics-main\IEEE2014DataChallengeData"
OUT_DIR_STR = r"E:\沈农信电\课题-预测质子交换膜燃料电池（PEMFC）的健康状态和剩余寿命\代码\New-PEMFC-Pytorch\消融实验outputs"

# ==================== 公共模板 ====================
# @@FC@@ @@INTERVAL@@ @@TITLE@@ @@DOC@@ @@MODEL_DEF@@ @@TRAIN_BODY@@ @@MODEL@@

HEADER = '''# -*- coding: utf-8 -*-
"""
消融实验 — @@TITLE@@ (@@FC@@)

@@DOC@@
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

DATA_PATH = Path(r"@@DATA_PATH@@")
USE_FC = '@@FC@@'
SAMPLING_INTERVAL = @@INTERVAL@@
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
@@MODEL_DEF@@

# ============ 训练 ============
@@TRAIN_BODY@@

print(f"=== @@TITLE@@ (@@FC@@, Loop {N_LOOPS}) ===")
r2s = [train_one(seed_value+li*10) for li in range(N_LOOPS)]
r2s = np.array(r2s)
print(f"Min={r2s.min():.4f}  Mean={r2s.mean():.4f}  Max={r2s.max():.4f}")
print(f"\\n本文主模型(Adaptive GRN+MCLSTM TFT) @@FC@@ Loop50: @@BASELINE@@")

# ============ 记录结果到消融实验outputs ============
_out_dir = Path(r"@@OUT_DIR@@")
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
_lines.append("[@@FC@@] @@MODEL@@: Min={:.4f} Mean={:.4f} Max={:.4f}  [倒数第1次]".format(r2s.min(), r2s.mean(), r2s.max()))
_summary.write_text(chr(10).join(_lines) + chr(10), encoding='utf-8')
print("结果已记录到 消融实验outputs/消融结果汇总.txt (倒数第1次)")
'''

# ==================== 单头训练模板 ====================
SINGLE_TRAIN = '''def train_one(seed):
    torch.manual_seed(seed)
    model = @@MODEL_CLASS@@(n_features).to(device)
    opt = optim.Adam(model.parameters(), lr=3e-4, weight_decay=1e-5)
    sched = optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=0.5, patience=20)
    mse = nn.MSELoss()
    best = float('inf'); pc = 0
    best_state = None
    swa_state = None; swa_n = 0
    for ep in range(500):
        model.train()
        for bx,by,_ in train_ld:
            bx,by = bx.to(device),by.to(device); opt.zero_grad()
            loss = mse(model(bx), by)
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),0.5); opt.step()
        model.eval(); vl=0.0
        with torch.no_grad():
            for bx,by,_ in val_ld:
                bx=bx.to(device); by=by.to(device)
                vl+=mse(model(bx), by).item()
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
            pr.append(model(bx.to(device)).cpu().numpy()); tr.append(by.numpy())
    yp=sy.inverse_transform(np.concatenate(pr)).ravel(); yt=sy.inverse_transform(np.concatenate(tr)).ravel()
    return r2_score(yt,yp)
'''

# ==================== 双头训练模板 ====================
# @@MODEL_CLASS@@ @@FTRAIN@@ @@LOSS@@ @@FVAL@@ @@FTEST@@
DUAL_TRAIN = '''def train_one(seed):
    torch.manual_seed(seed)
    model = @@MODEL_CLASS@@(n_features).to(device)
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
            @@FTRAIN@@
@@LOSS@@
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),0.5); opt.step()
        model.eval(); vl=0.0
        with torch.no_grad():
            for bx,by,_ in val_ld:
                bx=bx.to(device); by=by.to(device)
                @@FVAL@@
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
            @@FTEST@@
            pr.append(V.cpu().numpy()); tr.append(by.numpy())
    yp=sy.inverse_transform(np.concatenate(pr)).ravel(); yt=sy.inverse_transform(np.concatenate(tr)).ravel()
    return r2_score(yt,yp)
'''

# ==================== Adaptive GRN + TFTModel 主模型定义 ====================
ADAPTIVE_GRN = '''class GatedResidualNetwork(nn.Module):
    """Adaptive GRN: 门控受电压波动动态得分调制(纹波强→门控松)"""
    def __init__(self, d_in, d_hidden, d_out, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(d_in, d_hidden)
        self.fc2 = nn.Linear(d_hidden, d_out)
        self.gate = nn.Linear(d_hidden, d_out)
        self.gate_mod = nn.Linear(1, d_out)  # 动态得分 -> 门控偏移
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
    """TFT + Adaptive GRN + 双头(V + dV) + MCLSTM(cell state 可返回)"""
    def __init__(self, input_size, hidden_size=96, num_layers=1, output_size=1,
                 nhead=4, dim_feedforward=128, dropout=0.1):
        super().__init__()
        self.feat_weights = nn.Parameter(torch.zeros(input_size))
        self.feat_bias = nn.Parameter(torch.zeros(input_size))
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.grn = GatedResidualNetwork(hidden_size, dim_feedforward, hidden_size, dropout)
        self.self_attn = nn.MultiheadAttention(hidden_size, nhead, dropout=dropout, batch_first=True)
        self.attn_ln = nn.LayerNorm(hidden_size)
        self.dropout_layer = nn.Dropout(dropout)
        self.fc_V = nn.Linear(hidden_size, 1)
        self.fc_dV = nn.Linear(hidden_size, 1)
    def forward(self, x, return_cell_state=False):
        dyn_score = x[:, 0, :6].std(dim=1)
        w = torch.sigmoid(self.feat_weights + self.feat_bias)
        x_weighted = x * w.unsqueeze(0).unsqueeze(0)
        if self.training:
            x_weighted = x_weighted + torch.randn_like(x_weighted) * 0.01
        lstm_out, (h_n, c_n) = self.lstm(x_weighted)
        lstm_out = self.dropout_layer(lstm_out)
        grn_out = self.grn(lstm_out, dynamic_score=dyn_score)
        attn_out, _ = self.self_attn(grn_out, grn_out, grn_out)
        attn_out = self.attn_ln(attn_out + grn_out)
        shared = attn_out[:, -1, :]
        V_pred = self.fc_V(shared)
        dV_pred = self.fc_dV(shared)
        if return_cell_state:
            return V_pred, dV_pred, c_n
        return V_pred, dV_pred
'''

# 标准 GRN（无 gate_mod）
STANDARD_GRN_MODEL = '''class StandardGRN(nn.Module):
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
'''

# TFT 去掉 GRN
TFT_MINUS_GRN_MODEL = '''class TFTMinusGRN(nn.Module):
    """TFT - GRN: 去掉门控残差网络, LSTM 直连 Attention"""
    def __init__(self, input_size, hidden_size=96, num_layers=1, output_size=1,
                 nhead=4, dim_feedforward=128, dropout=0.1):
        super().__init__()
        self.feat_weights = nn.Parameter(torch.zeros(input_size))
        self.feat_bias = nn.Parameter(torch.zeros(input_size))
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
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
        attn_out, _ = self.self_attn(lstm_out, lstm_out, lstm_out)
        attn_out = self.attn_ln(attn_out + lstm_out)
        shared = attn_out[:, -1, :]
        V_pred = self.fc_V(shared)
        dV_pred = self.fc_dV(shared)
        if return_cell_state:
            return V_pred, dV_pred, c_n
        return V_pred, dV_pred
'''

# TFT 去掉 VS
TFT_MINUS_VS_MODEL = '''class TFTMinusVS(nn.Module):
    """TFT - Variable Selection: 去掉变量选择, 传感器等权输入"""
    def __init__(self, input_size, hidden_size=96, num_layers=1, output_size=1,
                 nhead=4, dim_feedforward=128, dropout=0.1):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.grn = GatedResidualNetwork(hidden_size, dim_feedforward, hidden_size, dropout)
        self.self_attn = nn.MultiheadAttention(hidden_size, nhead, dropout=dropout, batch_first=True)
        self.attn_ln = nn.LayerNorm(hidden_size)
        self.dropout_layer = nn.Dropout(dropout)
        self.fc_V = nn.Linear(hidden_size, 1)
        self.fc_dV = nn.Linear(hidden_size, 1)
    def forward(self, x, return_cell_state=False):
        dyn_score = x[:, 0, :6].std(dim=1)
        x_weighted = x
        if self.training:
            x_weighted = x_weighted + torch.randn_like(x_weighted) * 0.01
        lstm_out, (h_n, c_n) = self.lstm(x_weighted)
        lstm_out = self.dropout_layer(lstm_out)
        grn_out = self.grn(lstm_out, dynamic_score=dyn_score)
        attn_out, _ = self.self_attn(grn_out, grn_out, grn_out)
        attn_out = self.attn_ln(attn_out + grn_out)
        shared = attn_out[:, -1, :]
        V_pred = self.fc_V(shared)
        dV_pred = self.fc_dV(shared)
        if return_cell_state:
            return V_pred, dV_pred, c_n
        return V_pred, dV_pred
'''

# TFT 去掉 dV 头（单头）
TFT_MINUS_DV_MODEL = '''class TFTMinusdV(nn.Module):
    """TFT - dV头: 去掉退化速率辅助头, 单头只预测电压"""
    def __init__(self, input_size, hidden_size=96, num_layers=1, output_size=1,
                 nhead=4, dim_feedforward=128, dropout=0.1):
        super().__init__()
        self.feat_weights = nn.Parameter(torch.zeros(input_size))
        self.feat_bias = nn.Parameter(torch.zeros(input_size))
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.grn = GatedResidualNetwork(hidden_size, dim_feedforward, hidden_size, dropout)
        self.self_attn = nn.MultiheadAttention(hidden_size, nhead, dropout=dropout, batch_first=True)
        self.attn_ln = nn.LayerNorm(hidden_size)
        self.dropout_layer = nn.Dropout(dropout)
        self.fc_V = nn.Linear(hidden_size, 1)
    def forward(self, x, return_cell_state=False):
        dyn_score = x[:, 0, :6].std(dim=1)
        w = torch.sigmoid(self.feat_weights + self.feat_bias)
        x_weighted = x * w.unsqueeze(0).unsqueeze(0)
        if self.training:
            x_weighted = x_weighted + torch.randn_like(x_weighted) * 0.01
        lstm_out, (h_n, c_n) = self.lstm(x_weighted)
        lstm_out = self.dropout_layer(lstm_out)
        grn_out = self.grn(lstm_out, dynamic_score=dyn_score)
        attn_out, _ = self.self_attn(grn_out, grn_out, grn_out)
        attn_out = self.attn_ln(attn_out + grn_out)
        shared = attn_out[:, -1, :]
        V_pred = self.fc_V(shared)
        if return_cell_state:
            return V_pred, c_n
        return V_pred
'''

# ==================== 前人模型 ====================
CHOUWANG_MODEL = '''class GRN(nn.Module):
    """标准 GRN(Chou & Wang 2025)"""
    def __init__(self, d_in, d_hid, d_out):
        super().__init__()
        self.fc1=nn.Linear(d_in,d_hid); self.fc2=nn.Linear(d_hid,d_out); self.gate=nn.Linear(d_hid,d_out)
        self.ln=nn.LayerNorm(d_out); self.dp=nn.Dropout(0.1)
        self.proj=nn.Linear(d_in,d_out) if d_in!=d_out else nn.Identity()
    def forward(self,x):
        a=self.proj(x); h=F.elu(self.fc1(x)); g=torch.sigmoid(self.gate(h))
        return self.ln(a+self.dp(g*self.fc2(h)))

class ChouWangBiLSTMTFT(nn.Module):
    """Chou & Wang (2025): TFT 的 LSTM 编码器替换为 BiLSTM"""
    def __init__(self, input_size, hidden_size=96, num_layers=1, output_size=1,
                 nhead=4, dim_ff=128, dropout=0.1):
        super().__init__()
        self.feat_weights = nn.Parameter(torch.zeros(input_size))
        self.feat_bias = nn.Parameter(torch.zeros(input_size))
        self.bilstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True)
        self.grn = GRN(2*hidden_size, dim_ff, hidden_size)
        self.attn = nn.MultiheadAttention(hidden_size, nhead, dropout=dropout, batch_first=True)
        self.attn_ln = nn.LayerNorm(hidden_size)
        self.dp = nn.Dropout(dropout)
        self.fc_V = nn.Linear(hidden_size, 1)
        self.fc_dV = nn.Linear(hidden_size, 1)
    def forward(self, x):
        w = torch.sigmoid(self.feat_weights + self.feat_bias)
        xw = x * w.unsqueeze(0).unsqueeze(0)
        if self.training: xw = xw + torch.randn_like(xw)*0.01
        lstm_out, _ = self.bilstm(xw)      # (B, 1, 2*hidden)
        lstm_out = self.dp(lstm_out)
        grn_out = self.grn(lstm_out)       # (B, 1, hidden)
        attn_out, _ = self.attn(grn_out, grn_out, grn_out)
        attn_out = self.attn_ln(attn_out + grn_out)
        shared = attn_out[:, -1, :]
        return self.fc_V(shared), self.fc_dV(shared)
'''

RF_TCN_MODEL = '''class RF_TCN(nn.Module):
    """RF-TCN: 膨胀卷积(dilation=1,2,4,8) + 4 层全连接(48/32/16/1)"""
    def __init__(self, input_size, output_size=1):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 32, kernel_size=3, padding=1, dilation=1)
        self.conv2 = nn.Conv1d(32, 32, kernel_size=3, padding=2, dilation=2)
        self.conv3 = nn.Conv1d(32, 32, kernel_size=3, padding=4, dilation=4)
        self.conv4 = nn.Conv1d(32, 32, kernel_size=3, padding=8, dilation=8)
        self.fc = nn.Sequential(
            nn.Linear(32*input_size, 48), nn.ReLU(),
            nn.Linear(48, 32), nn.ReLU(),
            nn.Linear(32, 16), nn.ReLU(),
            nn.Linear(16, output_size),
        )
    def forward(self, x):
        x = x.squeeze(1).unsqueeze(1)  # (B, 1, input_size)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = x.flatten(1)
        return self.fc(x)
'''

# ==================== 通用模型 ====================
DLINEAR_MODEL = '''SEQUENCE_LENGTH = 1  # L=1 范式

class DLinearModel(nn.Module):
    """DLinear: 时序分解+线性映射 (L=1 退化为纯线性, 消融基线)"""
    def __init__(self, input_size, hidden_size=96, num_layers=1, output_size=1, dropout=0.0):
        super().__init__()
        k = min(5, SEQUENCE_LENGTH)
        if k % 2 == 0: k -= 1
        self.decomp = nn.AvgPool1d(kernel_size=k, stride=1, padding=k//2)
        self.trend_linear = nn.Linear(SEQUENCE_LENGTH, output_size)
        self.residual_linear = nn.Linear(SEQUENCE_LENGTH, output_size)
    def forward(self, x):
        x_t = x.permute(0, 2, 1)
        trend = self.decomp(x_t)
        residual = x_t - trend
        t_out = self.trend_linear(trend)
        r_out = self.residual_linear(residual)
        return (t_out + r_out).mean(dim=1)
'''

RESMLP_MODEL = '''class ResMLPModel(nn.Module):
    """残差 MLP: LayerNorm -> Linear x2 + Skip -> LayerNorm -> Linear -> Output"""
    def __init__(self, input_size, hidden_size=96, num_layers=1, output_size=1, dropout=0.1):
        super().__init__()
        self.norm_in = nn.LayerNorm(input_size)
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.norm_mid = nn.LayerNorm(hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size // 2)
        self.head = nn.Linear(hidden_size // 2, output_size)
        self.dropout = nn.Dropout(dropout)
        self.proj = nn.Linear(input_size, hidden_size) if input_size != hidden_size else nn.Identity()
    def forward(self, x):
        if x.dim() == 3:
            x = x.squeeze(1)
        h = self.norm_in(x)
        residual = self.proj(h)
        h = F.silu(self.fc1(h))
        h = self.dropout(h)
        h = F.silu(self.fc2(h))
        h = h + residual
        h = self.norm_mid(h)
        h = F.silu(self.fc3(h))
        return self.head(h)
'''

LSTM_BASELINE_MODEL = '''class LSTMBaseline(nn.Module):
    """LSTM 基线: 标准 LSTM + LayerNorm + 全连接"""
    def __init__(self, input_size, hidden_size=96, num_layers=1, output_size=1, dropout=0.0):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.norm = nn.LayerNorm(hidden_size)
        self.fc = nn.Linear(hidden_size, output_size)
    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        out = self.norm(out)
        return self.fc(out)
'''

TSMIXER_MODEL = '''class TSMixerBlock(nn.Module):
    """TSMixer 特征混合块: Linear -> ReLU -> Dropout -> Linear -> +残差"""
    def __init__(self, dim, hidden_dim, dropout):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x):
        residual = x
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x + residual

class TSMixerModel(nn.Module):
    """TSMixer: All-MLP 时序预测 (KDD 2023), L=1 下纯特征混合"""
    def __init__(self, input_size, hidden_size=96, num_layers=2, output_size=1, dropout=0.1):
        super().__init__()
        self.blocks = nn.ModuleList()
        for _ in range(max(1, num_layers)):
            self.blocks.append(TSMixerBlock(input_size, hidden_size, dropout))
        self.head = nn.Linear(input_size, output_size)
    def forward(self, x):
        if x.dim() == 3:
            x = x.squeeze(1)
        for block in self.blocks:
            x = block(x)
        return self.head(x)
'''

# ==================== loss 代码片段 ====================
# 完整物理约束(主模型)
LOSS_FULL = '''            L_mse = mse(V, by); L_dV = huber(dV, bdV)
            diff = V[1:]-V[:-1]; L_mono = torch.mean(torch.clamp(diff,min=0)**2)
            L_smooth = torch.mean((V[2:]-2*V[1:-1]+V[:-2])**2) if len(V)>=3 else torch.tensor(0.,device=device)
            health = c[-1].mean(dim=1); L_cell = torch.mean(torch.clamp(health[1:]-health[:-1],min=0)**2)
            loss = L_mse + 0.3*L_dV + 0.1*L_mono + 0.05*L_smooth + 0.05*L_cell'''

# 纯 MSE
LOSS_MSE_ONLY = '''            loss = mse(V, by)'''

# 无 dV(单头物理约束)
LOSS_NO_DV = '''            L_mse = mse(V, by)
            diff = V[1:]-V[:-1]; L_mono = torch.mean(torch.clamp(diff,min=0)**2)
            L_smooth = torch.mean((V[2:]-2*V[1:-1]+V[:-2])**2) if len(V)>=3 else torch.tensor(0.,device=device)
            health = c[-1].mean(dim=1); L_cell = torch.mean(torch.clamp(health[1:]-health[:-1],min=0)**2)
            loss = L_mse + 0.1*L_mono + 0.05*L_smooth + 0.05*L_cell'''

# 无 cell(MCLSTM 消融)
LOSS_NO_CELL = '''            L_mse = mse(V, by); L_dV = huber(dV, bdV)
            diff = V[1:]-V[:-1]; L_mono = torch.mean(torch.clamp(diff,min=0)**2)
            L_smooth = torch.mean((V[2:]-2*V[1:-1]+V[:-2])**2) if len(V)>=3 else torch.tensor(0.,device=device)
            loss = L_mse + 0.3*L_dV + 0.1*L_mono + 0.05*L_smooth'''

# 无 mono(单调消融)
LOSS_NO_MONO = '''            L_mse = mse(V, by); L_dV = huber(dV, bdV)
            L_smooth = torch.mean((V[2:]-2*V[1:-1]+V[:-2])**2) if len(V)>=3 else torch.tensor(0.,device=device)
            health = c[-1].mean(dim=1); L_cell = torch.mean(torch.clamp(health[1:]-health[:-1],min=0)**2)
            loss = L_mse + 0.3*L_dV + 0.05*L_smooth + 0.05*L_cell'''

# 无 smooth(平滑消融)
LOSS_NO_SMOOTH = '''            L_mse = mse(V, by); L_dV = huber(dV, bdV)
            diff = V[1:]-V[:-1]; L_mono = torch.mean(torch.clamp(diff,min=0)**2)
            health = c[-1].mean(dim=1); L_cell = torch.mean(torch.clamp(health[1:]-health[:-1],min=0)**2)
            loss = L_mse + 0.3*L_dV + 0.1*L_mono + 0.05*L_cell'''

# ChouWang 双头(dV 辅助, 无物理约束)
LOSS_CHOUWANG = '''            loss = mse(V, by) + 0.3*huber(dV, bdV)'''

# ==================== 模型注册表 ====================
# kind: 'single'(模板 B) 或 'dual'(模板 A)
# baseline: 打印的主模型成绩
MODELS = {}

def _add(name, title, doc, kind, model_def, model_class, baseline,
         ftrain=None, loss=None, fval=None, ftest=None):
    MODELS[name] = dict(
        name=name, title=title, doc=doc, kind=kind, model_def=model_def,
        model_class=model_class, baseline=baseline,
        ftrain=ftrain, loss=loss, fval=fval, ftest=ftest,
    )

BASE_FC1 = "Min=0.7786 Mean=0.9069 Max=0.9688"
BASE_FC2 = "Min=0.9458 Mean=0.9767 Max=0.9939"

# --- 单头通用模型 ---
_add('DLinear', 'DLinear 时序分解线性', '通用线性基线: 时序分解 + 线性映射, L=1 下退化为纯线性。', 'single',
     DLINEAR_MODEL, 'DLinearModel', BASE_FC1)
_add('ResMLP', 'ResMLP 残差MLP', '通用残差 MLP 基线: LayerNorm + 残差连接 + 全连接。', 'single',
     RESMLP_MODEL, 'ResMLPModel', BASE_FC1)
_add('LSTM-Baseline', 'LSTM 基线', '通用 LSTM 基线: 标准 LSTM + LayerNorm + 全连接。', 'single',
     LSTM_BASELINE_MODEL, 'LSTMBaseline', BASE_FC1)
_add('TSMixer', 'TSMixer 全MLP', '通用 All-MLP 时序模型 (KDD 2023)。', 'single',
     TSMIXER_MODEL, 'TSMixerModel', BASE_FC1)
_add('RF-TCN', 'RF-TCN 前人模型', '前人模型复现: RF-RFE 特征选择 + 改进 TCN(膨胀卷积 + 4 全连接)。', 'single',
     RF_TCN_MODEL, 'RF_TCN', BASE_FC1)

# --- 双头前人模型 ---
_add('ChouWang-BiLSTM-TFT', 'Chou & Wang BiLSTM-TFT', '前人模型复现: Chou & Wang (2025) 将 TFT 的 LSTM 替换为 BiLSTM。', 'dual',
     CHOUWANG_MODEL, 'ChouWangBiLSTMTFT', BASE_FC1,
     ftrain='V, dV = model(bx)',
     loss=LOSS_CHOUWANG,
     fval='V, _ = model(bx)',
     ftest='V, _ = model(bx.to(device))')

# --- TFT 主模型 + 变体 ---
_add('PINN-TFT-MT', 'PINN-TFT-MT 主模型', '本文主模型: Adaptive GRN + MCLSTM TFT, 双头 + 完整物理约束。', 'dual',
     ADAPTIVE_GRN, 'TFTModel', BASE_FC1,
     ftrain='V, dV, c = model(bx, return_cell_state=True)',
     loss=LOSS_FULL,
     fval='V, _, _ = model(bx, return_cell_state=True)',
     ftest='V, _, _ = model(bx.to(device), return_cell_state=True)')

_add('TFT-纯MSE', 'TFT 纯MSE', '消融: TFT 去掉全部物理约束 loss, 只保留纯 MSE(电压头)。', 'dual',
     ADAPTIVE_GRN, 'TFTModel', BASE_FC1,
     ftrain='V, dV = model(bx)',
     loss=LOSS_MSE_ONLY,
     fval='V, _ = model(bx)',
     ftest='V, _ = model(bx.to(device))')

_add('TFT-minus-GRN', 'TFT 去GRN', '消融: TFT 去掉门控残差网络(GRN), LSTM 直连 Attention。', 'dual',
     ADAPTIVE_GRN + '\n\n' + TFT_MINUS_GRN_MODEL, 'TFTMinusGRN', BASE_FC1,
     ftrain='V, dV, c = model(bx, return_cell_state=True)',
     loss=LOSS_FULL,
     fval='V, _, _ = model(bx, return_cell_state=True)',
     ftest='V, _, _ = model(bx.to(device), return_cell_state=True)')

_add('TFT-minus-VS', 'TFT 去变量选择', '消融: TFT 去掉变量选择(VS), 传感器等权输入。', 'dual',
     ADAPTIVE_GRN + '\n\n' + TFT_MINUS_VS_MODEL, 'TFTMinusVS', BASE_FC1,
     ftrain='V, dV, c = model(bx, return_cell_state=True)',
     loss=LOSS_FULL,
     fval='V, _, _ = model(bx, return_cell_state=True)',
     ftest='V, _, _ = model(bx.to(device), return_cell_state=True)')

_add('TFT-standard-GRN', 'TFT 标准GRN', '消融: TFT 的 GRN 去掉自适应门控(gate_mod), 退化为标准 GRN。', 'dual',
     STANDARD_GRN_MODEL, 'TFTStandardGRN', BASE_FC1,
     ftrain='V, dV, c = model(bx, return_cell_state=True)',
     loss=LOSS_FULL,
     fval='V, _, _ = model(bx, return_cell_state=True)',
     ftest='V, _, _ = model(bx.to(device), return_cell_state=True)')

_add('TFT-minus-dV', 'TFT 去dV头', '消融: TFT 去掉退化速率辅助头(单头), 验证 ΔV 辅助任务价值。', 'dual',
     ADAPTIVE_GRN + '\n\n' + TFT_MINUS_DV_MODEL, 'TFTMinusdV', BASE_FC1,
     ftrain='V, c = model(bx, return_cell_state=True)',
     loss=LOSS_NO_DV,
     fval='V, _ = model(bx, return_cell_state=True)',
     ftest='V, _ = model(bx.to(device), return_cell_state=True)')

_add('TFT-minus-MCLSTM', 'TFT 去MCLSTM', '消融: 关掉 cell 单调约束(λ_cell=0), 验证 MCLSTM 价值。', 'dual',
     ADAPTIVE_GRN, 'TFTModel', BASE_FC1,
     ftrain='V, dV = model(bx)',
     loss=LOSS_NO_CELL,
     fval='V, _ = model(bx)',
     ftest='V, _ = model(bx.to(device))')

_add('TFT-minus-mono', 'TFT 去单调约束', '消融: 关掉电压单调约束(λ_mono=0)。', 'dual',
     ADAPTIVE_GRN, 'TFTModel', BASE_FC1,
     ftrain='V, dV, c = model(bx, return_cell_state=True)',
     loss=LOSS_NO_MONO,
     fval='V, _, _ = model(bx, return_cell_state=True)',
     ftest='V, _, _ = model(bx.to(device), return_cell_state=True)')

_add('TFT-minus-smooth', 'TFT 去平滑约束', '消融: 关掉电压平滑约束(λ_smooth=0)。', 'dual',
     ADAPTIVE_GRN, 'TFTModel', BASE_FC1,
     ftrain='V, dV, c = model(bx, return_cell_state=True)',
     loss=LOSS_NO_SMOOTH,
     fval='V, _, _ = model(bx, return_cell_state=True)',
     ftest='V, _, _ = model(bx.to(device), return_cell_state=True)')

# ==================== 生成 ====================
FC_CONFIG = {'FC1': ('1.0', BASE_FC1), 'FC2': ('0.5', BASE_FC2)}

def render(model, fc):
    interval, base = FC_CONFIG[fc]
    text = HEADER
    text = text.replace('@@DATA_PATH@@', DATA_PATH_STR)
    text = text.replace('@@OUT_DIR@@', OUT_DIR_STR)
    text = text.replace('@@FC@@', fc)
    text = text.replace('@@INTERVAL@@', interval)
    text = text.replace('@@TITLE@@', model['title'])
    text = text.replace('@@DOC@@', model['doc'])
    text = text.replace('@@MODEL@@', model['name'])
    text = text.replace('@@MODEL_DEF@@', model['model_def'])
    text = text.replace('@@BASELINE@@', base)

    if model['kind'] == 'single':
        body = SINGLE_TRAIN.replace('@@MODEL_CLASS@@', model['model_class'])
    else:
        body = DUAL_TRAIN.replace('@@MODEL_CLASS@@', model['model_class'])
        body = body.replace('@@FTRAIN@@', model['ftrain'])
        body = body.replace('@@LOSS@@', model['loss'])
        body = body.replace('@@FVAL@@', model['fval'])
        body = body.replace('@@FTEST@@', model['ftest'])
    text = text.replace('@@TRAIN_BODY@@', body)
    return text

def main():
    os.makedirs(BASE, exist_ok=True)
    n = 0
    for name in MODELS:
        for fc in ('FC1', 'FC2'):
            content = render(MODELS[name], fc)
            fname = f"消融实验代码-{name}-{fc}.py"
            fpath = os.path.join(BASE, fname)
            with open(fpath, 'w', encoding='utf-8') as f:
                f.write(content)
            n += 1
            print(f"[OK] {fname}")
    print(f"\n共生成 {n} 个脚本 -> {BASE}")

if __name__ == '__main__':
    main()
