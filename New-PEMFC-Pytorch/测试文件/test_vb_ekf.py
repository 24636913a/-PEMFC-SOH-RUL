# -*- coding: utf-8 -*-
"""
VB-AKF (变分贝叶斯自适应KF) — FC1 简单验证

对比三种滤波替换 Utot 列:
  A. SG(21,2) 基线
  B. 普通 EKF (固定 R)
  C. VB-AKF (变分贝叶斯在线自适应 R, Sarkka 2009)

VB-AKF 核心: 观测噪声 R 服从逆 Gamma 分布, 用变分贝叶斯迭代递推估计时变 R,
让滤波器根据当前残差自适应判断"滤多少"。
"""
import sys, os, warnings
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
import matplotlib; matplotlib.use('Agg')

seed_value = 100
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DATA_PATH = Path(r"E:\沈农信电\课题-预测质子交换膜燃料电池（PEMFC）的健康状态和剩余寿命\代码\Long-Term-PEMFC-DL-Prognostics-main\Long-Term-PEMFC-DL-Prognostics-main\Long-Term-PEMFC-DL-Prognostics-main\IEEE2014DataChallengeData")
USE_FC = 'FC2'
SAMPLING_INTERVAL = 0.5
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
target_col = 'Utot (V)'
all_cols = list(meas_cols)
target_idx = all_cols.index(target_col)
n_features_orig = len(all_cols)
V_meas = s_df[target_col].values.astype(np.float64)

smoothed_df = pd.DataFrame({'Time (h)': sampled_df['Time (h)']})
for col in meas_cols:
    smoothed_df[col] = savgol_filter(s_df[col].values, 21, 2)
data_values = smoothed_df[all_cols].values.astype(np.float32)
N = len(data_values)
dt = SAMPLING_INTERVAL

# ============ 滤波函数 ============
def ekf(V, dt, q_v=1e-6, q_a=1e-8, R=2.5e-5):
    n = len(V); x = np.zeros((n,2)); P = np.zeros((n,2,2))
    x[0] = [V[0], 0.0]; P[0] = np.diag([R, 1e-4])
    F = np.array([[1.0,-dt],[0.0,1.0]]); H = np.array([[1.0,0.0]]); Q = np.diag([q_v,q_a]); I = np.eye(2)
    for k in range(1,n):
        xp = F @ x[k-1]; Pp = F @ P[k-1] @ F.T + Q
        S = H @ Pp @ H.T + R; K = Pp @ H.T / S
        x[k] = xp + K.flatten() * (V[k] - (H@xp)[0]); P[k] = (I - K@H) @ Pp
    return x[:, 0]

def vb_akf(V, dt, q_v=1e-6, q_a=1e-8, R0=2.5e-5, rho=0.98, N_iter=5):
    """VB-AKF: 观测噪声 R 时变, 逆 Gamma 先验, 变分贝叶斯迭代估计 (Sarkka 2009)"""
    n = len(V); x = np.zeros((n,2)); P = np.zeros((n,2,2))
    x[0] = [V[0], 0.0]; P[0] = np.diag([R0, 1e-4])
    F = np.array([[1.0,-dt],[0.0,1.0]]); H = np.array([[1.0,0.0]]); Q = np.diag([q_v,q_a]); I = np.eye(2)
    alpha = 3.0; beta = R0 * (alpha - 1.0)  # 初始 R ≈ R0
    for k in range(1, n):
        xp = F @ x[k-1]; Pp = F @ P[k-1] @ F.T + Q
        # VB 迭代: 交替估计 x 和 R
        a = 0.5 + alpha
        b = beta
        xk = xp.copy(); Pk = Pp.copy()
        for _ in range(N_iter):
            Rk = b / a
            S = H @ Pp @ H.T + Rk
            K = Pp @ H.T / S
            xk = xp + K.flatten() * (V[k] - (H@xp)[0])
            Pk = (I - K@H) @ Pp
            e = V[k] - (H@xk)[0]
            b = beta + 0.5 * e**2 + 0.5 * (H@Pk@H.T)[0,0]
        x[k] = xk; P[k] = Pk
        # 遗忘因子传播, 让 R 时变
        alpha = rho * a
        beta = rho * b
    return x[:, 0]

V_ekf = ekf(V_meas, dt)
V_vb = vb_akf(V_meas, dt)
print(f"滤波完成: SG/EKF/VB-AKF")

# ============ 数据构建 (替换 Utot 列) ============
def build_data(mode):
    dv = data_values.copy()
    if mode == 'ekf': dv[:, target_idx] = V_ekf.astype(np.float32)
    elif mode == 'vb': dv[:, target_idx] = V_vb.astype(np.float32)
    X_sections, y_sections = [], []
    for lag in LAGS:
        lm = lag/10.0
        X_lag = dv[MAX_LAG-lag : N-lag]
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
    tr_end=int(n_per_lag*0.5); vl_end=int(n_per_lag*0.6)
    Xtr=[];Xvl=[];Xts=[];ytr=[];yvl=[];yts=[]
    for k in range(len(LAGS)):
        st=k*n_per_lag
        Xtr.append(X_all[st:st+tr_end]); Xvl.append(X_all[st+tr_end:st+vl_end]); Xts.append(X_all[st+vl_end:st+n_per_lag])
        ytr.append(y_raw[st:st+tr_end]); yvl.append(y_raw[st+tr_end:st+vl_end]); yts.append(y_raw[st+vl_end:st+n_per_lag])
    return (np.concatenate(Xtr),np.concatenate(Xvl),np.concatenate(Xts),
            np.concatenate(ytr),np.concatenate(yvl),np.concatenate(yts)), X_all.shape[1]

# ============ TFT 模型 (同 test_ekf_loop) ============
class GRN(nn.Module):
    def __init__(self, d_in, d_hid, d_out):
        super().__init__()
        self.fc1=nn.Linear(d_in,d_hid); self.fc2=nn.Linear(d_hid,d_out); self.gate=nn.Linear(d_hid,d_out)
        self.gm=nn.Linear(1,d_out); self.ln=nn.LayerNorm(d_out); self.dp=nn.Dropout(0.1)
        self.proj=nn.Linear(d_in,d_out) if d_in!=d_out else nn.Identity()
    def forward(self,x,dyn=None):
        a=self.proj(x); h=F.elu(self.fc1(x)); g=self.gate(h)
        if dyn is not None: g=g+0.3*torch.tanh(self.gm(dyn.view(-1,1)))
        return self.ln(a+self.dp(torch.sigmoid(g)*self.fc2(h)))
class TFT(nn.Module):
    def __init__(self, nf, hidden=96, dim_ff=128):
        super().__init__()
        self.fw=nn.Parameter(torch.zeros(nf)); self.fb=nn.Parameter(torch.zeros(nf))
        self.lstm=nn.LSTM(nf,hidden,1,batch_first=True)
        self.grn=GRN(hidden,dim_ff,hidden); self.attn=nn.MultiheadAttention(hidden,4,batch_first=True)
        self.ln=nn.LayerNorm(hidden); self.dp=nn.Dropout(0.1); self.fcV=nn.Linear(hidden,1); self.fcdV=nn.Linear(hidden,1)
    def forward(self,x,ret_cell=False):
        dyn=x[:,0,:6].std(dim=1); w=torch.sigmoid(self.fw+self.fb); xw=x*w.unsqueeze(0).unsqueeze(0)
        if self.training: xw=xw+torch.randn_like(xw)*0.01
        o,(h,c)=self.lstm(xw); o=self.dp(o); g=self.grn(o,dyn=dyn); a,_=self.attn(g,g,g); s=self.ln(a+g)[:,-1,:]
        V=self.fcV(s); dV=self.fcdV(s)
        if ret_cell: return V,dV,c
        return V,dV

def run_loop(mode, name):
    (Xtr,Xvl,Xts,ytr,yvl,yts), nf = build_data(mode)
    sX=MinMaxScaler(); sy=MinMaxScaler()
    Xtr_s=sX.fit_transform(Xtr).reshape(-1,1,nf); Xvl_s=sX.transform(Xvl).reshape(-1,1,nf); Xts_s=sX.transform(Xts).reshape(-1,1,nf)
    ytr_s=sy.fit_transform(ytr.reshape(-1,1)).ravel(); yvl_s=sy.transform(yvl.reshape(-1,1)).ravel(); yts_s=sy.transform(yts.reshape(-1,1)).ravel()
    def _dV(y): return np.diff(y, prepend=y[0])
    ytr_dV=torch.tensor(_dV(ytr_s),dtype=torch.float32).view(-1,1); yvl_dV=torch.tensor(_dV(yvl_s),dtype=torch.float32).view(-1,1)
    tr_ld=DataLoader(TensorDataset(torch.tensor(Xtr_s,dtype=torch.float32),torch.tensor(ytr_s,dtype=torch.float32).view(-1,1),ytr_dV),batch_size=64,shuffle=False)
    vl_ld=DataLoader(TensorDataset(torch.tensor(Xvl_s,dtype=torch.float32),torch.tensor(yvl_s,dtype=torch.float32).view(-1,1),yvl_dV),batch_size=64,shuffle=False)
    ts_ld=DataLoader(TensorDataset(torch.tensor(Xts_s,dtype=torch.float32),torch.tensor(yts_s,dtype=torch.float32).view(-1,1)),batch_size=64,shuffle=False)
    r2s=[]
    for li in range(N_LOOPS):
        torch.manual_seed(seed_value+li*10)
        model=TFT(nf).to(device); opt=optim.Adam(model.parameters(),lr=3e-4,weight_decay=1e-5)
        mse=nn.MSELoss(); huber=nn.HuberLoss(delta=0.01); best=float('inf'); pc=0
        for ep in range(300):
            model.train()
            for bx,by,bdV in tr_ld:
                bx,by,bdV=bx.to(device),by.to(device),bdV.to(device); opt.zero_grad()
                Vp,dVp,c=model(bx,ret_cell=True)
                Lm=mse(Vp,by); Ld=huber(dVp,bdV)
                diff=Vp[1:]-Vp[:-1]; Lmo=torch.mean(torch.clamp(diff,min=0)**2)
                Ls=torch.mean((Vp[2:]-2*Vp[1:-1]+Vp[:-2])**2) if len(Vp)>=3 else torch.tensor(0.,device=device)
                h=c[-1].mean(dim=1); Lc=torch.mean(torch.clamp(h[1:]-h[:-1],min=0)**2)
                loss=Lm+0.3*Ld+0.1*Lmo+0.05*Ls+0.05*Lc
                loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),0.5); opt.step()
            model.eval(); vl=0.0
            with torch.no_grad():
                for bx,by,_ in vl_ld: bx=bx.to(device); vl+=mse(model(bx)[0],by.to(device)).item()
            if vl<best: best=vl; pc=0; torch.save(model.state_dict(),'_tmp.pth')
            else:
                pc+=1
                if pc>=40: break
        model.load_state_dict(torch.load('_tmp.pth')); os.remove('_tmp.pth')
        model.eval(); pr=[]; tr=[]
        with torch.no_grad():
            for bx,by in ts_ld: pr.append(model(bx.to(device))[0].cpu().numpy()); tr.append(by.numpy())
        yp=sy.inverse_transform(np.concatenate(pr)).ravel(); yt=sy.inverse_transform(np.concatenate(tr)).ravel()
        r2s.append(r2_score(yt,yp))
    r2s=np.array(r2s)
    print(f"{name}: Min={r2s.min():.4f} Mean={r2s.mean():.4f} Max={r2s.max():.4f}")
    return r2s

print(f"=== VB-AKF 验证 (FC2, Loop {N_LOOPS}) ===")
rA = run_loop('sg',  "A. SG 基线   ")
rC = run_loop('vb',  "C. VB-AKF    ")
print("\n=== 汇总 ===")
print(f"SG:    Min={rA.min():.4f} Mean={rA.mean():.4f} Max={rA.max():.4f}")
print(f"VB-AKF: Min={rC.min():.4f} Mean={rC.mean():.4f} Max={rC.max():.4f}")
print(f"\nVB-AKF vs SG: ΔMean={rC.mean()-rA.mean():+.4f}, ΔMin={rC.min()-rA.min():+.4f}, ΔMax={rC.max()-rA.max():+.4f}")
