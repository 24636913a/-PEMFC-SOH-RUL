# ================================================================
# PEMFC RUL — 电子科技大学5电堆数据集 + TFT
# ================================================================
# 数据: S3-S7, 恒流90A, 1Hz, ~1300-2400h/堆
# 特征: SG滤波电压(1D) + lag增强 → 2维输入
# 模型: TFT (VS+LSTM+GRN+Attention)
# ================================================================
import sys, os
sys.stdout.reconfigure(encoding='utf-8')
import torch, torch.nn as nn, torch.optim as optim
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from pathlib import Path; from datetime import datetime
import pickle, random, h5py
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# 配置
seed_value = 42
random.seed(seed_value); np.random.seed(seed_value)
torch.manual_seed(seed_value)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed_value)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

DATA_DIR = Path(r"E:\沈农信电\课题-预测质子交换膜燃料电池（PEMFC）的健康状态和剩余寿命\数据\基于LSTM的燃料电池剩余使用寿命预测模型的训练、验证和预测数据集-电子科技大学\2018YFB1502501-003-基于LSTM的燃料电池剩余使用寿命预测模型的训练、验证和预测数据集\2018YFB1502501-003-基于LSTM的燃料电池剩余使用寿命预测模型的训练、验证和预测数据集")
OUTPUT_DIR = Path("./outputs")
date_str = datetime.now().strftime('%Y%m%d')
n = 1
while (OUTPUT_DIR / f"{date_str}_{n}_UESTC-TFT").exists(): n += 1
RUN_DIR = OUTPUT_DIR / f"{date_str}_{n}_UESTC-TFT"
RUN_DIR.mkdir(parents=True, exist_ok=True)

STACKS = ['S3','S4','S5','S6','S7']
SAMPLING_H = 1.0; MAX_LAG = 4; LAGS = [1,2,4]

# ================================================================
# 1. 加载
# ================================================================
print("="*60)
print("加载电子科技大学数据")
all_data = {}
for st in STACKS:
    with h5py.File(DATA_DIR / f"data_{st}.mat", 'r') as f:
        d = f[st][:]  # (3, N)
    all_data[st] = pd.DataFrame({'Time_h': d[0,:], 'V_raw': d[1,:], 'V_sg': d[2,:]}
        ).dropna().sort_values('Time_h').reset_index(drop=True)
    print(f"  {st}: {len(all_data[st]):,}行, {all_data[st].Time_h.max():.0f}h")

# ================================================================
# 2. 模型
# ================================================================
class GRN(nn.Module):
    def __init__(self, di, dh, do, dp=0.1):
        super().__init__()
        self.f1 = nn.Linear(di, dh); self.f2 = nn.Linear(dh, do)
        self.g = nn.Linear(dh, do); self.ln = nn.LayerNorm(do)
        self.dp = nn.Dropout(dp)
        self.p = nn.Linear(di, do) if di != do else nn.Identity()
    def forward(self, x):
        a = self.p(x); h = torch.nn.functional.elu(self.f1(x))
        return self.ln(a + self.dp(torch.sigmoid(self.g(h)) * self.f2(h)))

class TFT(nn.Module):
    def __init__(self, ni, nh, nl, no, nhead=2, dff=64, dp=0.1):
        super().__init__()
        if nh % nhead: nhead = 2 if nh%2==0 else 1
        self.fw = nn.Parameter(torch.zeros(ni)); self.fb = nn.Parameter(torch.zeros(ni))
        self.lstm = nn.LSTM(ni, nh, nl, batch_first=True, dropout=dp if nl>1 else 0)
        self.grn = GRN(nh, dff, nh, dp)
        self.attn = nn.MultiheadAttention(nh, nhead, dropout=dp, batch_first=True)
        self.aln = nn.LayerNorm(nh); self.dp = nn.Dropout(dp)
        self.fc = nn.Linear(nh, no)
    def forward(self, x):
        w = torch.sigmoid(self.fw + self.fb)
        x = x * w.unsqueeze(0).unsqueeze(0)
        if self.training: x = x + torch.randn_like(x)*0.01
        o, _ = self.lstm(x); o = self.dp(o); o = self.grn(o)
        a, _ = self.attn(o, o, o); a = self.aln(a + o)
        return self.fc(a[:, -1, :])

# ================================================================
# 3. 预处理
# ================================================================
def prep(df):
    t = df.Time_h.values; idx = df.astype({'Time_h':'int32'}).drop_duplicates('Time_h').index
    s = df.iloc[idx].reset_index(drop=True)
    V = s.V_sg.values.astype(np.float32); N = len(V)

    Xp, yp = [], []
    for la in LAGS:
        Xp.append(np.c_[V[MAX_LAG-la:N-la], np.full(N-MAX_LAG, la/10., dtype=np.float32)])
        yp.append(V[MAX_LAG:])
    X = np.concatenate(Xp, 0); y = np.concatenate(yp, 0)
    nf = X.shape[1]

    npl = N - MAX_LAG; te = int(npl*.5); ve = int(npl*.6)
    Xt, Xv, Xs, yt, yv, ys = [], [], [], [], [], []
    for k in range(len(LAGS)):
        st = k*npl
        Xt.append(X[st:st+te]); Xv.append(X[st+te:st+ve]); Xs.append(X[st+ve:st+npl])
        yt.append(y[st:st+te]); yv.append(y[st+te:st+ve]); ys.append(y[st+ve:st+npl])
    Xtr = np.concatenate(Xt); Xva = np.concatenate(Xv); Xts = np.concatenate(Xs)
    ytr = np.concatenate(yt); yva = np.concatenate(yv); yts = np.concatenate(ys)

    sx = MinMaxScaler((0,1)); sy = MinMaxScaler((0,1))
    return (sx.fit_transform(Xtr).reshape(-1,1,nf), sx.transform(Xva).reshape(-1,1,nf),
            sx.transform(Xts).reshape(-1,1,nf), sy.fit_transform(ytr[:,None]).ravel(),
            sy.transform(yva[:,None]).ravel(), sy.transform(yts[:,None]).ravel(), sy, nf)

# ================================================================
# 4. 训练+评估
# ================================================================
def train_eval(Xtr, ytr, Xva, yva, Xts, yts, sy, nf, config, loops=20):
    from torch.utils.data import DataLoader, TensorDataset
    Xtt = torch.tensor(Xtr, dtype=torch.float32); ytt = torch.tensor(ytr, dtype=torch.float32).view(-1,1)
    Xvt = torch.tensor(Xva, dtype=torch.float32); yvt = torch.tensor(yva, dtype=torch.float32).view(-1,1)
    Xst = torch.tensor(Xts, dtype=torch.float32)

    res = []
    for li in range(loops):
        torch.manual_seed(seed_value + li*10)
        if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed_value + li*10)

        m = TFT(nf, config['hidden'], config['nlayers'], 1, nhead=2, dff=64, dp=config['dropout']).to(device)
        opt = optim.Adam(m.parameters(), lr=config['lr'], weight_decay=1e-5)
        sch = optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=20)
        crit = nn.MSELoss()
        tld = DataLoader(TensorDataset(Xtt, ytt), config['bs'], shuffle=False)
        vld = DataLoader(TensorDataset(Xvt, yvt), config['bs'], shuffle=False)

        best_v = float('inf'); best_s = None; pc = 0
        swa = None
        for ep in range(config['epochs']):
            m.train()
            for bx, by in tld:
                bx, by = bx.to(device), by.to(device)
                opt.zero_grad(); crit(m(bx), by).backward()
                torch.nn.utils.clip_grad_norm_(m.parameters(), 0.5); opt.step()
            m.eval(); vl = 0.0
            with torch.no_grad():
                for bx, by in vld: vl += crit(m(bx.to(device)), by.to(device)).item()
            vl /= len(vld); sch.step(vl)

            if vl < best_v: best_v = vl; pc = 0; best_s = {k:v.clone() for k,v in m.state_dict().items()}
            else: pc += 1
            if pc >= config['patience']: break

            if ep >= 30:
                if swa is None: swa = {k:v.clone().cpu() for k,v in m.state_dict().items()}; sc = 1
                else:
                    for k in swa: swa[k] = swa[k]*(sc/(sc+1)) + m.state_dict()[k].cpu()*(1/(sc+1))
                    sc += 1

        if swa is not None: best_s = {k:v.to(device) for k,v in swa.items()}
        m.load_state_dict(best_s); m.eval()
        with torch.no_grad():
            yp = m(Xst.to(device)).cpu().numpy().ravel()
        yp = sy.inverse_transform(yp[:,None]).ravel()
        yt = sy.inverse_transform(yts[:,None]).ravel()
        res.append((r2_score(yt, yp), np.sqrt(mean_squared_error(yt, yp)), mean_absolute_error(yt, yp)))
    return res

CFG = {'hidden': 64, 'nlayers': 1, 'dropout': 0.1, 'lr': 2e-4, 'bs': 72, 'epochs': 250, 'patience': 50}
N_LOOPS = 20

# ================================================================
# 5. 实验1: 各电堆独立
# ================================================================
print("\n"+"="*60)
print("各电堆独立训练 (TFT, Loop 20)")
all_r = {}
for st in STACKS:
    print(f"\n{st}...", end=' ', flush=True)
    Xtr, Xva, Xts, ytr, yva, yts, sy, nf = prep(all_data[st])
    r = train_eval(Xtr, ytr, Xva, yva, Xts, yts, sy, nf, CFG, N_LOOPS)
    r2s = [x[0] for x in r]
    print(f"Min={min(r2s):.4f} Mean={np.mean(r2s):.4f} Max={max(r2s):.4f}")
    all_r[st] = r

# ================================================================
# 6. 实验2: 跨电堆 (S3→其他)
# ================================================================
print("\n"+"="*60)
print("跨电堆泛化: S3训练 → 其他测试")
Xtr, Xva, _, ytr, yva, _, _, nf = prep(all_data['S3'])
for st in ['S4','S5','S6','S7']:
    _, _, Xts, _, _, yts, sy, _ = prep(all_data[st])
    print(f"  S3→{st}...", end=' ', flush=True)
    r = train_eval(Xtr, ytr, Xva, yva, Xts, yts, sy, nf, CFG, N_LOOPS)
    r2s = [x[0] for x in r]
    print(f"Min={min(r2s):.4f} Mean={np.mean(r2s):.4f} Max={max(r2s):.4f}")

# ================================================================
# 7. 汇总 & 退化图
# ================================================================
print("\n"+"="*60)
print("汇总")
for st in STACKS:
    r2s = [x[0] for x in all_r[st]]
    print(f"  {st}: Min={min(r2s):.4f} Mean={np.mean(r2s):.4f} Max={max(r2s):.4f}")

fig, axes = plt.subplots(2, 3, figsize=(16, 8))
for i, st in enumerate(STACKS):
    ax = axes.flat[i]; df = all_data[st]
    ax.plot(df.Time_h, df.V_sg, linewidth=0.3)
    ax.set_title(f'{st} ({df.Time_h.max()-df.Time_h.min():.0f}h)'); ax.grid(alpha=0.3)
axes.flat[-1].axis('off')
plt.suptitle('5电堆退化轨迹 (SG滤波电压)', fontweight='bold')
plt.tight_layout(); plt.savefig(RUN_DIR / 'trajectories.png', dpi=150); plt.show()

with open(RUN_DIR / 'results.pkl', 'wb') as f: pickle.dump(all_r, f)
print(f"\nDone → {RUN_DIR}")
