"""Streaming query-weighted actual-voltage statistics, float64 Welford merge."""
import numpy as np


class Metrics:
    def __init__(self, horizon):
        self.n = 0; self.mean = 0.0; self.m2 = 0.0
        self.sse = 0.0; self.sae = 0.0
        self.h_sse = np.zeros(horizon, dtype=np.float64)
        self.origins = 0

    def update(self, truth, prediction):
        y = np.asarray(truth, dtype=np.float64)
        p = np.asarray(prediction, dtype=np.float64)
        if y.shape != p.shape or not np.isfinite(y).all() or not np.isfinite(p).all():
            raise RuntimeError('Nonfinite or misaligned predictions; do not silently exclude queries')
        e = p-y; size = y.size
        mean = float(y.mean()); delta = mean-self.mean
        self.m2 += float(np.sum((y-mean)**2)) + delta*delta*self.n*size/(self.n+size)
        self.mean += delta*size/(self.n+size); self.n += size
        self.sse += float(np.sum(e*e)); self.sae += float(np.sum(np.abs(e)))
        self.h_sse += (e*e).sum(0); self.origins += len(y)

    def result(self):
        if not self.n:
            raise RuntimeError('Empty evaluation')
        return dict(RMSE_mV=float(np.sqrt(self.sse/self.n)*1000), MAE_mV=self.sae/self.n*1000,
                    R2=1-self.sse/self.m2 if self.m2 else None, origins=self.origins,
                    queries=self.n, target='actual mean10 voltage V', stride=1,
                    weighting='all legal origins x all 120 steps; overlap counted',
                    independent_physical_trials=False)

    def horizon_rmse(self):
        return np.sqrt(self.h_sse/self.origins)*1000
