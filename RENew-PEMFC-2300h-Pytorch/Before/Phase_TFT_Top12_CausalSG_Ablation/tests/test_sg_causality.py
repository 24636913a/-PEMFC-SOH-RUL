from pathlib import Path
import sys
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from run_d_ablation import one_sided_sg, causal_sg_audit

if __name__ == "__main__":
    x = np.arange(48.0, dtype=np.float32).reshape(1, 48, 1)
    x = np.pad(x, ((0, 0), (0, 0), (0, 13)), constant_values=0)
    original = one_sided_sg(x)
    changed = x.copy(); changed[:, 22:, 0] += 1000.0
    mutated = one_sided_sg(changed)
    assert np.allclose(original[:, :22], mutated[:, :22], atol=1e-7)
    assert causal_sg_audit()["pass"] is True
