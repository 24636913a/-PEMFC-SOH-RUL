"""The smoke mode must exercise the actual Canonical TFT, gradient update, and prediction path."""
from __future__ import annotations
import json
from pathlib import Path
import subprocess
import sys

PHASE = Path(__file__).resolve().parents[1]

if __name__ == "__main__":
    subprocess.run([sys.executable, str(PHASE / "scripts" / "run_c_ablation.py"), "--smoke"], check=True)
    result = json.loads((PHASE / "outputs" / "C_SYNTHETIC_SMOKE_TEST.json").read_text(encoding="utf-8"))
    assert result["pass"] is True
    assert result["optimizer_update_observed"] is True
    assert result["prediction_shape"] == [4, 24]
