from __future__ import annotations
import importlib.util
import sys
from pathlib import Path
import pandas as pd

sys.dont_write_bytecode = True
root = Path(__file__).resolve().parent.parent
source = Path(__file__).resolve().parent / "phase_rs_frozen_audit.py"
spec = importlib.util.spec_from_file_location("phase_rs_figures", source)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
module.make_figures(
    pd.read_csv(root / "outputs/frozen_model/frozen_model_summary.csv"),
    pd.read_csv(root / "outputs/horizon/frozen_model_per_horizon.csv"),
    pd.read_csv(root / "outputs/selector/selector_by_fold.csv"),
)
