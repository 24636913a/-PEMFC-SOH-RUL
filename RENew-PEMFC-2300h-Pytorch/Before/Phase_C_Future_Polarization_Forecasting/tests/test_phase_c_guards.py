"""Post-run invariants for Phase C's data firewall and sparse-label handling."""
from __future__ import annotations
import json
from pathlib import Path
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]

def main() -> None:
    audit = json.loads((ROOT / "RUN_AUDIT.json").read_text(encoding="utf-8"))
    decision = yaml.safe_load((ROOT / "FINAL_DECISION.yaml").read_text(encoding="utf-8"))
    grid = pd.read_csv(ROOT / "outputs" / "current_density_grid.csv")
    pred = pd.read_csv(ROOT / "outputs" / "all_curve_predictions.csv")
    assert audit["new_test_target_accessed"] is False and audit["forbidden_test_stages_opened"] == []
    assert decision["time_alignment_verified"] is True and decision["target_completion_purge"] is True
    assert set(grid.stage) == {"BoL", "300", "600", "850", "1150", "1500", "1750"}
    assert set(grid.j_A_cm2.round(1)) == {0.8, 1.0, 1.2, 1.4, 1.6}
    assert (pred[pred.status.ne("RUN")].prediction_voltage_V.isna()).all()
    assert set(pred[pred.status.ne("RUN")].model) >= {"M2_LSTM", "M3_TimeXer", "M4_Fusion"}
    print("Phase C leakage and sparse-supervision guard checks passed")

if __name__ == "__main__": main()
