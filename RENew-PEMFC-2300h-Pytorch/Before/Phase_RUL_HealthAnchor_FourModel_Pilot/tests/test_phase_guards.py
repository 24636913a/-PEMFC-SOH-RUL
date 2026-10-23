"""Dependency-light checks for the health-anchor pilot's fail-closed artifacts."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    audit = json.loads((ROOT / "RUN_AUDIT.json").read_text(encoding="utf-8"))
    decision = yaml.safe_load((ROOT / "FINAL_DECISION.yaml").read_text(encoding="utf-8"))
    anchors = pd.read_csv(ROOT / "outputs" / "health_anchor_table.csv")
    predictions = pd.read_csv(ROOT / "outputs" / "all_predictions.csv")
    assert audit["source_commit"] == "2950160fdb762511138c1bf4a159c6c898a9d193"
    assert audit["test_target_access_in_this_phase"] is False
    assert decision["new_test_target_accessed"] is False
    assert decision["test_metrics_computed"] is False
    assert set(anchors["stage"]) == {"BoL", "300", "600", "850", "1150", "1500", "1750"}
    assert not anchors["stage"].astype(str).str.contains("2000|2300").any()
    assert set(predictions.loc[predictions["status"] == "NOT_ESTIMABLE", "model"]) >= {"M1_LSTM", "M2_TimeXer", "M3_Fusion"}
    print("phase guard checks passed")


if __name__ == "__main__":
    main()
