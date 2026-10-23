"""Repack completed N2.8E development-only component arrays without pickle."""
from __future__ import annotations

from pathlib import Path

import numpy as np

ROOT = Path(r"E:\沈农信电\ PEMFC-SOH+RUL\RENew-PEMFC-2300h-Pytorch\Phase_N2_8E_Physics_Guided_Fast_Slow_TimeXer")
MODELS = ("FS_TimeXer_NoPhysics", "PGFS_TimeXer_Smooth", "PGFS_TimeXer_SmoothCurrent", "PGFS_TimeXer_Full")
FOLDS = ("fold_1", "fold_2", "fold_3")


def main() -> None:
    model_rows, fold_rows = [], []
    fields = {name: [] for name in ("y_true", "y_pred", "fast_prediction", "slow_component", "origin_age_h", "anchor", "target_end_age_h")}
    for model in MODELS:
        for fold in FOLDS:
            path = ROOT / "outputs" / "model_runs" / f"{model}_{fold}_seed42_predictions.npz"
            item = np.load(path, allow_pickle=False)
            count = len(item["origin_age_h"])
            model_rows.append(np.repeat(model, count))
            fold_rows.append(np.repeat(fold, count))
            for name in fields:
                fields[name].append(item[name])
    output = ROOT / "outputs" / "model_runs" / "fast_slow_component_predictions.npz"
    np.savez_compressed(
        output,
        model=np.concatenate(model_rows), fold=np.concatenate(fold_rows),
        **{name: np.concatenate(values, axis=0) for name, values in fields.items()},
    )
    check = np.load(output, allow_pickle=False)
    print({"safe_npz": True, "rows": int(len(check["model"])), "horizon_steps": int(check["y_true"].shape[1])})


if __name__ == "__main__":
    main()
