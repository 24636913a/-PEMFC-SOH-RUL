"""Generate the explicitly synthetic teaching data, kept separate from inference."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from model import recovery_age


def generate(config: dict, data_dir: Path) -> None:
    cfg = config["synthetic"]
    rng = np.random.default_rng(cfg["seed"])
    t = np.arange(0, cfg["end_time_h"] + 0.5 * cfg["step_h"], cfg["step_h"], dtype=float)
    restart = np.isin(t, cfg["restart_times_h"]).astype(int)
    age = recovery_age(t, restart)
    latent = cfg["initial_cell_voltage_v"] - cfg["long_term_decay_v_per_h"] * t
    q = cfg["reversible_loss_max_v"] * (1 - np.exp(-age / cfg["reversible_time_constant_h"]))
    # Constant reference conditions in this demo isolate the recovery problem.
    delta = np.zeros(len(t), dtype=float)
    noise = rng.normal(0, cfg["observation_noise_sd_v"], len(t))
    raw = latent + delta - q + noise
    observations = pd.DataFrame({
        "time_h": t, "stack_voltage_v": cfg["n_cells"] * raw,
        "n_cells": cfg["n_cells"], "restart": restart,
        "condition_offset_v": delta, "is_synthetic": True,
        "scenario": "FC1_like_synthetic",
    })
    truth = pd.DataFrame({
        "time_h": t, "latent_voltage_v": latent, "q_rev_v": q,
        "condition_offset_v": delta, "measurement_noise_v": noise,
        "noise_free_operating_voltage_v": latent + delta - q,
        "is_synthetic": True,
    })
    data_dir.mkdir(parents=True, exist_ok=True)
    observations.to_csv(data_dir / "observations.csv", index=False, float_format="%.12g")
    truth.to_csv(data_dir / "truth_for_evaluation_only.csv", index=False, float_format="%.12g")
    pd.DataFrame({"restart_time_h": cfg["restart_times_h"],
                  "event_type": "synthetic_full_recovery_restart"}).to_csv(
                      data_dir / "synthetic_events.csv", index=False)


if __name__ == "__main__":
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=root / "config.json")
    parser.add_argument("--data-dir", type=Path, default=root / "data")
    args = parser.parse_args()
    generate(json.loads(args.config.read_text(encoding="utf-8")), args.data_dir)
    print(f"Synthetic data saved to {args.data_dir}")
