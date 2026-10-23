# %% 唯一交互入口：默认只读，运行全部单元不会自动训练或使用GPU
"""VS Code Python Interactive entry for the frozen RealHyFC F2 study.

Change MODE deliberately to LOAD_MODEL, TRAIN_NEW or VERIFY. New training also
requires an explicit RUN_ID and CONFIRM_NEW_TRAINING=True. Completed runs are
reused, partial/failed runs are never overwritten. All paths follow __file__.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

sys.dont_write_bytecode = True

MODE = "READ_ONLY"              # READ_ONLY / LOAD_MODEL / TRAIN_NEW / VERIFY
RUN_ID = None                   # Explicitly choose A1, A2, A3, A4, B1 or B2
CONFIRM_NEW_TRAINING = False     # Second opt-in; leave False for analysis
SHOW_COVERAGE_FIGURE = False     # True displays only the already saved PNG
PREDICT_ONE_VALIDATION_ORIGIN = False
VALIDATION_ORIGIN_INDEX = 0      # Fixed index, never picked by prediction error
VERIFY_ARGS = ["--stage", "final"]  # Uses saved SHA-matched GPU proof; ["--help"] lists options

STUDY = Path(__file__).resolve().parent
SCRIPTS = STUDY / "scripts"
OUT = STUDY / "outputs"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))
if MODE not in {"READ_ONLY", "LOAD_MODEL", "TRAIN_NEW", "VERIFY"}:
    raise ValueError("Unknown MODE")
protocol_path = OUT / "frozen_protocol.json"
if not protocol_path.exists():
    raise FileNotFoundError("Frozen protocol not yet available; no training is attempted.")
CONFIG = json.loads(protocol_path.read_text(encoding="utf-8"))
RUN_SPECS = {item["id"]: item for item in CONFIG["runs"]}


def run_directory(run_id):
    if run_id not in RUN_SPECS:
        raise ValueError("Explicitly select RUN_ID from " + ", ".join(RUN_SPECS))
    item = RUN_SPECS[run_id]
    return STUDY / f"runs/{run_id}_{item['model']}_H{item['horizon']}"


# %% 数据、冻结采样与运行状态（CPU，只读）
import numpy as np
import pandas as pd

print("MODE:", MODE, "| Seed:", CONFIG["seed"], "| No automatic training")
print("History:", CONFIG["history_steps"], "records; raw information support:", CONFIG["history_information_support_rows"], "records")
print("Horizons:", CONFIG["horizons_seconds"], "s | rolling-origin grid:", CONFIG["sampling"]["candidate_grid_seconds"], "s")
selected_origins = pd.read_csv(OUT / "selected_train_origins.csv")
print("Frozen TRAIN origins:", len(selected_origins))
print(selected_origins.head().to_string(index=False))
print(selected_origins.selection.value_counts().to_string())
run_overview = []
for item in CONFIG["runs"]:
    directory = run_directory(item["id"])
    state_file = directory / "training_state.json"
    state = json.loads(state_file.read_text(encoding="utf-8")) if state_file.exists() else {}
    status = state.get("status", "NOT_STARTED")
    acceptance_file = OUT / "B1_acceptance.json"
    if item["id"] == "B2" and not state and acceptance_file.exists():
        acceptance = json.loads(acceptance_file.read_text(encoding="utf-8"))
        if acceptance.get("pass") and not acceptance.get("B2_allowed"):
            status = "CANCELED_BY_FROZEN_GATE"
    run_overview.append(dict(id=item["id"], model=item["model"], horizon=item["horizon"],
                             status=status,
                             optimizer_steps=state.get("optimizer_steps", 0),
                             best_epoch=state.get("best_epoch")))
print(pd.DataFrame(run_overview).to_string(index=False))

# %% 已保存的覆盖图与报告（可选显示；不生成新结果）
coverage_figure = STUDY / "figures/10_training_coverage.png"
coverage_report = STUDY / "01_训练样本及突变覆盖分析.md"
print("Coverage report:", coverage_report)
print("Saved figure:", coverage_figure)
if SHOW_COVERAGE_FIGURE:
    if coverage_figure.exists():
        import matplotlib.pyplot as plt
        image = plt.imread(coverage_figure)
        plt.figure(figsize=(13.2, 9.1))
        plt.imshow(image)
        plt.axis("off")
        plt.tight_layout()
        plt.show()
    else:
        print("Coverage figure has not been generated; nothing was recomputed.")

# %% 显式加载已完成模型：CPU默认，绝不调用训练器
LOADED_MODEL = None
LOADED_CHECKPOINT = None
if MODE == "LOAD_MODEL":
    import torch
    from core import assert_integrity, load_cache, model_factory, sha

    directory = run_directory(RUN_ID)
    state_path = directory / "training_state.json"
    if not state_path.exists():
        raise FileNotFoundError("Selected model has not completed training.")
    state = json.loads(state_path.read_text(encoding="utf-8"))
    if state.get("status") != "COMPLETE":
        raise RuntimeError("Only a completed immutable model may be loaded here.")
    assert_integrity(CONFIG)
    if state["protocol_sha256"] != sha(protocol_path) or state["checkpoint_sha256"] != sha(directory / "best.pt"):
        raise RuntimeError("Protocol or checkpoint identity mismatch.")
    item = RUN_SPECS[RUN_ID]
    LOADED_CHECKPOINT = torch.load(directory / "best.pt", map_location="cpu", weights_only=True)
    LOADED_MODEL = model_factory(item["model"], item["horizon"]).cpu()
    LOADED_MODEL.load_state_dict(LOADED_CHECKPOINT["state_dict"])
    LOADED_MODEL.eval()
    print("Loaded completed model on CPU:", directory)
    if PREDICT_ONE_VALIDATION_ORIGIN:
        _, arrays = load_cache()  # historical memmaps are strictly mode=r
        horizon = item["horizon"]
        values = np.load(OUT / f"origins_H{horizon}.npz")["VALIDATION"]
        if not 0 <= VALIDATION_ORIGIN_INDEX < len(values):
            raise IndexError("Validation origin index out of range")
        origin = int(values[VALIDATION_ORIGIN_INDEX])
        mean = np.asarray(CONFIG["scaler"]["mean"], dtype=np.float32)
        scale = np.asarray(CONFIG["scaler"]["scale"], dtype=np.float32)
        history = np.ascontiguousarray((arrays["x"][origin - 299:origin + 1] - mean) / scale)
        with torch.inference_mode():
            delta = LOADED_MODEL(torch.from_numpy(history[None])).cpu().numpy()[0] * CONFIG["training"]["target_scale_V"]
        targets = origin + np.arange(1, horizon + 1)
        prediction = pd.DataFrame(dict(target_DateTime=pd.to_datetime(arrays["time_ms"][targets], unit="ms"),
                                       lead_seconds=np.arange(1, horizon + 1),
                                       prediction_delta_V=delta,
                                       prediction_voltage_V=delta + arrays["voltage_V"][origin],
                                       actual_voltage_V=arrays["voltage_V"][targets]))
        print(prediction.to_string(index=False))

# %% 新训练双重选择：只运行一个明确RUN_ID；已完成模型只复用
if MODE == "TRAIN_NEW":
    directory = run_directory(RUN_ID)
    if CONFIRM_NEW_TRAINING is not True:
        raise RuntimeError("New training disabled: explicitly set CONFIRM_NEW_TRAINING=True.")
    state_file = directory / "training_state.json"
    if directory.exists():
        state = json.loads(state_file.read_text(encoding="utf-8")) if state_file.exists() else {}
        if state.get("status") == "COMPLETE":
            print("REUSE_COMPLETE_NO_TRAINING:", directory, "| Use MODE='LOAD_MODEL' for CPU loading.")
        else:
            raise RuntimeError("Existing partial/failed/unknown directory requires investigation; it will not be retrained or overwritten.")
    else:
        states = list((STUDY / "runs").glob("*/training_state.json"))
        if len(states) >= CONFIG["max_trained_instances"]:
            raise RuntimeError("Frozen maximum number of model instances reached.")
        for other_file in states:
            other = json.loads(other_file.read_text(encoding="utf-8"))
            if other.get("status") not in {"COMPLETE"}:
                raise RuntimeError("Another run is incomplete/failed/running; serial training requires investigation first.")
        if RUN_ID == "B2":
            acceptance = OUT / "B1_acceptance.json"
            if not acceptance.exists():
                raise RuntimeError("B2 requires the independent B1 acceptance record.")
            gate = json.loads(acceptance.read_text(encoding="utf-8"))
            if not gate.get("pass") or not gate.get("B2_allowed"):
                raise RuntimeError("B1 did not satisfy the frozen B2 gate.")
        subprocess.run([sys.executable, "-B", str(SCRIPTS / "train.py"), "--run", RUN_ID], cwd=STUDY, check=True)

# %% 显式独立核验：默认不运行，也不训练
if MODE == "VERIFY":
    verifier = SCRIPTS / "verify_optimization.py"
    if verifier.exists():
        subprocess.run([sys.executable, "-B", str(verifier), *VERIFY_ARGS], cwd=STUDY, check=True)
    else:
        print("Independent verifier has not yet been generated:", verifier)

# %% 无修改式结果检查：已经存在的全样本指标和报告路径
for filename in ("all_metrics.csv", "validation_metrics.csv"):
    metric_file = OUT / filename
    if metric_file.exists() and metric_file.stat().st_size > 1:
        print("Saved metrics:", metric_file)
        print(pd.read_csv(metric_file).head(20).to_string(index=False))
for path in sorted(STUDY.glob("0[0-5]_*.md")):
    print("Report:", path)
