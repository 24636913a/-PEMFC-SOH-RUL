"""Development-only chronology-safe transfer from frozen DeltaV TimesNet encoders.

The runner creates outputs only within Phase CSTL.  It reads the frozen old
phases as sources, verifies original preprocessing using old prediction caches,
then freezes every encoder and fits deterministic chronological Ridge probes.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler


sys.dont_write_bytecode = True
PHASE = Path(__file__).resolve().parents[1]
PROJECT = PHASE.parent
REPO = PROJECT.parent
OUT = PHASE / "outputs"
EMBED = OUT / "embeddings"
FIG = PHASE / "figures"
REPORT = PHASE / "reports"
CONFIG = PHASE / "configs" / "CSTL_FROZEN_PROTOCOL.yaml"
BS = PROJECT / "Phase_BS_Final_Backbone_Screening"
BS_PROTOCOL = BS / "configs" / "PREREGISTERED_PROTOCOL.yaml"
BS_PROTOCOL_SHA = BS / "configs" / "PREREGISTERED_PROTOCOL.sha256"
TSHE_EVENTS = PROJECT / "Phase_TSHE_TimesNet_P1S_B2_Health_State_Estimation" / "outputs" / "HEALTH_EVENT_DATASET.csv"
ANCHORS = PROJECT / "Phase_CNP1S_Condition_Normalized_P1S_Development_Study" / "outputs" / "CN_P1S_ANCHOR_COMPARISON.csv"
AHC_AFFINE = PROJECT / "Phase_AHC_Sparse_Anchor_Affine_Health_Calibration" / "scripts" / "run_affine_characterization.py"
CAUSAL = PROJECT / "Phase_N1_Data_Foundation" / "outputs" / "causal_feature_tables" / "causal_features_30min.csv"
BS_SCRIPTS = BS / "scripts"

PALETTE = {"blue_main": "#0F4D92", "blue_secondary": "#3775BA", "red_strong": "#B64342", "teal": "#42949E", "violet": "#9A4D8E", "neutral": "#CFCECE"}
FEATURES = [
    "cell_voltage_mean", "cell_voltage_std", "cell_voltage_range", "cell_voltage_min", "cell_voltage_max",
    "actual_current_A_mean", "actual_current_A_std", "actual_current_A_peak_to_peak",
    "actual_current_density_A_cm2_mean", "actual_current_density_A_cm2_std",
    "actual_current_density_A_cm2_peak_to_peak", "load_transition_count", "operation_point_transitions",
    "zero_current_fraction",
]
FOLDS = (
    ("fold_1", 72.788690, 946.393214, 946.393214, 1164.794345),
    ("fold_2", 72.788690, 1164.794345, 1164.794345, 1383.195476),
    ("fold_3", 72.788690, 1383.195476, 1383.195476, 1819.997738),
)
SEEDS = (42, 2024, 3407)
REPRODUCTION_TOLERANCE_MV = 1.0


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, encoding="utf-8-sig", lineterminator="\n")


def write_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def markdown_table(frame: pd.DataFrame, decimals: int = 6) -> str:
    if frame.empty:
        return "_No rows._"
    copied = frame.copy()
    for column in copied.columns:
        if pd.api.types.is_float_dtype(copied[column]):
            copied[column] = copied[column].map(lambda value: "" if pd.isna(value) else f"{value:.{decimals}f}")
    return "| " + " | ".join(copied.columns) + " |\n|" + "|".join(["---"] * len(copied.columns)) + "|\n" + "\n".join(
        "| " + " | ".join(map(str, row)) + " |" for row in copied.itertuples(index=False, name=None)
    )


def load_authorities() -> tuple[dict[str, Any], dict[str, Any], Any, Any]:
    expected_head = "a85c07e5bb2b09b9d1ea58f3fa8c844508365162"
    current_head = subprocess.check_output(["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True).strip()
    if current_head != expected_head:
        raise RuntimeError(f"CSTL base HEAD mismatch: {current_head}")
    cfg = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    if cfg.get("base_commit") != expected_head or cfg.get("test_access") != "prohibited":
        raise RuntimeError("CSTL frozen protocol mismatch")
    if list(cfg["input"]["feature_order"]) != FEATURES:
        raise RuntimeError("CSTL Raw F2 feature-order mismatch")
    raw_protocol = BS_PROTOCOL.read_bytes()
    if hashlib.sha256(raw_protocol).hexdigest() != BS_PROTOCOL_SHA.read_text(encoding="utf-8").split()[0].lower():
        raise RuntimeError("BS frozen protocol hash mismatch")
    bs_cfg = yaml.safe_load(raw_protocol)
    timesnet = bs_cfg["official_sources"]["TimesNet"]["config"]
    expected_body = {"seq_len": 48, "pred_len": 24, "enc_in": 14, "c_out": 14, "d_model": 32,
                     "d_ff": 64, "e_layers": 1, "top_k": 2, "num_kernels": 3, "dropout": 0.1}
    if {name: timesnet.get(name) for name in expected_body} != expected_body:
        raise RuntimeError("Frozen TimesNet body configuration mismatch")
    for source in (TSHE_EVENTS, ANCHORS, AHC_AFFINE, CAUSAL, BS_SCRIPTS / "backbone_common.py"):
        if not source.is_file():
            raise FileNotFoundError(source)
    if str(BS_SCRIPTS) not in sys.path:
        sys.path.insert(0, str(BS_SCRIPTS))
    import backbone_common as common
    spec = importlib.util.spec_from_file_location("ahc_affine_frozen", AHC_AFFINE)
    if spec is None or spec.loader is None:
        raise RuntimeError("Cannot load frozen AHC affine function")
    ahc = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ahc)
    return cfg, bs_cfg, common, ahc.affine_parameters


def compute_metrics(truth: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    true = np.asarray(truth, float).reshape(-1)
    pred = np.asarray(prediction, float).reshape(-1)
    if len(true) != len(pred) or not len(true) or not np.isfinite(pred).all():
        raise RuntimeError("Invalid health metric input")
    error = pred - true
    denominator = float(np.sum((true - true.mean()) ** 2))
    spearman = pd.Series(true).corr(pd.Series(pred), method="spearman")
    return {
        "MAE": float(np.mean(abs(error))), "RMSE": float(np.sqrt(np.mean(error ** 2))),
        "R2": float(1 - np.sum(error ** 2) / denominator) if denominator > 0 else math.nan,
        "Spearman": float(spearman) if pd.notna(spearman) else math.nan,
        "MAE_percentage_points": float(np.mean(abs(error)) * 100),
        "RMSE_percentage_points": float(np.sqrt(np.mean(error ** 2)) * 100),
    }


def fit_linear_predict(x_train: np.ndarray, y_train: np.ndarray, x_eval: np.ndarray) -> np.ndarray:
    design = np.column_stack([np.asarray(x_train, float), np.ones(len(x_train))])
    coefficients, *_ = np.linalg.lstsq(design, np.asarray(y_train, float), rcond=None)
    return (coefficients[0] * np.asarray(x_eval, float) + coefficients[1]).astype(float)


def load_and_validate_events() -> tuple[pd.DataFrame, np.ndarray]:
    events = pd.read_csv(TSHE_EVENTS)
    required = {"cycle_interval_id", "raw_cycle_id", "measurement_end_ophrs2_h", "input_start_ophrs2_h",
                "input_end_ophrs2_h", "target_cycle_start_ophrs2_h", "target_P1S_retention", "window_legal",
                "target_cycle_isolated", "input_start_causal_row", "input_end_causal_row"}
    if required - set(events.columns):
        raise RuntimeError("TSHE health-event table schema mismatch")
    legal = events.loc[events["window_legal"].astype(bool)].copy().sort_values("measurement_end_ophrs2_h").reset_index(drop=True)
    if len(legal) != 105 or not legal["target_cycle_isolated"].astype(bool).all():
        raise RuntimeError("TSHE legal health-event authority mismatch")
    if not (legal["input_end_ophrs2_h"] < legal["target_cycle_start_ophrs2_h"]).all():
        raise RuntimeError("Target-cycle isolation audit failed")
    input_columns = ["bin_start_ophrs2_h", "bin_end_ophrs2_h", "bin_valid_for_windows", "temporal_segment_id", *FEATURES]
    causal = pd.read_csv(CAUSAL, usecols=input_columns, nrows=3007).reset_index(names="causal_row")
    causal = causal.loc[causal.bin_end_ophrs2_h.to_numpy(float) < 1819.997738].reset_index(drop=True)
    if len(causal) != 3005:
        raise RuntimeError("CSTL Development prefix firewall mismatch")
    windows: list[np.ndarray] = []
    for event in legal.itertuples(index=False):
        start, end = int(event.input_start_causal_row), int(event.input_end_causal_row)
        if end >= len(causal) or start < 0 or end - start + 1 != 48:
            raise RuntimeError("Recorded TSHE health window is outside Development prefix")
        window = causal.iloc[start:end + 1]
        if not np.allclose(window.bin_start_ophrs2_h.to_numpy(float)[0], float(event.input_start_ophrs2_h), atol=1e-9):
            raise RuntimeError("Health-window start replay mismatch")
        if not np.allclose(window.bin_end_ophrs2_h.to_numpy(float)[-1], float(event.input_end_ophrs2_h), atol=1e-9):
            raise RuntimeError("Health-window end replay mismatch")
        if not window.bin_valid_for_windows.astype(bool).all() or window.temporal_segment_id.nunique() != 1:
            raise RuntimeError("Health-window legality changed during CSTL replay")
        if not np.allclose(np.diff(window.bin_start_ophrs2_h.to_numpy(float)), .5, atol=1e-6, rtol=0):
            raise RuntimeError("Health-window continuity replay mismatch")
        values = window[FEATURES].to_numpy(dtype=np.float32)
        if not np.isfinite(values).all():
            raise RuntimeError("Non-finite Raw F2 history")
        windows.append(values)
    return legal, np.stack(windows).astype(np.float32)


def calibrations(affine_parameters: Any) -> tuple[pd.DataFrame, dict[str, tuple[float, float]]]:
    anchor = pd.read_csv(ANCHORS, usecols=["checkpoint", "Y", "R_raw"])
    anchor["checkpoint"] = anchor["checkpoint"].astype(int)
    expected = [300, 600, 850, 1150, 1500, 1750]
    if anchor.checkpoint.tolist() != expected or not np.isfinite(anchor[["Y", "R_raw"]].to_numpy(float)).all():
        raise RuntimeError("CNP1S health-anchor authority mismatch")
    availability = {"fold_1": [300, 600, 850], "fold_2": [300, 600, 850, 1150], "fold_3": [300, 600, 850, 1150]}
    records: list[dict[str, Any]] = []
    parameters: dict[str, tuple[float, float]] = {}
    for fold, checkpoints in availability.items():
        subset = anchor.loc[anchor.checkpoint.isin(checkpoints)].copy()
        a, b = affine_parameters(subset.R_raw.to_numpy(float), subset.Y.to_numpy(float))
        prediction = a * subset.R_raw.to_numpy(float) + b
        stats = compute_metrics(subset.Y.to_numpy(float), prediction)
        records.append({"fold": fold, "analysis_role": "TARGET_AUTHORITY", "train_end_h": next(item[2] for item in FOLDS if item[0] == fold),
                        "anchor_count": len(subset), "anchor_checkpoints": " / ".join(map(str, checkpoints)),
                        "a_fold": a, "b_fold": b, "anchor_MAE": stats["MAE"], "anchor_RMSE": stats["RMSE"],
                        "anchor_Spearman": stats["Spearman"], "future_anchor_used": False})
        parameters[fold] = (float(a), float(b))
    a_global, b_global = affine_parameters(anchor.R_raw.to_numpy(float), anchor.Y.to_numpy(float))
    global_stats = compute_metrics(anchor.Y.to_numpy(float), a_global * anchor.R_raw.to_numpy(float) + b_global)
    records.append({"fold": "global_six_anchor_reference", "analysis_role": "DESCRIPTIVE_REFERENCE_ONLY", "train_end_h": math.nan,
                    "anchor_count": 6, "anchor_checkpoints": " / ".join(map(str, expected)), "a_fold": a_global, "b_fold": b_global,
                    "anchor_MAE": global_stats["MAE"], "anchor_RMSE": global_stats["RMSE"], "anchor_Spearman": global_stats["Spearman"],
                    "future_anchor_used": "NOT_APPLICABLE_DESCRIPTIVE_REFERENCE"})
    return pd.DataFrame(records), parameters


def fold_expanded_events(legal: pd.DataFrame, parameters: dict[str, tuple[float, float]]) -> tuple[pd.DataFrame, dict[str, tuple[np.ndarray, np.ndarray]]]:
    rows: list[dict[str, Any]] = []
    indices: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    age = legal.measurement_end_ophrs2_h.to_numpy(float)
    for fold, train_start, train_end, val_start, val_end in FOLDS:
        train = np.flatnonzero((age >= train_start) & (age < train_end))
        valid = np.flatnonzero((age >= val_start) & (age < val_end))
        if not len(train) or not len(valid) or set(train) & set(valid):
            raise RuntimeError(f"{fold}: invalid chronology-safe health population")
        if np.max(age[valid]) >= 1819.997738:
            raise RuntimeError(f"{fold}: Test boundary crossed")
        a, b = parameters[fold]
        for role, indexes in (("train", train), ("validation", valid)):
            for index in indexes:
                source = legal.iloc[int(index)]
                if not bool(source.target_cycle_isolated) or not bool(source.window_legal):
                    raise RuntimeError("Fold-expanded event lost TSHE isolation/legality")
                rows.append({"fold": fold, "role": role, "cycle_interval_id": source.cycle_interval_id,
                             "raw_cycle_id": int(source.raw_cycle_id), "measurement_end_ophrs2_h": float(source.measurement_end_ophrs2_h),
                             "input_start_ophrs2_h": float(source.input_start_ophrs2_h), "input_end_ophrs2_h": float(source.input_end_ophrs2_h),
                             "target_cycle_start_ophrs2_h": float(source.target_cycle_start_ophrs2_h),
                             "target_P1S_retention": float(source.target_P1S_retention), "B2_a": a, "B2_b": b,
                             "SOH_proxy_fold": float(a * source.target_P1S_retention + b),
                             "target_cycle_isolated": True, "window_legal": True})
        indices[fold] = (train, valid)
    output = pd.DataFrame(rows).sort_values(["fold", "role", "measurement_end_ophrs2_h"]).reset_index(drop=True)
    if not (output.input_end_ophrs2_h < output.target_cycle_start_ophrs2_h).all():
        raise RuntimeError("Chronology-safe expanded events contain target-cycle leakage")
    return output, indices


def checkpoint_path(fold: str, seed: int) -> Path:
    return BS / "checkpoints" / f"bs_timesnet_{fold}_seed{seed}.pt"


def prediction_path(fold: str, seed: int) -> Path:
    return BS / "outputs" / "predictions" / f"bs_timesnet_{fold}_seed{seed}.npz"


def checkpoint_scaler(common: Any, checkpoint: dict[str, Any]) -> Any:
    data = checkpoint.get("preprocessor", {})
    expected = {"outer_train_feature_medians", "outer_train_scaler_mean", "outer_train_scaler_scale"}
    if expected - set(data):
        raise RuntimeError("Checkpoint does not contain the original outer-train scaler")
    scaler = common.FoldPreprocessor()
    scaler.medians = np.asarray(data["outer_train_feature_medians"], dtype=float)
    scaler.mean = np.asarray(data["outer_train_scaler_mean"], dtype=float)
    scaler.scale = np.asarray(data["outer_train_scaler_scale"], dtype=float)
    if scaler.medians.shape != (14,) or scaler.mean.shape != (14,) or scaler.scale.shape != (14,) or np.any(scaler.scale <= 0):
        raise RuntimeError("Checkpoint scaler dimensionality/value mismatch")
    return scaler


def freeze_model(common: Any, bs_cfg: dict[str, Any], checkpoint: dict[str, Any]) -> Any:
    model = common.create_model("TimesNet", bs_cfg)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad = False
    if any(parameter.requires_grad for parameter in model.parameters()):
        raise RuntimeError("Encoder freeze contract failed")
    return model


def original_windows(common: Any, bs_cfg: dict[str, Any]) -> dict[str, Any]:
    frame, audit = common.read_development_prefix(bs_cfg)
    if audit.get("test_features_read") != 0 or len(frame) != 3005:
        raise RuntimeError("Original BS Development prefix reconstruction failed")
    return {fold: validation for fold, _train, validation in common.build_outer_folds(frame, bs_cfg)}


def pre_projection_latent(adapter: Any, x_scaled: np.ndarray) -> np.ndarray:
    """Replay the official forecast body through layer_norm, immediately before projection."""
    official = adapter.official_model
    with torch.no_grad():
        x = torch.from_numpy(np.asarray(x_scaled, dtype=np.float32))
        means = x.mean(1, keepdim=True).detach()
        normalized = x.sub(means)
        stdev = torch.sqrt(torch.var(normalized, dim=1, keepdim=True, unbiased=False) + 1e-5)
        encoded = normalized.div(stdev)
        encoded = official.enc_embedding(encoded, None)
        encoded = official.predict_linear(encoded.permute(0, 2, 1)).permute(0, 2, 1)
        for block in official.model:
            encoded = official.layer_norm(block(encoded))
    latent = encoded.detach().cpu().numpy().astype(np.float32)
    if latent.ndim != 3 or latent.shape[1:] != (72, 32) or not np.isfinite(latent).all():
        raise RuntimeError(f"Unexpected pre-projection latent shape: {latent.shape}")
    return latent


def reproduce_preprocessing(common: Any, bs_cfg: dict[str, Any]) -> pd.DataFrame:
    originals = original_windows(common, bs_cfg)
    records: list[dict[str, Any]] = []
    for fold in ("fold_1", "fold_2", "fold_3"):
        validation = originals[fold]
        lookup = {int(origin): position for position, origin in enumerate(validation.origin_row)}
        for seed in SEEDS:
            cpath, ppath = checkpoint_path(fold, seed), prediction_path(fold, seed)
            if not cpath.is_file() or not ppath.is_file():
                raise FileNotFoundError(f"Missing frozen checkpoint/prediction: {fold}, seed {seed}")
            checkpoint = torch.load(cpath, map_location="cpu", weights_only=False)
            scaler = checkpoint_scaler(common, checkpoint)
            with np.load(ppath, allow_pickle=False) as saved:
                origins, expected = saved["origin_row"][:3].astype(int), saved["y_pred_delta_V"][:3].astype(float)
            if any(int(origin) not in lookup for origin in origins):
                raise RuntimeError("Stored BS prediction origin not found in reconstructed fold")
            positions = np.asarray([lookup[int(origin)] for origin in origins], dtype=int)
            model = freeze_model(common, bs_cfg, checkpoint)
            with torch.no_grad():
                predicted_scaled, _ = common.model_delta_prediction(model, torch.from_numpy(scaler.transform(validation.x[positions])))
            predicted = predicted_scaled.detach().cpu().numpy() * scaler.voltage_scale
            error_mV = np.abs(predicted - expected) * 1000.0
            maximum = float(error_mV.max())
            records.append({"fold": fold, "seed": seed, "checkpoint": str(cpath.relative_to(REPO)),
                            "sampled_origin_rows": "|".join(map(str, origins)), "sample_count": len(origins),
                            "reconstructed_preprocessor": "checkpoint.outer_train_feature_medians_mean_scale",
                            "max_abs_prediction_difference_mV": maximum, "mean_abs_prediction_difference_mV": float(error_mV.mean()),
                            "tolerance_mV": REPRODUCTION_TOLERANCE_MV, "pass": maximum <= REPRODUCTION_TOLERANCE_MV})
            del model
    result = pd.DataFrame(records)
    if not result["pass"].all():
        raise RuntimeError("PRETRAINED_PREPROCESSING_NOT_REPRODUCED")
    return result


def embeddings_for_fold_seed(common: Any, bs_cfg: dict[str, Any], fold: str, seed: int, x: np.ndarray) -> np.ndarray:
    checkpoint = torch.load(checkpoint_path(fold, seed), map_location="cpu", weights_only=False)
    scaler = checkpoint_scaler(common, checkpoint)
    adapter = freeze_model(common, bs_cfg, checkpoint)
    latent = pre_projection_latent(adapter, scaler.transform(x))
    vector = latent.mean(axis=1)
    if vector.shape != (len(x), 32) or not np.isfinite(vector).all():
        raise RuntimeError("Invalid temporal-mean pretrained embedding")
    del adapter
    return vector.astype(np.float64)


def select_ridge_alpha(embedding: np.ndarray, target: np.ndarray, train_idx: np.ndarray) -> tuple[float, pd.DataFrame]:
    stopper_count = int(math.ceil(len(train_idx) * .15))
    fit_idx, stop_idx = train_idx[:-stopper_count], train_idx[-stopper_count:]
    if len(fit_idx) < 2 or len(stop_idx) < 1 or not np.all(fit_idx < stop_idx.min()):
        raise RuntimeError("Cannot form chronological 85/15 Ridge stopper")
    records: list[dict[str, Any]] = []
    for alpha in (1e-6, 1e-4, 1e-2, 1.0, 100.0):
        scaler = StandardScaler().fit(embedding[fit_idx])
        probe = Ridge(alpha=alpha, fit_intercept=True).fit(scaler.transform(embedding[fit_idx]), target[fit_idx])
        prediction = probe.predict(scaler.transform(embedding[stop_idx]))
        records.append({"alpha": alpha, "inner_fit_count": len(fit_idx), "inner_stopper_count": len(stop_idx),
                        "inner_chronological_MAE": float(np.mean(abs(prediction - target[stop_idx])))})
    frame = pd.DataFrame(records).sort_values(["inner_chronological_MAE", "alpha"], kind="stable").reset_index(drop=True)
    return float(frame.iloc[0].alpha), frame


def fit_probe(embedding: np.ndarray, target: np.ndarray, train_idx: np.ndarray, valid_idx: np.ndarray, alpha: float) -> tuple[np.ndarray, dict[str, Any]]:
    scaler = StandardScaler().fit(embedding[train_idx])
    probe = Ridge(alpha=alpha, fit_intercept=True).fit(scaler.transform(embedding[train_idx]), target[train_idx])
    prediction = probe.predict(scaler.transform(embedding[valid_idx]))
    return prediction, {"embedding_scaler_fit_population": "full_outer_train", "ridge_alpha": alpha,
                        "ridge_intercept": float(probe.intercept_), "ridge_coefficient_l2_norm": float(np.linalg.norm(probe.coef_))}


def style() -> None:
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11, "axes.linewidth": 1.5,
                         "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
                         "figure.dpi": 120, "savefig.dpi": 300})


def figures(calibration: pd.DataFrame, predictions: pd.DataFrame, baselines: pd.DataFrame, stability: pd.DataFrame) -> None:
    FIG.mkdir(parents=True, exist_ok=True); style()
    anchor = pd.read_csv(ANCHORS, usecols=["checkpoint", "Y", "R_raw"])
    fig, ax = plt.subplots(figsize=(7.3, 4.8))
    ax.scatter(anchor.R_raw, anchor.Y, s=55, color="black", label="Sparse Development anchors")
    grid = np.linspace(anchor.R_raw.min() - .001, anchor.R_raw.max() + .001, 150)
    colors = {"global_six_anchor_reference": PALETTE["neutral"], "fold_1": PALETTE["blue_main"], "fold_2": PALETTE["teal"], "fold_3": PALETTE["violet"]}
    for item in calibration.itertuples(index=False):
        label = "Global six-anchor (reference only)" if item.fold == "global_six_anchor_reference" else item.fold.replace("_", " ") + " chronology-safe B2"
        ax.plot(grid, item.a_fold * grid + item.b_fold, color=colors[item.fold], linewidth=2,
                linestyle="--" if item.fold == "global_six_anchor_reference" else "-", label=label)
    ax.set(xlabel="Raw P1S retention $R_{raw}$", ylabel="Polarization health target $Y$", title="Fold-specific chronology-safe B2 calibration")
    ax.grid(alpha=.22); ax.legend(fontsize=8); fig.tight_layout(); fig.savefig(FIG / "Fig1_FoldSpecific_B2_Calibration.png"); plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.5, 5.2))
    for seed, group in predictions.groupby("pretrained_seed", sort=True):
        ax.scatter(group.SOH_proxy_fold, group.predicted_SOH_proxy, alpha=.7, s=32, label=f"pretrained seed {seed}")
    low = float(min(predictions.SOH_proxy_fold.min(), predictions.predicted_SOH_proxy.min())); high = float(max(predictions.SOH_proxy_fold.max(), predictions.predicted_SOH_proxy.max()))
    ax.plot([low, high], [low, high], color="black", linewidth=1, linestyle="--")
    ax.set(xlabel="Fold-specific chronology-safe SOH proxy", ylabel="Ridge prediction", title="Frozen TimesNet representation health probe")
    ax.grid(alpha=.22); ax.legend(fontsize=8); fig.tight_layout(); fig.savefig(FIG / "Fig2_Pretrained_Health_Probe_Predictions.png"); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.0, 4.8))
    probe_fold = predictions.groupby(["fold", "pretrained_seed"], as_index=False).agg(MAE=("absolute_error", "mean"))
    values = [probe_fold.MAE.mean(), baselines.Age_MAE.mean(), baselines.LastVoltage_MAE.mean()]
    errors = [probe_fold.MAE.std(ddof=0), 0.0, 0.0]
    bars = ax.bar(["Pretrained\nTimesNet + Ridge", "Age Linear", "LastVoltage Linear"], np.asarray(values) * 100,
                  yerr=np.asarray(errors) * 100, capsize=4, color=[PALETTE["blue_main"], PALETTE["red_strong"], PALETTE["teal"]])
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{value * 100:.3f}", ha="center", va="bottom", fontsize=9)
    ax.set(ylabel="MAE (percentage points)", title="Chronology-safe Development comparison")
    ax.grid(axis="y", alpha=.22); fig.tight_layout(); fig.savefig(FIG / "Fig3_Pretrained_vs_Simple_Baselines.png"); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    fold_order = ["fold_1", "fold_2", "fold_3"]
    for seed, group in predictions.groupby("pretrained_seed", sort=True):
        metrics = group.groupby("fold", sort=False).agg(MAE=("absolute_error", "mean")).reindex(fold_order)
        ax.plot(fold_order, metrics.MAE.to_numpy() * 100, marker="o", linewidth=2, label=f"seed {seed}")
    ax.set(ylabel="MAE (percentage points)", xlabel="Chronological outer fold", title="Pretrained-seed stability by fold")
    ax.grid(alpha=.22); ax.legend(); fig.tight_layout(); fig.savefig(FIG / "Fig4_Seed_Stability_By_Fold.png"); plt.close(fig)


def main() -> None:
    os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")
    cfg, bs_cfg, common, affine_parameters = load_authorities()
    OUT.mkdir(parents=True, exist_ok=True); EMBED.mkdir(parents=True, exist_ok=True); FIG.mkdir(parents=True, exist_ok=True); REPORT.mkdir(parents=True, exist_ok=True)
    source_hashes = {path.name: sha256(path) for path in (CONFIG, BS_PROTOCOL, TSHE_EVENTS, ANCHORS, AHC_AFFINE)}
    calibration, b2_parameters = calibrations(affine_parameters)
    write_csv(calibration, OUT / "FOLD_SPECIFIC_CHRONOLOGY_SAFE_B2.csv")
    legal, raw_windows = load_and_validate_events()
    expanded, fold_indices = fold_expanded_events(legal, b2_parameters)
    write_csv(expanded, OUT / "CHRONOLOGY_SAFE_HEALTH_EVENT_DATASET.csv")
    global_row = calibration.loc[calibration.fold.eq("global_six_anchor_reference")].iloc[0]
    global_a, global_b = float(global_row.a_fold), float(global_row.b_fold)
    shift_rows: list[dict[str, Any]] = []
    for (fold, role), group in expanded.groupby(["fold", "role"], sort=True):
        global_proxy = global_a * group.target_P1S_retention.to_numpy(float) + global_b
        difference = group.SOH_proxy_fold.to_numpy(float) - global_proxy
        shift_rows.append({"fold": fold, "role": role, "event_count": len(group),
                           "mean_signed_shift": float(difference.mean()), "mean_absolute_shift": float(abs(difference).mean()),
                           "max_absolute_shift": float(abs(difference).max()),
                           "mean_absolute_shift_percentage_points": float(abs(difference).mean() * 100),
                           "max_absolute_shift_percentage_points": float(abs(difference).max() * 100)})
    target_shifts = pd.DataFrame(shift_rows)
    write_csv(target_shifts, OUT / "B2_TARGET_SHIFT_VS_GLOBAL_REFERENCE.csv")
    reproduction = reproduce_preprocessing(common, bs_cfg)
    write_csv(reproduction, OUT / "PRETRAINED_SCALER_REPRODUCTION_CHECK.csv")

    result_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    baseline_rows: list[dict[str, Any]] = []
    ridge_selection_rows: list[pd.DataFrame] = []
    for fold, _train_start, _train_end, _val_start, _val_end in FOLDS:
        train_idx, valid_idx = fold_indices[fold]
        a, b = b2_parameters[fold]
        target = a * legal.target_P1S_retention.to_numpy(float) + b
        age_prediction = fit_linear_predict(legal.measurement_end_ophrs2_h.to_numpy(float)[train_idx], target[train_idx], legal.measurement_end_ophrs2_h.to_numpy(float)[valid_idx])
        last_prediction = fit_linear_predict(raw_windows[train_idx, -1, 0], target[train_idx], raw_windows[valid_idx, -1, 0])
        age_stats, last_stats = compute_metrics(target[valid_idx], age_prediction), compute_metrics(target[valid_idx], last_prediction)
        baseline_rows.append({"fold": fold, "n_train": len(train_idx), "n_validation": len(valid_idx),
                              **{f"Age_{key}": value for key, value in age_stats.items()},
                              **{f"LastVoltage_{key}": value for key, value in last_stats.items()}})
        combined = np.concatenate([train_idx, valid_idx])
        roles = np.asarray(["train"] * len(train_idx) + ["validation"] * len(valid_idx))
        for seed in SEEDS:
            embedding = embeddings_for_fold_seed(common, bs_cfg, fold, seed, raw_windows)
            alpha, selection = select_ridge_alpha(embedding, target, train_idx)
            ridge_selection_rows.append(selection.assign(fold=fold, pretrained_seed=seed))
            prediction, probe_meta = fit_probe(embedding, target, train_idx, valid_idx, alpha)
            metrics = compute_metrics(target[valid_idx], prediction)
            result_rows.append({"fold": fold, "pretrained_seed": seed, "n_train": len(train_idx), "n_validation": len(valid_idx),
                                "selected_ridge_alpha": alpha, **probe_meta, **metrics,
                                "encoder_frozen": True, "timesnet_parameters_updated": 0})
            embedding_frame = legal.iloc[combined][["cycle_interval_id", "measurement_end_ophrs2_h"]].copy().reset_index(drop=True)
            embedding_frame.insert(1, "role", roles)
            embedding_frame.insert(0, "fold", fold); embedding_frame.insert(2, "pretrained_seed", seed)
            embedding_frame["SOH_proxy_fold"] = target[combined]
            for dimension in range(32):
                embedding_frame[f"z_{dimension:02d}"] = embedding[combined, dimension]
            write_csv(embedding_frame, EMBED / f"{fold}_seed{seed}.csv")
            for index, value in zip(valid_idx, prediction):
                event = legal.iloc[int(index)]
                prediction_rows.append({"fold": fold, "pretrained_seed": seed, "cycle_interval_id": event.cycle_interval_id,
                                        "raw_cycle_id": int(event.raw_cycle_id), "measurement_end_ophrs2_h": float(event.measurement_end_ophrs2_h),
                                        "SOH_proxy_fold": float(target[index]), "predicted_SOH_proxy": float(value),
                                        "error": float(value - target[index]), "absolute_error": float(abs(value - target[index])),
                                        "selected_ridge_alpha": alpha, "target_cycle_isolated": True})
    results = pd.DataFrame(result_rows).sort_values(["fold", "pretrained_seed"]).reset_index(drop=True)
    predictions = pd.DataFrame(prediction_rows).sort_values(["fold", "pretrained_seed", "measurement_end_ophrs2_h"]).reset_index(drop=True)
    baselines = pd.DataFrame(baseline_rows).sort_values("fold").reset_index(drop=True)
    stability = results.groupby("fold", as_index=False).agg(MAE_mean=("MAE", "mean"), MAE_std=("MAE", lambda values: float(np.std(values, ddof=0))),
        R2_mean=("R2", "mean"), R2_std=("R2", lambda values: float(np.std(values, ddof=0))),
        Spearman_mean=("Spearman", "mean"), Spearman_std=("Spearman", lambda values: float(np.std(values, ddof=0)),),
        worst_seed_MAE=("MAE", "max"), best_seed_MAE=("MAE", "min"))
    stability["MAE_coefficient_of_variation"] = stability.MAE_std / stability.MAE_mean
    stability["worst_seed_over_best_seed_MAE"] = stability.worst_seed_MAE / stability.best_seed_MAE
    write_csv(results, OUT / "PRETRAINED_HEALTH_PROBE_RESULTS.csv")
    write_csv(predictions, OUT / "PRETRAINED_HEALTH_PROBE_PREDICTIONS.csv")
    write_csv(baselines, OUT / "CHRONOLOGY_SAFE_BASELINE_RESULTS.csv")
    write_csv(stability, OUT / "SEED_STABILITY_SUMMARY.csv")
    write_csv(pd.concat(ridge_selection_rows, ignore_index=True), OUT / "RIDGE_CHRONOLOGICAL_SELECTION_AUDIT.csv")

    overall_mae = float(results.MAE.mean()); overall_rmse = float(results.RMSE.mean())
    age_mae = float(baselines.Age_MAE.mean()); last_mae = float(baselines.LastVoltage_MAE.mean())
    fold3 = results.loc[results.fold == "fold_3"].sort_values("pretrained_seed")
    baseline3 = baselines.loc[baselines.fold == "fold_3"].iloc[0]
    gate_a, gate_b = overall_mae < age_mae, overall_mae < last_mae
    gate_c = float(fold3.MAE.mean()) < float(baseline3.Age_MAE) and float(fold3.MAE.mean()) < float(baseline3.LastVoltage_MAE)
    gate_d, positive_r2 = float(fold3.Spearman.median()) > .5, int((fold3.R2 > 0).sum())
    gate_e = positive_r2 >= 2
    status = "PRETRAINED_TIMESNET_HEALTH_REPRESENTATION_SUPPORTED" if all((gate_a, gate_b, gate_c, gate_d, gate_e)) else "PRETRAINED_TIMESNET_HEALTH_REPRESENTATION_NOT_YET_SUPPORTED"
    if status.endswith("NOT_YET_SUPPORTED"):
        if not gate_a and not gate_b:
            diagnosis = "REPRESENTATION_NOT_HEALTH_INFORMATIVE"
        elif not gate_d or not gate_e:
            diagnosis = "SEED_REPRESENTATION_INSTABILITY"
        else:
            diagnosis = "24H_INFORMATION_HORIZON_TOO_SHORT"
        next_phase = "RESEARCH_DECISION_REQUIRED_NO_AUTOMATIC_MULTITASK_TRAINING"
    else:
        diagnosis = "TRANSFERABLE_REPRESENTATION_SUPPORTED_FOR_PROXY_ONLY"
        next_phase = "TIMESNET_HEALTH_TRAJECTORY_FORECASTING"
    summary = {"backbone": "TimesNet", "backbone_search_reopened": False, "scratch_neural_health_training": False,
               "pretrained_DeltaV_checkpoints_loaded": "9/9", "pretrained_scaler_reproduction": "PASS",
               "encoder_frozen": True, "timesnet_parameters_updated": 0, "health_probe": "Ridge",
               "health_labels": "FOLD-SPECIFIC CHRONOLOGY-SAFE P1S-B2", "future_calibration_anchor_used": False,
               "legal_health_events": len(legal), "target_cycle_isolation": "PASS", "overall_MAE": overall_mae,
               "overall_RMSE": overall_rmse, "age_baseline_MAE": age_mae, "last_voltage_baseline_MAE": last_mae,
               "fold3_mean_MAE": float(fold3.MAE.mean()), "fold3_median_Spearman": float(fold3.Spearman.median()),
               "fold3_positive_R2_seeds": positive_r2, "gate_A": "PASS" if gate_a else "FAIL", "gate_B": "PASS" if gate_b else "FAIL",
               "gate_C": "PASS" if gate_c else "FAIL", "gate_D": "PASS" if gate_d else "FAIL", "gate_E": "PASS" if gate_e else "FAIL",
               "final_status": status, "most_likely_diagnosis_if_not_supported": diagnosis,
               "true_SOH_independently_validated": False, "RUL_trained": False, "multitask_training_performed": False,
               "test_accessed": False, "recommended_next_phase": next_phase, "source_sha256": source_hashes}
    write_json(summary, OUT / "CSTL_SUMMARY.json")
    figures(calibration, predictions, baselines, stability)
    report = f"""# Pretrained TimesNet health transfer

## 1. Why scratch TSHE was insufficient

The earlier TSHE scratch end-to-end experiment retained strict target-cycle isolation but did not meet its Fold 3 explained-variation gate. It remains an immutable historical baseline and is not numerically ranked against this phase because its targets came from a global six-anchor B2 mapping.

## 2. Chronology-safe B2 supervision

This phase uses only `checkpoint`, `Y`, and `R_raw` from the CNP1S anchor authority. Fold 1 calibrates with 300/600/850 h; folds 2 and 3 use 300/600/850/1150 h. The 1500 and 1750 h anchors are excluded from Fold 3 target construction. The global six-anchor B2 fit is retained only as a descriptive reference.

## 3. Fold-specific calibration coefficients

{markdown_table(calibration, 8)}

The chronology-safe target displacement relative to the historical global-six-anchor B2 reference is:

{markdown_table(target_shifts, 8)}

This demonstrates the numerical scale change induced by chronology-safe calibration. It is a descriptive target-definition comparison, not a post-hoc statistical test or selection criterion.

## 4. DeltaV-pretrained TimesNet authority

All nine formal Final Backbone Screening TimesNet checkpoints were loaded. Inputs use their checkpoint-serialized outer-train median/mean/scale preprocessing, not a health-event scaler.

## 5. Pretrained preprocessing reproduction

{markdown_table(reproduction, 6)}

Each checkpoint reproduced a small saved Development-origin sample within {REPRODUCTION_TOLERANCE_MV:.1f} mV. This is an integrity replay, not a new forecast experiment.

## 6. Frozen latent extraction

The original forecasting body was replayed unchanged through the frozen 48→72 temporal transformation, DataEmbedding, TimesBlock, and layer normalization. The extracted representation is the actual `[B, 72, 32]` pre-output-projection latent sequence. Temporal mean pooling yields a 32-dimensional vector. All TimesNet parameters have `requires_grad=False`; zero encoder parameters were updated.

## 7. Ridge health probe

For each fold and pretrained seed, chronological inner selection uses the last 15% of outer-train health events and the fixed alpha grid. Only embeddings are standardized; proxy targets remain on their original scale. The final Ridge refit uses the full outer train and is evaluated once on the outer validation interval.

## 8. Development results

{markdown_table(results[["fold", "pretrained_seed", "n_train", "n_validation", "selected_ridge_alpha", "MAE", "RMSE", "R2", "Spearman"]], 6)}

## 9. Fold3 forward-generalization

{markdown_table(fold3[["pretrained_seed", "MAE", "RMSE", "R2", "Spearman", "selected_ridge_alpha"]], 6)}

## 10. Seed stability

{markdown_table(stability, 6)}

## 11. Comparison with Age and LastVoltage

{markdown_table(baselines[["fold", "Age_MAE", "Age_RMSE", "Age_R2", "LastVoltage_MAE", "LastVoltage_RMSE", "LastVoltage_R2"]], 6)}

## 12. What DeltaV pretraining contributed

The formal comparison is only against these chronology-safe same-target baselines. A positive result may be described only as: **DeltaV-pretrained TimesNet representations contain transferable information associated with the P1S-B2 calibrated health proxy.** It is not independent validation of true SOH.

## 13. Remaining limitations

The labels are sparse calibrated proxies, not independently validated true SOH. This phase reads only the 3007-row Development prefix of the causal feature table and never constructs post-1819 h labels, uses 2000/2300 targets, or trains RUL.

## 14. Decision for the next research phase

**{status}**. Gates A–E: {summary['gate_A']}, {summary['gate_B']}, {summary['gate_C']}, {summary['gate_D']}, {summary['gate_E']}. If unsupported, the most likely diagnosis is `{diagnosis}` and the next step requires a separate research decision; this phase does not begin multitask training automatically.
"""
    (REPORT / "PRETRAINED_TIMESNET_HEALTH_TRANSFER_REPORT.md").write_text(report, encoding="utf-8")
    print(json.dumps({"final_status": status, "overall_MAE": overall_mae, "fold3_positive_R2_seeds": positive_r2,
                      "gates": {letter: summary[f"gate_{letter}"] for letter in "ABCDE"}}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
