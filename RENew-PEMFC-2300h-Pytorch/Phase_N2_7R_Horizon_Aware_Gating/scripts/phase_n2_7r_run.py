"""Phase N2.7R: strictly Development-only horizon-aware gated fusion.

This program reads frozen expert prediction caches and causal feature histories.
It never constructs a Test window, prediction, gate weight, or Test metric.
"""
from __future__ import annotations

import json
import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml
from scipy.stats import pearsonr, spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
from sklearn.preprocessing import StandardScaler


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
PROJECT = ROOT.parent
CONFIG = ROOT / "configs/phase_n2_7r_config.yaml"
N25_SCRIPTS = PROJECT / "Phase_N2_5_Cycle_Aware_xLSTM/scripts"
if str(N25_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(N25_SCRIPTS))
from phase_n2_5_common import (  # noqa: E402
    fold_windows,
    load_development_data,
    metric_dict,
    paired_moving_block_bootstrap,
)


FOLDS = ["fold_1", "fold_2", "fold_3"]
SEEDS = [42, 2024, 3407]
META = [
    ("meta_1_fold2_oos", ["fold_1"], "fold_2"),
    ("meta_2_fold3_oos", ["fold_1", "fold_2"], "fold_3"),
]
EPS = 1e-12


def ensure(path: Path) -> Path:
    root = ROOT.resolve()
    target = path.resolve()
    if target.parent != root and root not in target.parents:
        raise RuntimeError(f"Refusing output outside N2.7R: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    return target


def write_csv(frame: pd.DataFrame, relative: str) -> Path:
    path = ensure(ROOT / relative)
    frame.to_csv(path, index=False, encoding="utf-8-sig", lineterminator="\n")
    return path


def write_json(payload: Any, relative: str) -> Path:
    path = ensure(ROOT / relative)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path


def write_text(text: str, relative: str) -> Path:
    path = ensure(ROOT / relative)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")
    return path


def load_cfg() -> dict[str, Any]:
    cfg = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    if Path(cfg["phase_n2_7r_root"]).resolve() != ROOT.resolve():
        raise RuntimeError("N2.7R root in config does not equal script root")
    frozen_path = Path(cfg["project_root"]) / cfg["n2_7_frozen_config"]
    frozen = yaml.safe_load(frozen_path.read_text(encoding="utf-8"))
    protocol = frozen.get("protocol", {})
    required = {
        "sampling_resolution_hours": 0.5,
        "feature_group": "F2",
        "lookback_hours": 24,
        "lookback_steps": 48,
        "horizon_hours": 12,
        "horizon_steps": 24,
    }
    for key, wanted in required.items():
        actual = protocol.get(key)
        if actual != wanted:
            raise RuntimeError(f"Frozen N2.7 protocol mismatch {key}: {actual!r}")
    if frozen.get("final_lstm_expert", {}).get("preprocessing") != "P2_causal_sg":
        raise RuntimeError("Frozen LSTM is not N2.7 P2 causal SG")
    if frozen.get("final_timesnet_expert", {}).get("tuning") != "T0_original":
        raise RuntimeError("Frozen TimesNet is not T0 original")
    if list(protocol.get("exact_feature_names", [])) != list(cfg["features"]):
        raise RuntimeError("F2 column order differs from frozen protocol")
    cfg["frozen"] = frozen
    return cfg


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def safe_corr(x: np.ndarray, y: np.ndarray, kind: str = "pearson") -> float:
    x, y = np.asarray(x, float), np.asarray(y, float)
    mask = np.isfinite(x) & np.isfinite(y)
    if int(mask.sum()) < 3 or np.std(x[mask]) < EPS or np.std(y[mask]) < EPS:
        return math.nan
    result = pearsonr(x[mask], y[mask]) if kind == "pearson" else spearmanr(x[mask], y[mask])
    return float(result.statistic)


def prediction_paths(cfg: dict[str, Any], kind: str, seed: int, fold: str) -> tuple[Path, Path, str]:
    n27 = Path(cfg["phase_n2_7_root"])
    n26r = Path(cfg["phase_n2_6r_root"])
    n2 = Path(cfg["phase_n2_root"])
    if kind == "sg_lstm":
        run_id = f"n27_preproc_lstm_P2_causal_sg_{fold}_seed{seed}"
        root = n27
    elif kind == "timesnet":
        run_id = f"screening_timesnet_hf_t1_{fold}_seed42" if seed == 42 else f"multiseed_timesnet_hf_t1_{fold}_seed{seed}"
        root = n26r
    elif kind == "raw_lstm":
        run_id = f"reproduction_LSTM_{fold}_seed42" if seed == 42 else f"multiseed_LSTM_F2_L24p0h_H12p0h_R0p5h_{fold}_seed{seed}"
        root = n26r if seed == 42 else n2
    else:
        raise KeyError(kind)
    return root / "outputs/model_runs" / f"{run_id}.json", root / "outputs/model_runs" / f"{run_id}_predictions.npz", run_id


def load_prediction(cfg: dict[str, Any], kind: str, seed: int, fold: str) -> tuple[dict[str, Any], dict[str, np.ndarray], str]:
    result_path, array_path, run_id = prediction_paths(cfg, kind, seed, fold)
    result = json.loads(result_path.read_text(encoding="utf-8"))
    arrays = dict(np.load(array_path, allow_pickle=False))
    required = {"y_true", "y_pred", "anchor", "origin_age_h", "target_end_age_h", "row_origin_index"}
    if result.get("status") != "complete" or not required.issubset(arrays):
        raise RuntimeError(f"Invalid frozen cache: {run_id}")
    if arrays["y_true"].shape != arrays["y_pred"].shape or arrays["y_true"].shape[1] != 24:
        raise RuntimeError(f"Shape mismatch in {run_id}")
    if not np.isfinite(arrays["y_true"]).all() or not np.isfinite(arrays["y_pred"]).all():
        raise RuntimeError(f"Non-finite frozen cache: {run_id}")
    if float(np.max(arrays["target_end_age_h"])) >= float(cfg["split"]["development_end_h"]):
        raise RuntimeError(f"TEST FIREWALL: frozen cache reaches Test: {run_id}")
    return result, arrays, run_id


def macro_metrics(frame: pd.DataFrame, column: str) -> dict[str, float]:
    rows = []
    for _, part in frame.groupby(["seed", "evaluation_fold"], sort=True):
        rows.append(metric_dict(part["y_true_V"].to_numpy(float), part[column].to_numpy(float)))
    if not rows:
        return {key: math.nan for key in ("rmse_V", "rmse_mV", "mae_V", "mae_mV", "bias_V", "bias_mV", "r2")}
    return {key: float(np.mean([row[key] for row in rows])) for key in rows[0]}


def mrow(frame: pd.DataFrame, column: str, method: str, scope: str) -> dict[str, Any]:
    metrics = macro_metrics(frame, column)
    return {"method": method, "scope": scope, "prediction_point_count": int(len(frame)), **metrics}


def linear_slope(values: np.ndarray) -> np.ndarray:
    t = np.arange(values.shape[1], dtype=float)
    t = t - t.mean()
    return (values @ t) / float(np.sum(t * t))


def g0_for_windows(windows: Any, feature_names: list[str], spec: dict[str, list[str]]) -> tuple[np.ndarray, list[str]]:
    col_index = {name: index for index, name in enumerate(feature_names)}
    output, names = [], []
    for feature, statistics in spec.items():
        if feature == "mahalanobis_shift_score":
            continue
        if feature not in col_index:
            raise RuntimeError(f"G0 feature absent from exact F2: {feature}")
        values = windows.x[:, :, col_index[feature]].astype(float)
        values = np.where(np.isfinite(values), values, np.nanmedian(values))
        for stat in statistics:
            if stat == "mean": value = values.mean(axis=1)
            elif stat == "std": value = values.std(axis=1)
            elif stat == "min": value = values.min(axis=1)
            elif stat == "max": value = values.max(axis=1)
            elif stat == "last": value = values[:, -1]
            elif stat == "sum": value = values.sum(axis=1)
            elif stat == "slope": value = linear_slope(values)
            else: raise KeyError(stat)
            output.append(value)
            names.append(f"g0__{feature}__{stat}")
    return np.column_stack(output).astype(np.float64), names


def stages_from_windows(windows_by_fold: dict[str, Any]) -> tuple[float, float]:
    values = np.concatenate([part.origin_age_h for part in windows_by_fold.values()])
    return tuple(np.quantile(values, [1 / 3, 2 / 3]).tolist())


def age_stage(ages: np.ndarray, cuts: tuple[float, float]) -> np.ndarray:
    return np.where(ages < cuts[0], "Early Development", np.where(ages < cuts[1], "Middle Development", "Late Development"))


@dataclass
class GateFit:
    name: str
    scaler: StandardScaler
    logistic: LogisticRegression | None = None
    constant: float | None = None
    mlp: nn.Module | None = None
    used_features: list[str] | None = None
    epochs: int = 0
    validation_loss: float = math.nan

    def predict(self, x: np.ndarray) -> np.ndarray:
        z = self.scaler.transform(x).astype(np.float32)
        if self.logistic is not None:
            return self.logistic.predict_proba(z)[:, 1]
        if self.constant is not None:
            return np.full(len(z), self.constant, dtype=float)
        if self.mlp is None:
            raise RuntimeError("Unfitted gate")
        self.mlp.eval()
        with torch.no_grad():
            return self.mlp(torch.from_numpy(z)).cpu().numpy().reshape(-1)


class TinyGateMLP(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(width, 32), nn.GELU(), nn.Dropout(0.1), nn.Linear(32, 16), nn.GELU(), nn.Linear(16, 1), nn.Sigmoid())
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def fit_logistic(name: str, x: np.ndarray, labels: np.ndarray, names: list[str]) -> GateFit:
    scaler = StandardScaler().fit(x)
    if len(np.unique(labels)) < 2:
        return GateFit(name=name, scaler=scaler, constant=float(np.mean(labels)), used_features=names)
    model = LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000, random_state=0)
    model.fit(scaler.transform(x), labels.astype(int))
    return GateFit(name=name, scaler=scaler, logistic=model, used_features=names)


def fit_mlp(name: str, x: np.ndarray, expert_difference: np.ndarray, lstm_error: np.ndarray, origin_age: np.ndarray, seed: int, names: list[str]) -> GateFit:
    set_seed(seed)
    scaler = StandardScaler().fit(x)
    z = scaler.transform(x).astype(np.float32)
    unique_age = np.unique(origin_age)
    cutoff = unique_age[max(0, int(math.floor(len(unique_age) * 0.8)) - 1)]
    train_mask = origin_age <= cutoff
    validation_mask = ~train_mask
    if not validation_mask.any():
        validation_mask[-max(1, len(validation_mask)//5):] = True
        train_mask = ~validation_mask
    model = TinyGateMLP(z.shape[1]).cpu()
    opt = torch.optim.Adam(model.parameters(), lr=0.001)
    x_train = torch.from_numpy(z[train_mask]); difference_train = torch.from_numpy(expert_difference[train_mask].astype(np.float32)).reshape(-1, 1); base_error_train = torch.from_numpy(lstm_error[train_mask].astype(np.float32)).reshape(-1, 1)
    x_val = torch.from_numpy(z[validation_mask]); difference_val = torch.from_numpy(expert_difference[validation_mask].astype(np.float32)).reshape(-1, 1); base_error_val = torch.from_numpy(lstm_error[validation_mask].astype(np.float32)).reshape(-1, 1)
    generator = torch.Generator().manual_seed(seed)
    best_state, best_loss, bad, epoch_final = None, math.inf, 0, 0
    for epoch in range(1, 301):
        model.train()
        order = torch.randperm(len(x_train), generator=generator)
        for start in range(0, len(order), 256):
            take = order[start:start + 256]
            pred = model(x_train[take])
            loss = torch.mean((pred * difference_train[take] + base_error_train[take]) ** 2)
            opt.zero_grad(); loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            value = float(torch.mean((model(x_val) * difference_val + base_error_val) ** 2).item())
        epoch_final = epoch
        if value < best_loss - 1e-6:
            best_loss, bad = value, 0
            best_state = {key: item.detach().clone() for key, item in model.state_dict().items()}
        else:
            bad += 1
            if bad >= 35:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return GateFit(name=name, scaler=scaler, mlp=model, used_features=names, epochs=epoch_final, validation_loss=best_loss)


def classifier_metrics(labels: np.ndarray, score: np.ndarray) -> dict[str, float]:
    labels, score = np.asarray(labels, int), np.asarray(score, float)
    predicted = score >= 0.5
    return {
        "winner_auc": float(roc_auc_score(labels, score)) if len(np.unique(labels)) > 1 else math.nan,
        "winner_accuracy": float(accuracy_score(labels, predicted)),
        "winner_precision_timesnet": float(precision_score(labels, predicted, zero_division=0)),
        "winner_recall_timesnet": float(recall_score(labels, predicted, zero_division=0)),
        "timesnet_winner_support": int(labels.sum()),
        "point_support": int(len(labels)),
    }


def build_base_points(cfg: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, Any], dict[str, Any], dict[str, np.ndarray], list[str]]:
    frame, inherited = load_development_data(cfg, cfg["features"])
    if inherited.get("target_rows_read_from_test") != 0 or frame["split"].eq("test").any():
        raise RuntimeError("TEST FIREWALL: Test feature/target values entered frame")
    raw_windows = fold_windows(frame, cfg)
    windows = {fold: validation for fold, _train, validation in raw_windows}
    for fold, window in windows.items():
        if float(window.target_end_age_h.max()) >= float(cfg["split"]["development_end_h"]):
            raise RuntimeError(f"TEST FIREWALL: {fold} reaches Test")
    shift = pd.read_csv(Path(cfg["phase_n2_7_root"]) / "outputs/shift_diagnostics/window_shift_score.csv")
    shift = shift[shift["dataset_role"].eq("validation")].copy()
    if shift.duplicated(["fold", "row_origin_index"]).any():
        raise RuntimeError("Shift score duplicate keys")
    shift_lookup = shift.set_index(["fold", "row_origin_index"])
    g0_by_fold, g0_names = {}, None
    for fold, window in windows.items():
        matrix, names = g0_for_windows(window, cfg["features"], cfg["g0_features"])
        g0_by_fold[fold] = matrix
        g0_names = names
    assert g0_names is not None
    cuts = stages_from_windows(windows)
    arrays: dict[tuple[str, int, str], dict[str, np.ndarray]] = {}
    audit_rows, records = [], []
    for seed in SEEDS:
        for fold in FOLDS:
            _, sg, sg_id = load_prediction(cfg, "sg_lstm", seed, fold)
            _, times, time_id = load_prediction(cfg, "timesnet", seed, fold)
            raw_result, raw, raw_id = load_prediction(cfg, "raw_lstm", seed, fold)
            window = windows[fold]
            for key in ["row_origin_index", "y_true", "anchor", "origin_age_h", "target_end_age_h"]:
                window_value = window.y_abs if key == "y_true" else window.__getattribute__(key)
                if not np.array_equal(sg[key], times[key]) or not np.array_equal(sg[key], window_value):
                    raise RuntimeError(f"Paired mismatch seed={seed} fold={fold} key={key}")
                if not np.array_equal(sg[key], raw[key]):
                    raise RuntimeError(f"SG/raw mismatch seed={seed} fold={fold} key={key}")
            arrays[("sg", seed, fold)], arrays[("times", seed, fold)], arrays[("raw", seed, fold)] = sg, times, raw
            audit_rows.append({"seed": seed, "fold": fold, "lstm_run": sg_id, "timesnet_run": time_id, "raw_lstm_run": raw_id,
                               "matched_origin_indices": True, "matched_target_timestamps": True, "matched_y_true": True,
                               "matched_anchor": True, "shape": str(sg["y_true"].shape), "target_max_h": float(sg["target_end_age_h"].max()), "status": "PASS"})
            stage = age_stage(sg["origin_age_h"], cuts)
            for position, origin_idx in enumerate(sg["row_origin_index"]):
                try:
                    shift_row = shift_lookup.loc[(fold, int(origin_idx))]
                except KeyError as exc:
                    raise RuntimeError(f"Missing causal shift record {fold}/{origin_idx}") from exc
                if isinstance(shift_row, pd.DataFrame):
                    raise RuntimeError("Nonunique shift lookup")
                g0 = g0_by_fold[fold][position]
                for step in range(24):
                    truth = float(sg["y_true"][position, step]); lstm = float(sg["y_pred"][position, step]); times_pred = float(times["y_pred"][position, step])
                    row = {"seed": seed, "fold": fold, "row_origin_index": int(origin_idx), "origin_position": position,
                           "origin_age_h": float(sg["origin_age_h"][position]), "target_end_age_h": float(sg["target_end_age_h"][position]),
                           "origin_datetime": str(window.origin_datetime[position]), "ageing_stage": str(stage[position]),
                           "horizon_step": step + 1, "horizon_hours": (step + 1) * 0.5, "horizon_normalized": (step + 1) / 24.0,
                           "shift_score": float(shift_row["shift_score"]), "anchor_V": float(sg["anchor"][position]),
                           "y_true_V": truth, "lstm_pred_V": lstm, "timesnet_pred_V": times_pred,
                           "lstm_error_mV": (lstm - truth) * 1000.0, "timesnet_error_mV": (times_pred - truth) * 1000.0,
                           "lstm_absolute_error_mV": abs(lstm - truth) * 1000.0, "timesnet_absolute_error_mV": abs(times_pred - truth) * 1000.0,
                           "timesnet_winner": int(abs(times_pred - truth) < abs(lstm - truth)),
                           "lstm_delta_from_anchor_V": lstm - float(sg["anchor"][position]),
                           "timesnet_delta_from_anchor_V": times_pred - float(sg["anchor"][position]),
                           "signed_prediction_difference_V": times_pred - lstm, "absolute_prediction_difference_V": abs(times_pred - lstm)}
                    row.update({name: float(value) for name, value in zip(g0_names, g0)})
                    records.append(row)
    points = pd.DataFrame(records)
    firewall = {**inherited, "test_target_rows_read": 0, "test_predictions_constructed": False,
                "test_gate_predictions_constructed": False, "test_gate_weights_constructed": False,
                "test_metrics_computed": False, "test_performance_accessed": False,
                "development_target_max_h": float(points.target_end_age_h.max()), "firewall_status": "PASS"}
    return points, firewall, windows, arrays, g0_names


def feature_groups(g0_names: list[str], use_g2: bool, add_age: bool = False) -> tuple[list[str], list[str]]:
    g1 = ["horizon_normalized"]
    g2 = ["lstm_delta_from_anchor_V", "timesnet_delta_from_anchor_V", "signed_prediction_difference_V", "absolute_prediction_difference_V"]
    used = list(g0_names) + g1 + (g2 if use_g2 else []) + (["origin_age_h"] if add_age else [])
    return used, g2


def hstatic_predictions(train: pd.DataFrame, evaluation: pd.DataFrame, candidate: list[float]) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    output = evaluation.copy(); rows = []
    for step, test_part in output.groupby("horizon_step", sort=True):
        tr = train[train.horizon_step.eq(step)]
        weights = {}
        for alpha in candidate:
            prediction = (1 - alpha) * tr.lstm_pred_V.to_numpy(float) + alpha * tr.timesnet_pred_V.to_numpy(float)
            weights[alpha] = metric_dict(tr.y_true_V.to_numpy(float), prediction)["rmse_mV"]
        best = min(weights, key=weights.get)
        index = test_part.index
        output.loc[index, "horizon_static_weight"] = best
        output.loc[index, "horizon_static_pred_V"] = (1 - best) * output.loc[index, "lstm_pred_V"] + best * output.loc[index, "timesnet_pred_V"]
        rows.append({"horizon_step": int(step), "horizon_hours": float(test_part.horizon_hours.iloc[0]), "selected_timesnet_weight": best,
                     "selection_train_rmse_mV": weights[best], "selection_scope": "chronological meta-train"})
    return output, rows


def run_gate_meta(points: pd.DataFrame, cfg: dict[str, Any], g0_names: list[str], gate_name: str, use_g2: bool, add_age: bool = False) -> tuple[pd.DataFrame, list[dict[str, Any]], list[dict[str, Any]], dict[tuple[int, str], np.ndarray]]:
    used, _ = feature_groups(g0_names, use_g2, add_age)
    all_eval, h_rows, fit_rows, prediction_matrices = [], [], [], {}
    for seed in SEEDS:
        seed_points = points[points.seed.eq(seed)]
        for meta_name, train_folds, eval_fold in META:
            train = seed_points[seed_points.fold.isin(train_folds)].copy().sort_values(["origin_age_h", "horizon_step"])
            evaluation = seed_points[seed_points.fold.eq(eval_fold)].copy().sort_values(["origin_age_h", "horizon_step"])
            if add_age:
                # Origin age scaling is fit only on chronological gate-training points by the gate scaler.
                pass
            x_train, x_eval = train[used].to_numpy(float), evaluation[used].to_numpy(float)
            labels = train.timesnet_winner.to_numpy(int)
            if gate_name.startswith("Gate_C"):
                target_delta = train.timesnet_pred_V.to_numpy(float) - train.lstm_pred_V.to_numpy(float)
                lstm_error = train.lstm_pred_V.to_numpy(float) - train.y_true_V.to_numpy(float)
                fit = fit_mlp(gate_name, x_train, target_delta, lstm_error, train.origin_age_h.to_numpy(float), seed, used)
                fit_kind = "direct_fusion_raw_voltage_mse"
            else:
                fit = fit_logistic(gate_name, x_train, labels, used)
                fit_kind = "pointwise_winner_logistic"
            weight = np.clip(fit.predict(x_eval), 0.0, 1.0)
            evaluation["gate_weight"] = weight
            evaluation["gate_pred_V"] = (1.0 - weight) * evaluation.lstm_pred_V.to_numpy(float) + weight * evaluation.timesnet_pred_V.to_numpy(float)
            evaluation["fixed_025_pred_V"] = 0.75 * evaluation.lstm_pred_V.to_numpy(float) + 0.25 * evaluation.timesnet_pred_V.to_numpy(float)
            static, selected = hstatic_predictions(train, evaluation, cfg["horizon_static_candidate_weights"])
            evaluation["horizon_static_pred_V"] = static["horizon_static_pred_V"]
            evaluation["horizon_static_weight"] = static["horizon_static_weight"]
            evaluation["meta_evaluation"] = meta_name; evaluation["evaluation_fold"] = eval_fold; evaluation["gate_name"] = gate_name
            evaluation["gate_feature_mode"] = "G0+G1+G2" if use_g2 else "G0+G1"
            evaluation["gate_uses_ophrs2"] = add_age
            all_eval.append(evaluation)
            for row in selected:
                h_rows.append({"seed": seed, "meta_evaluation": meta_name, "evaluation_fold": eval_fold, **row})
            fit_rows.append({"seed": seed, "meta_evaluation": meta_name, "evaluation_fold": eval_fold, "gate_name": gate_name,
                             "fit_kind": fit_kind, "train_fold_count": len(train_folds), "train_point_count": len(train),
                             "train_timesnet_winner_rate": float(labels.mean()), "fit_constant_probability": fit.constant,
                             "mlp_epochs": fit.epochs, "mlp_chronological_validation_mse_V2": fit.validation_loss,
                             "feature_count": len(used), "feature_names": ";".join(used)})
            order = evaluation[["row_origin_index", "horizon_step"]].drop_duplicates().sort_values(["row_origin_index", "horizon_step"])
            matrix = evaluation.set_index(["row_origin_index", "horizon_step"]).loc[pd.MultiIndex.from_frame(order), "gate_pred_V"].to_numpy().reshape(-1, 24)
            prediction_matrices[(seed, eval_fold)] = matrix
    return pd.concat(all_eval, ignore_index=True), h_rows, fit_rows, prediction_matrices


def bootstrap_rows(pairs: dict[tuple[int, str], tuple[np.ndarray, np.ndarray, np.ndarray]], cfg: dict[str, Any], label: str) -> list[dict[str, Any]]:
    packed = {f"seed{seed}_{fold}": values for (seed, fold), values in pairs.items()}
    rows = [paired_moving_block_bootstrap(packed, int(cfg["bootstrap"]["repetitions"]), int(cfg["bootstrap"]["block_length_origins"]), int(cfg["bootstrap"]["seed"]), label, label.split(" vs ")[0], label.split(" vs ")[1])]
    for key, values in packed.items():
        row = paired_moving_block_bootstrap({key: values}, int(cfg["bootstrap"]["repetitions"]), int(cfg["bootstrap"]["block_length_origins"]), int(cfg["bootstrap"]["seed"]), label, label.split(" vs ")[0], label.split(" vs ")[1])
        row["series"] = key; rows.append(row)
    for row in rows:
        row["analysis_scope"] = "meta_oos" if "Gate" in label else "all_development_folds"
        for key in list(row):
            if key.endswith("_V"):
                row[key[:-2] + "_mV"] = float(row[key]) * 1000.0
    return rows


def make_figures(points: pd.DataFrame, evaluations: dict[str, pd.DataFrame], best: str, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    # only Development Meta-OOS is visualized below.
    oos = evaluations[best]
    fig, ax = plt.subplots(figsize=(8, 4.5)); table = points.groupby("horizon_hours").timesnet_winner.mean(); ax.plot(table.index, 100*table.values, marker="o", color="#d85a38"); ax.axhline(50, color="gray", ls="--"); ax.set(xlabel="Horizon (h)", ylabel="TimesNet win rate (%)", title="Development-only pointwise TimesNet win rate"); fig.tight_layout(); fig.savefig(out/"timesnet_win_rate_by_horizon.png", dpi=160); plt.close(fig)
    fig, ax = plt.subplots(figsize=(6.7,4.7)); sample=oos.sample(min(6000,len(oos)),random_state=0); ax.scatter(sample.absolute_prediction_difference_V*1000, sample.timesnet_absolute_error_mV, s=5, alpha=.25,label="TimesNet"); ax.scatter(sample.absolute_prediction_difference_V*1000, sample.lstm_absolute_error_mV, s=5, alpha=.18,label="LSTM"); ax.set(xlabel="Expert disagreement |T−L| (mV)",ylabel="Absolute error (mV)",title="Meta-OOS disagreement and error"); ax.legend(); fig.tight_layout(); fig.savefig(out/"expert_disagreement_vs_error.png",dpi=160);plt.close(fig)
    for grouping, filename, xlabel, order in [("horizon_hours","gate_weight_by_horizon.png","Horizon (h)",None),("ageing_stage","gate_weight_by_ageing_stage.png","Development ageing stage",["Early Development","Middle Development","Late Development"]),("disagreement_level","gate_weight_by_disagreement.png","Train-defined disagreement level",["Low","Medium","High"])]:
        fig,ax=plt.subplots(figsize=(7.4,4.4)); temp=oos.groupby(grouping).gate_weight.agg(["mean","std"]).reindex(order) if order else oos.groupby(grouping).gate_weight.agg(["mean","std"]); ax.bar(temp.index.astype(str),temp["mean"],yerr=temp["std"],color="#8055a5",alpha=.85); ax.set(xlabel=xlabel,ylabel="TimesNet gate weight g",ylim=(0,1.05),title=f"{best}: Meta-OOS gate response"); fig.tight_layout();fig.savefig(out/filename,dpi=160);plt.close(fig)
    metric_methods=[("LSTM","lstm_pred_V"),("TimesNet","timesnet_pred_V"),("Fixed .25","fixed_025_pred_V"),("Horizon Static","horizon_static_pred_V"),(best,"gate_pred_V")]
    for scope, filename in [("all","meta_oos_rmse_comparison.png"),("fold_2","fold2_rmse_comparison.png"),("fold_3","fold3_rmse_comparison.png")]:
        part=oos if scope=="all" else oos[oos.evaluation_fold.eq(scope)]; values=[macro_metrics(part,column)["rmse_mV"] for _name,column in metric_methods]; fig,ax=plt.subplots(figsize=(8,4.5));ax.bar([x[0] for x in metric_methods],values,color=["#2864b7","#d85a38","#8055a5","#5f9e6e","#b074c6"]);ax.tick_params(axis="x",rotation=20);ax.set(ylabel="Macro RMSE (mV)",title=f"{'Meta-OOS' if scope=='all' else scope} fusion comparison");fig.tight_layout();fig.savefig(out/filename,dpi=160);plt.close(fig)
    rows=[]
    for pct in [5,10]:
        for _,part in oos.groupby(["seed","meta_evaluation"]):
            take=part.timesnet_absolute_error_mV>=part.timesnet_absolute_error_mV.quantile(1-pct/100);rows.append({"top":pct,"g":part.loc[take,"gate_weight"].mean()})
    tail=pd.DataFrame(rows).groupby("top").g.mean();fig,ax=plt.subplots(figsize=(6.2,4.2));ax.bar([f"Top {x}%" for x in tail.index],tail.values,color="#8055a5");ax.set(ylim=(0,1.05),ylabel="Mean TimesNet gate weight",title="Gate response in TimesNet high-error tails");fig.tight_layout();fig.savefig(out/"tail_error_gate_response.png",dpi=160);plt.close(fig)
    oracle=oos.copy();oracle["oracle_pred_V"]=np.where(oracle.timesnet_absolute_error_mV<oracle.lstm_absolute_error_mV,oracle.timesnet_pred_V,oracle.lstm_pred_V); items=[("Pointwise oracle","oracle_pred_V"),("Fixed .25","fixed_025_pred_V"),("Horizon Static","horizon_static_pred_V"),(best,"gate_pred_V")];fig,ax=plt.subplots(figsize=(7.3,4.4));ax.bar([x[0] for x in items],[macro_metrics(oracle,c)["rmse_mV"] for _,c in items],color=["#2e8b57","#8055a5","#5f9e6e","#b074c6"]);ax.tick_params(axis="x",rotation=15);ax.set(ylabel="Macro RMSE (mV)",title="Oracle, fixed, horizon-static and adaptive Meta-OOS");fig.tight_layout();fig.savefig(out/"oracle_fixed_horizonstatic_adaptive_comparison.png",dpi=160);plt.close(fig)


def markdown_table(rows: list[list[str]], headers: list[str]) -> str:
    return "\n".join(["| " + " | ".join(headers) + " |", "|" + "|".join(["---"]*len(headers)) + "|"] + ["| " + " | ".join(row) + " |" for row in rows])


def main() -> None:
    cfg = load_cfg()
    points, firewall, windows, arrays, g0_names = build_base_points(cfg)
    write_json(firewall, "outputs/integrity_audit/test_firewall_audit.json")
    write_csv(pd.DataFrame([{"feature_name": name, "feature_group": "G0", "causal_source": "raw 48-step F2 history", "availability": "past-known", "used_main_gate": True, "used_ophrs_ablation": True} for name in g0_names] +
        [{"feature_name":"horizon_step","feature_group":"G1","causal_source":"forecast task index","availability":"origin-known","used_main_gate":False,"used_ophrs_ablation":False},{"feature_name":"horizon_hours","feature_group":"G1","causal_source":"forecast task index","availability":"origin-known","used_main_gate":False,"used_ophrs_ablation":False},{"feature_name":"horizon_normalized","feature_group":"G1","causal_source":"forecast task index / 24","availability":"origin-known","used_main_gate":True,"used_ophrs_ablation":True}] +
        [{"feature_name":x,"feature_group":"G2","causal_source":"frozen deployable expert outputs at origin","availability":"origin-known","used_main_gate":True,"used_ophrs_ablation":True} for x in ["lstm_delta_from_anchor_V","timesnet_delta_from_anchor_V","signed_prediction_difference_V","absolute_prediction_difference_V"]] +
        [{"feature_name":"shift_score","feature_group":"G0 auxiliary","causal_source":"fold-train-fitted causal 48-step history diagnostic","availability":"past-known","used_main_gate":False,"used_ophrs_ablation":False},{"feature_name":"origin_age_h","feature_group":"diagnostic ablation","causal_source":"OpHrs2 at origin","availability":"origin-known","used_main_gate":False,"used_ophrs_ablation":True}]), "outputs/gate_diagnostics/gate_feature_definition.csv")
    write_csv(pd.DataFrame([{ "seed":r["seed"],"fold":r["fold"],"lstm_run":r["lstm_run"],"timesnet_run":r["timesnet_run"],"raw_lstm_run":r["raw_lstm_run"],"matched_origin_indices":r["matched_origin_indices"],"matched_target_timestamps":r["matched_target_timestamps"],"matched_y_true":r["matched_y_true"],"matched_anchor":r["matched_anchor"],"shape":r["shape"],"target_max_h":r["target_max_h"],"status":r["status"]} for r in []]), "outputs/integrity_audit/paired_expert_prediction_audit.csv")
    # rewrite paired audit directly from verified array keys to retain all 9 audit rows.
    audit=[]
    for seed in SEEDS:
        for fold in FOLDS:
            sg,times,raw=arrays[("sg",seed,fold)],arrays[("times",seed,fold)],arrays[("raw",seed,fold)]
            audit.append({"seed":seed,"fold":fold,"matched_origin_indices":bool(np.array_equal(sg["row_origin_index"],times["row_origin_index"])),"matched_target_timestamps":bool(np.array_equal(sg["target_end_age_h"],times["target_end_age_h"])),"matched_y_true":bool(np.array_equal(sg["y_true"],times["y_true"])),"matched_sg_raw":bool(np.array_equal(sg["y_true"],raw["y_true"])),"origin_count":len(sg["anchor"]),"horizon_steps":sg["y_true"].shape[1],"maximum_target_end_h":float(sg["target_end_age_h"].max()),"status":"PASS"})
    write_csv(pd.DataFrame(audit), "outputs/integrity_audit/paired_expert_prediction_audit.csv")
    sg_pairs={(seed,fold):(arrays[("sg",seed,fold)]["y_true"],arrays[("sg",seed,fold)]["y_pred"],arrays[("raw",seed,fold)]["y_pred"]) for seed in SEEDS for fold in FOLDS}
    write_csv(pd.DataFrame(bootstrap_rows(sg_pairs,cfg,"SG-LSTM vs Raw-LSTM")), "outputs/stability/sg_lstm_stability_bootstrap.csv")
    win=[]
    for (fold,step),part in points.groupby(["fold","horizon_step"],sort=True): win.append({"scope":"development_fold","fold":fold,"horizon_step":step,"horizon_hours":float(part.horizon_hours.iloc[0]),"horizon_normalized":float(part.horizon_normalized.iloc[0]),"point_count":len(part),"timesnet_winner_count":int(part.timesnet_winner.sum()),"timesnet_win_rate":float(part.timesnet_winner.mean())})
    write_csv(pd.DataFrame(win),"outputs/gate_diagnostics/pointwise_winner_by_horizon.csv")
    gate_specs=[("Gate_A_Horizon_Logistic",False),("Gate_B_Disagreement_Logistic",True),("Gate_C_Continuous_MLP",True)]
    evaluations, hstatic_rows, fitrows, matrices = {}, [], [], {}
    for name,g2 in gate_specs:
        result, hs, fit, mat = run_gate_meta(points,cfg,g0_names,name,g2)
        evaluations[name]=result; hstatic_rows.extend([{**x,"gate_name":name} for x in hs]);fitrows.extend(fit);matrices[name]=mat
    # Train-defined disagreement terciles are attached separately for every chronological meta task.
    for name,result in evaluations.items():
        chunks=[]
        for seed in SEEDS:
            for meta_name,train_folds,eval_fold in META:
                train=points[(points.seed.eq(seed))&points.fold.isin(train_folds)]
                lo,hi=np.quantile(train.absolute_prediction_difference_V,[1/3,2/3])
                take=result[(result.seed.eq(seed))&(result.meta_evaluation.eq(meta_name))].copy()
                take["disagreement_low_cut_V"],take["disagreement_high_cut_V"]=lo,hi
                take["disagreement_level"]=np.where(take.absolute_prediction_difference_V<=lo,"Low",np.where(take.absolute_prediction_difference_V<=hi,"Medium","High"));chunks.append(take)
        evaluations[name]=pd.concat(chunks,ignore_index=True)
    # Baseline and all gate summaries, exact same OOS records.
    method_rows=[]
    for name,result in evaluations.items():
        for scope,part in [("fold_2",result[result.evaluation_fold.eq("fold_2")]),("fold_3",result[result.evaluation_fold.eq("fold_3")]),("meta_oos",result)]:
            for method,column in [("LSTM","lstm_pred_V"),("TimesNet","timesnet_pred_V"),("Fixed .25","fixed_025_pred_V"),("Horizon Static","horizon_static_pred_V"),(name,"gate_pred_V")]:
                row=mrow(part,column,method,scope); row.update({"gate_name":name,"winner_auc":classifier_metrics(part.timesnet_winner,part.gate_weight)["winner_auc"] if method==name else math.nan});method_rows.append(row)
    gate_results=pd.DataFrame(method_rows)
    # Gate selection by Development-only Meta-OOS averaged over all pre-registered seeds.
    selected=gate_results[gate_results.scope.eq("meta_oos") & gate_results.method.isin([x[0] for x in gate_specs])].sort_values("rmse_mV").iloc[0]
    best=str(selected.method); best_eval=evaluations[best]
    write_csv(gate_results,"outputs/gate_diagnostics/adaptive_gate_meta_oos.csv")
    for name,result in evaluations.items():
        compact=gate_results[(gate_results.gate_name.eq(name))&gate_results.method.eq(name)].copy(); compact["gate_weight_mean"]=float(result.gate_weight.mean());compact["gate_weight_std"]=float(result.gate_weight.std());compact["p10"]=float(result.gate_weight.quantile(.1));compact["p50"]=float(result.gate_weight.quantile(.5));compact["p90"]=float(result.gate_weight.quantile(.9));compact["auc"]=classifier_metrics(result.timesnet_winner,result.gate_weight)["winner_auc"]
        file="gate_logistic_state_horizon.csv" if name.startswith("Gate_A") else ("gate_logistic_disagreement.csv" if name.startswith("Gate_B") else "gate_continuous_mlp.csv")
        write_csv(pd.concat([compact,pd.DataFrame(fitrows)[pd.DataFrame(fitrows).gate_name.eq(name)]],ignore_index=True,sort=False),"outputs/gate_diagnostics/"+file)
    best_static=pd.DataFrame(hstatic_rows)[pd.DataFrame(hstatic_rows).gate_name.eq(best)].copy()
    write_csv(best_static,"outputs/gate_diagnostics/horizon_static_fusion.csv")
    base=gate_results[(gate_results.gate_name.eq(best))&gate_results.method.isin(["Fixed .25","Horizon Static"])].copy();write_csv(base,"outputs/gate_diagnostics/fixed_fusion_meta_oos.csv")
    dis_rows=[]
    for level,part in best_eval.groupby("disagreement_level",sort=False):
        dis_rows.append({"scope":"meta_oos","disagreement_level":level,"point_count":len(part),"train_defined_low_cut_V_mean":float(part.disagreement_low_cut_V.mean()),"train_defined_high_cut_V_mean":float(part.disagreement_high_cut_V.mean()),"disagreement_vs_lstm_abs_error_pearson":safe_corr(part.absolute_prediction_difference_V,part.lstm_absolute_error_mV),"disagreement_vs_times_abs_error_pearson":safe_corr(part.absolute_prediction_difference_V,part.timesnet_absolute_error_mV),"lstm_rmse_mV":macro_metrics(part,"lstm_pred_V")["rmse_mV"],"timesnet_rmse_mV":macro_metrics(part,"timesnet_pred_V")["rmse_mV"],"fixed_rmse_mV":macro_metrics(part,"fixed_025_pred_V")["rmse_mV"],"adaptive_rmse_mV":macro_metrics(part,"gate_pred_V")["rmse_mV"]})
    write_csv(pd.DataFrame(dis_rows),"outputs/gate_diagnostics/expert_disagreement_analysis.csv")
    for grouping,file in [("horizon_hours","gate_weight_by_horizon.csv"),("ageing_stage","gate_weight_by_ageing_stage.csv"),("disagreement_level","gate_weight_by_disagreement.csv")]:
        rows=[]
        for value,part in best_eval.groupby(grouping,sort=True):
            rows.append({"gate_name":best,"grouping":grouping,"group":str(value),"point_count":len(part),"mean_gate_weight":float(part.gate_weight.mean()),"std_gate_weight":float(part.gate_weight.std()),"p10_gate_weight":float(part.gate_weight.quantile(.1)),"p50_gate_weight":float(part.gate_weight.quantile(.5)),"p90_gate_weight":float(part.gate_weight.quantile(.9)),"lstm_rmse_mV":macro_metrics(part,"lstm_pred_V")["rmse_mV"],"timesnet_rmse_mV":macro_metrics(part,"timesnet_pred_V")["rmse_mV"],"adaptive_rmse_mV":macro_metrics(part,"gate_pred_V")["rmse_mV"]})
        write_csv(pd.DataFrame(rows),"outputs/gate_diagnostics/"+file)
    tail=[]
    for pct in [5,10]:
        parts=[]
        for (seed,meta),part in best_eval.groupby(["seed","meta_evaluation"]):
            threshold=part.timesnet_absolute_error_mV.quantile(1-pct/100);parts.append(part[part.timesnet_absolute_error_mV.ge(threshold)])
        t=pd.concat(parts);tail.append({"tail_definition":f"TimesNet top {pct}% absolute error within each seed/meta-OOS series","top_error_percent":pct,"point_count":len(t),"mean_gate_weight":float(t.gate_weight.mean()),"mean_disagreement_mV":float(t.absolute_prediction_difference_V.mean()*1000),"mean_horizon_hours":float(t.horizon_hours.mean()),"mean_voltage_anchor_V":float(t.anchor_V.mean()),"mean_current_A":float(t[[c for c in t if c.endswith('actual_current_A_mean__last')][0]].mean()),"lstm_rmse_mV":macro_metrics(t,"lstm_pred_V")["rmse_mV"],"timesnet_rmse_mV":macro_metrics(t,"timesnet_pred_V")["rmse_mV"],"adaptive_rmse_mV":macro_metrics(t,"gate_pred_V")["rmse_mV"]})
    write_csv(pd.DataFrame(tail),"outputs/gate_diagnostics/timesnet_tail_error_gate_analysis.csv")
    # OpHrs2 is a diagnostic-only after-selection ablation; it does not replace the main gate.
    best_g2=best.startswith("Gate_B") or best.startswith("Gate_C")
    age_name=best+"_plus_OpHrs2_diagnostic"
    age_eval,_,age_fit,_=run_gate_meta(points,cfg,g0_names,age_name,best_g2,add_age=True)
    age_rows=[]
    for scope,part in [("fold_2",age_eval[age_eval.evaluation_fold.eq("fold_2")]),("fold_3",age_eval[age_eval.evaluation_fold.eq("fold_3")]),("meta_oos",age_eval)]:
        row=mrow(part,"gate_pred_V",age_name,scope);row["main_gate_rmse_mV"]=macro_metrics(best_eval[best_eval.evaluation_fold.eq(scope) if scope!='meta_oos' else best_eval.index.notna()],"gate_pred_V")["rmse_mV"];row["delta_rmse_mV_plus_ophrs2_minus_main"]=row["rmse_mV"]-row["main_gate_rmse_mV"];age_rows.append(row)
    write_csv(pd.DataFrame(age_rows+age_fit),"outputs/gate_diagnostics/ophrs2_ablation.csv")
    # multiseed compact view, classifiers and metric stability.
    multi=[]
    for name,result in evaluations.items():
        for seed,part in result.groupby("seed"):
            multi.append({"gate_name":name,"seed":seed,"meta_oos_rmse_mV":macro_metrics(part,"gate_pred_V")["rmse_mV"],"meta_oos_mae_mV":macro_metrics(part,"gate_pred_V")["mae_mV"],"winner_auc":classifier_metrics(part.timesnet_winner,part.gate_weight)["winner_auc"],"mean_gate_weight":float(part.gate_weight.mean()),"std_gate_weight":float(part.gate_weight.std())})
    write_csv(pd.DataFrame(multi),"outputs/gate_diagnostics/gate_multiseed_summary.csv")
    pairs={}
    for seed in SEEDS:
        for fold in ["fold_2","fold_3"]:
            p=best_eval[(best_eval.seed.eq(seed))&(best_eval.evaluation_fold.eq(fold))].sort_values(["row_origin_index","horizon_step"])
            truth=p.y_true_V.to_numpy().reshape(-1,24);candidate=p.gate_pred_V.to_numpy().reshape(-1,24)
            pairs[(seed,fold)]=(truth,candidate,p.lstm_pred_V.to_numpy().reshape(-1,24))
    boots=[]
    for label,ref in [(f"{best} vs LSTM","lstm_pred_V"),(f"{best} vs Fixed .25","fixed_025_pred_V"),(f"{best} vs Horizon Static","horizon_static_pred_V")]:
        q={}
        for seed in SEEDS:
            for fold in ["fold_2","fold_3"]:
                p=best_eval[(best_eval.seed.eq(seed))&(best_eval.evaluation_fold.eq(fold))].sort_values(["row_origin_index","horizon_step"]);q[(seed,fold)]=(p.y_true_V.to_numpy().reshape(-1,24),p.gate_pred_V.to_numpy().reshape(-1,24),p[ref].to_numpy().reshape(-1,24))
        boots.extend(bootstrap_rows(q,cfg,label))
    write_csv(pd.DataFrame(boots),"outputs/stability/gate_bootstrap_comparison.csv")
    figures=ROOT/"outputs/figures";make_figures(points,evaluations,best,figures)
    # Result report and pre-registered decision.
    summary={}
    for method in ["LSTM","TimesNet","Fixed .25","Horizon Static",best]:
        part=gate_results[(gate_results.gate_name.eq(best))&(gate_results.method.eq(method))]
        summary[method]={row.scope:row._asdict() for row in part.itertuples(index=False)}
    best_meta=summary[best]["meta_oos"]; lstm_meta=summary["LSTM"]["meta_oos"]; fixed_meta=summary["Fixed .25"]["meta_oos"]; hs_meta=summary["Horizon Static"]["meta_oos"]
    auc=classifier_metrics(best_eval.timesnet_winner,best_eval.gate_weight)["winner_auc"]; weight_std=float(best_eval.gate_weight.std()); h95=float(best_eval[np.isclose(best_eval.horizon_hours,9.5)].gate_weight.mean()); tail10=float(pd.DataFrame(tail).query('top_error_percent == 10').mean_gate_weight.iloc[0])
    imp_lstm=(lstm_meta["rmse_mV"]-best_meta["rmse_mV"])/lstm_meta["rmse_mV"];imp_fixed=(fixed_meta["rmse_mV"]-best_meta["rmse_mV"])/fixed_meta["rmse_mV"];imp_hs=(hs_meta["rmse_mV"]-best_meta["rmse_mV"])/hs_meta["rmse_mV"]
    fold3_good=best_meta is not None and summary[best]["fold_3"]["rmse_mV"]<=summary["Fixed .25"]["fold_3"]["rmse_mV"]
    go=bool(imp_fixed>=.02 and fold3_good and imp_lstm>=.05 and weight_std>=.01)
    hstatic_close=abs(imp_hs)<=.01
    all_methods=["LSTM","TimesNet","Fixed .25","Horizon Static","Gate_A_Horizon_Logistic","Gate_B_Disagreement_Logistic","Gate_C_Continuous_MLP"]
    rows=[]
    for method in all_methods:
        part=gate_results[(gate_results.gate_name.eq(best))&(gate_results.method.eq(method))] if method in ["LSTM","TimesNet","Fixed .25","Horizon Static"] else gate_results[(gate_results.method.eq(method))&(gate_results.scope.eq("meta_oos"))]
        if method not in ["LSTM","TimesNet","Fixed .25","Horizon Static"]:
            # Retrieve its metrics across identical meta OOS series from its own gate result.
            local=gate_results[(gate_results.method.eq(method))]; fold2=local[local.scope.eq("fold_2")].iloc[0];fold3=local[local.scope.eq("fold_3")].iloc[0];meta=local[local.scope.eq("meta_oos")].iloc[0]
        else:
            fold2=part[part.scope.eq("fold_2")].iloc[0];fold3=part[part.scope.eq("fold_3")].iloc[0];meta=part[part.scope.eq("meta_oos")].iloc[0]
        rows.append([method,f"{fold2.rmse_mV:.3f}",f"{fold3.rmse_mV:.3f}",f"{meta.rmse_mV:.3f}",f"{meta.mae_mV:.3f}",f"{meta.r2:.5f}"])
    gate_diag=[]
    for name in [x[0] for x in gate_specs]:
        ev=evaluations[name]; meta=gate_results[(gate_results.method.eq(name))&(gate_results.scope.eq("meta_oos"))].iloc[0];fixed=gate_results[(gate_results.gate_name.eq(name))&(gate_results.method.eq("Fixed .25"))&(gate_results.scope.eq("meta_oos"))].iloc[0]; gate_diag.append([name,f"{classifier_metrics(ev.timesnet_winner,ev.gate_weight)['winner_auc']:.3f}",f"{ev.gate_weight.mean():.3f}",f"{ev.gate_weight.std():.3f}",f"{(fixed.rmse_mV-meta.rmse_mV)/fixed.rmse_mV*100:.2f}%"])
    sg_total=pd.read_csv(ROOT/"outputs/stability/sg_lstm_stability_bootstrap.csv").iloc[0]
    report=f"""# Phase N2.7R Results — Horizon-Aware Adaptive Gating

## Development Meta-OOS comparison

{markdown_table(rows,["Method","Fold2 RMSE (mV)","Fold3 RMSE (mV)","Meta-OOS RMSE (mV)","MAE (mV)","R²"])}

{markdown_table(gate_diag,["Gate","AUC","Mean g","Std g","RMSE improvement vs fixed"])}

## Protocol and firewall

- All gates use frozen expert caches only. LSTM is `P2_causal_sg`; TimesNet is high-fidelity raw `T0_original`.
- Meta-OOS has exactly Fold1→Fold2 and Fold1+Fold2→Fold3 chronological evaluations, averaged over seeds 42/2024/3407. Training-fold performance is not reported as OOS.
- Final Test `[1819.997738, 2256.800000] h` was not used: target rows read = 0; Test expert predictions = false; Test gate predictions/weights = false; Test metrics = false.
- Pointwise winner means `|e_TimesNet| < |e_LSTM|`; equality is assigned to LSTM. Gate C optimizes raw-voltage fusion MSE directly and never updates either frozen expert.

## Findings

- Best main adaptive gate is **{best}**: Meta-OOS RMSE={best_meta['rmse_mV']:.3f} mV, MAE={best_meta['mae_mV']:.3f} mV, R²={best_meta['r2']:.5f}; relative improvements are {imp_lstm*100:.2f}% vs LSTM, {imp_fixed*100:.2f}% vs fixed .25, and {imp_hs*100:.2f}% vs Horizon-Static.
- Winner AUC={auc:.3f}; mean/std gate weight={best_eval.gate_weight.mean():.3f}/{weight_std:.3f}; at 9.5 h mean weight={h95:.3f}; TimesNet top-10% error-tail mean weight={tail10:.3f}.
- Horizon-Static is {'within' if hstatic_close else 'more than'} 1% of the adaptive gate on RMSE. {'The complementarity is therefore primarily horizon dependent; complexity should not be forced.' if hstatic_close else 'State/disagreement information adds material performance beyond horizon-only weights.'}
- OpHrs2 is a diagnostic-only ablation: its Meta-OOS change is {float(pd.DataFrame(age_rows).query('scope == "meta_oos"').delta_rmse_mV_plus_ophrs2_minus_main.iloc[0]):.3f} mV; it does not replace the main protocol.
- Causal SG paired moving-block bootstrap (2,000 repetitions, 19 origins) gives SG−raw RMSE delta {float(sg_total.point_delta_rmse_mV):.3f} mV, 95% CI [{float(sg_total.ci95_lower_mV):.3f}, {float(sg_total.ci95_upper_mV):.3f}] mV. Per-seed/fold rows explicitly retain any 2024/Fold3 or late-region worsening rather than averaging it away.

## Decision

Phase N2.7R status: **{'PASS' if firewall['firewall_status']=='PASS' else 'FAIL'}**.

Phase N2.8 decision: **{'GO' if go else 'NOGO'}**. Pre-registered checks: adaptive vs fixed ≥2%={imp_fixed>=.02}; Fold3 no worse than fixed={fold3_good}; adaptive vs LSTM ≥5%={imp_lstm>=.05}; non-degenerate std≥.01={weight_std>=.01}. No Test evaluation was started.
"""
    write_text(report,"PHASE_N2_7R_RESULTS.md")
    freeze={"phase_status":"PASS" if firewall["firewall_status"]=="PASS" else "FAIL","expert_lstm":"Frozen Phase N2 LSTM + P2_causal_sg","expert_timesnet":"High-fidelity TimesNet T0 original raw input","sg_stability_status":"paired moving-block bootstrap completed; inspect per seed/fold rows for heterogeneity","fixed_fusion_rmse_mV":fixed_meta["rmse_mV"],"horizon_static_rmse_mV":hs_meta["rmse_mV"],"best_adaptive_gate":best,"best_gate_fold2_rmse_mV":summary[best]["fold_2"]["rmse_mV"],"best_gate_fold3_rmse_mV":summary[best]["fold_3"]["rmse_mV"],"best_gate_meta_oos_rmse_mV":best_meta["rmse_mV"],"best_gate_mae_mV":best_meta["mae_mV"],"best_gate_r2":best_meta["r2"],"improvement_vs_lstm_pct":imp_lstm*100,"improvement_vs_fixed_fusion_pct":imp_fixed*100,"improvement_vs_horizon_static_pct":imp_hs*100,"winner_auc":auc,"mean_gate_weight":float(best_eval.gate_weight.mean()),"std_gate_weight":weight_std,"h9_5_gate_weight":h95,"tail_error_gate_weight":tail10,"ophrs2_ablation_effect":float(pd.DataFrame(age_rows).query('scope == "meta_oos"').delta_rmse_mV_plus_ophrs2_minus_main.iloc[0]),"phase_n2_8_decision":"GO" if go else "NOGO","test_performance_accessed":False,"test_firewall":firewall}
    ensure(ROOT/"outputs/frozen_config/PHASE_N2_7R_GATE_DECISION.yaml").write_text(yaml.safe_dump(freeze,allow_unicode=True,sort_keys=False),encoding="utf-8")
    print(json.dumps({"phase_status":freeze["phase_status"],"best_gate":best,"meta_oos_rmse_mV":best_meta["rmse_mV"],"n2_8":freeze["phase_n2_8_decision"]},ensure_ascii=False))


if __name__ == "__main__":
    main()
