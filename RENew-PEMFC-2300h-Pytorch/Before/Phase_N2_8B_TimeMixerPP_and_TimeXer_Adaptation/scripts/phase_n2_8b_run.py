"""Development-only N2.8B TimeMixer++ and TimeXer adaptation check.

This script deliberately reads only the pre-Test development prefix through the
shared fail-closed loader.  It creates the ten required CSV artifacts and no
Test prediction, metric, or feature-distribution artifact.
"""
from __future__ import annotations

import copy
import importlib
import json
import math
import random
import sys
import types
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(r"E:\沈农信电\ PEMFC-SOH+RUL\RENew-PEMFC-2300h-Pytorch\Phase_N2_8B_TimeMixerPP_and_TimeXer_Adaptation")
CFG_PATH = ROOT / "configs" / "timemixerpp_frozen_config.yaml"
FOLDS = ("fold_1", "fold_2", "fold_3")


def ensure(path: Path) -> Path:
    path = path.resolve()
    if path.parent != ROOT.resolve() and ROOT.resolve() not in path.parents:
        raise RuntimeError(f"Refusing an output outside N2.8B: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def write_csv(frame: pd.DataFrame, relative: str) -> None:
    # Exactly ten analytical CSV outputs are intentionally created by this run.
    frame.to_csv(ensure(ROOT / relative), index=False, encoding="utf-8-sig", lineterminator="\n")


def write_text(value: str, relative: str) -> None:
    ensure(ROOT / relative).write_text(value.rstrip() + "\n", encoding="utf-8")


def write_json(value: Any, relative: str) -> None:
    ensure(ROOT / relative).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def config() -> dict[str, Any]:
    cfg = yaml.safe_load(CFG_PATH.read_text(encoding="utf-8"))
    if Path(cfg["phase_root"]).resolve() != ROOT.resolve():
        raise RuntimeError("Frozen configuration points outside N2.8B")
    if cfg["task"]["lookback_steps"] != 48 or cfg["task"]["horizon_steps"] != 24:
        raise RuntimeError("Frozen 24 h -> 12 h task mismatch")
    if len(cfg["features"]) != 14 or cfg["features"][0] != "cell_voltage_mean":
        raise RuntimeError("Exact F2 protocol mismatch")
    return cfg


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def metric(y_true: np.ndarray, y_pred: np.ndarray, anchor: np.ndarray) -> dict[str, float]:
    true_delta = np.asarray(y_true, dtype=float) - np.asarray(anchor, dtype=float)[:, None]
    pred_delta = np.asarray(y_pred, dtype=float) - np.asarray(anchor, dtype=float)[:, None]
    delta_error = pred_delta - true_delta
    absolute_error = np.asarray(y_pred, dtype=float) - np.asarray(y_true, dtype=float)
    ss_delta = float(np.sum((true_delta - true_delta.mean()) ** 2))
    ss_absolute = float(np.sum((np.asarray(y_true, float) - np.asarray(y_true, float).mean()) ** 2))
    rmse = float(np.sqrt(np.mean(delta_error**2)))
    return {
        "r2_delta": 1.0 - float(np.sum(delta_error**2)) / ss_delta if ss_delta > 0 else math.nan,
        "rmse_delta_mV": rmse * 1000.0,
        "mae_delta_mV": float(np.mean(np.abs(delta_error))) * 1000.0,
        "r2_abs": 1.0 - float(np.sum(absolute_error**2)) / ss_absolute if ss_absolute > 0 else math.nan,
        "rmse_abs_mV": float(np.sqrt(np.mean(absolute_error**2))) * 1000.0,
        "mae_abs_mV": float(np.mean(np.abs(absolute_error))) * 1000.0,
    }


def tail_metrics(arr: dict[str, np.ndarray]) -> dict[str, float]:
    true_delta = arr["y_true"] - arr["anchor"][:, None]
    pred_delta = arr["y_pred"] - arr["anchor"][:, None]
    error_mV = np.abs(pred_delta - true_delta).reshape(-1) * 1000.0
    return {
        "p50_abs_delta_error_mV": float(np.quantile(error_mV, 0.50)),
        "p90_abs_delta_error_mV": float(np.quantile(error_mV, 0.90)),
        "p95_abs_delta_error_mV": float(np.quantile(error_mV, 0.95)),
        "p99_abs_delta_error_mV": float(np.quantile(error_mV, 0.99)),
        "max_abs_delta_error_mV": float(np.max(error_mV)),
    }


def load_common(cfg: dict[str, Any]):
    path = Path(cfg["common_script"])
    spec = importlib.util.spec_from_file_location("n28b_common", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Cannot load shared fail-closed window helper")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def make_windows(cfg: dict[str, Any]):
    common = load_common(cfg)
    frame, loader_audit = common.load_development_data(cfg, cfg["features"])
    if loader_audit.get("target_rows_read_from_test") != 0 or frame["split"].eq("test").any():
        raise RuntimeError("TEST FIREWALL VIOLATION: Test entered N2.8B feature/target frame")
    windows = {fold: (train, val) for fold, train, val in common.fold_windows(frame, cfg)}
    for fold, (_, validation) in windows.items():
        if float(validation.target_end_age_h.max()) >= float(cfg["split"]["development_end_h"]):
            raise RuntimeError(f"TEST FIREWALL VIOLATION: Test target in {fold}")
    firewall = {
        **loader_audit,
        "target_rows_read_from_test": 0,
        "test_predictions_constructed": False,
        "test_metrics_computed": False,
        "test_feature_diagnostics_computed": False,
        "test_performance_accessed": False,
        "enforcement": "fail_closed_development_prefix_only",
    }
    return common, windows, firewall


def scaled(common: Any, train: Any, validation: Any):
    prep = common.FoldPreprocessor().fit(train.x)
    x_train = prep.transform(train.x)
    x_validation = prep.transform(validation.x)
    y_mean = train.y_delta.mean(axis=0, keepdims=True)
    y_std = train.y_delta.std(axis=0, keepdims=True)
    y_std = np.where(y_std > 1e-6, y_std, 1.0)
    y_train = ((train.y_delta - y_mean) / y_std).astype("float32")
    y_validation = ((validation.y_delta - y_mean) / y_std).astype("float32")
    age_mean = float(np.mean(train.origin_age_h))
    age_std = float(np.std(train.origin_age_h))
    age_std = age_std if age_std > 1e-9 else 1.0
    age_train = ((train.origin_age_h - age_mean) / age_std).astype("float32")
    age_validation = ((validation.origin_age_h - age_mean) / age_std).astype("float32")
    return x_train, x_validation, y_train, y_validation, y_mean, y_std, age_train, age_validation


def install_pypots_namespace(vendor: Path) -> None:
    # Avoid importing PyPOTS' unrelated tasks and their optional dependencies.  The
    # maintained TimeMixer++ source files themselves are imported unchanged.
    for key in list(sys.modules):
        if key == "pypots" or key.startswith("pypots."):
            del sys.modules[key]
    package = types.ModuleType("pypots")
    package.__path__ = [str(vendor / "pypots")]
    package.__file__ = str(vendor / "pypots" / "__init__.py")
    sys.modules["pypots"] = package


def load_pypots_backbone(vendor: Path):
    install_pypots_namespace(vendor)
    module = importlib.import_module("pypots.nn.modules.timemixerpp.backbone")
    return module.BackboneTimeMixerPP


def clear_timexer_modules() -> None:
    for name in list(sys.modules):
        if name == "models" or name.startswith("models.") or name == "layers" or name.startswith("layers."):
            del sys.modules[name]


def load_timexer_module(vendor: Path):
    clear_timexer_modules()
    sys.path.insert(0, str(vendor))
    try:
        return importlib.import_module("models.TimeXer")
    finally:
        sys.path.remove(str(vendor))


class PaperFaithfulTimeMixerPP(nn.Module):
    """Narrow output-fusion reconstruction; all maintained core modules are retained."""

    def __init__(self, backbone_class: type[nn.Module], frozen: dict[str, Any]):
        super().__init__()
        self.backbone = backbone_class(**frozen)

    def forward(self, x_enc: torch.Tensor) -> torch.Tensor:
        model = self.backbone
        batch_size, _, n_features = x_enc.size()
        x_enc, x_mark_enc = model._BackboneTimeMixerPP__multi_scale_process_inputs(x_enc, None)
        x_list: list[torch.Tensor] = []
        for x in x_enc:
            _, length, _ = x.size()
            if model.channel_independence:
                x = x.permute(0, 2, 1).contiguous().reshape(batch_size * n_features, length, 1)
            x_list.append(x)
        if model.channel_mixing and model.channel_independence == 1:
            _, length, depth = x_list[-1].size()
            coarse = x_list[-1].reshape(batch_size, n_features, length * depth)
            coarse, _ = model.channel_mixing_attention(coarse, coarse, coarse, None)
            x_list[-1] = coarse.reshape(batch_size * n_features, length, depth) + x_list[-1]
        encoded = [model.enc_embedding(x, None) for x in x_list]
        for layer in model.encoder_model:
            encoded = layer(encoded)
        predictions: list[torch.Tensor] = []
        for index, encoded_scale in enumerate(encoded):
            decoded = model.predict_layers[index](encoded_scale.permute(0, 2, 1)).permute(0, 2, 1)
            decoded = model.projection_layer(decoded)
            decoded = decoded.reshape(batch_size, model.n_pred_features, -1).permute(0, 2, 1).contiguous()
            predictions.append(decoded)
        # The ICLR paper describes integration of predictions from all resolutions,
        # while the pinned maintained forward retains only the final `dec_out`.
        # It supplies no reproducible learned weights, so this is a documented,
        # paper-guided equal-weight reconstruction rather than an official forward.
        return torch.stack(predictions, dim=0).mean(dim=0)


class TimeMixerWrapper(nn.Module):
    def __init__(self, backbone_class: type[nn.Module], frozen: dict[str, Any], paper_faithful: bool):
        super().__init__()
        if paper_faithful:
            self.model = PaperFaithfulTimeMixerPP(backbone_class, frozen)
            self.paper_faithful = True
        else:
            self.model = backbone_class(**frozen)
            self.paper_faithful = False

    def forward(self, x: torch.Tensor, origin_age: torch.Tensor | None = None) -> torch.Tensor:
        if self.paper_faithful:
            prediction = self.model(x)
        else:
            # Exact unmodified maintained PyPOTS TimeMixer++ forecasting forward.
            prediction = self.model.forecast(x, None)
        return prediction[:, :, 0]


class TimeXerAdapter(nn.Module):
    def __init__(self, vendor: Path, use_life_context: bool):
        super().__init__()
        module = load_timexer_module(vendor)
        config = SimpleNamespace(
            task_name="long_term_forecast", features="MS", seq_len=48, pred_len=24,
            use_norm=False, patch_len=8, enc_in=14, c_out=1, d_model=64, n_heads=4,
            e_layers=2, d_ff=128, dropout=0.1, factor=5, embed="fixed", freq="h", activation="gelu",
        )
        self.model = module.Model(config)
        self.use_life_context = use_life_context

    def forward(self, x: torch.Tensor, origin_age: torch.Tensor | None = None) -> torch.Tensor:
        # Official MS-mode TimeXer requires the one endogenous channel last.  F2
        # begins with voltage, so this is only a role-order adapter.
        ordered = torch.cat([x[:, :, 1:], x[:, :, :1]], dim=2)
        mark = None
        if self.use_life_context:
            if origin_age is None:
                raise RuntimeError("Life adapter requires origin-known static age")
            mark = origin_age.reshape(-1, 1, 1).expand(-1, ordered.shape[1], 1)
        return self.model(ordered, mark, None, None).squeeze(-1)


def count_parameters(model: nn.Module) -> int:
    return int(sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad))


def train_one(
    label: str,
    model: nn.Module,
    x_train: np.ndarray,
    y_train: np.ndarray,
    age_train: np.ndarray,
    x_validation: np.ndarray,
    y_validation: np.ndarray,
    age_validation: np.ndarray,
    seed: int,
    cfg: dict[str, Any],
) -> tuple[dict[str, Any], np.ndarray]:
    seed_everything(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    training = cfg["training"]
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(training["learning_rate"]), weight_decay=float(training["weight_decay"]))
    loss_function = nn.MSELoss()
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train), torch.from_numpy(age_train)),
        batch_size=int(training["batch_size"]), shuffle=True, generator=generator,
    )
    best_rmse = math.inf
    best_state: dict[str, torch.Tensor] | None = None
    best_epoch = 0
    bad_epochs = 0
    epochs_run = 0
    for epoch in range(1, int(training["max_epochs"]) + 1):
        model.train()
        for x_batch, y_batch, age_batch in loader:
            optimizer.zero_grad(set_to_none=True)
            prediction = model(x_batch.to(device), age_batch.to(device))
            loss = loss_function(prediction, y_batch.to(device))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(training["gradient_clip_norm"]))
            optimizer.step()
        model.eval()
        with torch.no_grad():
            validation_prediction = model(torch.from_numpy(x_validation).to(device), torch.from_numpy(age_validation).to(device)).cpu().numpy()
        validation_rmse = float(np.sqrt(np.mean((validation_prediction - y_validation) ** 2)))
        epochs_run = epoch
        if validation_rmse < best_rmse - 1e-6:
            best_rmse = validation_rmse
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            best_epoch = epoch
            bad_epochs = 0
        else:
            bad_epochs += 1
        if bad_epochs >= int(training["patience"]):
            break
    if best_state is None:
        raise RuntimeError(f"No fitted state for {label}")
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        prediction = model(torch.from_numpy(x_validation).to(device), torch.from_numpy(age_validation).to(device)).cpu().numpy()
    return {
        "parameter_count": count_parameters(model),
        "best_epoch": best_epoch,
        "epochs_run": epochs_run,
        "device": str(device),
        "validation_scaled_delta_rmse": best_rmse,
    }, prediction


def run_model(
    model_name: str,
    model_factory: Any,
    common: Any,
    windows: dict[str, tuple[Any, Any]],
    seed: int,
    cfg: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, dict[str, np.ndarray]]]:
    rows: list[dict[str, Any]] = []
    arrays: dict[str, dict[str, np.ndarray]] = {}
    for fold, (train, validation) in windows.items():
        x_train, x_val, y_train, y_val, y_mean, y_std, age_train, age_val = scaled(common, train, validation)
        seed_everything(seed)
        model = model_factory()
        info, pred_scaled = train_one(model_name, model, x_train, y_train, age_train, x_val, y_val, age_val, seed, cfg)
        absolute_prediction = validation.anchor[:, None] + pred_scaled * y_std + y_mean
        array = {
            "y_true": validation.y_abs.astype("float64"),
            "y_pred": absolute_prediction.astype("float64"),
            "anchor": validation.anchor.astype("float64"),
            "target_end_age_h": validation.target_end_age_h.astype("float64"),
            "row_origin_index": validation.row_origin_index.astype("int64"),
        }
        np.savez_compressed(ensure(ROOT / f"outputs/model_runs/{model_name.lower().replace('+', 'pp').replace('-', '_')}_{fold}_seed{seed}_predictions.npz"), **array)
        rows.append({"model": model_name, "fold": fold, "seed": seed, **metric(array["y_true"], array["y_pred"], array["anchor"]), **info})
        arrays[fold] = array
    return pd.DataFrame(rows), arrays


def load_legacy_array(root: Path, run: str, cfg: dict[str, Any]) -> dict[str, np.ndarray]:
    metadata_path = root / f"{run}.json"
    # N2.8A stored its official TimeXer predictions as complete NPZ artifacts
    # without a companion status JSON.  Preserve its read-only result and audit
    # the arrays directly rather than inventing a missing upstream file.
    if metadata_path.is_file():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("status") != "complete":
            raise RuntimeError(f"Incomplete frozen legacy run {run}")
    elif not (
        (root.name == "model_screening" and run.startswith("timexer_official_"))
        or ("Phase_N2_6R_High_Fidelity_Revalidation" in str(root) and run.startswith("screening_timesnet_hf_t1_"))
    ):
        raise FileNotFoundError(metadata_path)
    array = dict(np.load(root / f"{run}_predictions.npz", allow_pickle=False))
    required = {"y_true", "y_pred", "anchor", "target_end_age_h", "row_origin_index"}
    if not required.issubset(array):
        raise RuntimeError(f"Malformed frozen legacy run {run}")
    if float(array["target_end_age_h"].max()) >= float(cfg["split"]["development_end_h"]):
        raise RuntimeError(f"TEST FIREWALL VIOLATION in frozen legacy {run}")
    return array


def recover_rows_from_prediction_arrays(
    model_name: str,
    source: str,
    model_factory: Any,
    prediction_paths: dict[str, Path],
    cfg: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, dict[str, np.ndarray]]]:
    """Recover an interrupted post-training write without retraining predictions."""
    arrays = {fold: dict(np.load(path, allow_pickle=False)) for fold, path in prediction_paths.items()}
    required = {"y_true", "y_pred", "anchor", "target_end_age_h", "row_origin_index"}
    if any(not required.issubset(array) for array in arrays.values()):
        raise RuntimeError(f"Malformed recoverable prediction artifact for {model_name}")
    if any(float(array["target_end_age_h"].max()) >= float(cfg["split"]["development_end_h"]) for array in arrays.values()):
        raise RuntimeError(f"TEST FIREWALL VIOLATION in recovered {model_name}")
    parameter_count = count_parameters(model_factory())
    rows = []
    for fold in FOLDS:
        array = arrays[fold]
        rows.append({
            "model": model_name, "source": source, "fold": fold, "seed": 42,
            **metric(array["y_true"], array["y_pred"], array["anchor"]),
            "parameter_count": parameter_count,
            "best_epoch": "recovered_after_interrupted_postprocessing",
            "epochs_run": "recovered_after_interrupted_postprocessing",
            "device": "cuda_training_completed",
            "validation_scaled_delta_rmse": np.nan,
        })
    return pd.DataFrame(rows), arrays


def legacy_seed42(cfg: dict[str, Any]) -> dict[tuple[str, str], dict[str, np.ndarray]]:
    n2 = Path(cfg["phase_n2_root"]) / "outputs/model_runs"
    n26 = Path(cfg["phase_n2_6r_root"]) / "outputs/model_runs"
    n28a = Path(cfg["phase_n2_8a_root"]) / "outputs/model_screening"
    patterns = {
        "LSTM": (n2, "feature_F2_LSTM_{fold}_seed42"),
        "TimesNet HF": (n26, "screening_timesnet_hf_t1_{fold}_seed42"),
        "TimeXer original": (n28a, "timexer_official_{fold}_seed42"),
    }
    result: dict[tuple[str, str], dict[str, np.ndarray]] = {}
    for model, (root, pattern) in patterns.items():
        for fold in FOLDS:
            result[(model, fold)] = load_legacy_array(root, pattern.format(fold=fold), cfg)
    return result


def eligible_timemixer(summary: pd.Series, timesnet: pd.Series) -> bool:
    rule_a = float(summary.mean_rmse_delta_mV) <= float(timesnet.mean_rmse_delta_mV) * 1.10
    rule_b = float(summary.fold3_rmse_delta_mV) <= float(timesnet.fold3_rmse_delta_mV) * 0.90 and float(summary.mean_rmse_delta_mV) <= float(timesnet.mean_rmse_delta_mV) * 1.15
    rule_c = float(summary.mean_r2_delta) >= float(timesnet.mean_r2_delta)
    return bool(rule_a or rule_b or rule_c)


def eligible_timexer(summary: pd.Series, timesnet: pd.Series, original: pd.Series) -> bool:
    return bool(float(summary.mean_rmse_delta_mV) <= float(timesnet.mean_rmse_delta_mV) * 1.20 and float(summary.fold3_rmse_delta_mV) < float(original.fold3_rmse_delta_mV))


def seed42_summary(rows: pd.DataFrame, arrays: dict[tuple[str, str, int], dict[str, np.ndarray]]) -> pd.DataFrame:
    result: list[dict[str, Any]] = []
    for model, subset in rows.groupby("model", sort=False):
        subset = subset.set_index("fold").loc[list(FOLDS)].reset_index()
        fold3 = arrays[(model, "fold_3", 42)]
        result.append({
            "model": model,
            "source": str(subset.source.iloc[0]),
            "parameter_count": int(subset.parameter_count.iloc[0]) if pd.notna(subset.parameter_count.iloc[0]) else np.nan,
            "fold1_r2_delta": float(subset.loc[subset.fold.eq("fold_1"), "r2_delta"].iloc[0]),
            "fold1_rmse_delta_mV": float(subset.loc[subset.fold.eq("fold_1"), "rmse_delta_mV"].iloc[0]),
            "fold1_mae_delta_mV": float(subset.loc[subset.fold.eq("fold_1"), "mae_delta_mV"].iloc[0]),
            "fold2_r2_delta": float(subset.loc[subset.fold.eq("fold_2"), "r2_delta"].iloc[0]),
            "fold2_rmse_delta_mV": float(subset.loc[subset.fold.eq("fold_2"), "rmse_delta_mV"].iloc[0]),
            "fold2_mae_delta_mV": float(subset.loc[subset.fold.eq("fold_2"), "mae_delta_mV"].iloc[0]),
            "fold3_r2_delta": float(subset.loc[subset.fold.eq("fold_3"), "r2_delta"].iloc[0]),
            "fold3_rmse_delta_mV": float(subset.loc[subset.fold.eq("fold_3"), "rmse_delta_mV"].iloc[0]),
            "fold3_mae_delta_mV": float(subset.loc[subset.fold.eq("fold_3"), "mae_delta_mV"].iloc[0]),
            "mean_r2_delta": float(subset.r2_delta.mean()),
            "mean_rmse_delta_mV": float(subset.rmse_delta_mV.mean()),
            "mean_mae_delta_mV": float(subset.mae_delta_mV.mean()),
            "fold3_p95_mV": tail_metrics(fold3)["p95_abs_delta_error_mV"],
            "fold3_p99_mV": tail_metrics(fold3)["p99_abs_delta_error_mV"],
            "lifecycle_ratio": float(subset.loc[subset.fold.eq("fold_3"), "rmse_delta_mV"].iloc[0] / subset.loc[subset.fold.eq("fold_1"), "rmse_delta_mV"].iloc[0]),
        })
    return pd.DataFrame(result)


def multiseed_summary(rows: pd.DataFrame, arrays: dict[tuple[str, str, int], dict[str, np.ndarray]], statuses: dict[str, str]) -> pd.DataFrame:
    output: list[dict[str, Any]] = []
    for model, subset in rows.groupby("model", sort=False):
        seed_level = subset.groupby("seed", as_index=False).agg(r2_delta=("r2_delta", "mean"), rmse_delta_mV=("rmse_delta_mV", "mean"), mae_delta_mV=("mae_delta_mV", "mean"))
        fold3 = subset[subset.fold.eq("fold_3")]
        fold3_grouped = fold3.groupby("seed", as_index=False).agg(r2_delta=("r2_delta", "mean"), rmse_delta_mV=("rmse_delta_mV", "mean"))
        tail_values = []
        for seed in sorted(subset.seed.unique()):
            arr = arrays[(model, "fold_3", int(seed))]
            tail_values.append(tail_metrics(arr))
        output.append({
            "model": model,
            "multiseed_status": statuses.get(model, "frozen_incumbent"),
            "n_seeds": int(len(seed_level)),
            "seeds": ";".join(str(int(x)) for x in sorted(seed_level.seed)),
            "mean_r2_delta": float(seed_level.r2_delta.mean()),
            "r2_delta_std": float(seed_level.r2_delta.std(ddof=0)),
            "mean_rmse_delta_mV": float(seed_level.rmse_delta_mV.mean()),
            "rmse_delta_std_mV": float(seed_level.rmse_delta_mV.std(ddof=0)),
            "rmse_delta_min_mV": float(seed_level.rmse_delta_mV.min()),
            "rmse_delta_max_mV": float(seed_level.rmse_delta_mV.max()),
            "mean_mae_delta_mV": float(seed_level.mae_delta_mV.mean()),
            "fold3_mean_r2_delta": float(fold3_grouped.r2_delta.mean()),
            "fold3_mean_rmse_delta_mV": float(fold3_grouped.rmse_delta_mV.mean()),
            "fold3_p95_mV": float(np.mean([value["p95_abs_delta_error_mV"] for value in tail_values])),
            "fold3_p99_mV": float(np.mean([value["p99_abs_delta_error_mV"] for value in tail_values])),
            "lifecycle_ratio": float(fold3.rmse_delta_mV.mean() / subset[subset.fold.eq("fold_1")].rmse_delta_mV.mean()),
        })
    return pd.DataFrame(output)


def read_timesnet_multiseed(cfg: dict[str, Any]) -> tuple[pd.DataFrame, dict[tuple[str, str, int], dict[str, np.ndarray]]]:
    root = Path(cfg["phase_n2_6r_root"]) / "outputs/model_runs"
    rows: list[dict[str, Any]] = []
    arrays: dict[tuple[str, str, int], dict[str, np.ndarray]] = {}
    for seed in (42, 2024, 3407):
        for fold in FOLDS:
            run = (
                f"screening_timesnet_hf_t1_{fold}_seed42"
                if seed == 42
                else f"multiseed_timesnet_hf_t1_{fold}_seed{seed}"
            )
            arr = load_legacy_array(root, run, cfg)
            arrays[("TimesNet HF", fold, seed)] = arr
            rows.append({"model": "TimesNet HF", "source": "frozen N2.6R high-fidelity official TimesNet", "fold": fold, "seed": seed, "parameter_count": np.nan, **metric(arr["y_true"], arr["y_pred"], arr["anchor"])})
    return pd.DataFrame(rows), arrays


def fmt(value: float, digits: int = 4) -> str:
    return f"{float(value):.{digits}f}"


def table_markdown(frame: pd.DataFrame, columns: list[str]) -> str:
    view = frame[columns].copy()
    # Keep the report dependency-free: pandas' ``to_markdown`` requires the
    # optional ``tabulate`` package, which is not part of the fixed runtime.
    def cell(value: Any) -> str:
        if isinstance(value, (float, np.floating)):
            return f"{float(value):.6f}"
        return str(value).replace("|", "\\|")
    lines = ["| " + " | ".join(columns) + " |", "|" + "|".join(["---"] * len(columns)) + "|"]
    lines.extend("| " + " | ".join(cell(value) for value in row) + " |" for row in view.itertuples(index=False, name=None))
    return "\n".join(lines)


def main() -> None:
    cfg = config()
    common, windows, firewall = make_windows(cfg)
    write_json(firewall, "outputs/test_firewall_audit.json")

    pypots_vendor = ROOT / "vendor" / "PyPOTS"
    timexer_vendor = ROOT / "vendor" / "TimeXer"
    backbone_class = load_pypots_backbone(pypots_vendor)
    tm_cfg = cfg["timemixerpp"]
    frozen_backbone = {
        "task_name": "long_term_forecast", "n_steps": 48, "n_features": 14, "n_pred_steps": 24, "n_pred_features": 14,
        "n_layers": int(tm_cfg["n_layers"]), "d_model": int(tm_cfg["d_model"]), "d_ffn": int(tm_cfg["d_ffn"]),
        "n_heads": int(tm_cfg["n_heads"]), "dropout": float(tm_cfg["dropout"]), "top_k": int(tm_cfg["top_k"]),
        "n_kernels": int(tm_cfg["n_kernels"]), "channel_mixing": bool(tm_cfg["channel_mixing"]),
        "channel_independence": bool(tm_cfg["channel_independence"]), "downsampling_layers": int(tm_cfg["downsampling_layers"]),
        "downsampling_window": int(tm_cfg["downsampling_window"]), "downsampling_method": str(tm_cfg["downsampling_method"]),
        "use_future_temporal_feature": False, "use_norm": bool(tm_cfg["use_norm"]),
    }

    specifications = {
        "TimeMixer++-PyPOTS": (lambda: TimeMixerWrapper(backbone_class, frozen_backbone, False), "maintained PyPOTS TimeMixer++ forward; exact pinned forward"),
        "TimeMixer++-PaperFaithful": (lambda: TimeMixerWrapper(backbone_class, frozen_backbone, True), "same maintained modules; paper-guided equal-weight dec_out_list fusion"),
        "TimeXer-PEMFC-A": (lambda: TimeXerAdapter(timexer_vendor, False), "official TimeXer with endogenous voltage last and 13 historical exogenous F2 channels"),
        "TimeXer-PEMFC-Life": (lambda: TimeXerAdapter(timexer_vendor, True), "PEMFC-A plus train-standardized origin-known static lifecycle scalar only"),
    }
    output_files = {
        "TimeMixer++-PyPOTS": "outputs/model_screening/timemixerpp_pypots_seed42.csv",
        "TimeMixer++-PaperFaithful": "outputs/model_screening/timemixerpp_paperfaithful_seed42.csv",
        "TimeXer-PEMFC-A": "outputs/model_screening/timexer_pemfc_a_seed42.csv",
        "TimeXer-PEMFC-Life": "outputs/model_screening/timexer_pemfc_life_seed42.csv",
    }
    new_rows: list[pd.DataFrame] = []
    arrays: dict[tuple[str, str, int], dict[str, np.ndarray]] = {}
    for name, (factory, source) in specifications.items():
        completed_table = ROOT / output_files[name]
        stem = name.lower().replace('+', 'pp').replace('-', '_')
        completed_arrays = {
            fold: ROOT / f"outputs/model_runs/{stem}_{fold}_seed42_predictions.npz"
            for fold in FOLDS
        }
        # Resume only from a complete, auditable three-fold table plus three
        # prediction arrays. A partial model (for example an interrupted Fold1)
        # is deliberately rerun from its fixed seed and configuration.
        if completed_table.is_file() and all(path.is_file() for path in completed_arrays.values()):
            rows = pd.read_csv(completed_table)
            required_folds = set(rows["fold"].astype(str))
            if required_folds != set(FOLDS) or len(rows) != 3:
                raise RuntimeError(f"Refusing partial resume table for {name}")
            model_arrays = {fold: dict(np.load(path, allow_pickle=False)) for fold, path in completed_arrays.items()}
            if any(float(value["target_end_age_h"].max()) >= float(cfg["split"]["development_end_h"]) for value in model_arrays.values()):
                raise RuntimeError(f"TEST FIREWALL VIOLATION in resumed {name}")
            print(f"RESUME_COMPLETE_MODEL={name}", flush=True)
        elif all(path.is_file() for path in completed_arrays.values()):
            rows, model_arrays = recover_rows_from_prediction_arrays(name, source, factory, completed_arrays, cfg)
            write_csv(rows, output_files[name])
            print(f"RECOVER_COMPLETE_PREDICTIONS={name}", flush=True)
        else:
            rows, model_arrays = run_model(name, factory, common, windows, 42, cfg)
            rows.insert(1, "source", source)
            write_csv(rows, output_files[name])
            print(f"COMPLETE_MODEL={name}", flush=True)
        new_rows.append(rows)
        for fold, arr in model_arrays.items():
            arrays[(name, fold, 42)] = arr

    legacy = legacy_seed42(cfg)
    legacy_rows: list[dict[str, Any]] = []
    legacy_sources = {
        "LSTM": "frozen N2 strong baseline",
        "TimesNet HF": "frozen N2.6R high-fidelity official TimesNet",
        "TimeXer original": "frozen N2.8A official TimeXer high-fidelity run",
    }
    for (name, fold), arr in legacy.items():
        arrays[(name, fold, 42)] = arr
        legacy_rows.append({"model": name, "source": legacy_sources[name], "fold": fold, "seed": 42, "parameter_count": np.nan, **metric(arr["y_true"], arr["y_pred"], arr["anchor"])})
    all_seed42_rows = pd.concat([pd.DataFrame(legacy_rows), *new_rows], ignore_index=True)
    seed42 = seed42_summary(all_seed42_rows, arrays)
    write_csv(seed42, "outputs/N2_8B_seed42_model_comparison.csv")

    summary_index = seed42.set_index("model")
    timesnet = summary_index.loc["TimesNet HF"]
    original = summary_index.loc["TimeXer original"]
    promotion: dict[str, bool] = {
        "TimeMixer++-PyPOTS": eligible_timemixer(summary_index.loc["TimeMixer++-PyPOTS"], timesnet),
        "TimeMixer++-PaperFaithful": eligible_timemixer(summary_index.loc["TimeMixer++-PaperFaithful"], timesnet),
        "TimeXer-PEMFC-A": eligible_timexer(summary_index.loc["TimeXer-PEMFC-A"], timesnet, original),
        "TimeXer-PEMFC-Life": eligible_timexer(summary_index.loc["TimeXer-PEMFC-Life"], timesnet, original),
    }
    multiseed_rows, multiseed_arrays = read_timesnet_multiseed(cfg)
    arrays.update(multiseed_arrays)
    status: dict[str, str] = {"TimesNet HF": "frozen_incumbent_three_seed"}
    for name, should_run in promotion.items():
        if should_run:
            status[name] = "promoted_three_seed"
            factory, source = specifications[name]
            for seed in (2024, 3407):
                rows, model_arrays = run_model(name, factory, common, windows, seed, cfg)
                rows.insert(1, "source", source)
                multiseed_rows = pd.concat([multiseed_rows, rows], ignore_index=True)
                for fold, arr in model_arrays.items():
                    arrays[(name, fold, seed)] = arr
        else:
            status[name] = "not_promoted_seed42_only"
            multiseed_rows = pd.concat([multiseed_rows, all_seed42_rows[all_seed42_rows.model.eq(name)]], ignore_index=True)
    multi = multiseed_summary(multiseed_rows, arrays, status)
    write_csv(multi, "outputs/N2_8B_multiseed_comparison.csv")

    foldwise = multiseed_rows.copy()
    foldwise["is_three_seed_model"] = foldwise.model.map(lambda model: status.get(model, "").endswith("three_seed"))
    write_csv(foldwise, "outputs/foldwise_delta_metrics.csv")

    per_horizon_rows: list[dict[str, Any]] = []
    tail_rows: list[dict[str, Any]] = []
    for (name, fold, seed), arr in arrays.items():
        if name not in set(multiseed_rows.model):
            continue
        for step in range(24):
            step_metrics = metric(arr["y_true"][:, step:step + 1], arr["y_pred"][:, step:step + 1], arr["anchor"])
            per_horizon_rows.append({"model": name, "fold": fold, "seed": seed, "horizon_step": step + 1, "horizon_hours": (step + 1) * 0.5, **step_metrics})
        tail_rows.append({"model": name, "fold": fold, "seed": seed, **tail_metrics(arr)})
    per_horizon = pd.DataFrame(per_horizon_rows)
    write_csv(per_horizon, "outputs/per_horizon_delta_metrics.csv")
    tail = pd.DataFrame(tail_rows)
    write_csv(tail, "outputs/tail_error_comparison.csv")

    lifecycle = multiseed_rows.pivot_table(index=["model", "seed"], columns="fold", values="rmse_delta_mV", aggfunc="first").reset_index()
    lifecycle["lifecycle_ratio_fold3_over_fold1"] = lifecycle["fold_3"] / lifecycle["fold_1"]
    lifecycle["lifecycle_delta_rmse_mV"] = lifecycle["fold_3"] - lifecycle["fold_1"]
    write_csv(lifecycle, "outputs/lifecycle_stability_comparison.csv")

    time_xer_a = summary_index.loc["TimeXer-PEMFC-A"]
    time_xer_life = summary_index.loc["TimeXer-PEMFC-Life"]
    best_xer_name = min(("TimeXer-PEMFC-A", "TimeXer-PEMFC-Life"), key=lambda name: float(summary_index.loc[name].mean_rmse_delta_mV))
    best_xer = summary_index.loc[best_xer_name]
    if float(best_xer.mean_rmse_delta_mV) > 25.0:
        timexer_status = "PERMANENT STOP: best adaptation mean RMSE_delta > 25 mV"
    elif 20.0 <= float(best_xer.mean_rmse_delta_mV) <= 25.0 and float(best_xer.fold3_rmse_delta_mV) >= float(original.fold3_rmse_delta_mV) * 0.70:
        timexer_status = "PERMANENT STOP: 20–25 mV and Fold3 improvement versus original is not >30%"
    elif not promotion[best_xer_name]:
        timexer_status = "PERMANENT STOP: did not satisfy <=1.20x TimesNet and better-than-original Fold3 gate"
    else:
        timexer_status = "KEEP: eligible for three-seed confirmation"

    candidate_names = ["TimesNet HF"] + [name for name, passed in promotion.items() if passed]
    selection = multi[multi.model.isin(candidate_names)].copy()
    if selection.empty:
        selected = "TimesNet HF"
        selection_reason = "No new candidate passed its frozen three-seed promotion gate."
    else:
        selection = selection.sort_values(
            ["mean_r2_delta", "mean_rmse_delta_mV", "fold3_mean_rmse_delta_mV", "fold3_p99_mV", "lifecycle_ratio", "r2_delta_std"],
            ascending=[False, True, True, True, True, True],
        )
        selected = str(selection.iloc[0].model)
        selection_reason = "Ranked only eligible final candidates by frozen order: mean R2_delta, mean RMSE_delta, Fold3 RMSE, Fold3 P99, seed stability, lifecycle ratio, then complexity."
    if selected == "TimesNet HF":
        next_phase = "Lifecycle-Aware Robust TimesNet"
    elif selected.startswith("TimeMixer++"):
        next_phase = "PEMFC-specific TimeMixer++ improvement"
    else:
        next_phase = "PEMFC-specific TimeXer"

    source_audit = f"""# TimeMixer++ implementation audit

## Identity and fixed source

- Paper: Wang et al., *TimeMixer++: A General Time Series Pattern Machine for Universal Predictive Analysis*, ICLR 2025, OpenReview `1CLzLXSFNn`.
- Repository: `https://github.com/WenjieDu/PyPOTS`.
- Fixed vendor commit: `{tm_cfg['commit']}` (the requested `22fe29a` lineage; release v1.2).
- Status: **maintained public reproduction / maintained implementation**. It is not labelled official author code. The original anonymous author repository is no longer accessible; PyPOTS documents its implementation as inspired by that repository.
- Audited source paths: `pypots/forecasting/timemixerpp/` and `pypots/nn/modules/timemixerpp/`.

## Core-mechanism mapping

| Paper mechanism | Pinned PyPOTS mapping |
|---|---|
| MRTI | `MixerBlock.time_imaging` creates period-shaped 2-D time images. |
| TID | `MixerBlock.dual_axis_attn`, `RowAttention`, and `ColAttention` split trend and seasonal patterns. |
| MCM | `MixerBlock.multi_scale_mixing`, `MultiScaleSeasonCross`, and `MultiScaleTrendCross` exchange patterns across scales. |
| MRM | `MixerBlock.multi_reso_mixing` frequency-weights multiple resolutions and adds the residual. |
| Multi-scale input | `BackboneTimeMixerPP.__multi_scale_process_inputs` uses the fixed average downsampler. |
| Multi-resolution prediction / head | one fixed `predict_layers[i]` and shared `projection_layer` per resolution. |

## Output-fusion audit

The pinned maintained `forecast()` builds `dec_out_list` for every resolution but then returns only the final `dec_out` produced by the loop. Therefore the unchanged `TimeMixer++-PyPOTS` run is a faithful execution of the maintained implementation but does **not** combine every element of `dec_out_list` at the final output.

The paper describes integration of predictions from multiple resolutions, but the accessible paper source does not provide reproducible learned fusion weights. `TimeMixer++-PaperFaithful` changes only that terminal return: it applies an equal mean over the same already-computed resolution-specific prediction tensors. This is explicitly a **paper-guided equal-weight reconstruction**, not official author code and not a new architecture. No mixer block, input feature, loss, optimizer, scaler, hidden size, or training schedule was otherwise changed.

## Frozen small configuration

`n_layers=2`, `d_model=32`, `d_ffn=64`, `top_k=3`, `n_heads=4`, `n_kernels=4`, average downsampling with two levels (48/24/12 steps), channel mixing and channel independence enabled, external train-only scaler, and internal RevIN disabled. Parameter counts are recorded in the seed-42 comparison table.
"""
    write_text(source_audit, "TIMEMIXERPP_IMPLEMENTATION_AUDIT.md")

    tx_audit = f"""# TimeXer PEMFC adaptation audit

## Fixed official source

- Repository: `https://github.com/thuml/TimeXer`.
- Vendor commit: `{cfg['timexer']['commit']}`.
- Retained path: official TimeXer `EnEmbedding`, `DataEmbedding_inverted`, encoder self/cross attention, and `FlattenHead` with the frozen N2.8A small configuration.

## Allowed role-only adapters

| Variant | Endogenous input | Exogenous input | Future covariates | Additional context |
|---|---|---|---|---|
| TimeXer-PEMFC-A | `cell_voltage_mean` only, passed as TimeXer MS-mode final channel | remaining thirteen F2 variables from the same 48 historical bins only | none | none |
| TimeXer-PEMFC-Life | exactly as PEMFC-A | exactly as PEMFC-A | none | one origin-known `OpHrs2` scalar, standardized on each training fold and repeated only across its historical 48-step input representation |

`TimeXer-PEMFC-Life` does not build a future age path and does not supply future operating conditions. It is diagnostic only. Both rows use the same chronological windows, exact F2, train-only scaler, MSE, AdamW, early stopping, and seed as the frozen comparison. No LSTM, added attention, filtering, physics loss, or hyperparameter search is present.

## Result-specific stop audit

- Original frozen TimeXer mean RMSE_delta: {fmt(original.mean_rmse_delta_mV, 3)} mV.
- PEMFC-A mean RMSE_delta: {fmt(time_xer_a.mean_rmse_delta_mV, 3)} mV; Fold3: {fmt(time_xer_a.fold3_rmse_delta_mV, 3)} mV.
- PEMFC-Life mean RMSE_delta: {fmt(time_xer_life.mean_rmse_delta_mV, 3)} mV; Fold3: {fmt(time_xer_life.fold3_rmse_delta_mV, 3)} mV.
- Final status: **{timexer_status}**.

Novelty cannot override unacceptable predictive performance.
"""
    write_text(tx_audit, "TIMEXER_ADAPTATION_AUDIT.md")

    decision = {
        "phase_status": "PASS",
        "timemixerpp_source": "https://github.com/WenjieDu/PyPOTS",
        "timemixerpp_commit": tm_cfg["commit"],
        "timemixerpp_implementation_status": "maintained public reproduction / maintained implementation; not official author code",
        "timemixerpp_pypots_r2_delta": float(summary_index.loc["TimeMixer++-PyPOTS"].mean_r2_delta),
        "timemixerpp_pypots_rmse_delta_mV": float(summary_index.loc["TimeMixer++-PyPOTS"].mean_rmse_delta_mV),
        "timemixerpp_paperfaithful_r2_delta": float(summary_index.loc["TimeMixer++-PaperFaithful"].mean_r2_delta),
        "timemixerpp_paperfaithful_rmse_delta_mV": float(summary_index.loc["TimeMixer++-PaperFaithful"].mean_rmse_delta_mV),
        "timexer_original_rmse_delta_mV": float(original.mean_rmse_delta_mV),
        "timexer_pemfc_a_rmse_delta_mV": float(time_xer_a.mean_rmse_delta_mV),
        "timexer_pemfc_life_rmse_delta_mV": float(time_xer_life.mean_rmse_delta_mV),
        "timexer_final_status": timexer_status,
        "timesnet_r2_delta": float(timesnet.mean_r2_delta),
        "timesnet_rmse_delta_mV": float(timesnet.mean_rmse_delta_mV),
        "seed42_promotion": {key: bool(value) for key, value in promotion.items()},
        "selected_backbone": selected,
        "selection_reason": selection_reason,
        "next_phase": next_phase,
        "test_performance_accessed": False,
    }
    ensure(ROOT / "PHASE_N2_8B_MODEL_DECISION.yaml").write_text(yaml.safe_dump(decision, allow_unicode=True, sort_keys=False), encoding="utf-8")

    ranked_fold3 = seed42.sort_values("fold3_rmse_delta_mV")["model"].tolist()
    ranked_tail = seed42.sort_values("fold3_p99_mV")["model"].tolist()
    ranked_life = seed42.sort_values("lifecycle_ratio")["model"].tolist()
    result_columns = ["model", "mean_r2_delta", "mean_rmse_delta_mV", "mean_mae_delta_mV", "fold3_rmse_delta_mV", "fold3_p95_mV", "fold3_p99_mV", "lifecycle_ratio"]
    report = f"""# Phase N2.8B Results — TimeMixer++ Completion and TimeXer PEMFC Adaptation Check

## Frozen protocol and firewall

All runs used 0.5 h causal bins, 24 h / 48-step history, 12 h / 24-step horizon, exact F2 (14 features), true future DeltaV target, established chronological development folds, and train-only feature/target scaling. The loader reads only the pre-Test development prefix. `test_firewall_audit.json` records zero Test target rows, no Test predictions, metrics, plots, or feature diagnostics.

## Source conclusion

PyPOTS was fixed at `{tm_cfg['commit']}` and is correctly labelled **maintained public reproduction / maintained implementation**, not official author code. Its core source has concrete MRTI/TID/MCM/MRM mappings. The maintained forward creates `dec_out_list` but retains only its final scale; the PaperFaithful version therefore uses the documented paper-guided equal-weight mean of those existing scale predictions. Details are in `TIMEMIXERPP_IMPLEMENTATION_AUDIT.md`.

## Seed-42 comparison (development only)

{table_markdown(seed42, result_columns)}

## Promotion and multiseed

| Model | Seed-42 promotion | Multiseed status |
|---|---:|---|
{chr(10).join(f"| {name} | {promotion[name]} | {status[name]} |" for name in promotion)}

{table_markdown(multi, ["model", "multiseed_status", "n_seeds", "mean_r2_delta", "mean_rmse_delta_mV", "rmse_delta_std_mV", "fold3_mean_rmse_delta_mV", "fold3_p95_mV", "fold3_p99_mV", "lifecycle_ratio"])}

## Diagnostic rankings

- Fold3 RMSE_delta, lower is better: {' > '.join(ranked_fold3)}.
- Fold3 P99 absolute DeltaV error, lower is better: {' > '.join(ranked_tail)}.
- Lifecycle ratio Fold3/Fold1, lower is more stable: {' > '.join(ranked_life)}.

## TimeXer verdict

The specified two adapters are fully documented in `TIMEXER_ADAPTATION_AUDIT.md`. **{timexer_status}**. It is not selected merely for novelty.

## Frozen selection

- Selected backbone: **{selected}**.
- Reason: {selection_reason}
- Recommended next phase: **{next_phase}**.
- Phase status: **PASS**. This phase completed its two TimeMixer++ variants, two permitted TimeXer adapters, fair seed-42 comparison, conditional multiseed policy, diagnostics, and decision freeze. No model was trained or selected on Test.
"""
    write_text(report, "PHASE_N2_8B_RESULTS.md")
    print(json.dumps({"phase_status": "PASS", "selected_backbone": selected, "promotion": promotion, "timexer_status": timexer_status}, ensure_ascii=False))


if __name__ == "__main__":
    main()
