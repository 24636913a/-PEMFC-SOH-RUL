"""Development-only Phase N2.8E physics-guided fast-slow TimeXer study.

The script is intentionally self-contained in the new phase directory.  It reads
frozen data and baseline artifacts only, never writes outside this phase root,
and fails closed before any Test feature/target/prediction can be constructed.
"""
from __future__ import annotations

import importlib.util
import json
import math
import random
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset


ROOT = Path(r"E:\沈农信电\ PEMFC-SOH+RUL\RENew-PEMFC-2300h-Pytorch\Phase_N2_8E_Physics_Guided_Fast_Slow_TimeXer")
CONFIG = ROOT / "configs" / "physics_guided_fast_slow_timexer_frozen_config.yaml"
FOLDS = ("fold_1", "fold_2", "fold_3")
SEED = 42


def out(relative: str) -> Path:
    """Return a verified output path beneath this phase root only."""
    path = (ROOT / relative).resolve()
    root = ROOT.resolve()
    if path.parent != root and root not in path.parents:
        raise RuntimeError(f"Output escaped Phase N2.8E root: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def write_csv(frame: pd.DataFrame, relative: str) -> None:
    frame.to_csv(out(relative), index=False, encoding="utf-8-sig", lineterminator="\n")


def write_text(content: str, relative: str) -> None:
    out(relative).write_text(content.rstrip() + "\n", encoding="utf-8")


def write_json(obj: Any, relative: str) -> None:
    out(relative).write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_yaml(obj: dict[str, Any], relative: str) -> None:
    out(relative).write_text(yaml.safe_dump(obj, allow_unicode=True, sort_keys=False) + "\n", encoding="utf-8")


def markdown_table(frame: pd.DataFrame, columns: list[str]) -> str:
    cols = [x for x in columns if x in frame.columns]
    if frame.empty or not cols:
        return "| status |\n|---|\n| no records |"
    show = frame.loc[:, cols].copy()
    for col in show.columns:
        if pd.api.types.is_float_dtype(show[col]):
            show[col] = show[col].map(lambda x: "" if pd.isna(x) else f"{x:.6f}")
    head = "| " + " | ".join(show.columns) + " |"
    sep = "|" + "|".join("---" for _ in show.columns) + "|"
    body = ["| " + " | ".join(str(v).replace("|", "/") for v in row) + " |" for row in show.itertuples(index=False, name=None)]
    return "\n".join([head, sep, *body])


def load_config() -> dict[str, Any]:
    cfg = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    if Path(cfg["phase_root"]).resolve() != ROOT.resolve():
        raise RuntimeError("Phase root in config does not match current phase")
    frozen = (cfg["task"]["resolution_hours"], cfg["task"]["lookback_steps"], cfg["task"]["horizon_steps"])
    if frozen != (0.5, 48, 24):
        raise RuntimeError(f"Frozen task mismatch: {frozen}")
    if len(cfg["features"]) != 14 or len(cfg["future_exogenous_features"]) != 9:
        raise RuntimeError("F2 or future-exogenous feature contract changed")
    if any("voltage" in x.lower() for x in cfg["future_exogenous_features"]):
        raise RuntimeError("Future voltage is prohibited")
    return cfg


def set_seed(value: int) -> None:
    random.seed(value)
    np.random.seed(value)
    torch.manual_seed(value)
    torch.cuda.manual_seed_all(value)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def common_module(cfg: dict[str, Any]):
    spec = importlib.util.spec_from_file_location("phase_n2_8e_common", Path(cfg["common_script"]))
    if spec is None or spec.loader is None:
        raise RuntimeError("Cannot load frozen N2.5 common module")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_development(cfg: dict[str, Any]):
    common = common_module(cfg)
    frame, audit = common.load_development_data(cfg, cfg["features"])
    if audit.get("target_rows_read_from_test") != 0 or frame["split"].eq("test").any():
        raise RuntimeError("TEST FIREWALL: Test rows entered the development frame")
    windows = {name: (train, valid) for name, train, valid in common.fold_windows(frame, cfg)}
    for name, (_, valid) in windows.items():
        if valid.target_end_age_h.max() >= cfg["split"]["development_end_h"]:
            raise RuntimeError(f"TEST FIREWALL: {name} validation target reaches Test")
    audit.update(
        {
            "target_rows_read_from_test": 0,
            "test_predictions_constructed": False,
            "test_metrics_computed": False,
            "test_feature_diagnostics_computed": False,
            "test_performance_accessed": False,
            "enforcement": "fail_closed_development_prefix_only",
        }
    )
    return common, frame, windows, audit


def future_exogenous(frame: pd.DataFrame, windows, cfg: dict[str, Any]) -> np.ndarray:
    h = int(cfg["task"]["horizon_steps"])
    result = []
    for origin in windows.row_origin_index.astype(int):
        part = frame.iloc[origin + 1 : origin + 1 + h].loc[:, cfg["future_exogenous_features"]].to_numpy(float)
        if part.shape != (h, len(cfg["future_exogenous_features"])):
            raise RuntimeError("Incomplete future-exogenous window")
        result.append(part)
    return np.asarray(result, dtype=float)


class FutureScaler:
    """Training-history-only scaler for Ridge and neural FiLM inputs."""

    def __init__(self):
        self.median: np.ndarray | None = None
        self.scaler = StandardScaler()

    def fit(self, history: np.ndarray):
        flat = history.reshape(-1, history.shape[-1])
        self.median = np.nanmedian(flat, axis=0)
        self.scaler.fit(np.where(np.isfinite(flat), flat, self.median))
        return self

    def fill(self, values: np.ndarray) -> np.ndarray:
        if self.median is None:
            raise RuntimeError("FutureScaler was not fitted")
        shape = (1,) * (values.ndim - 1) + (len(self.median),)
        return np.where(np.isfinite(values), values, self.median.reshape(shape))

    def transform(self, values: np.ndarray) -> np.ndarray:
        clean = self.fill(values)
        return self.scaler.transform(clean.reshape(-1, clean.shape[-1])).reshape(clean.shape).astype("float32")


class RidgeMultiHorizon:
    """One multi-output Ridge model fitted only on fold training histories."""

    def __init__(self, alpha: float):
        self.x_scaler = StandardScaler()
        self.y_scaler = StandardScaler()
        self.model = Ridge(alpha=alpha)

    def fit(self, history: np.ndarray, future: np.ndarray, scaler: FutureScaler):
        x = scaler.fill(history).reshape(len(history), -1)
        y = scaler.fill(future).reshape(len(future), -1)
        self.x_scaler.fit(x)
        self.y_scaler.fit(y)
        self.model.fit(self.x_scaler.transform(x), self.y_scaler.transform(y))
        return self

    def predict(self, history: np.ndarray, scaler: FutureScaler) -> np.ndarray:
        x = scaler.fill(history).reshape(len(history), -1)
        pred = self.y_scaler.inverse_transform(self.model.predict(self.x_scaler.transform(x)))
        return pred.reshape(len(history), 24, 9)


def expanding_oof_ridge(history: np.ndarray, future: np.ndarray, cfg: dict[str, Any], scaler: FutureScaler):
    """Build strict chronological OOF future-exogenous pseudo-inputs for training."""
    n = len(history)
    warmup = int(cfg["future_condition"]["oof_warmup_windows"])
    blocks = int(cfg["future_condition"]["oof_blocks"])
    if n <= warmup + 1:
        raise RuntimeError("Insufficient windows for strict chronological OOF Ridge")
    pred = np.full_like(future, np.nan, dtype=float)
    audit: list[dict[str, Any]] = []
    start = warmup
    size = math.ceil((n - warmup) / blocks)
    while start < n:
        end = min(n, start + size)
        model = RidgeMultiHorizon(cfg["future_condition"]["ridge_alpha"]).fit(history[:start], future[:start], scaler)
        pred[start:end] = model.predict(history[start:end], scaler)
        audit.append(
            {
                "fit_start_index": 0,
                "fit_end_exclusive": int(start),
                "predict_start_index": int(start),
                "predict_end_exclusive": int(end),
                "uses_prediction_row_future_label": False,
            }
        )
        start = end
    keep = np.isfinite(pred).all(axis=(1, 2))
    return pred, keep, audit


def clear_timexer_modules() -> None:
    for name in list(sys.modules):
        if name == "models" or name.startswith("models.") or name == "layers" or name.startswith("layers."):
            del sys.modules[name]


def import_timexer(vendor: Path):
    clear_timexer_modules()
    sys.path.insert(0, str(vendor))
    try:
        return __import__("models.TimeXer", fromlist=["Model"])
    finally:
        sys.path.remove(str(vendor))


class StepAlignedTimeXer(nn.Module):
    """Official TimeXer backbone plus strictly h-to-h FiLM correction.

    `future[h]` enters only `film(future[h])`; no module receives a sequence of
    future steps jointly. The base forecast remains the official TimeXer head.
    """

    def __init__(self, cfg: dict[str, Any], fast_slow: bool):
        super().__init__()
        tc = cfg["timexer"]
        source = import_timexer(Path(cfg["timexer_vendor"]))
        args = SimpleNamespace(
            task_name="long_term_forecast", features="MS", seq_len=48, pred_len=24,
            use_norm=False, patch_len=tc["patch_len"], enc_in=14, c_out=1,
            d_model=tc["d_model"], n_heads=tc["n_heads"], e_layers=tc["e_layers"],
            d_ff=tc["d_ff"], dropout=tc["dropout"], factor=5, embed="fixed", freq="h", activation="gelu",
        )
        self.base = source.Model(args)
        self.d_model = int(tc["d_model"])
        self.fast_slow = bool(fast_slow)
        hidden = int(cfg["future_condition"]["film_hidden_dim"])
        self.horizon_state = nn.Parameter(torch.empty(1, 24, self.d_model))
        nn.init.normal_(self.horizon_state, std=0.02)
        self.film = nn.Sequential(
            nn.Linear(9, hidden), nn.GELU(), nn.Linear(hidden, 2 * self.d_model)
        )
        # Exact zero gamma/beta at initialization: no future condition is injected until learned.
        nn.init.zeros_(self.film[-1].weight)
        nn.init.zeros_(self.film[-1].bias)
        self.correction = nn.Linear(self.d_model, 1)
        nn.init.zeros_(self.correction.weight)
        nn.init.zeros_(self.correction.bias)
        if self.fast_slow:
            dslow = int(cfg["slow_branch"]["hidden_dim"])
            if dslow > 32:
                raise RuntimeError("Slow branch violates dslow <= 32")
            self.slow = nn.Sequential(
                nn.LayerNorm(self.d_model), nn.Linear(self.d_model, dslow), nn.GELU(), nn.Linear(dslow, 24)
            )
            nn.init.zeros_(self.slow[-1].weight)
            nn.init.zeros_(self.slow[-1].bias)
        else:
            self.slow = None

    def _historical_state_and_base(self, x: torch.Tensor):
        if x.shape[1:] != (48, 14):
            raise RuntimeError(f"Frozen historical shape mismatch: {tuple(x.shape)}")
        # Frozen F2 has target in first feature; official MS TimeXer expects target last.
        ordered = torch.cat([x[:, :, 1:], x[:, :, :1]], dim=2)
        en_embed, n_vars = self.base.en_embedding(ordered[:, :, -1:].permute(0, 2, 1))
        ex_embed = self.base.ex_embedding(ordered[:, :, :-1], None)
        encoded = self.base.encoder(en_embed, ex_embed)
        core = encoded.reshape(-1, n_vars, encoded.shape[-2], encoded.shape[-1]).permute(0, 1, 3, 2)
        base_prediction = self.base.head(core).permute(0, 2, 1).squeeze(-1)
        historical_global = encoded[:, -1, :]
        return historical_global, base_prediction

    def forward(self, x: torch.Tensor, future: torch.Tensor, return_components: bool = False):
        if future.shape[1:] != (24, 9):
            raise RuntimeError(f"Frozen future-exogenous shape mismatch: {tuple(future.shape)}")
        historical_global, base_prediction = self._historical_state_and_base(x)
        h = historical_global.unsqueeze(1) + self.horizon_state
        gamma, beta = self.film(future).chunk(2, dim=-1)
        h_modulated = (1.0 + gamma) * h + beta
        fast = base_prediction + self.correction(h_modulated).squeeze(-1)
        slow = self.slow(historical_global) if self.slow is not None else torch.zeros_like(fast)
        total = fast + slow
        if return_components:
            return total, fast, slow
        return total


def parameter_count(model: nn.Module) -> int:
    return sum(x.numel() for x in model.parameters() if x.requires_grad)


def delta_metrics(y_abs: np.ndarray, p_abs: np.ndarray, anchor: np.ndarray) -> dict[str, float]:
    y = y_abs - anchor[:, None]
    p = p_abs - anchor[:, None]
    error = p - y
    ae = np.abs(error).ravel() * 1000.0
    return {
        "r2_delta": float(r2_score(y.ravel(), p.ravel())),
        "rmse_delta_mV": float(np.sqrt(np.mean(error ** 2)) * 1000.0),
        "mae_delta_mV": float(np.mean(np.abs(error)) * 1000.0),
        "p50_mV": float(np.quantile(ae, 0.50)),
        "p90_mV": float(np.quantile(ae, 0.90)),
        "p95_mV": float(np.quantile(ae, 0.95)),
        "p99_mV": float(np.quantile(ae, 0.99)),
        "max_mV": float(ae.max()),
    }


def prepare_fold(common, train, valid, feature_idx: list[int]):
    prep = common.FoldPreprocessor().fit(train.x)
    x_train, x_valid = prep.transform(train.x), prep.transform(valid.x)
    y_mean, y_std = float(train.y_delta.mean()), float(train.y_delta.std())
    if not np.isfinite(y_std) or y_std <= 1e-10:
        raise RuntimeError("Degenerate DeltaV target scale")
    y_train = ((train.y_delta - y_mean) / y_std).astype("float32")
    y_valid = ((valid.y_delta - y_mean) / y_std).astype("float32")
    z_scaler = FutureScaler().fit(train.x[:, :, feature_idx])
    return x_train, x_valid, y_train, y_valid, y_mean, y_std, z_scaler


def initial_loss_normalizers(losses: dict[str, torch.Tensor], data_loss: torch.Tensor) -> dict[str, float]:
    """Apply the permitted one-batch relative scale fix only for >100x mismatch."""
    denominator = max(float(data_loss.detach().item()), 1e-12)
    result: dict[str, float] = {}
    for name, value in losses.items():
        ratio = float(value.detach().item()) / denominator
        result[name] = ratio if ratio > 100.0 or (0.0 < ratio < 0.01) else 1.0
        if result[name] <= 0.0 or not np.isfinite(result[name]):
            result[name] = 1.0
    return result


def physics_terms(
    fast: torch.Tensor,
    slow: torch.Tensor,
    future: torch.Tensor,
    raw_future: torch.Tensor,
    cfg: dict[str, Any],
    need_current: bool,
) -> dict[str, torch.Tensor]:
    zero = fast.new_zeros(())
    smooth = ((slow[:, 2:] - 2.0 * slow[:, 1:-1] + slow[:, :-2]) ** 2).mean()
    mono = torch.relu(slow[:, 1:] - slow[:, :-1]).mean()
    current = zero
    if need_current:
        density_idx = cfg["future_exogenous_features"].index(cfg["physics_loss"]["current_density_feature"])
        gradients = torch.autograd.grad(
            fast.sum(), future, create_graph=True, retain_graph=True, allow_unused=False
        )[0][:, :, density_idx]
        active = (
            raw_future[:, :, density_idx] > float(cfg["physics_loss"]["active_current_threshold_A_cm2"])
        ) & (
            raw_future[:, :, -1] <= float(cfg["physics_loss"]["maximum_zero_current_fraction"])
        )
        current = torch.relu(gradients[active]).mean() if bool(active.any()) else zero
    return {"smooth": smooth, "current": current, "mono": mono}


def train_variant(
    model: StepAlignedTimeXer,
    train_pack: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    valid_pack: tuple[np.ndarray, np.ndarray, np.ndarray],
    cfg: dict[str, Any],
    physics_mode: str,
    checkpoint_relative: str,
):
    """Train one fold and return prediction plus epoch-loss evidence."""
    dev = device()
    x_train, y_train, z_train, raw_z_train = train_pack
    x_valid, y_valid, z_valid = valid_pack
    model.to(dev)
    loader = DataLoader(
        TensorDataset(
            torch.from_numpy(x_train), torch.from_numpy(y_train), torch.from_numpy(z_train), torch.from_numpy(raw_z_train.astype("float32"))
        ),
        batch_size=int(cfg["training"]["batch_size"]), shuffle=True,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(cfg["training"]["learning_rate"]), weight_decay=float(cfg["training"]["weight_decay"])
    )
    mse = nn.MSELoss()
    use_current = physics_mode in {"smooth_current", "full"}
    use_smooth = physics_mode in {"smooth", "smooth_current", "full"}
    use_mono = physics_mode == "full"
    lambdas = {
        "smooth": float(cfg["physics_loss"]["smooth_lambda"]) if use_smooth else 0.0,
        "current": float(cfg["physics_loss"]["current_lambda"]) if use_current else 0.0,
        "mono": float(cfg["physics_loss"]["mono_lambda"]) if use_mono else 0.0,
    }
    normalizers: dict[str, float] | None = None
    best = float("inf")
    best_epoch = 0
    stale = 0
    state: dict[str, torch.Tensor] | None = None
    history: list[dict[str, Any]] = []
    times: list[float] = []
    xv, yv, zv = torch.from_numpy(x_valid).to(dev), torch.from_numpy(y_valid).to(dev), torch.from_numpy(z_valid).to(dev)
    if dev.type == "cuda":
        torch.cuda.reset_peak_memory_stats(dev)
    for epoch in range(1, int(cfg["training"]["max_epochs"]) + 1):
        start = time.perf_counter()
        model.train()
        accumulator = {"data": [], "smooth": [], "current": [], "mono": [], "weighted_smooth": [], "weighted_current": [], "weighted_mono": []}
        for xb, yb, zb, raw_zb in loader:
            xb, yb = xb.to(dev), yb.to(dev)
            raw_zb = raw_zb.to(dev)
            zb = zb.to(dev).detach().requires_grad_(use_current)
            optimizer.zero_grad(set_to_none=True)
            total, fast, slow = model(xb, zb, return_components=True)
            data_loss = mse(total, yb)
            raw = physics_terms(fast, slow, zb, raw_zb, cfg, use_current)
            if normalizers is None:
                normalizers = initial_loss_normalizers(raw, data_loss)
            weighted = {name: lambdas[name] * raw[name] / normalizers[name] for name in raw}
            loss = data_loss + sum(weighted.values())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg["training"]["gradient_clip_norm"]))
            optimizer.step()
            accumulator["data"].append(float(data_loss.detach().item()))
            for name in raw:
                accumulator[name].append(float(raw[name].detach().item()))
                accumulator[f"weighted_{name}"].append(float(weighted[name].detach().item()))
        if dev.type == "cuda":
            torch.cuda.synchronize(dev)
        times.append(time.perf_counter() - start)
        model.eval()
        with torch.no_grad():
            validation_rmse = float(torch.sqrt(mse(model(xv, zv), yv)).item())
        if validation_rmse < best - 1e-7:
            best, best_epoch, stale = validation_rmse, epoch, 0
            state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        else:
            stale += 1
        hist = {
            "epoch": epoch,
            "train_data_loss": float(np.mean(accumulator["data"])),
            "train_slow_smooth_raw": float(np.mean(accumulator["smooth"])),
            "train_current_raw": float(np.mean(accumulator["current"])),
            "train_slow_mono_raw": float(np.mean(accumulator["mono"])),
            "weighted_slow_smooth": float(np.mean(accumulator["weighted_smooth"])),
            "weighted_current": float(np.mean(accumulator["weighted_current"])),
            "weighted_slow_mono": float(np.mean(accumulator["weighted_mono"])),
            "validation_scaled_delta_rmse": validation_rmse,
            "normalizer_smooth": normalizers["smooth"],
            "normalizer_current": normalizers["current"],
            "normalizer_mono": normalizers["mono"],
        }
        history.append(hist)
        if stale >= int(cfg["training"]["patience"]):
            break
    if state is None:
        raise RuntimeError("No best checkpoint was recorded")
    model.load_state_dict(state)
    torch.save(state, out(checkpoint_relative))
    model.eval()
    started = time.perf_counter()
    with torch.no_grad():
        prediction, fast, slow = model(xv, zv, return_components=True)
    if dev.type == "cuda":
        torch.cuda.synchronize(dev)
    info = {
        "parameter_count": parameter_count(model),
        "best_epoch": best_epoch,
        "epochs_run": len(history),
        "validation_scaled_delta_rmse": best,
        "training_time_per_epoch_s": float(np.mean(times)),
        "peak_gpu_memory_mb": float(torch.cuda.max_memory_allocated(dev) / 1024**2) if dev.type == "cuda" else 0.0,
        "inference_time_batch_s": float(time.perf_counter() - started),
        "device": str(dev),
        "loss_normalizer_smooth": normalizers["smooth"],
        "loss_normalizer_current": normalizers["current"],
        "loss_normalizer_mono": normalizers["mono"],
        "physics_loss_mode": physics_mode,
    }
    return info, prediction.cpu().numpy(), fast.cpu().numpy(), slow.cpu().numpy(), history


def sensitivity_audit(model: StepAlignedTimeXer, x: np.ndarray, z: np.ndarray, raw_z: np.ndarray, cfg: dict[str, Any], label: str, fold: str):
    """Verify h-to-h derivative alignment and report current-density sensitivity."""
    dev = device()
    model.eval()
    density_idx = cfg["future_exogenous_features"].index(cfg["physics_loss"]["current_density_feature"])
    take = min(16, len(x))
    xb = torch.from_numpy(x[:take]).to(dev)
    zb = torch.from_numpy(z[:take]).to(dev).detach().requires_grad_(True)
    _, fast, _ = model(xb, zb, return_components=True)
    rows = []
    diagonals = []
    off_diagonals = []
    for h in range(24):
        gradient = torch.autograd.grad(fast[:, h].sum(), zb, retain_graph=h < 23)[0][:, :, density_idx].detach().cpu().numpy()
        diag = gradient[:, h]
        off = np.delete(gradient, h, axis=1)
        active = (raw_z[:take, h, density_idx] > float(cfg["physics_loss"]["active_current_threshold_A_cm2"])) & (raw_z[:take, h, -1] <= float(cfg["physics_loss"]["maximum_zero_current_fraction"]))
        rows.append(
            {
                "model": label, "fold": fold, "horizon_step": h + 1, "horizon_hours": (h + 1) * 0.5,
                "mean_dfast_d_future_current_density": float(np.mean(diag)),
                "positive_derivative_fraction": float(np.mean(diag[active] > 0.0)) if active.any() else np.nan,
                "active_samples": int(active.sum()),
                "mean_abs_off_diagonal_derivative": float(np.mean(np.abs(off))),
                "max_abs_off_diagonal_derivative": float(np.max(np.abs(off))),
            }
        )
        diagonals.append(np.abs(diag))
        off_diagonals.append(np.abs(off))
    return rows, float(np.mean(np.concatenate(diagonals))), float(np.max(np.concatenate(off_diagonals)))


def original_reference(cfg: dict[str, Any], windows):
    source = Path(cfg["phase_n2_8a_root"]) / "outputs" / "model_screening"
    rows, arrays = [], {}
    for fold in FOLDS:
        array = dict(np.load(source / f"timexer_official_{fold}_seed42_predictions.npz", allow_pickle=False))
        validation = windows[fold][1]
        if not np.array_equal(array["row_origin_index"], validation.row_origin_index):
            raise RuntimeError(f"Frozen original TimeXer rows differ for {fold}")
        if array["target_end_age_h"].max() >= cfg["split"]["development_end_h"]:
            raise RuntimeError("TEST FIREWALL: original reference contains Test target")
        arrays[fold] = array
        rows.append({"model": "TimeXer-Original", "fold": fold, "seed": 42, "eligibility": "FROZEN_REFERENCE", **delta_metrics(array["y_true"], array["y_pred"], array["anchor"])})
    return pd.DataFrame(rows), arrays


def run_model(
    label: str,
    architecture: str,
    future_mode: str,
    physics_mode: str,
    packages: dict[str, Any],
    common,
    windows,
    cfg: dict[str, Any],
    seed_value: int,
):
    feature_idx = [cfg["features"].index(x) for x in cfg["future_exogenous_features"]]
    rows, arrays, histories, current_rows, component_rows = [], {}, [], [], []
    alignment = []
    for fold_index, fold in enumerate(FOLDS):
        train, valid = windows[fold]
        xtr, xva, ytr, yva, y_mean, y_std, z_scaler = prepare_fold(common, train, valid, feature_idx)
        package = packages[fold]
        if future_mode == "oracle":
            raw_train, raw_valid = package["true_train"], package["true_valid"]
            keep = np.ones(len(train.x), dtype=bool)
            eligibility = "DIAGNOSTIC_ORACLE_ONLY"
        elif future_mode == "zero":
            raw_train, raw_valid = np.zeros_like(package["oof_train"]), np.zeros_like(package["ridge_valid"])
            keep = package["keep"]
            eligibility = "FORMAL_ZERO_CONTROL"
        else:
            raw_train, raw_valid = package["oof_train"], package["ridge_valid"]
            keep = package["keep"]
            eligibility = "NEGATIVE_CONTROL" if future_mode == "shuffled" else "FORMAL"
            if future_mode == "shuffled":
                raw_valid = raw_valid[np.random.default_rng(seed_value + fold_index).permutation(len(raw_valid))]
        if not keep.any():
            raise RuntimeError(f"No causal OOF training rows in {fold}")
        set_seed(seed_value)
        model = StepAlignedTimeXer(cfg, fast_slow=(architecture == "fast_slow"))
        info, prediction, fast, slow, history = train_variant(
            model,
            (xtr[keep], ytr[keep], z_scaler.transform(raw_train[keep]), raw_train[keep]),
            (xva, yva, z_scaler.transform(raw_valid)),
            cfg, physics_mode, f"checkpoints/{label}_{fold}_seed{seed_value}.pt",
        )
        abs_prediction = valid.anchor[:, None] + prediction * y_std + y_mean
        abs_fast = valid.anchor[:, None] + fast * y_std + y_mean
        slow_abs = slow * y_std
        array = {
            "y_true": valid.y_abs.astype(float), "y_pred": abs_prediction.astype(float), "anchor": valid.anchor.astype(float),
            "fast_prediction": abs_fast.astype(float), "slow_component": slow_abs.astype(float),
            "target_end_age_h": valid.target_end_age_h.astype(float), "origin_age_h": valid.origin_age_h.astype(float),
            "row_origin_index": valid.row_origin_index.astype(int), "future_exogenous_input": raw_valid.astype(float),
            "future_exogenous_true": package["true_valid"].astype(float),
        }
        if array["target_end_age_h"].max() >= cfg["split"]["development_end_h"]:
            raise RuntimeError("TEST FIREWALL: constructed prediction enters Test")
        np.savez_compressed(out(f"outputs/model_runs/{label}_{fold}_seed{seed_value}_predictions.npz"), **array)
        arrays[fold] = array
        rows.append({
            "model": label, "fold": fold, "seed": seed_value, "eligibility": eligibility,
            "architecture": architecture, "future_condition": future_mode, **delta_metrics(array["y_true"], array["y_pred"], array["anchor"]),
            **info, "oof_train_windows": int(keep.sum()), "oof_warmup_windows_dropped": int((~keep).sum()),
        })
        for entry in history:
            histories.append({"model": label, "fold": fold, "seed": seed_value, **entry})
        if architecture == "fast_slow":
            component_rows.append({"model": label, "fold": fold, "array": array})
            sensitivity, diagonal, max_off = sensitivity_audit(model, xva, z_scaler.transform(raw_valid), raw_valid, cfg, label, fold)
            current_rows.extend(sensitivity)
            alignment.append({"model": label, "fold": fold, "mean_abs_diagonal_derivative": diagonal, "max_abs_off_diagonal_derivative": max_off})
    return pd.DataFrame(rows), arrays, histories, current_rows, component_rows, alignment


def summarize_models(rows: pd.DataFrame) -> pd.DataFrame:
    result = []
    for model, group in rows.groupby("model", sort=False):
        seed42 = group[group["seed"] == 42]
        if len(seed42) != 3:
            continue
        record: dict[str, Any] = {
            "model": model, "eligibility": seed42["eligibility"].iloc[0], "architecture": seed42.get("architecture", pd.Series(["reference"])).iloc[0],
            "future_condition": seed42.get("future_condition", pd.Series(["none"])).iloc[0],
            "parameter_count": float(seed42["parameter_count"].dropna().iloc[0]) if seed42["parameter_count"].notna().any() else np.nan,
        }
        for fold in FOLDS:
            item = seed42[seed42["fold"] == fold].iloc[0]
            for key in ("r2_delta", "rmse_delta_mV", "mae_delta_mV", "p95_mV", "p99_mV"):
                record[f"{fold}_{key}"] = float(item[key])
        for key in ("r2_delta", "rmse_delta_mV", "mae_delta_mV", "p95_mV", "p99_mV"):
            record[f"mean_{key}"] = float(seed42[key].mean())
        record["lifecycle_ratio"] = record["fold_3_rmse_delta_mV"] / record["fold_1_rmse_delta_mV"]
        result.append(record)
    return pd.DataFrame(result)


def horizon_rows(label: str, arrays: dict[str, Any]) -> list[dict[str, Any]]:
    result = []
    for fold, array in arrays.items():
        y = array["y_true"] - array["anchor"][:, None]
        p = array["y_pred"] - array["anchor"][:, None]
        for h in range(24):
            result.append({
                "model": label, "fold": fold, "horizon_step": h + 1, "horizon_hours": (h + 1) * 0.5,
                "r2_delta": float(r2_score(y[:, h], p[:, h])),
                "rmse_delta_mV": float(np.sqrt(np.mean((p[:, h] - y[:, h]) ** 2)) * 1000.0),
                "mae_delta_mV": float(np.mean(np.abs(p[:, h] - y[:, h])) * 1000.0),
            })
    return result


def tail_row(label: str, arrays: dict[str, Any]) -> dict[str, Any]:
    errors = []
    for array in arrays.values():
        errors.extend(np.abs((array["y_pred"] - array["anchor"][:, None]) - (array["y_true"] - array["anchor"][:, None])).ravel() * 1000.0)
    errors = np.asarray(errors)
    return {"model": label, "p50_mV": float(np.quantile(errors, 0.5)), "p90_mV": float(np.quantile(errors, 0.9)), "p95_mV": float(np.quantile(errors, 0.95)), "p99_mV": float(np.quantile(errors, 0.99)), "max_mV": float(errors.max())}


def plot_outputs(comparison: pd.DataFrame, horizon: pd.DataFrame, tail: pd.DataFrame, components: list[dict[str, Any]], current: pd.DataFrame, history: pd.DataFrame):
    figures = out("outputs/figures/.keep").parent
    ordered = comparison.sort_values("mean_rmse_delta_mV")
    ax = ordered.set_index("model")["mean_rmse_delta_mV"].plot(kind="bar", figsize=(11, 4), color="#3b82f6")
    ax.set_ylabel("Mean RMSE DeltaV (mV)"); ax.set_xlabel(""); plt.xticks(rotation=30, ha="right"); plt.tight_layout(); plt.savefig(figures / "model_rmse_comparison.png", dpi=180); plt.close()
    ax = ordered.set_index("model")["mean_r2_delta"].plot(kind="bar", figsize=(11, 4), color="#0f766e")
    ax.set_ylabel("Mean R2 DeltaV"); ax.set_xlabel(""); plt.xticks(rotation=30, ha="right"); plt.tight_layout(); plt.savefig(figures / "model_r2_delta_comparison.png", dpi=180); plt.close()
    fig, ax = plt.subplots(figsize=(10, 4))
    for name, group in horizon.groupby("model"):
        ax.plot(group.groupby("horizon_hours")["rmse_delta_mV"].mean().index, group.groupby("horizon_hours")["rmse_delta_mV"].mean().values, label=name)
    ax.set_xlabel("Forecast horizon (h)"); ax.set_ylabel("RMSE DeltaV (mV)"); ax.legend(fontsize=6, ncol=2); ax.grid(alpha=.25); plt.tight_layout(); plt.savefig(figures / "per_horizon_rmse.png", dpi=180); plt.close()
    fold3 = comparison.set_index("model")["fold_3_rmse_delta_mV"].sort_values()
    ax = fold3.plot(kind="bar", figsize=(11, 4), color="#8b5cf6"); ax.set_ylabel("Fold-3 RMSE DeltaV (mV)"); ax.set_xlabel(""); plt.xticks(rotation=30, ha="right"); plt.tight_layout(); plt.savefig(figures / "fold3_error_comparison.png", dpi=180); plt.close()
    ax = tail.set_index("model")[["p95_mV", "p99_mV"]].plot(kind="bar", figsize=(11, 4)); ax.set_ylabel("Absolute DeltaV error (mV)"); ax.set_xlabel(""); plt.xticks(rotation=30, ha="right"); plt.tight_layout(); plt.savefig(figures / "tail_error_comparison.png", dpi=180); plt.close()
    if components:
        best = min(components, key=lambda x: float(np.sqrt(np.mean((x["array"]["y_pred"] - x["array"]["y_true"]) ** 2))))
        arr = best["array"]; index = min(1, len(arr["y_true"]) - 1); h = np.arange(1, 25) * 0.5
        fig, ax = plt.subplots(figsize=(9, 4)); ax.plot(h, arr["y_true"][index] - arr["anchor"][index], label="true DeltaV", lw=2); ax.plot(h, arr["y_pred"][index] - arr["anchor"][index], label="total predicted"); ax.plot(h, arr["fast_prediction"][index] - arr["anchor"][index], label="fast component"); ax.plot(h, arr["slow_component"][index], label="slow component"); ax.set_title(f"{best['model']} {best['fold']} example"); ax.set_xlabel("Forecast horizon (h)"); ax.set_ylabel("DeltaV (V)"); ax.legend(fontsize=8); ax.grid(alpha=.25); plt.tight_layout(); plt.savefig(figures / "fast_slow_example_predictions.png", dpi=180); plt.close()
        fig, ax = plt.subplots(figsize=(9, 4))
        for item in components:
            arr = item["array"]; ax.scatter(arr["origin_age_h"], arr["slow_component"].mean(axis=1) * 1000.0, s=6, alpha=.35, label=f"{item['model']} {item['fold']}")
        ax.set_xlabel("Origin operating age (h)"); ax.set_ylabel("Mean slow component (mV)"); ax.legend(fontsize=5, ncol=2); ax.grid(alpha=.25); plt.tight_layout(); plt.savefig(figures / "slow_component_vs_lifecycle.png", dpi=180); plt.close()
    if not current.empty:
        fig, ax = plt.subplots(figsize=(9, 4))
        for name, group in current.groupby("model"):
            q = group.groupby("horizon_hours")["mean_dfast_d_future_current_density"].mean(); ax.plot(q.index, q.values, label=name)
        ax.axhline(0, color="black", lw=.7); ax.set_xlabel("Forecast horizon (h)"); ax.set_ylabel("Mean dFast/dFutureCurrentDensity"); ax.legend(fontsize=7); ax.grid(alpha=.25); plt.tight_layout(); plt.savefig(figures / "current_density_sensitivity.png", dpi=180); plt.close()
    if not history.empty:
        chosen = history[history["model"].str.contains("Full")].copy()
        if chosen.empty: chosen = history.copy()
        fig, ax = plt.subplots(figsize=(9, 4))
        for key, label in (("train_data_loss", "data"), ("weighted_slow_smooth", "weighted smooth"), ("weighted_current", "weighted current"), ("weighted_slow_mono", "weighted mono")):
            q = chosen.groupby("epoch")[key].mean(); ax.plot(q.index, q.values, label=label)
        ax.set_yscale("log"); ax.set_xlabel("Epoch"); ax.set_ylabel("Loss (scaled DeltaV space)"); ax.legend(fontsize=8); ax.grid(alpha=.25); plt.tight_layout(); plt.savefig(figures / "physics_loss_training_curve.png", dpi=180); plt.close()


def main() -> None:
    cfg = load_config()
    write_text(CONFIG.read_text(encoding="utf-8"), "outputs/frozen_config/physics_guided_fast_slow_timexer_frozen_config.yaml")
    common, frame, windows, firewall = load_development(cfg)
    write_json(firewall, "outputs/test_firewall_audit.json")
    feature_idx = [cfg["features"].index(x) for x in cfg["future_exogenous_features"]]
    packages: dict[str, Any] = {}
    oof_audit: list[dict[str, Any]] = []
    for fold in FOLDS:
        train, valid = windows[fold]
        historical_train = train.x[:, :, feature_idx]
        historical_valid = valid.x[:, :, feature_idx]
        true_train = future_exogenous(frame, train, cfg)
        true_valid = future_exogenous(frame, valid, cfg)
        scaler = FutureScaler().fit(historical_train)
        oof, keep, blocks = expanding_oof_ridge(historical_train, true_train, cfg, scaler)
        ridge = RidgeMultiHorizon(cfg["future_condition"]["ridge_alpha"]).fit(historical_train, true_train, scaler)
        ridge_valid = ridge.predict(historical_valid, scaler)
        packages[fold] = {"true_train": true_train, "true_valid": true_valid, "oof_train": oof, "ridge_valid": ridge_valid, "keep": keep}
        oof_audit.extend({"fold": fold, **item, "warmup_windows_dropped": int((~keep).sum())} for item in blocks)
        np.savez_compressed(out(f"outputs/future_exogenous/ridge_future_exogenous_{fold}.npz"), predicted_future_exogenous=ridge_valid, true_future_exogenous=true_valid, row_origin_index=valid.row_origin_index, origin_datetime=valid.origin_datetime)
    write_text(
        "# Future-exogenous OOF audit\n\nFormal training uses expanding chronological OOF Ridge pseudo-inputs. Each training prediction is generated by a model fitted only on earlier training-window rows. Warm-up rows without a prior-only fit are removed from formal model training, never filled with true future conditions. Validation uses a Ridge model fitted on the full fold training block.\n\n" + markdown_table(pd.DataFrame(oof_audit), ["fold", "fit_start_index", "fit_end_exclusive", "predict_start_index", "predict_end_exclusive", "warmup_windows_dropped", "uses_prediction_row_future_label"]),
        "FUTURE_EXOGENOUS_OOF_AUDIT.md",
    )
    set_seed(SEED)
    base = StepAlignedTimeXer(cfg, fast_slow=False)
    full = StepAlignedTimeXer(cfg, fast_slow=True)
    base_count, sa_total, fs_total = parameter_count(base.base), parameter_count(base), parameter_count(full)
    if (sa_total - base_count) / base_count >= 0.30 or (fs_total - base_count) / base_count >= 0.30:
        raise RuntimeError("New branch violates 30% parameter cap")
    write_text(
        f"# Fast-slow architecture\n\nThe official TimeXer historical backbone is preserved at commit `{cfg['timexer']['commit']}`. It produces the historical global latent and official base forecast. Each future condition row `z_h` goes only through `MLP(z_h) -> gamma_h,beta_h`; the corresponding horizon state is modulated as `(1+gamma_h)H_h+beta_h`. No CrossAttention, pooling over future steps, or future-time concatenation is present.\n\nThe fast prediction is the frozen-form official head plus a zero-initialized step-aligned correction. The slow prediction is `LayerNorm(Hhist) -> Linear(64,32) -> GELU -> Linear(32,24)` with a zero-initialized final layer, and is supervised only by the DeltaV data loss plus optional regularizers.\n\n| component | parameters | extra/original |\n|---|---:|---:|\n| official TimeXer backbone | {base_count} | 0.000000% |\n| SA-TimeXer total | {sa_total} | {(sa_total-base_count)/base_count:.6%} |\n| FS/PGFS-TimeXer total | {fs_total} | {(fs_total-base_count)/base_count:.6%} |\n\nThe model graph enforces strict h-to-h conditioning; a derivative audit is emitted for every fast-slow fold.\n",
        "FAST_SLOW_ARCHITECTURE.md",
    )
    original_rows, original_arrays = original_reference(cfg, windows)
    all_rows = [original_rows]
    all_arrays: dict[str, dict[str, Any]] = {"TimeXer-Original": original_arrays}
    all_history: list[dict[str, Any]] = []
    all_current: list[dict[str, Any]] = []
    components: list[dict[str, Any]] = []
    alignment: list[dict[str, Any]] = []
    variants = [
        ("SA_TimeXer_Zero", "step_aligned", "zero", "none"),
        ("SA_TimeXer_Ridge", "step_aligned", "ridge", "none"),
        ("SA_TimeXer_Oracle", "step_aligned", "oracle", "none"),
        ("SA_TimeXer_Shuffled", "step_aligned", "shuffled", "none"),
        ("FS_TimeXer_NoPhysics", "fast_slow", "ridge", "none"),
        ("PGFS_TimeXer_Smooth", "fast_slow", "ridge", "smooth"),
        ("PGFS_TimeXer_SmoothCurrent", "fast_slow", "ridge", "smooth_current"),
        ("PGFS_TimeXer_Full", "fast_slow", "ridge", "full"),
    ]
    for label, architecture, future_mode, physics_mode in variants:
        row, arrays, history, current, comps, aligned = run_model(label, architecture, future_mode, physics_mode, packages, common, windows, cfg, SEED)
        all_rows.append(row); all_arrays[label] = arrays; all_history.extend(history); all_current.extend(current); components.extend(comps); alignment.extend(aligned)
    fold_metrics = pd.concat(all_rows, ignore_index=True)
    comparison = summarize_models(fold_metrics)
    original_summary = comparison.loc[comparison["model"] == "TimeXer-Original"].iloc[0]
    comparison["gain_vs_original_pct"] = (float(original_summary["mean_rmse_delta_mV"]) - comparison["mean_rmse_delta_mV"]) / float(original_summary["mean_rmse_delta_mV"]) * 100.0
    comparison.loc[comparison["model"] == "TimeXer-Original", "gain_vs_original_pct"] = 0.0
    physics_models = ["FS_TimeXer_NoPhysics", "PGFS_TimeXer_Smooth", "PGFS_TimeXer_SmoothCurrent", "PGFS_TimeXer_Full"]
    physics_ablation = comparison[comparison["model"].isin(physics_models)].copy()
    horizon = pd.DataFrame([item for model, arrays in all_arrays.items() for item in horizon_rows(model, arrays)])
    tail = pd.DataFrame([tail_row(model, arrays) for model, arrays in all_arrays.items()])
    history = pd.DataFrame(all_history)
    current = pd.DataFrame(all_current)
    variance_rows = []
    mono_rows = []
    for item in components:
        array = item["array"]
        variance_rows.append({"model": item["model"], "fold": item["fold"], "fast_variance_mV2": float(np.var((array["fast_prediction"] - array["anchor"][:, None]) * 1000.0)), "slow_variance_mV2": float(np.var(array["slow_component"] * 1000.0)), "total_variance_mV2": float(np.var((array["y_pred"] - array["anchor"][:, None]) * 1000.0))})
        difference = np.diff(array["slow_component"], axis=1)
        mono_rows.append({"model": item["model"], "fold": item["fold"], "non_increasing_fraction": float(np.mean(difference <= 0.0)), "positive_increment_fraction": float(np.mean(difference > 0.0)), "mean_positive_increment_mV": float(np.mean(np.maximum(difference, 0.0)) * 1000.0), "mean_slow_second_difference_mV": float(np.mean(np.abs(array["slow_component"][:, 2:] - 2*array["slow_component"][:, 1:-1] + array["slow_component"][:, :-2])) * 1000.0)})
    variance = pd.DataFrame(variance_rows); monotonicity = pd.DataFrame(mono_rows); alignment_frame = pd.DataFrame(alignment)
    # Save a consolidated, explicitly development-only component artifact.  All
    # fields are flat numeric/string arrays so it can be safely read with
    # np.load(..., allow_pickle=False).
    if components:
        row_model, row_fold = [], []
        fields = {name: [] for name in ("y_true", "y_pred", "fast_prediction", "slow_component", "origin_age_h", "anchor", "target_end_age_h")}
        for item in components:
            array = item["array"]
            count = len(array["origin_age_h"])
            row_model.append(np.repeat(item["model"], count))
            row_fold.append(np.repeat(item["fold"], count))
            for name in fields:
                fields[name].append(array[name])
        np.savez_compressed(
            out("outputs/model_runs/fast_slow_component_predictions.npz"),
            model=np.concatenate(row_model), fold=np.concatenate(row_fold),
            **{name: np.concatenate(parts, axis=0) for name, parts in fields.items()},
        )
    # Promotion is intentionally restricted to PGFS variants, after the fixed seed-42 ablation.
    candidates = comparison[comparison["model"].str.startswith("PGFS_")].sort_values("mean_rmse_delta_mV")
    best_label = str(candidates.iloc[0]["model"])
    best_seed42 = candidates.iloc[0]
    fold3_gain = float(best_seed42["gain_vs_original_pct"])
    promote = float(best_seed42["mean_rmse_delta_mV"]) < float(cfg["thresholds"]["multiseed_rmse_mV"]) or (float(best_seed42["mean_rmse_delta_mV"]) < float(cfg["thresholds"]["conditional_rmse_mV"]) and float(original_summary["fold_3_rmse_delta_mV"] - best_seed42["fold_3_rmse_delta_mV"]) / float(original_summary["fold_3_rmse_delta_mV"]) * 100.0 >= float(cfg["thresholds"]["conditional_fold3_gain_pct"]))
    if promote:
        lookup = {label: (architecture, future_mode, physics_mode) for label, architecture, future_mode, physics_mode in variants}
        architecture, future_mode, physics_mode = lookup[best_label]
        for multi_seed in (2024, 3407):
            row, arrays, more_history, more_current, more_components, more_alignment = run_model(best_label, architecture, future_mode, physics_mode, packages, common, windows, cfg, multi_seed)
            fold_metrics = pd.concat([fold_metrics, row], ignore_index=True)
            all_history.extend(more_history); all_current.extend(more_current); components.extend(more_components); alignment.extend(more_alignment)
    best_seed_rows = fold_metrics[fold_metrics["model"] == best_label].groupby("seed", as_index=False)[["r2_delta", "rmse_delta_mV", "mae_delta_mV"]].mean()
    multiseed = pd.DataFrame([{
        "model": best_label, "promotion_status": "three_seed_completed" if promote else "not_promoted_seed42_only", "n_seeds": int(len(best_seed_rows)),
        "seeds": ";".join(best_seed_rows["seed"].astype(str)), "mean_r2_delta": float(best_seed_rows["r2_delta"].mean()), "mean_rmse_delta_mV": float(best_seed_rows["rmse_delta_mV"].mean()), "rmse_std_mV": float(best_seed_rows["rmse_delta_mV"].std(ddof=0)),
    }])
    history = pd.DataFrame(all_history); current = pd.DataFrame(all_current); alignment_frame = pd.DataFrame(alignment)
    # Audit whether any weighted regularizer dominated data loss for three consecutive epochs.
    dominance = []
    for (model, fold, seed_value), group in history.groupby(["model", "fold", "seed"]):
        group = group.sort_values("epoch")
        maximum_run, run = 0, 0
        for _, item in group.iterrows():
            weight = item["weighted_slow_smooth"] + item["weighted_current"] + item["weighted_slow_mono"]
            run = run + 1 if weight > item["train_data_loss"] else 0
            maximum_run = max(maximum_run, run)
        dominance.append({"model": model, "fold": fold, "seed": seed_value, "max_consecutive_physics_dominance_epochs": maximum_run, "physics_loss_status": "DOMINATED_RETRY_REQUIRED" if maximum_run >= 3 else "BALANCED"})
    dominance_frame = pd.DataFrame(dominance)
    if (dominance_frame["physics_loss_status"] == "DOMINATED_RETRY_REQUIRED").any():
        raise RuntimeError("Physics loss dominated data loss; required normalization retry was not performed")
    write_csv(comparison, "outputs/model_screening/N2_8E_seed42_comparison.csv")
    write_csv(physics_ablation, "outputs/ablation/N2_8E_physics_ablation.csv")
    write_csv(horizon, "outputs/horizon_analysis/N2_8E_per_horizon.csv")
    write_csv(fold_metrics, "outputs/model_screening/N2_8E_fold_metrics.csv")
    write_csv(tail, "outputs/tail_analysis/N2_8E_tail_metrics.csv")
    write_csv(variance, "outputs/diagnostics/fast_slow_variance_audit.csv")
    write_csv(current, "outputs/diagnostics/current_sensitivity_audit.csv")
    write_csv(monotonicity, "outputs/diagnostics/slow_monotonicity_audit.csv")
    write_csv(multiseed, "outputs/model_screening/N2_8E_multiseed_comparison.csv")
    write_csv(history, "outputs/diagnostics/physics_loss_training_history.csv")
    write_csv(alignment_frame, "outputs/diagnostics/step_alignment_derivative_audit.csv")
    write_csv(dominance_frame, "outputs/diagnostics/physics_loss_dominance_audit.csv")
    plot_outputs(comparison, horizon, tail, components, current, history)
    loss_audit = pd.DataFrame([{ "model": row["model"], "fold": row["fold"], "seed": row["seed"], "smooth_normalizer": row["loss_normalizer_smooth"], "current_normalizer": row["loss_normalizer_current"], "mono_normalizer": row["loss_normalizer_mono"], "physics_mode": row["physics_loss_mode"] } for _, row in fold_metrics[fold_metrics["model"] != "TimeXer-Original"].iterrows()])
    write_text(
        "# Physics-loss audit\n\nLosses are evaluated in standardized DeltaV target space. The first training batch may rescale only a raw regularizer whose magnitude differs from data MSE by more than 100×; the detached normalizers used are listed below. The fixed relative multipliers remain smooth=0.01, current=0.001, mono=0.001. A fold would fail if the sum of weighted physics losses exceeded data loss for three consecutive epochs. All completed runs are balanced.\n\n" + markdown_table(loss_audit, ["model", "fold", "seed", "physics_mode", "smooth_normalizer", "current_normalizer", "mono_normalizer"]) + "\n\n" + markdown_table(dominance_frame, ["model", "fold", "seed", "max_consecutive_physics_dominance_epochs", "physics_loss_status"]),
        "PHYSICS_LOSS_AUDIT.md",
    )
    # Oracle is a diagnostic ceiling and shuffled future inputs are a negative
    # control. Neither can select a formal best model or open Phase N2.8F.
    formal_comparison = comparison.loc[
        comparison["eligibility"].isin(["FROZEN_REFERENCE", "FORMAL", "FORMAL_ZERO_CONTROL"])
    ].copy()
    if formal_comparison.empty:
        raise RuntimeError("No formal development-only model is eligible for the decision")
    best = formal_comparison.sort_values("mean_rmse_delta_mV").iloc[0]
    rmse = float(best["mean_rmse_delta_mV"])
    if rmse >= 25.0: recovery = "WEAK"
    elif rmse >= 20.0: recovery = "MEANINGFUL_RECOVERY"
    elif rmse >= 15.0: recovery = "PROMISING"
    elif rmse > 12.0: recovery = "STRONG"
    else: recovery = "REVIVED"
    go = rmse < float(cfg["thresholds"]["n2_8f_entry_rmse_mV"])
    decision = {
        "phase_status": "PASS", "study_commit": cfg["study_commit"], "test_performance_accessed": False,
        "best_seed42_model": str(best["model"]), "best_seed42_rmse_delta_mV": rmse, "best_seed42_r2_delta": float(best["mean_r2_delta"]), "recovery_status": recovery,
        "multiseed_promoted": bool(promote), "multiseed_model": best_label, "multiseed_summary": multiseed.iloc[0].to_dict(),
        "time_xer_original_rmse_delta_mV": float(original_summary["mean_rmse_delta_mV"]), "enter_phase_n2_8f": "GO" if go else "NO-GO",
        "n2_8f_reason": "Best development-only seed-42 RMSE is below 20 mV." if go else "Best development-only seed-42 RMSE did not clear the <20 mV N2.8F entry gate.",
    }
    write_yaml(decision, "PHASE_N2_8E_DECISION.yaml")
    report = f"""# Phase N2.8E results

## 1. Frozen protocol and Test firewall

This is a development-only 0.5 h / 48-step / 24-step / F2 / DeltaV experiment. Test feature/target rows read: **0**. Test predictions, metrics, diagnostics, and model-selection access: **false**. Formal future exogenous inputs are expanding-OOF Ridge pseudo-forecasts in training and Train-fit Ridge forecasts in validation. Oracle inputs are explicitly diagnostic-only.

## 2. Step-aligned conditioning controls

{markdown_table(comparison, ["model", "eligibility", "mean_r2_delta", "mean_rmse_delta_mV", "mean_mae_delta_mV", "fold_3_rmse_delta_mV", "mean_p95_mV", "mean_p99_mV", "gain_vs_original_pct", "lifecycle_ratio"])}

`SA_TimeXer_Ridge` is the formal step-aligned future-condition comparison. `SA_TimeXer_Zero` is the zero-input control, `SA_TimeXer_Shuffled` is a negative control, and `SA_TimeXer_Oracle` is diagnostic-only. The architecture document and derivative audit establish h-to-h construction rather than an attention-alignment claim.

## 3. Fast-slow and physics ablation

{markdown_table(physics_ablation, ["model", "mean_r2_delta", "mean_rmse_delta_mV", "fold_3_rmse_delta_mV", "gain_vs_original_pct", "mean_p95_mV", "mean_p99_mV"])}

The slow branch has no interpolated SOH or fabricated degradation target. It is an additive historical-latent component constrained only by the explicitly listed data/smoothness/current/monotonicity objectives.

## 4. Physics and strict-alignment audits

{markdown_table(dominance_frame, ["model", "fold", "seed", "max_consecutive_physics_dominance_epochs", "physics_loss_status"])}

{markdown_table(alignment_frame, ["model", "fold", "mean_abs_diagonal_derivative", "max_abs_off_diagonal_derivative"])}

Current-density derivative diagnostics apply only to active future operating points (`J > 0.01 A cm^-2`, `zero_current_fraction <= 0.5`). They are a soft tendency constraint on the fast component, not a claim of a complete PEMFC mechanistic model.

## 5. Horizon, Fold-3, tails, and components

The full 0.5–12 h horizon table, Fold-3 metrics, tail quantiles, fast/slow variance audit, and slow monotonicity audit are stored as CSV artifacts. They are all development-only. The figures use no Test targets or annotations.

## 6. Multi-seed rule

{markdown_table(multiseed, ["model", "promotion_status", "n_seeds", "seeds", "mean_r2_delta", "mean_rmse_delta_mV", "rmse_std_mV"])}

The fixed promotion gate was: seed-42 mean RMSE <20 mV, or <22 mV and at least 30% Fold-3 improvement vs TimeXer-Original. No other tuning or selection occurred.

## 7. Decision

- Best formal seed-42 model: **{best['model']}**
- Best development mean RMSE DeltaV: **{rmse:.6f} mV**
- Best development mean R2 DeltaV: **{float(best['mean_r2_delta']):.6f}**
- Recovery category: **{recovery}**
- Phase N2.8F (EIS/polarization) entry: **{'GO' if go else 'NO-GO'}**

N2.8E is an execution PASS. Its modeling conclusion is limited to the frozen development folds. It does not train an N2.8F model, inspect final Test performance, fit an EOL threshold, or claim strict RUL ground truth.
"""
    write_text(report, "PHASE_N2_8E_RESULTS.md")
    print(json.dumps({"phase_status": "PASS", "best_model": str(best["model"]), "best_rmse_mV": rmse, "recovery": recovery, "n2_8f": "GO" if go else "NO-GO", "multiseed_promoted": bool(promote)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
