"""Leakage-safe shared utilities for Phase POT.

This module is deliberately independent of the historical Phase BS/DF runners:
those runners retain their own frozen HEAD guards and write locations.  POT
imports only Phase BS pure window/scaler/metric helpers and never calls a
historical write function.  The only N1-table reader below is prefix-bounded.
"""
from __future__ import annotations

import hashlib
import importlib
import importlib.util
import json
import math
import random
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any, Iterable

import numpy as np
import pandas as pd
import torch
import yaml
from torch import nn


PHASE = Path(__file__).resolve().parents[1]
PROJECT = PHASE.parent
REPO = PROJECT.parent
BS = PROJECT / "Phase_BS_Final_Backbone_Screening"
DF = PROJECT / "Phase_DF_Leakage_Safe_Decomposition_Features"
PROTOCOL = PHASE / "configs" / "PREREGISTERED_POT_PROTOCOL.yaml"
PROTOCOL_SHA = PHASE / "configs" / "PREREGISTERED_POT_PROTOCOL.sha256"
BS_PROTOCOL = BS / "configs" / "PREREGISTERED_PROTOCOL.yaml"
BS_PROTOCOL_SHA = BS / "configs" / "PREREGISTERED_PROTOCOL.sha256"
DF_PROTOCOL = DF / "configs" / "PREREGISTERED_DF_PROTOCOL.yaml"
DF_PROTOCOL_SHA = DF / "configs" / "PREREGISTERED_DF_PROTOCOL.sha256"
N1_TABLE = PROJECT / "Phase_N1_Data_Foundation" / "outputs" / "causal_feature_tables" / "causal_features_30min.csv"


class POTFailure(RuntimeError):
    """Fail-closed protocol, cache, or numerical error."""


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    """File hash for small frozen artifacts only; never use on N1_TABLE."""
    if path.resolve() == N1_TABLE.resolve():
        raise POTFailure("DATA_FIREWALL_FAIL: full causal CSV hashing is forbidden")
    return sha256_bytes(path.read_bytes())


def phase_path(*parts: str) -> Path:
    path = PHASE.joinpath(*parts)
    resolved = path.resolve()
    if resolved != PHASE.resolve() and PHASE.resolve() not in resolved.parents:
        raise POTFailure(f"Refusing Phase POT path escape: {path}")
    return path


def write_json(value: Any, *parts: str) -> Path:
    path = phase_path(*parts)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path


def write_text(value: str, *parts: str) -> Path:
    path = phase_path(*parts)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value.rstrip() + "\n", encoding="utf-8")
    return path


def write_csv(frame: pd.DataFrame, *parts: str) -> Path:
    path = phase_path(*parts)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, encoding="utf-8-sig", lineterminator="\n")
    return path


def git_head() -> str:
    return subprocess.check_output(["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True).strip()


def _load_hashed_yaml(path: Path, digest_path: Path) -> tuple[dict[str, Any], str]:
    expected = digest_path.read_text(encoding="utf-8").split()[0].lower()
    observed = sha256_file(path)
    if observed != expected:
        raise POTFailure(f"PROTOCOL_DRIFT: SHA256 mismatch for {path.name}")
    return yaml.safe_load(path.read_text(encoding="utf-8")), observed


def load_protocol() -> tuple[dict[str, Any], str]:
    cfg, digest = _load_hashed_yaml(PROTOCOL, PROTOCOL_SHA)
    if cfg.get("status") != "FROZEN_BEFORE_STAGE0_AND_TRAINING":
        raise POTFailure("PROTOCOL_DRIFT: POT protocol is not frozen")
    base = str(cfg["base_commit"])
    if subprocess.run(["git", "-C", str(REPO), "merge-base", "--is-ancestor", base, "HEAD"], check=False).returncode != 0:
        raise POTFailure("BASE_COMMIT_NOT_ANCESTOR")
    firewall = cfg["data_firewall"]
    if int(firewall["pandas_read_csv_nrows"]) != 3007 or int(firewall["expected_development_rows"]) != 3005:
        raise POTFailure("PROTOCOL_DRIFT: registered prefix contract changed")
    if cfg["task"]["context_steps"] != 48 or cfg["task"]["horizon_steps"] != 24:
        raise POTFailure("PROTOCOL_DRIFT: forecasting information set changed")
    return cfg, digest


def load_inherited_protocols(cfg: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any], ModuleType]:
    bs_cfg, bs_digest = _load_hashed_yaml(BS_PROTOCOL, BS_PROTOCOL_SHA)
    df_cfg, df_digest = _load_hashed_yaml(DF_PROTOCOL, DF_PROTOCOL_SHA)
    inherited = cfg["inherited_evidence"]
    if bs_digest != inherited["phase_bs_protocol_sha256"] or df_digest != inherited["phase_df_protocol_sha256"]:
        raise POTFailure("PROTOCOL_DRIFT: inherited Phase BS/DF protocol fingerprint differs")
    if bs_cfg["task"]["input_shape"] != ["batch", 48, 14] or df_cfg["task"]["augmented_dimension"] != 16:
        raise POTFailure("PROTOCOL_DRIFT: inherited input contract differs")
    spec = importlib.util.spec_from_file_location("pot_phase_bs_common", BS / "scripts" / "backbone_common.py")
    if spec is None or spec.loader is None:
        raise POTFailure("IMPLEMENTATION_FAIL: Phase BS common helper is unavailable")
    module = importlib.util.module_from_spec(spec)
    sys.modules["pot_phase_bs_common"] = module
    spec.loader.exec_module(module)
    return bs_cfg, df_cfg, module


def source_audit(common: ModuleType, bs_cfg: dict[str, Any]) -> pd.DataFrame:
    audit = common.source_fingerprint_audit(bs_cfg)
    timesnet = audit.loc[audit["model"].eq("TimesNet")].copy()
    if timesnet.empty or not bool(timesnet["pass"].all()):
        raise POTFailure("IMPLEMENTATION_FAIL: frozen official TimesNet source verification failed")
    return timesnet


def safe_read_development(cfg: dict[str, Any], bs_cfg: dict[str, Any], extra_columns: Iterable[str] = ()) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Read exactly N1 rows [0, 3006], then retain the registered Development rows.

    Extra columns can only support a Stage-0 audit; they never become forecast
    model inputs.  This function intentionally has no full-file hash or read.
    """
    firewall = cfg["data_firewall"]
    if N1_TABLE.name != "causal_features_30min.csv" or int(firewall["pandas_read_csv_nrows"]) != 3007:
        raise POTFailure("DATA_FIREWALL_FAIL: source or registered prefix changed")
    base = list(dict.fromkeys(bs_cfg["data_firewall"]["required_columns"]))
    columns = list(dict.fromkeys(base + list(extra_columns)))
    prefix = pd.read_csv(N1_TABLE, usecols=columns, nrows=3007)
    if len(prefix) != 3007:
        raise POTFailure("DEVELOPMENT_PREFIX_ROW_COUNT_MISMATCH")
    development = prefix.loc[prefix["bin_end_ophrs2_h"].to_numpy(float) < 1819.997738].copy().reset_index(drop=True)
    if len(development) != 3005 or np.any(development["bin_end_ophrs2_h"].to_numpy(float) >= 1819.997738):
        raise POTFailure("DEVELOPMENT_PREFIX_ROW_COUNT_MISMATCH")
    if not np.all(np.diff(development["bin_start_ophrs2_h"].to_numpy(float)) > 0):
        raise POTFailure("DATA_FIREWALL_FAIL: Development prefix is not time ordered")
    payload = development.to_csv(index=False, header=True, lineterminator="\n", float_format="%.17g").encode("utf-8")
    audit = {
        "source_relative_path": str(N1_TABLE.relative_to(PROJECT)).replace("\\", "/"),
        "pandas_read_csv_nrows": 3007,
        "prefix_rows_read": int(len(prefix)),
        "development_filter": "bin_end_ophrs2_h < 1819.997738",
        "development_rows_retained": int(len(development)),
        "rows_after_registered_prefix_read": 0,
        "full_causal_file_hash_computed": False,
        "full_causal_file_read_detected": False,
        "development_prefix_fingerprint_sha256": sha256_bytes(payload),
        "fingerprint_columns": list(development.columns),
        "test_accessed": False,
        "test_rows_read": 0,
        "test_targets_read": 0,
    }
    return development, audit


def build_frozen_folds(frame: pd.DataFrame, cfg: dict[str, Any], bs_cfg: dict[str, Any], common: ModuleType) -> list[tuple[str, Any, Any]]:
    folds = common.build_outer_folds(frame, bs_cfg)
    expected = cfg["expected_windows"]
    manifest: list[dict[str, Any]] = []
    for fold, outer_train, outer_validation in folds:
        fit, stopper = common.split_inner_fit_stopper(outer_train, float(cfg["inner_stopping"]["stopper_fraction"]))
        observed = {
            "outer_train_origins": len(outer_train.origin_row),
            "outer_validation_origins": len(outer_validation.origin_row),
            "inner_fit_origins": len(fit.origin_row),
            "inner_stopper_origins": len(stopper.origin_row),
        }
        if observed != expected[fold]:
            raise POTFailure(f"WINDOW_UNIVERSE_MISMATCH: {fold}: {observed}")
        if set(fit.target_row.reshape(-1)) & set(stopper.target_row.reshape(-1)):
            raise POTFailure("TARGET_ROW_ISOLATION_FAIL")
        if np.any(outer_validation.target_end_age_h >= 1819.997738):
            raise POTFailure("TEST_FIREWALL_FAIL: validation target crosses Development boundary")
        manifest.append({"fold": fold, **observed, "inner_target_overlap_count": 0, "all_validation_origins_retained": True})
    if [fold for fold, _, _ in folds] != ["fold_1", "fold_2", "fold_3"]:
        raise POTFailure("WINDOW_UNIVERSE_MISMATCH: fold identity")
    write_csv(pd.DataFrame(manifest), "outputs", "POT_WINDOW_MANIFEST.csv")
    return folds


def _input_voltage_sha(values: np.ndarray) -> str:
    return sha256_bytes(np.ascontiguousarray(np.asarray(values, dtype=np.float64)).tobytes())


def load_emd_cache_additions(windows: Any) -> dict[int, np.ndarray]:
    """Load only frozen EMD caches for the given legal Development origins."""
    manifest_path = DF / "outputs" / "CACHE_PROVENANCE_MANIFEST.csv"
    manifest_sha = DF / "outputs" / "CACHE_PROVENANCE_MANIFEST.sha256"
    if sha256_file(manifest_path) != manifest_sha.read_text(encoding="utf-8").split()[0].lower():
        raise POTFailure("EMD_CACHE_MANIFEST_HASH_MISMATCH")
    manifest = pd.read_csv(manifest_path)
    emd = manifest.loc[manifest["method"].eq("EMD")].copy().set_index("origin_row")
    if not emd.index.is_unique:
        raise POTFailure("EMD_CACHE_MANIFEST_DUPLICATE_ORIGIN")
    additions: dict[int, np.ndarray] = {}
    root = (DF / "cache" / "EMD").resolve()
    for i, origin in enumerate(windows.origin_row.tolist()):
        origin = int(origin)
        if origin not in emd.index:
            raise POTFailure(f"EMD_CACHE_MISSING_ORIGIN_{origin}")
        row = emd.loc[origin]
        path = Path(str(row["cache_path"])).resolve()
        if root not in path.parents:
            raise POTFailure("EMD_CACHE_PATH_OUTSIDE_FROZEN_ROOT")
        with np.load(path, allow_pickle=False) as archive:
            features = np.asarray(archive["features"])
            stored_origin = int(np.asarray(archive["origin_row"]).item())
            stored_input_sha = str(np.asarray(archive["input_voltage_sha256"]).item())
        if stored_origin != origin or features.shape != (48, 2) or features.dtype != np.float32 or not np.isfinite(features).all():
            raise POTFailure(f"EMD_CACHE_INVALID_{origin}")
        history = np.asarray(windows.x[i, :, 0], dtype=np.float64)
        if stored_input_sha != _input_voltage_sha(history) or str(row["cache_sha256"]) != sha256_bytes(np.ascontiguousarray(features).tobytes()):
            raise POTFailure(f"EMD_CACHE_PROVENANCE_MISMATCH_{origin}")
        additions[origin] = features
    return additions


def augment_emd(windows: Any, additions: dict[int, np.ndarray]) -> Any:
    extra = []
    for origin in windows.origin_row.tolist():
        value = additions.get(int(origin))
        if value is None:
            raise POTFailure("EMD_CACHE_MISSING_FOR_WINDOW")
        extra.append(value)
    x = np.concatenate([windows.x, np.asarray(extra, dtype=np.float32)], axis=-1)
    if x.shape != (len(windows.origin_row), 48, 16) or not np.array_equal(x[:, :, :14], windows.x):
        raise POTFailure("EMD_AUGMENTATION_CONTRACT_FAIL")
    return replace(windows, x=x)


def set_seed(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TimesNetVoltageAdapter(nn.Module):
    """Frozen official TimesNet body with only registered channel count adaptation."""
    def __init__(self, bs_cfg: dict[str, Any], input_dimension: int, dropout: float) -> None:
        super().__init__()
        source = dict(bs_cfg["official_sources"]["TimesNet"]["config"])
        source.update({"enc_in": int(input_dimension), "c_out": int(input_dimension), "dropout": float(dropout)})
        root = BS / bs_cfg["official_sources"]["TimesNet"]["vendor_root"]
        for name in list(sys.modules):
            if name in {"models", "layers"} or name.startswith(("models.", "layers.")):
                del sys.modules[name]
        sys.path.insert(0, str(root))
        try:
            official = importlib.import_module("models.TimesNet")
        finally:
            sys.path.remove(str(root))
        self.official_model = official.Model(SimpleNamespace(label_len=0, num_class=0, **source))
        self.input_dimension = int(input_dimension)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.official_model(x, None, None, None)
        if out.shape != (x.shape[0], 24, self.input_dimension):
            raise POTFailure(f"TIMESNET_OUTPUT_SHAPE_FAIL: {tuple(out.shape)}")
        return out[:, :, 0]


def model_delta_prediction(model: nn.Module, x_scaled: torch.Tensor) -> torch.Tensor:
    delta = model(x_scaled) - x_scaled[:, -1, 0:1]
    if delta.shape != (x_scaled.shape[0], 24):
        raise POTFailure("TIMESNET_DELTA_OUTPUT_CONTRACT_FAIL")
    return delta


def parameter_count(model: nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))


def candidate_polarization_order_loss(predicted_absolute_voltage: torch.Tensor, high_index: torch.Tensor, low_index: torch.Tensor) -> torch.Tensor:
    """Differentiable candidate only; Stage 0 must approve it before training use."""
    if high_index.numel() == 0:
        return predicted_absolute_voltage.new_zeros(())
    return torch.relu(predicted_absolute_voltage[high_index] - predicted_absolute_voltage[low_index]).mean()


def finite_gradient_norm(loss: torch.Tensor, parameters: Iterable[torch.nn.Parameter]) -> float:
    grads = torch.autograd.grad(loss, tuple(parameters), allow_unused=True, retain_graph=False)
    pieces = [g.detach().reshape(-1) for g in grads if g is not None]
    if not pieces:
        return 0.0
    value = float(torch.linalg.vector_norm(torch.cat(pieces)).item())
    if not math.isfinite(value):
        raise POTFailure("PHYSICS_GRADIENT_NONFINITE")
    return value

