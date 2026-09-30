"""Read-only adapter for the frozen Phase BS RealHyFC protocol.

The historical Phase BS ``load_config`` head guard remains untouched.  This
adapter verifies its frozen YAML and source fingerprints, then reuses the
historical window/scaler/metric behavior without requiring the current Git HEAD
to equal Phase BS's original execution commit.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
import torch
import yaml
from torch import nn


DF = Path(__file__).resolve().parents[1]
PROJECT = DF.parent
REPO = PROJECT.parent
BS = PROJECT / "Phase_BS_Final_Backbone_Screening"
DF_PROTOCOL = DF / "configs" / "PREREGISTERED_DF_PROTOCOL.yaml"
DF_PROTOCOL_SHA = DF / "configs" / "PREREGISTERED_DF_PROTOCOL.sha256"
BS_PROTOCOL = BS / "configs" / "PREREGISTERED_PROTOCOL.yaml"
BS_PROTOCOL_SHA = BS / "configs" / "PREREGISTERED_PROTOCOL.sha256"


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def safe_df_path(*parts: str) -> Path:
    path = DF.joinpath(*parts)
    resolved = path.resolve()
    if resolved != DF.resolve() and DF.resolve() not in resolved.parents:
        raise RuntimeError(f"Refusing output outside Phase DF: {path}")
    return path


def write_json(payload: Any, *parts: str) -> Path:
    path = safe_df_path(*parts)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path


def write_text(text: str, *parts: str) -> Path:
    path = safe_df_path(*parts)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")
    return path


def write_csv(frame: pd.DataFrame, *parts: str) -> Path:
    path = safe_df_path(*parts)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, encoding="utf-8-sig", lineterminator="\n")
    return path


def _verify_yaml_hash(path: Path, sha_path: Path) -> tuple[dict[str, Any], str]:
    expected = sha_path.read_text(encoding="utf-8").split()[0].lower()
    observed = sha256_file(path)
    if observed != expected:
        raise RuntimeError(f"PROTOCOL_DRIFT: SHA256 mismatch for {path.name}")
    return yaml.safe_load(path.read_bytes()), observed


def load_df_protocol() -> tuple[dict[str, Any], str]:
    cfg, observed = _verify_yaml_hash(DF_PROTOCOL, DF_PROTOCOL_SHA)
    if cfg.get("status") != "FROZEN_BEFORE_REALHYFC_STAGE0":
        raise RuntimeError("PROTOCOL_DRIFT: Phase DF protocol not frozen")
    if cfg["task"]["context_steps"] != 48 or cfg["task"]["horizon_steps"] != 24:
        raise RuntimeError("PROTOCOL_DRIFT: information set changed")
    if cfg["vmd"]["K"] != 2 or cfg["vmd"]["DC"] != 0:
        raise RuntimeError("PROTOCOL_DRIFT: VMD structure changed")
    return cfg, observed


def load_bs_protocol(df_cfg: dict[str, Any]) -> tuple[dict[str, Any], str]:
    cfg, observed = _verify_yaml_hash(BS_PROTOCOL, BS_PROTOCOL_SHA)
    expected = str(df_cfg["phase_bs_protocol"]["sha256"])
    if observed != expected:
        raise RuntimeError("PROTOCOL_DRIFT: Phase BS protocol does not match frozen DF reference")
    if cfg["task"]["input_shape"] != ["batch", 48, 14]:
        raise RuntimeError("PROTOCOL_DRIFT: Phase BS task shape changed")
    if list(cfg["data_firewall"]["f2_raw_feature_order"]) != list(df_cfg["task"]["raw_f2_feature_order"]):
        raise RuntimeError("PROTOCOL_DRIFT: Raw F2 order changed")
    if float(cfg["development_end_h_exclusive"]) != float(df_cfg["data_firewall"]["development_end_h_exclusive"]):
        raise RuntimeError("PROTOCOL_DRIFT: Development boundary changed")
    return cfg, observed


def load_bs_common_readonly() -> ModuleType:
    """Import Phase BS helpers without invoking its historical HEAD guard."""
    path = BS / "scripts" / "backbone_common.py"
    name = "phase_bs_backbone_common_readonly_adapter"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError("PROTOCOL_DRIFT: Phase BS backbone_common cannot be imported")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def verify_bs_source_and_adapter(df_cfg: dict[str, Any], bs_cfg: dict[str, Any]) -> pd.DataFrame:
    common = load_bs_common_readonly()
    # The historical fingerprint utility checks the frozen vendor commits/blob
    # hashes.  It never reads the causal CSV or invokes historical load_config().
    fingerprints = common.source_fingerprint_audit(bs_cfg)
    timesnet = fingerprints.loc[fingerprints["model"].eq("TimesNet")].copy()
    if timesnet.empty or not bool(timesnet["pass"].all()):
        raise RuntimeError("PROTOCOL_DRIFT: frozen TimesNet source fingerprint failed")
    if str(bs_cfg["official_sources"]["TimesNet"]["commit"]) != str(df_cfg["phase_bs_protocol"]["timesnet_official_commit"]):
        raise RuntimeError("PROTOCOL_DRIFT: official TimesNet commit differs from DF protocol")
    timesnet["adapter_uses_historical_windows_scaler_metrics"] = True
    timesnet["historical_head_guard_preserved_not_invoked"] = True
    return timesnet


def read_development_prefix(df_cfg: dict[str, Any], bs_cfg: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Read exactly the protected N1 prefix and reject all post-boundary rows."""
    firewall = df_cfg["data_firewall"]
    registered_relative_path = Path(firewall["source_relative_path"])
    expected_relative_path = Path("Phase_N1_Data_Foundation/outputs/causal_feature_tables/causal_features_30min.csv")
    if registered_relative_path != expected_relative_path:
        raise RuntimeError("DATA_FIREWALL_FAIL: only the frozen N1 30-minute source path is permitted")
    if any("test" in part.lower() for part in registered_relative_path.parts):
        raise RuntimeError("DATA_FIREWALL_FAIL: Test-labelled input paths are forbidden")
    nrows = int(firewall["pandas_read_csv_nrows"])
    if nrows != 3007:
        raise RuntimeError("PROTOCOL_DRIFT: DF reader must use nrows=3007")
    source = PROJECT / registered_relative_path
    columns = list(dict.fromkeys(bs_cfg["data_firewall"]["required_columns"]))
    prefix = pd.read_csv(source, usecols=columns, nrows=nrows)
    if len(prefix) != int(firewall["expected_raw_prefix_rows"]):
        raise RuntimeError("DATA_FIREWALL_FAIL: unexpected protected prefix length")
    development_end = float(firewall["development_end_h_exclusive"])
    development = prefix.loc[prefix["bin_end_ophrs2_h"].to_numpy(float) < development_end].copy().reset_index(drop=True)
    if len(development) != int(firewall["expected_development_rows_after_filter"]):
        raise RuntimeError("DATA_FIREWALL_FAIL: expected 3005 Development rows after end filter")
    if np.any(development["bin_end_ophrs2_h"].to_numpy(float) >= development_end):
        raise RuntimeError("DATA_FIREWALL_FAIL: post-boundary row entered Development")
    audit = {
        "source": str(source),
        "raw_prefix_rows_read": int(len(prefix)),
        "development_rows_after_filter": int(len(development)),
        "boundary_rows_filtered": int(len(prefix) - len(development)),
        "post_boundary_rows_entered_pipeline": 0,
        "test_accessed": False,
        "test_rows_read": 0,
        "test_targets_read": 0,
        "test_predictions_constructed": False,
        "test_metrics_computed": False,
    }
    if audit["boundary_rows_filtered"] != 2:
        raise RuntimeError("DATA_FIREWALL_FAIL: registered prefix did not filter exactly two boundary rows")
    return development, audit


def build_frozen_folds(frame: pd.DataFrame, bs_cfg: dict[str, Any], df_cfg: dict[str, Any]) -> list[tuple[str, Any, Any]]:
    common = load_bs_common_readonly()
    folds = common.build_outer_folds(frame, bs_cfg)
    expected = df_cfg["expected_windows"]
    manifest_rows: list[dict[str, Any]] = []
    for fold, train, validation in folds:
        fit, stopper = common.split_inner_fit_stopper(train, .15)
        expected_row = expected[fold]
        observed = {
            "outer_train_origins": len(train.origin_row),
            "outer_validation_origins": len(validation.origin_row),
            "inner_fit_origins": len(fit.origin_row),
            "inner_stopper_origins": len(stopper.origin_row),
            "validation_max_target_end_age_h": float(validation.target_end_age_h.max()),
        }
        if any(not np.isclose(float(observed[key]), float(value), rtol=0, atol=1e-9) for key, value in expected_row.items()):
            raise RuntimeError(f"PROTOCOL_DRIFT: {fold} frozen window population changed: {observed}")
        if set(fit.target_row.reshape(-1)) & set(stopper.target_row.reshape(-1)):
            raise RuntimeError(f"PROTOCOL_DRIFT: {fold} inner target overlap")
        manifest_rows.append({"fold": fold, **observed, "inner_target_overlap_count": 0})
    if [fold for fold, _, _ in folds] != ["fold_1", "fold_2", "fold_3"]:
        raise RuntimeError("PROTOCOL_DRIFT: frozen fold identity changed")
    write_csv(pd.DataFrame(manifest_rows), "outputs", "WINDOW_MANIFEST_REVALIDATED.csv")
    return folds


def augment_windows(windows: Any, additions_by_origin: dict[int, np.ndarray]) -> Any:
    additions = []
    for origin in windows.origin_row.tolist():
        value = additions_by_origin.get(int(origin))
        if value is None or value.shape != (48, 2) or value.dtype != np.float32 or not np.isfinite(value).all():
            raise RuntimeError(f"CACHE_AUDIT_FAIL: invalid origin-conditioned cache for origin {origin}")
        additions.append(value)
    augmented = np.concatenate([windows.x, np.asarray(additions, dtype=np.float32)], axis=-1)
    if augmented.shape != (len(windows.origin_row), 48, 16):
        raise RuntimeError("CACHE_AUDIT_FAIL: augmented feature shape differs from 48x16 contract")
    return replace(windows, x=augmented)


class TimesNet16VoltageAdapter(nn.Module):
    """Official TimesNet with only the required 16-channel I/O dimensions changed."""
    def __init__(self, bs_cfg: dict[str, Any]) -> None:
        super().__init__()
        source = dict(bs_cfg["official_sources"]["TimesNet"]["config"])
        source["enc_in"] = 16
        source["c_out"] = 16
        root = BS / bs_cfg["official_sources"]["TimesNet"]["vendor_root"]
        common = load_bs_common_readonly()
        common._clear_absolute_model_packages()
        sys.path.insert(0, str(root))
        try:
            official = __import__("models.TimesNet", fromlist=["Model"])
        finally:
            sys.path.remove(str(root))
        self.official_model = official.Model(SimpleNamespace(label_len=0, num_class=0, **source))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.official_model(x, None, None, None)
        if out.shape != (x.shape[0], 24, 16):
            raise RuntimeError(f"IMPLEMENTATION_FAIL: TimesNet16 output shape {tuple(out.shape)}")
        return out[:, :, 0]


def model_delta_prediction(model: nn.Module, x_scaled: torch.Tensor) -> torch.Tensor:
    absolute = model(x_scaled)
    delta = absolute - x_scaled[:, -1, 0:1]
    if delta.shape != (x_scaled.shape[0], 24):
        raise RuntimeError("IMPLEMENTATION_FAIL: DeltaV adapter shape mismatch")
    return delta


def current_git_head() -> str:
    return subprocess.check_output(["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True).strip()
