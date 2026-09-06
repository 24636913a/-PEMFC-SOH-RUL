"""Shared, fail-closed utilities for Phase N1.

The module never writes outside the configured Phase N1 directory and never
modifies source data.  All CSV output uses UTF-8 with BOM for Excel-friendly
inspection while preserving machine-readable column names.
"""

from __future__ import annotations

import hashlib
import math
import re
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import yaml


SCRIPT_DIR = Path(__file__).resolve().parent
PHASE_ROOT = SCRIPT_DIR.parent
CONFIG_PATH = PHASE_ROOT / "configs" / "phase_n1_config.yaml"


def load_config() -> dict[str, Any]:
    with CONFIG_PATH.open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    required = ("raw_data_root", "project_root", "phase_root", "ageing_coordinate")
    missing = [key for key in required if key not in config]
    if missing:
        raise KeyError(f"Missing required config keys: {missing}")
    configured_phase = Path(config["phase_root"]).resolve()
    if configured_phase != PHASE_ROOT.resolve():
        raise RuntimeError(
            f"Configured phase root {configured_phase} does not match script location {PHASE_ROOT.resolve()}"
        )
    raw_root = Path(config["raw_data_root"])
    if not raw_root.is_dir():
        raise FileNotFoundError(f"Raw data root not found: {raw_root}")
    return config


def ensure_output_path(path: Path) -> Path:
    resolved_phase = PHASE_ROOT.resolve()
    resolved_parent = path.parent.resolve()
    if resolved_parent != resolved_phase and resolved_phase not in resolved_parent.parents:
        raise RuntimeError(f"Refusing to write outside Phase N1: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def write_csv(frame: pd.DataFrame, path: Path) -> None:
    ensure_output_path(path)
    frame.to_csv(path, index=False, encoding="utf-8-sig", lineterminator="\n")


def write_text(text: str, path: Path) -> None:
    ensure_output_path(path)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def format_number(value: Any, digits: int = 6) -> str:
    if value is None or (isinstance(value, (float, np.floating)) and not math.isfinite(float(value))):
        return ""
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.{digits}g}"
    return str(value)


def normalize_numeric(series: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(series, errors="coerce")
    values = series.astype("string").str.replace("\ufeff", "", regex=False).str.strip()
    has_comma = values.str.contains(",", regex=False, na=False)
    has_dot = values.str.contains(".", regex=False, na=False)
    values = values.mask(has_comma & has_dot, values.str.replace(",", "", regex=False))
    values = values.mask(has_comma & ~has_dot, values.str.replace(",", ".", regex=False))
    return pd.to_numeric(values, errors="coerce")


def loadcycle_order(path: Path) -> tuple[int, int, str]:
    match = re.search(r"Loadcycle_(\d+)-(\d+)", path.stem, flags=re.IGNORECASE)
    if not match:
        return (10**9, 10**9, path.name)
    return (int(match.group(1)), int(match.group(2)), path.name)


def polarization_stage(path: Path) -> tuple[str, float, bool]:
    lower = path.stem.lower()
    after = "afterairair" in lower
    if "bol" in lower:
        return ("BoL", 0.0, after)
    match = re.search(r"(\d+)h", lower)
    if not match:
        raise ValueError(f"Cannot parse polarization ageing checkpoint from {path.name}")
    age = float(match.group(1))
    label = f"{int(age)}h" + ("_afterAirAirStart" if after else "")
    return (label, age, after)


def assign_checkpoint_split(age_h: float, cut60: float, cut80: float) -> str:
    if age_h < cut60:
        return "train"
    if age_h < cut80:
        return "validation"
    return "test"


def sha256_file(path: Path, block_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while True:
            block = stream.read(block_size)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def safe_mode(values: pd.Series) -> float:
    clean = values.dropna()
    if clean.empty:
        return math.nan
    counts = clean.value_counts(dropna=True)
    return float(counts.index[0])


def population_std(values: pd.Series) -> float:
    numeric = pd.to_numeric(values, errors="coerce")
    return float(numeric.std(ddof=0)) if numeric.notna().any() else math.nan


def finite_or_nan(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return math.nan
    return number if math.isfinite(number) else math.nan


def consecutive_true_counts(mask: np.ndarray) -> int:
    return int(np.count_nonzero(mask))


def markdown_table(frame: pd.DataFrame, columns: Iterable[str] | None = None) -> str:
    use = frame[list(columns)].copy() if columns is not None else frame.copy()
    if use.empty:
        return "_No rows._"
    rendered = use.fillna("").astype(str)
    header = "| " + " | ".join(rendered.columns) + " |"
    rule = "| " + " | ".join(["---"] * len(rendered.columns)) + " |"
    rows = ["| " + " | ".join(row) + " |" for row in rendered.itertuples(index=False, name=None)]
    return "\n".join([header, rule, *rows])
