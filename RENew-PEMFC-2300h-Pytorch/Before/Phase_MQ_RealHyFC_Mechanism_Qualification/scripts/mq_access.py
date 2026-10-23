"""Attempt-2 local-only data guard for Phase MQ."""
from __future__ import annotations

from pathlib import Path
from typing import Any
import json
import pandas as pd
import yaml

PHASE = Path(__file__).resolve().parents[1]
ALLOWLIST = PHASE / "configs" / "ATTEMPT_2_LOCAL_SOURCE_ALLOWLIST.yaml"


class DataAccessGuard:
    def __init__(self, n1_columns: list[str]):
        self.spec = yaml.safe_load(ALLOWLIST.read_text(encoding="utf-8"))
        self.n1 = Path(self.spec["dynamic_source"]["path"]).resolve()
        root = Path(self.spec["polarization_sources"]["raw_root"]).resolve()
        self.polar = {(root / name).resolve() for name in self.spec["polarization_sources"]["allowed_files"]}
        self.n1_columns = list(n1_columns)
        self.events: list[dict[str, Any]] = []

    def _check(self, path: Path, *, nrows: int | None, usecols: list[str] | None) -> str:
        raw = str(path)
        if "://" in raw or any(ch in raw for ch in "*?["):
            raise RuntimeError("URL and wildcard data paths are forbidden")
        target = path.resolve()
        name = target.name.lower()
        if "polarization_checkpoint_jstar_measurements" in name or any(x in name for x in ("2000", "2300", "afterairair")):
            raise RuntimeError("Mixed/Test polarization source is forbidden")
        if target == self.n1:
            if nrows != 3007 or usecols != self.n1_columns:
                raise RuntimeError("N1 must be read with exactly nrows=3007 and frozen effective columns")
            return "n1_prefix"
        if target not in self.polar:
            raise RuntimeError(f"Path is not in Attempt-2 allowlist: {target}")
        return "development_polarization"

    def read_csv(self, path: str | Path, **kwargs: Any) -> pd.DataFrame:
        target = Path(path)
        kind = self._check(target, nrows=kwargs.get("nrows"), usecols=kwargs.get("usecols"))
        if kwargs.get("chunksize"):
            raise RuntimeError("Chunk iterators must use iter_csv(), so reads are logged once")
        frame = pd.read_csv(target, **kwargs)
        self.events.append({"path": str(target.resolve()), "scope": kind, "nrows": kwargs.get("nrows"), "usecols": kwargs.get("usecols"), "rows_returned": len(frame)})
        return frame

    def iter_csv(self, path: str | Path, **kwargs: Any):
        target = Path(path)
        kind = self._check(target, nrows=kwargs.get("nrows"), usecols=kwargs.get("usecols"))
        chunksize = kwargs.get("chunksize")
        if not chunksize:
            raise RuntimeError("iter_csv requires a fixed chunksize")
        self.events.append({"path": str(target.resolve()), "scope": kind, "nrows": kwargs.get("nrows"), "usecols": kwargs.get("usecols"), "chunksize": chunksize})
        yield from pd.read_csv(target, **kwargs)

    def write_audit(self, path: Path, **extra: Any) -> None:
        payload = {"read_events": self.events, **extra}
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
