"""Fail-closed, source-only CFPT protocol gate for Phase BG.

It verifies immutable source blobs and inspects the two official ASTs.  It
does not import a forecasting model, read the Development CSV, or construct
any model score.  The gate specifically asks whether official CFPT can be
challenged under the frozen Raw-F2 protocol or a *functionally* equal
known-time protocol with unmodified official TimesNet.
"""
from __future__ import annotations

import ast
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from legacy_protocol import PHASE, load_legacy_contract, write_csv, write_json


CANDIDATE = "CFPT"
ROOT = PHASE / "03_CFPT"
CONFIG = ROOT / "PREREGISTERED_CONFIG.yaml"
CONFIG_SHA = ROOT / "PREREGISTERED_CONFIG.sha256"


def load_config() -> dict[str, Any]:
    raw = CONFIG.read_bytes()
    expected = CONFIG_SHA.read_text(encoding="utf-8").split()[0].lower()
    observed = hashlib.sha256(raw).hexdigest()
    if observed != expected:
        raise RuntimeError("CFPT preregistered protocol-gate SHA256 mismatch")
    cfg = yaml.safe_load(raw)
    if cfg.get("status") != "FROZEN_PROTOCOL_GATE_BEFORE_DEVELOPMENT_RESULTS":
        raise RuntimeError("CFPT protocol configuration is not frozen")
    return cfg


def _fingerprint(repo: Path, source: dict[str, Any], label: str) -> list[dict[str, Any]]:
    observed_commit = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
    records: list[dict[str, Any]] = []
    for relative, expected in source["canonical_source_sha256"].items():
        raw = subprocess.check_output(["git", "-C", str(repo), "show", f"{source['commit']}:{relative}"])
        observed = hashlib.sha256(raw).hexdigest()
        records.append({"role": label, "repository": source["repository"], "expected_commit": source["commit"], "observed_commit": observed_commit, "relative_path": relative, "expected_sha256": expected, "observed_sha256": observed, "pass": observed_commit == source["commit"] and observed == expected})
    return records


def source_fingerprints(cfg: dict[str, Any]) -> pd.DataFrame:
    cfpt = ROOT / "vendor" / "CFPT_official"
    timesnet = PHASE.parent / cfg["control_source"]["vendor_root"]
    records = _fingerprint(cfpt, cfg["source"], "CFPT") + _fingerprint(timesnet, cfg["control_source"], "TimesNet_control")
    frame = pd.DataFrame(records)
    if not frame["pass"].all():
        raise RuntimeError("CFPT or TimesNet source fingerprint mismatch")
    return frame


def _class_method(tree: ast.AST, class_name: str, method_name: str) -> ast.FunctionDef:
    for item in ast.walk(tree):
        if isinstance(item, ast.ClassDef) and item.name == class_name:
            for child in item.body:
                if isinstance(child, ast.FunctionDef) and child.name == method_name:
                    return child
    raise RuntimeError(f"Cannot locate {class_name}.{method_name} in verified source")


def _body_name_uses(function: ast.FunctionDef, name: str) -> int:
    return sum(isinstance(node, ast.Name) and node.id == name for node in ast.walk(ast.Module(body=function.body, type_ignores=[])))


def analyze_protocol_source(cfg: dict[str, Any]) -> dict[str, Any]:
    """Analyse source-controlled use of decoder timestamp marks, not scores."""
    cfpt_source = (ROOT / "vendor" / "CFPT_official" / "models" / "CFPT.py").read_text(encoding="utf-8")
    timesnet_source = (PHASE.parent / cfg["control_source"]["vendor_root"] / "models" / "TimesNet.py").read_text(encoding="utf-8")
    cfpt_tree, timesnet_tree = ast.parse(cfpt_source), ast.parse(timesnet_source)
    cfpt_forecast = _class_method(cfpt_tree, "Model", "forecast")
    cfpt_forward = _class_method(cfpt_tree, "Model", "forward")
    timesnet_forecast = _class_method(timesnet_tree, "Model", "forecast")
    checks = {
        "cfpt_forecast_uses_x_mark_dec": _body_name_uses(cfpt_forecast, "x_mark_dec") > 0,
        "cfpt_forward_uses_x_mark_dec": _body_name_uses(cfpt_forward, "x_mark_dec") > 0,
        "cfpt_time_enc_call_present": "self.time_enc(x_mark_dec)" in cfpt_source,
        "cfpt_time_frequency_blend_present": "self.beta * freq_pred + (1 - self.beta) * time_pred" in cfpt_source,
        "timesnet_forecast_uses_x_mark_enc": _body_name_uses(timesnet_forecast, "x_mark_enc") > 0,
        "timesnet_forecast_uses_x_mark_dec": _body_name_uses(timesnet_forecast, "x_mark_dec") > 0,
        "timesnet_encoder_embedding_receives_x_mark_enc": "self.enc_embedding(x_enc, x_mark_enc)" in timesnet_source,
    }
    required_cfpt_timestamp = all(checks[key] for key in ["cfpt_forecast_uses_x_mark_dec", "cfpt_forward_uses_x_mark_dec", "cfpt_time_enc_call_present", "cfpt_time_frequency_blend_present"])
    timesnet_functionally_consumes_decoder_timestamp = checks["timesnet_forecast_uses_x_mark_dec"]
    route_a_supported = not required_cfpt_timestamp
    route_b_supported = required_cfpt_timestamp and timesnet_functionally_consumes_decoder_timestamp
    route = "RAW_F2_HISTORY_ONLY" if route_a_supported else "EQUAL_KNOWN_TIME" if route_b_supported else "INCOMPATIBLE"
    return {
        "candidate": CANDIDATE,
        "route": route,
        "cfpt_requires_decoder_timestamp": required_cfpt_timestamp,
        "timesnet_functionally_consumes_decoder_timestamp": timesnet_functionally_consumes_decoder_timestamp,
        "route_a_supported_by_author_source": route_a_supported,
        "route_b_functional_information_equality_supported": route_b_supported,
        "checks": checks,
        "reason": "CFPT unconditionally encodes x_mark_dec and blends it into prediction, while verified official TimesNet ignores x_mark_dec in forecast; equality would require a prohibited source/body or Raw-F2 change.",
        "development_data_read": False,
        "development_score_computed": False,
        "resource_preflight": False,
        "model_training": False,
        "test_accessed": False,
    }


def main() -> None:
    cfg = load_config()
    _legacy_cfg, _helpers, legacy = load_legacy_contract()
    fingerprints = source_fingerprints(cfg)
    result = analyze_protocol_source(cfg)
    if result["route"] != cfg["protocol_gate"]["frozen_result_before_development_results"]:
        raise RuntimeError("CFPT protocol result differs from frozen pre-result determination")
    payload = {**result, "legacy_protocol_sha256": legacy["legacy_protocol_sha256"], "legacy_protected_bs_drift": legacy["protected_bs_drift"]}
    write_csv(fingerprints, "03_CFPT", "SOURCE_FINGERPRINTS.csv")
    write_json(payload, "03_CFPT", "PROTOCOL_GATE_AUDIT.json")
    write_json({"candidate": CANDIDATE, "status": "NOT_RUN", "reason": "CFPT_PROTOCOL_INCOMPATIBLE before resource gate; no model construction, score, training, or Test access.", "test_accessed": False}, "03_CFPT", "RESOURCE_AUDIT.json")
    write_json({"candidate": CANDIDATE, "final_status": "CFPT_PROTOCOL_INCOMPATIBLE", "protocol": result["route"], "next_action": "BACKBONE_SEARCH_CLOSED", "development_score_computed": False, "test_accessed": False}, "03_CFPT", "CANDIDATE_DECISION.json")
    print(json.dumps({"candidate": CANDIDATE, "protocol": result["route"], "status": "CFPT_PROTOCOL_INCOMPATIBLE"}, ensure_ascii=False))


if __name__ == "__main__":
    main()
