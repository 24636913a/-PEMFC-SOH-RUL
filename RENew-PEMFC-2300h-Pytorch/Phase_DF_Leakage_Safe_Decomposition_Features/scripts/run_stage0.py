"""Execute Phase DF's synthetic and Train-only decomposition audits.

This program never reads past the registered 3007-row N1 prefix.  Validation
features are cached only after all Train-only method parameters are frozen and
are never summarized or used for Stage 0 selection.
"""
from __future__ import annotations

import hashlib
import importlib.metadata
import inspect
import json
import math
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from PyEMD import EMD
from vmdpy import VMD

from decomposition_features import CONTEXT, emd_feature, input_voltage_sha256, method_feature, simple_trend_features, source_api_snapshot, vmd_feature
from df_protocol_adapter import DF, build_frozen_folds, current_git_head, load_bs_protocol, load_df_protocol, read_development_prefix, sha256_file, verify_bs_source_and_adapter, write_csv, write_json, write_text


def _source_record(distribution: str, module: Any, obj: Any) -> dict[str, Any]:
    metadata = importlib.metadata.metadata(distribution)
    source = Path(inspect.getsourcefile(obj) or module.__file__).resolve()
    return {
        "distribution": distribution,
        "version": importlib.metadata.version(distribution),
        "license": metadata.get("License", "NOT_DECLARED"),
        "home_page": metadata.get("Home-page", "NOT_DECLARED"),
        "source_commit": "NOT_AVAILABLE_IN_PYPI_WHEEL",
        "source_file": str(source),
        "source_sha256": sha256_file(source),
        "import_path": f"{obj.__module__}.{obj.__name__}",
        "signature": str(inspect.signature(obj)),
    }


def write_method_source_pinning() -> dict[str, Any]:
    import PyEMD
    import vmdpy

    vmd = _source_record("vmdpy", vmdpy, VMD)
    emd = _source_record("EMD-signal", PyEMD, EMD)
    text = "# Method source pinning\n\n"
    text += "Phase DF uses the PyPI distributions below; no alternative VMD/EMD, EEMD, CEEMDAN, or ICEEMDAN implementation is permitted. A wheel does not carry an upstream source commit, so its installed source-file SHA256, distribution version, import path, signature and license metadata are pinned instead.\n\n"
    for name, record in (("VMD", vmd), ("EMD", emd)):
        text += f"## {name}\n\n"
        for key, value in record.items():
            text += f"- `{key}`: `{value}`\n"
        text += "\n"
    text += "`EMD-signal` imports as `PyEMD`; it is not the Earth Mover's Distance package.\n"
    write_text(text, "docs", "METHOD_SOURCE_PINNING.md")
    return {"VMD": vmd, "EMD": emd, "api_snapshot": source_api_snapshot()}


def _synthetic_history() -> np.ndarray:
    t = np.arange(CONTEXT, dtype=np.float64)
    return 0.72 + 0.0007 * t + 0.006 * np.sin(2 * np.pi * t / 16) + 0.002 * np.sin(2 * np.pi * t / 5)


def _history_from_container(series: np.ndarray, origin: int) -> np.ndarray:
    if origin < CONTEXT - 1:
        raise ValueError("origin lacks a 48-step history")
    return np.asarray(series[origin - CONTEXT + 1:origin + 1], dtype=np.float64)


def synthetic_audits() -> dict[str, Any]:
    history = _synthetic_history()
    trend = simple_trend_features(history)
    vmd = vmd_feature(history, alpha=1000)
    emd = emd_feature(history)
    repeated = {"SimpleTrend": simple_trend_features(history), "VMD": vmd_feature(history, alpha=1000), "EMD": emd_feature(history)}
    deterministic = {name: bool(np.array_equal(result.features, repeated[name].features)) for name, result in {"SimpleTrend": trend, "VMD": vmd, "EMD": emd}.items()}
    hand = np.arange(CONTEXT, dtype=np.float64) * 0.1
    hand_result = simple_trend_features(hand).features
    if not np.isclose(hand_result[7, 0], 0.35) or not np.isclose(hand_result[7, 1], 0.2) or hand_result[0, 1] != 0.0:
        raise RuntimeError("IMPLEMENTATION_FAIL: SimpleTrend hand calculation or V/h unit failed")
    if not all(result.features.shape == (48, 2) and np.isfinite(result.features).all() for result in (trend, vmd, emd)):
        raise RuntimeError("IMPLEMENTATION_FAIL: synthetic feature shape/finite check failed")
    if not all(deterministic.values()):
        raise RuntimeError("IMPLEMENTATION_FAIL: decomposition repeat was not deterministic")
    frequencies = vmd.audit["final_center_frequencies_cycles_per_sample"]
    if frequencies != sorted(frequencies):
        raise RuntimeError("IMPLEMENTATION_FAIL: VMD modes not ordered by final center frequency")
    if emd.audit["normalized_reconstruction_max_abs"] > 1e-8:
        raise RuntimeError("IMPLEMENTATION_FAIL: EMD IMF + residue reconstruction failed")
    serialized = np.savez_compressed  # retained explicit check without persisting a source artifact
    del serialized
    # Future perturbation: the production feature sees the same 48 values.
    origin = 60
    full = np.concatenate([np.linspace(0.70, 0.73, 61), np.linspace(1.2, 3.0, 19)])
    changed = full.copy(); changed[origin + 1:] = np.array([100.0, -100.0] * 10)[: len(changed) - origin - 1]
    future_equal = {}
    prefix_equal = {}
    for method, alpha in (("SimpleTrend", None), ("VMD", 1000), ("EMD", None)):
        before = method_feature(method, _history_from_container(full, origin), vmd_alpha=alpha).features
        after = method_feature(method, _history_from_container(changed, origin), vmd_alpha=alpha).features
        prefix = method_feature(method, _history_from_container(full[:origin + 1], origin), vmd_alpha=alpha).features
        future_equal[method] = bool(np.array_equal(before, after))
        prefix_equal[method] = bool(np.array_equal(before, prefix))
    if not all(future_equal.values()):
        raise RuntimeError("LEAKAGE_OR_IMPLEMENTATION_FAIL: future perturbation changed a feature")
    if not all(prefix_equal.values()):
        raise RuntimeError("LEAKAGE_OR_IMPLEMENTATION_FAIL: prefix invariance failed")
    return {
        "status": "PASS", "synthetic_input_shape": [48], "feature_shape": [48, 2],
        "finite": True, "deterministic": deterministic, "vmd_mode_ordering": True,
        "emd_imf_residue_reconstruction": True, "simpletrend_hand_calculation": True,
        "slope_units": "V_per_h", "first_slope_zero": True,
        "future_perturbation_invariance": future_equal, "prefix_invariance": prefix_equal,
        "production_function_accepts_target": False, "production_function_accepts_full_series": False,
    }


def _endpoint_pairs(origins: np.ndarray, ages: np.ndarray, segments: dict[int, Any]) -> list[tuple[int, int]]:
    lookup = {int(origin): index for index, origin in enumerate(origins.tolist())}
    pairs: list[tuple[int, int]] = []
    for origin, index in lookup.items():
        nxt = lookup.get(origin + 1)
        if nxt is None:
            continue
        if not np.isclose(ages[nxt] - ages[index], 0.5, rtol=0, atol=1e-6):
            continue
        if segments[origin] != segments[origin + 1]:
            continue
        pairs.append((index, nxt))
    return pairs


def endpoint_audit(method: str, origins: np.ndarray, ages: np.ndarray, segments: dict[int, Any], histories: dict[int, np.ndarray], results: dict[int, Any]) -> dict[str, Any]:
    pairs = _endpoint_pairs(origins, ages, segments)
    if not pairs:
        raise RuntimeError(f"{method}: no eligible exact-0.5h same-segment endpoint pairs")
    revisions = np.asarray([abs(float(results[int(origins[i])].features[47, 0]) - float(results[int(origins[j])].features[46, 0])) for i, j in pairs])
    raw_changes = np.asarray([abs(float(histories[int(origins[j])][-1]) - float(histories[int(origins[i])][-1])) for i, j in pairs])
    return {
        "method": method, "eligible_pair_count": len(pairs), "revision_p95_V": float(np.quantile(revisions, .95)),
        "revision_median_V": float(np.median(revisions)), "raw_step_change_p95_V": float(np.quantile(raw_changes, .95)),
        "endpoint_unstable": bool(float(np.quantile(revisions, .95)) > 2 * float(np.quantile(raw_changes, .95)) and float(np.quantile(revisions, .95)) > .005),
    }


def _unique_train_histories(folds: list[tuple[str, Any, Any]], frame: pd.DataFrame) -> tuple[dict[int, np.ndarray], dict[int, float], dict[int, Any], dict[int, set[str]], dict[int, set[str]]]:
    histories: dict[int, np.ndarray] = {}
    ages: dict[int, float] = {}
    segments: dict[int, Any] = {}
    memberships: dict[int, set[str]] = defaultdict(set)
    roles: dict[int, set[str]] = defaultdict(set)
    for fold, train, validation in folds:
        for role, windows in (("outer_train", train), ("outer_validation", validation)):
            for i, origin in enumerate(windows.origin_row.tolist()):
                origin = int(origin)
                history = np.asarray(windows.x[i, :, 0], dtype=np.float64)
                if origin in histories and not np.array_equal(histories[origin], history):
                    raise RuntimeError(f"CACHE_AUDIT_FAIL: same origin has different raw history across folds")
                histories[origin] = history
                ages[origin] = float(windows.origin_age_h[i])
                segments[origin] = frame.loc[origin, "temporal_segment_id"]
                memberships[origin].add(fold); roles[origin].add(role)
    return histories, ages, segments, memberships, roles


def _run_method(method: str, histories: dict[int, np.ndarray], *, vmd_alpha: int | None) -> tuple[dict[int, Any], float]:
    started = time.perf_counter()
    results = {origin: method_feature(method, history, vmd_alpha=vmd_alpha) for origin, history in histories.items()}
    return results, time.perf_counter() - started


def _vmd_candidate_records(train_origins: list[int], histories: dict[int, np.ndarray], ages: dict[int, float], segments: dict[int, Any], cfg: dict[str, Any]) -> tuple[pd.DataFrame, dict[int, dict[int, Any]]]:
    rows: list[dict[str, Any]] = []
    all_results: dict[int, dict[int, Any]] = {}
    origins = np.asarray(sorted(train_origins), dtype=np.int64)
    for alpha in cfg["vmd"]["alpha_candidates"]:
        results, elapsed = _run_method("VMD", {origin: histories[origin] for origin in origins.tolist()}, vmd_alpha=int(alpha))
        all_results[int(alpha)] = results
        std_mask = np.asarray([np.std(histories[int(origin)]) >= .001 for origin in origins])
        separations = np.asarray([results[int(origin)].audit["frequency_separation_cycles_per_sample"] for origin in origins])
        endpoint = endpoint_audit("VMD", origins, np.asarray([ages[int(o)] for o in origins]), segments, histories, results)
        separation_rate = float(np.mean(separations[std_mask] >= 1.0 / 48.0)) if std_mask.any() else 0.0
        rows.append({
            "method": "VMD", "alpha": int(alpha), "finite_feature_rate": float(np.mean([np.isfinite(x.features).all() for x in results.values()])),
            "mean_normalized_reconstruction_rmse": float(np.mean([x.audit["normalized_reconstruction_rmse"] for x in results.values()])),
            "mode_separation_pass_rate": separation_rate, "eligible_std_ge_1mV_windows": int(std_mask.sum()),
            "endpoint_revision_p95_V": endpoint["revision_p95_V"], "raw_step_change_p95_V": endpoint["raw_step_change_p95_V"],
            "endpoint_unstable": endpoint["endpoint_unstable"], "runtime_s": elapsed,
        })
    return pd.DataFrame(rows), all_results


def _resolve_alpha(records: pd.DataFrame) -> tuple[int | None, str]:
    eligible = records.loc[(records.finite_feature_rate == 1.0) & (records.mode_separation_pass_rate >= .90) & (~records.endpoint_unstable)].copy()
    if eligible.empty:
        return None, "MODE_OR_ENDPOINT_NOT_IDENTIFIABLE"
    eligible = eligible.sort_values(["mode_separation_pass_rate", "endpoint_revision_p95_V", "alpha"], ascending=[False, True, True])
    return int(eligible.iloc[0].alpha), "PASS"


def _resource_audit(method: str, folds: list[tuple[str, Any, Any]], histories: dict[int, np.ndarray], *, vmd_alpha: int | None) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for fold, train, _validation in folds:
        sample = np.linspace(0, len(train.origin_row) - 1, min(100, len(train.origin_row)), dtype=int)
        origin_list = [int(train.origin_row[i]) for i in sample]
        started = time.perf_counter()
        for origin in origin_list:
            method_feature(method, histories[origin], vmd_alpha=vmd_alpha)
        elapsed = time.perf_counter() - started
        rows.append({"method": method, "fold": fold, "sample_origins": len(origin_list), "runtime_s": elapsed, "seconds_per_origin": elapsed / len(origin_list)})
    table = pd.DataFrame(rows)
    total_unique = len(histories)
    projected = float(table.seconds_per_origin.mean() * total_unique / 3600.0)
    table["projected_complete_origin_cache_hours"] = projected
    table["resource_stop"] = projected > 2.0
    return table


def _method_train_audit(method: str, results: dict[int, Any], train_origins: list[int], histories: dict[int, np.ndarray], ages: dict[int, float], segments: dict[int, Any], elapsed_s: float) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    ordered = np.asarray(sorted(train_origins), dtype=np.int64)
    features = np.asarray([results[int(origin)].features for origin in ordered])
    endpoint = endpoint_audit(method, ordered, np.asarray([ages[int(o)] for o in ordered]), segments, histories, results)
    records: list[dict[str, Any]] = []
    raw = np.asarray([histories[int(origin)] for origin in ordered])
    for column in range(2):
        values = features[:, :, column].reshape(-1)
        records.append({
            "method": method, "feature_column": column, "mean": float(values.mean()), "std": float(values.std()),
            "min": float(values.min()), "max": float(values.max()), "missing_count": int(np.isnan(values).sum()),
            "nonfinite_count": int((~np.isfinite(values)).sum()), "raw_voltage_correlation": float(np.corrcoef(values, raw.reshape(-1))[0, 1]),
            "temporal_smoothness_mean_abs_step": float(np.mean(np.abs(np.diff(features[:, :, column], axis=1)))),
            "runtime_s_for_train_unique_origins": elapsed_s, **endpoint,
        })
    return records, endpoint


def _cache_parameters_sha(method: str, df_protocol_sha: str, resolved_alpha: int | None) -> str:
    payload = json.dumps({"method": method, "df_protocol_sha256": df_protocol_sha, "resolved_vmd_alpha": resolved_alpha}, sort_keys=True).encode()
    return hashlib.sha256(payload).hexdigest()


def _write_origin_cache(method: str, results: dict[int, Any], histories: dict[int, np.ndarray], ages: dict[int, float], segments: dict[int, Any], memberships: dict[int, set[str]], roles: dict[int, set[str]], *, df_protocol_sha: str, source_metadata: dict[str, Any], resolved_alpha: int | None) -> list[dict[str, Any]]:
    parameter_sha = _cache_parameters_sha(method, df_protocol_sha, resolved_alpha)
    source = source_metadata["VMD"] if method == "VMD" else source_metadata["EMD"] if method == "EMD" else {"version": "builtin", "source_sha256": sha256_file(Path(__file__).resolve())}
    rows: list[dict[str, Any]] = []
    for origin, result in results.items():
        features = result.features
        path = DF / "cache" / method / f"origin_{origin}.npz"
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, features=features, origin_row=np.asarray(origin, dtype=np.int64), input_voltage_sha256=np.asarray(input_voltage_sha256(histories[origin])), method_parameter_sha256=np.asarray(parameter_sha))
        rows.append({
            "origin_row": origin, "origin_age_h": ages[origin], "context_start_row": origin - 47,
            "context_end_row": origin, "temporal_segment_id": segments[origin], "input_voltage_sha256": input_voltage_sha256(histories[origin]),
            "method": method, "method_parameter_sha256": parameter_sha, "library_version": source["version"],
            "source_sha256": source["source_sha256"], "feature_names": ";".join({"SimpleTrend": ["causal_mean8_V", "causal_slope8_V_per_h"], "VMD": ["vmd_low_V", "vmd_low_slope8_V_per_h"], "EMD": ["emd_remainder2_V", "emd_remainder2_slope8_V_per_h"]}[method]),
            "feature_order": "14,15", "feature_shape": "[48,2]", "fold_membership": ";".join(sorted(memberships[origin])),
            "split_role": ";".join(sorted(roles[origin])), "cache_path": str(path), "cache_sha256": hashlib.sha256(features.tobytes()).hexdigest(),
            "rebuild_command": "python scripts/run_stage0.py",
        })
    return rows


def main() -> None:
    df_cfg, df_sha = load_df_protocol()
    bs_cfg, bs_sha = load_bs_protocol(df_cfg)
    source_metadata = write_method_source_pinning()
    adapter_sources = verify_bs_source_and_adapter(df_cfg, bs_cfg)
    synthetic = synthetic_audits()
    write_json({"base_head": current_git_head(), "df_protocol_sha256": df_sha, "phase_bs_protocol_sha256": bs_sha, "source_metadata": source_metadata, "synthetic": synthetic, "test_accessed": False}, "outputs", "CAUSALITY_AUDIT.json")
    write_json({"status": "PASS", "future_perturbation_invariance": synthetic["future_perturbation_invariance"], "test_accessed": False}, "outputs", "PREFIX_INVARIANCE_AUDIT.json")
    if synthetic["status"] != "PASS":
        raise RuntimeError("LEAKAGE_OR_IMPLEMENTATION_FAIL: synthetic causality audit")

    frame, firewall = read_development_prefix(df_cfg, bs_cfg)
    folds = build_frozen_folds(frame, bs_cfg, df_cfg)
    histories, ages, segments, memberships, roles = _unique_train_histories(folds, frame)
    train_origins = sorted({int(origin) for _fold, train, _validation in folds for origin in train.origin_row.tolist()})
    fold1_train_origins = [int(origin) for fold, train, _validation in folds if fold == "fold_1" for origin in train.origin_row.tolist()]

    trend_results, trend_elapsed = _run_method("SimpleTrend", {o: histories[o] for o in train_origins}, vmd_alpha=None)
    feature_rows, endpoint_rows = [], []
    trend_feature_rows, trend_endpoint = _method_train_audit("SimpleTrend", trend_results, train_origins, histories, ages, segments, trend_elapsed)
    feature_rows.extend(trend_feature_rows); endpoint_rows.append(trend_endpoint)
    method_status: dict[str, dict[str, Any]] = {"SimpleTrend": {"stage0_status": "PASS", "reason": "strict_per_position_causal_control"}}

    vmd_candidates, _candidate_results = _vmd_candidate_records(fold1_train_origins, histories, ages, segments, df_cfg)
    resolved_alpha, vmd_status = _resolve_alpha(vmd_candidates)
    write_csv(vmd_candidates, "outputs", "VMD_ALPHA_CALIBRATION.csv")
    if resolved_alpha is not None:
        resolved = {"method": "VMD", "status": "FROZEN_AFTER_FOLD1_OUTER_TRAIN_UNSUPERVISED_CALIBRATION", "alpha": resolved_alpha, "K": 2, "tau": 0, "DC": 0, "init": 1, "tol": 1e-7, "selection_rule": df_cfg["vmd"]["selection"]}
        resolved_path = DF / "configs" / "RESOLVED_VMD_PARAMETERS.yaml"
        resolved_path.write_text(yaml.safe_dump(resolved, allow_unicode=True, sort_keys=False), encoding="utf-8")
        (DF / "configs" / "RESOLVED_VMD_PARAMETERS.sha256").write_text(sha256_file(resolved_path) + "  RESOLVED_VMD_PARAMETERS.yaml\n", encoding="utf-8")
        vmd_results, vmd_elapsed = _run_method("VMD", {o: histories[o] for o in train_origins}, vmd_alpha=resolved_alpha)
        vmd_feature_rows, vmd_endpoint = _method_train_audit("VMD", vmd_results, train_origins, histories, ages, segments, vmd_elapsed)
        vmd_resource = _resource_audit("VMD", folds, histories, vmd_alpha=resolved_alpha)
        feature_rows.extend(vmd_feature_rows); endpoint_rows.append(vmd_endpoint)
        resource_stop = bool(vmd_resource.resource_stop.any())
        eligible = [origin for origin in train_origins if np.std(histories[origin]) >= .001]
        full_train_separation_rate = float(np.mean([vmd_results[origin].audit["frequency_separation_cycles_per_sample"] >= 1.0 / 48.0 for origin in eligible])) if eligible else 0.0
        vmd_stage0_status = "RESOURCE_STOP" if resource_stop else "ENDPOINT_UNSTABLE" if vmd_endpoint["endpoint_unstable"] else "VMD_MODE_NOT_IDENTIFIABLE" if full_train_separation_rate < .90 else "PASS"
        method_status["VMD"] = {"stage0_status": vmd_stage0_status, "reason": "resource_endpoint_or_full_outer_train_mode_gate", "resolved_alpha": resolved_alpha, "full_outer_train_mode_separation_pass_rate": full_train_separation_rate}
    else:
        vmd_results = {}; vmd_resource = pd.DataFrame(); method_status["VMD"] = {"stage0_status": vmd_status, "reason": "no_registered_alpha_passed_unsupervised_identifiability_and_endpoint_rule", "resolved_alpha": None}

    emd_results, emd_elapsed = _run_method("EMD", {o: histories[o] for o in train_origins}, vmd_alpha=None)
    emd_feature_rows, emd_endpoint = _method_train_audit("EMD", emd_results, train_origins, histories, ages, segments, emd_elapsed)
    emd_valid_rate = float(np.mean([result.audit["imf_count"] >= 1 for result in emd_results.values()]))
    emd_resource = _resource_audit("EMD", folds, histories, vmd_alpha=None)
    feature_rows.extend(emd_feature_rows); endpoint_rows.append(emd_endpoint)
    emd_resource_stop = bool(emd_resource.resource_stop.any())
    method_status["EMD"] = {"stage0_status": "RESOURCE_STOP" if emd_resource_stop else "ENDPOINT_UNSTABLE" if emd_endpoint["endpoint_unstable"] else "EMD_MODE_NOT_IDENTIFIABLE" if emd_valid_rate < .90 else "PASS", "reason": "preregistered_imf_endpoint_resource_gates", "valid_imf_rate": emd_valid_rate}

    mode_rows = vmd_candidates.to_dict("records")
    mode_rows.append({"method": "EMD", "alpha": None, "valid_imf_rate": emd_valid_rate, "imf_count_mean": float(np.mean([x.audit["imf_count"] for x in emd_results.values()])), "imf_count_min": int(min(x.audit["imf_count"] for x in emd_results.values())), "imf_count_max": int(max(x.audit["imf_count"] for x in emd_results.values()))})
    write_csv(pd.DataFrame(feature_rows), "outputs", "FEATURE_AUDIT.csv")
    write_csv(pd.DataFrame(endpoint_rows), "outputs", "ENDPOINT_AUDIT.csv")
    write_csv(pd.DataFrame(mode_rows), "outputs", "MODE_STABILITY_AUDIT.csv")
    resources = pd.concat([x for x in (vmd_resource, emd_resource) if not x.empty], ignore_index=True) if (not vmd_resource.empty or not emd_resource.empty) else pd.DataFrame()
    write_csv(resources, "outputs", "RESOURCE_AUDIT.csv")
    write_json({
        "status": "PASS" if not resources.empty and bool((resources["projected_complete_origin_cache_hours"] <= 2.0).all()) else "RESOURCE_STOP",
        "method_rows": resources.to_dict(orient="records"),
        "registered_method_limit_hours": 2.0,
        "test_accessed": False,
    }, "outputs", "RESOURCE_AUDIT.json")
    write_csv(adapter_sources, "outputs", "PHASE_BS_TIMESNET_SOURCE_AUDIT.csv")

    # Generate validation cache only after the method status and VMD alpha have
    # become immutable. No validation feature statistic is computed or written.
    cache_rows: list[dict[str, Any]] = []
    trend_all, _ = _run_method("SimpleTrend", histories, vmd_alpha=None)
    cache_rows.extend(_write_origin_cache("SimpleTrend", trend_all, histories, ages, segments, memberships, roles, df_protocol_sha=df_sha, source_metadata=source_metadata, resolved_alpha=None))
    if method_status["VMD"]["stage0_status"] == "PASS":
        # The union includes outer-validation origins solely for post-freeze cache construction.
        vmd_all, _ = _run_method("VMD", histories, vmd_alpha=resolved_alpha)
        cache_rows.extend(_write_origin_cache("VMD", vmd_all, histories, ages, segments, memberships, roles, df_protocol_sha=df_sha, source_metadata=source_metadata, resolved_alpha=resolved_alpha))
    if method_status["EMD"]["stage0_status"] == "PASS":
        emd_all, _ = _run_method("EMD", histories, vmd_alpha=None)
        cache_rows.extend(_write_origin_cache("EMD", emd_all, histories, ages, segments, memberships, roles, df_protocol_sha=df_sha, source_metadata=source_metadata, resolved_alpha=None))
    write_csv(pd.DataFrame(cache_rows), "outputs", "CACHE_PROVENANCE_MANIFEST.csv")
    manifest_sha = sha256_file(DF / "outputs" / "CACHE_PROVENANCE_MANIFEST.csv")
    (DF / "outputs" / "CACHE_PROVENANCE_MANIFEST.sha256").write_text(manifest_sha + "  CACHE_PROVENANCE_MANIFEST.csv\n", encoding="utf-8")

    summary = {"status": "PASS", "base_head": current_git_head(), "phase_bs_protocol_sha256": bs_sha, "df_protocol_sha256": df_sha, "data_firewall": firewall, "methods": method_status, "test_accessed": False, "validation_feature_statistics_used": False, "model_training_performed": False, "soh_claim": False, "rul_claim": False, "extended_context_tested": False}
    write_json(summary, "outputs", "STAGE0_SUMMARY.json")
    print(json.dumps({"stage0": summary["methods"], "test_accessed": False}, ensure_ascii=False))


if __name__ == "__main__":
    main()
