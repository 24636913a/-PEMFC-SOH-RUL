"""Phase DF seed-gated TimesNet training on frozen origin-window feature caches."""
from __future__ import annotations

import hashlib
import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

from df_protocol_adapter import BS, DF, TimesNet16VoltageAdapter, augment_windows, build_frozen_folds, current_git_head, load_bs_common_readonly, load_bs_protocol, load_df_protocol, model_delta_prediction, read_development_prefix, sha256_file, verify_bs_source_and_adapter, write_csv, write_json, write_text


METHOD_TO_CACHE = {
    "SimpleTrend_TimesNet": "SimpleTrend",
    "VMDFeature_TimesNet": "VMD",
    "EMDFeature_TimesNet": "EMD",
}
RAW = "RawF2_TimesNet"
SEEDS = (42, 2024, 3407)


def _loader(x: np.ndarray, y: np.ndarray, seed: int) -> DataLoader:
    return DataLoader(TensorDataset(torch.from_numpy(x), torch.from_numpy(y)), batch_size=64, shuffle=True, generator=torch.Generator().manual_seed(seed), num_workers=0)


def _synthetic_timesnet16_preflight(bs_common: Any, bs_cfg: dict[str, Any]) -> dict[str, Any]:
    """Validate the permitted 16-channel projection change before data access."""
    bs_common.set_seed(42)
    dev = bs_common.device()
    model = TimesNet16VoltageAdapter(bs_cfg).to(dev)
    if dev.type == "cuda":
        torch.cuda.reset_peak_memory_stats(dev)
    x = torch.randn(4, 48, 16, device=dev, requires_grad=True)
    target = torch.zeros(4, 24, device=dev)
    prediction = model_delta_prediction(model, x)
    loss = F.mse_loss(prediction, target)
    loss.backward()
    parameter_gradients_finite = all(parameter.grad is None or torch.isfinite(parameter.grad).all() for parameter in model.parameters())
    result = {
        "status": "PASS" if prediction.shape == (4, 24) and torch.isfinite(prediction).all() and torch.isfinite(loss) and torch.isfinite(x.grad).all() and parameter_gradients_finite else "FAIL",
        "synthetic_input_shape": [4, 48, 16], "delta_output_shape": list(prediction.shape),
        "output_finite": bool(torch.isfinite(prediction).all()), "loss_finite": bool(torch.isfinite(loss)),
        "input_gradient_finite": bool(torch.isfinite(x.grad).all()), "parameter_gradients_finite": bool(parameter_gradients_finite),
        "parameter_count": int(sum(p.numel() for p in model.parameters() if p.requires_grad)),
        "device": str(dev), "peak_gpu_memory_MiB": float(torch.cuda.max_memory_allocated(dev) / 1024**2) if dev.type == "cuda" else 0.0,
        "development_accessed": False, "test_accessed": False,
    }
    del model
    if dev.type == "cuda":
        torch.cuda.empty_cache()
    if result["status"] != "PASS":
        raise RuntimeError("IMPLEMENTATION_FAIL: synthetic 16-channel TimesNet preflight")
    return result


def _predict_delta(model: torch.nn.Module, x: np.ndarray, dev: torch.device) -> np.ndarray:
    rows = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(x), 256):
            rows.append(model_delta_prediction(model, torch.from_numpy(x[start:start + 256]).to(dev)).detach().cpu().numpy())
    return np.concatenate(rows, axis=0)


def _load_cache(method: str) -> dict[int, np.ndarray]:
    manifest = pd.read_csv(DF / "outputs" / "CACHE_PROVENANCE_MANIFEST.csv")
    rows = manifest.loc[manifest.method.eq(method)].copy()
    if rows.empty:
        raise RuntimeError(f"CACHE_AUDIT_FAIL: no Stage 0 cache manifest rows for {method}")
    cache: dict[int, np.ndarray] = {}
    for row in rows.itertuples(index=False):
        path = Path(row.cache_path)
        if not path.is_file():
            raise RuntimeError(f"CACHE_AUDIT_FAIL: missing {path}")
        with np.load(path, allow_pickle=False) as loaded:
            feature = loaded["features"]
            origin = int(loaded["origin_row"])
            source_hash = str(loaded["input_voltage_sha256"])
        if origin != int(row.origin_row) or source_hash != str(row.input_voltage_sha256):
            raise RuntimeError(f"CACHE_AUDIT_FAIL: cache metadata mismatch for {path.name}")
        if feature.shape != (48, 2) or feature.dtype != np.float32 or not np.isfinite(feature).all():
            raise RuntimeError(f"CACHE_AUDIT_FAIL: feature shape/finite violation for {path.name}")
        if hashlib.sha256(feature.tobytes()).hexdigest() != str(row.cache_sha256):
            raise RuntimeError(f"CACHE_AUDIT_FAIL: feature checksum mismatch for {path.name}")
        if origin in cache:
            raise RuntimeError(f"CACHE_AUDIT_FAIL: duplicate origin cache {origin}")
        cache[origin] = feature
    return cache


def _train_one(method: str, fold: str, outer_train: Any, outer_validation: Any, seed: int, bs_common: Any, bs_cfg: dict[str, Any], additions: dict[int, np.ndarray]) -> dict[str, Any]:
    cache_method = METHOD_TO_CACHE[method]
    train = augment_windows(outer_train, additions)
    validation = augment_windows(outer_validation, additions)
    if not np.array_equal(train.x[:, :, :14], outer_train.x) or not np.array_equal(validation.x[:, :, :14], outer_validation.x):
        raise RuntimeError("PROTOCOL_DRIFT: Raw F2 differs after tail-only feature append")
    fit, stopper = bs_common.split_inner_fit_stopper(train, .15)
    if set(fit.target_row.reshape(-1)) & set(stopper.target_row.reshape(-1)):
        raise RuntimeError(f"{method}/{fold}: inner target isolation failure")
    bs_common.set_seed(seed)
    dev = bs_common.device()
    scaler = bs_common.FoldPreprocessor().fit(fit.x)
    x_fit, x_stop = scaler.transform(fit.x), scaler.transform(stopper.x)
    y_fit = bs_common.standardized_delta_target(fit, scaler)
    model = TimesNet16VoltageAdapter(bs_cfg).to(dev)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=1e-5)
    best, best_epoch, stale = math.inf, 0, 0
    inner_logs: list[dict[str, Any]] = []
    if dev.type == "cuda": torch.cuda.reset_peak_memory_stats(dev)
    started = time.perf_counter()
    for epoch in range(1, 301):
        model.train(); losses: list[float] = []
        for bx, by in _loader(x_fit, y_fit, seed + epoch):
            bx, by = bx.to(dev), by.to(dev)
            optimizer.zero_grad(set_to_none=True)
            loss = F.mse_loss(model_delta_prediction(model, bx), by)
            if not torch.isfinite(loss):
                raise RuntimeError(f"{method}/{fold}: non-finite inner loss")
            loss.backward()
            grad = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            if not torch.isfinite(grad):
                raise RuntimeError(f"{method}/{fold}: non-finite inner gradient")
            optimizer.step(); losses.append(float(loss.detach().cpu()))
        stopper_pred = _predict_delta(model, x_stop, dev) * scaler.voltage_scale
        stopper_rmse = bs_common.delta_metrics(stopper.y_delta_V, stopper_pred, stopper.target_row)["rmse_mV"]
        inner_logs.append({"phase": "inner_stopper", "method": method, "cache_method": cache_method, "fold": fold, "seed": seed, "epoch": epoch, "train_standardized_delta_mse": float(np.mean(losses)), "stopper_deltaV_RMSE_mV": stopper_rmse})
        if stopper_rmse < best - 1e-10:
            best, best_epoch, stale = stopper_rmse, epoch, 0
        else:
            stale += 1
        if stale >= 30:
            break
    inner_time = time.perf_counter() - started
    inner_peak = float(torch.cuda.max_memory_allocated(dev) / 1024**2) if dev.type == "cuda" else 0.0
    del model
    if dev.type == "cuda": torch.cuda.empty_cache()
    if best_epoch <= 0:
        raise RuntimeError(f"{method}/{fold}: no selected inner epoch")

    bs_common.set_seed(seed)
    scaler = bs_common.FoldPreprocessor().fit(train.x)
    x_train, x_valid = scaler.transform(train.x), scaler.transform(validation.x)
    y_train = bs_common.standardized_delta_target(train, scaler)
    model = TimesNet16VoltageAdapter(bs_cfg).to(dev)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=1e-5)
    if dev.type == "cuda": torch.cuda.reset_peak_memory_stats(dev)
    refit_logs: list[dict[str, Any]] = []
    started = time.perf_counter()
    for epoch in range(1, best_epoch + 1):
        model.train(); losses = []
        for bx, by in _loader(x_train, y_train, seed + epoch):
            bx, by = bx.to(dev), by.to(dev)
            optimizer.zero_grad(set_to_none=True)
            loss = F.mse_loss(model_delta_prediction(model, bx), by)
            if not torch.isfinite(loss):
                raise RuntimeError(f"{method}/{fold}: non-finite outer loss")
            loss.backward()
            grad = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            if not torch.isfinite(grad):
                raise RuntimeError(f"{method}/{fold}: non-finite outer gradient")
            optimizer.step(); losses.append(float(loss.detach().cpu()))
        refit_logs.append({"phase": "outer_refit", "method": method, "cache_method": cache_method, "fold": fold, "seed": seed, "epoch": epoch, "train_standardized_delta_mse": float(np.mean(losses))})
    prediction = _predict_delta(model, x_valid, dev) * scaler.voltage_scale
    outer_time = time.perf_counter() - started
    outer_peak = float(torch.cuda.max_memory_allocated(dev) / 1024**2) if dev.type == "cuda" else 0.0
    arrays = {
        "y_true_delta_V": validation.y_delta_V, "y_pred_delta_V": prediction, "anchor_V": validation.anchor_V,
        "origin_row": validation.origin_row, "target_row": validation.target_row, "origin_age_h": validation.origin_age_h,
        "target_end_age_h": validation.target_end_age_h, "origin_datetime": validation.origin_datetime,
        "target_datetime": validation.target_datetime, "horizon_step": np.arange(1, 25, dtype=np.int64),
    }
    if float(np.max(arrays["target_end_age_h"])) >= 1819.997738:
        raise RuntimeError(f"DATA_FIREWALL_FAIL: {method}/{fold} target crosses Development boundary")
    metrics = bs_common.delta_metrics(arrays["y_true_delta_V"], arrays["y_pred_delta_V"], arrays["target_row"])
    run_id = f"df_{method.lower()}_{fold}_seed{seed}"
    logs_path = DF / "logs" / f"{run_id}_epochs.csv"; logs_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(inner_logs + refit_logs).to_csv(logs_path, index=False, encoding="utf-8-sig", lineterminator="\n")
    prediction_path = DF / "outputs" / "predictions" / f"{run_id}.npz"; prediction_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(prediction_path, **arrays)
    checkpoint_path = DF / "checkpoints" / f"{run_id}.pt"; checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    metadata = {"parameter_count": int(sum(p.numel() for p in model.parameters() if p.requires_grad)), "outer_train_scaler_mean": scaler.mean.tolist(), "outer_train_scaler_scale": scaler.scale.tolist(), "outer_train_feature_medians": scaler.medians.tolist()}
    torch.save({"model": method, "seed": seed, "fold": fold, "selected_epoch": best_epoch, "model_state_dict": model.state_dict(), "preprocessor": metadata}, checkpoint_path)
    peak = max(inner_peak, outer_peak)
    del model
    if dev.type == "cuda": torch.cuda.empty_cache()
    result = {"run_id": run_id, "model": method, "fold": fold, "seed": seed, "status": "complete", "selected_epoch": best_epoch, "inner_fit_origins": len(fit.origin_row), "inner_stopper_origins": len(stopper.origin_row), "inner_target_overlap_count": 0, "parameter_count": metadata["parameter_count"], "peak_gpu_memory_MiB": peak, "inner_time_s": inner_time, "outer_refit_time_s": outer_time, "total_inner_plus_refit_time_s": inner_time + outer_time, "max_target_end_age_h": float(np.max(validation.target_end_age_h)), "cache_method": cache_method, "input_dimension": 16, "future_covariates_used": False, "auxiliary_output_supervision": False, "test_accessed": False, "all_legal_validation_origins_retained": True, **metrics}
    result_path = DF / "outputs" / "predictions" / f"{run_id}.json"
    result_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def _raw_control(folds: list[tuple[str, Any, Any]], bs_common: Any, bs_cfg: dict[str, Any], seed: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for fold, _train, validation in folds:
        stem = BS / "outputs" / "predictions" / f"bs_timesnet_{fold}_seed{seed}"
        result_path, npz_path = stem.with_suffix(".json"), stem.with_suffix(".npz")
        if not result_path.is_file() or not npz_path.is_file():
            raise RuntimeError(f"RAW_CONTROL_IDENTITY_FAIL: missing frozen TimesNet cache for {fold}/seed{seed}")
        result = json.loads(result_path.read_text(encoding="utf-8"))
        with np.load(npz_path, allow_pickle=False) as z:
            required = ("y_true_delta_V", "y_pred_delta_V", "anchor_V", "origin_row", "target_row", "target_end_age_h", "horizon_step")
            if any(key not in z.files for key in required):
                raise RuntimeError("RAW_CONTROL_IDENTITY_FAIL: incomplete prediction cache")
            for key, expected in (("y_true_delta_V", validation.y_delta_V), ("anchor_V", validation.anchor_V), ("origin_row", validation.origin_row), ("target_row", validation.target_row), ("target_end_age_h", validation.target_end_age_h)):
                if not np.array_equal(z[key], expected):
                    raise RuntimeError(f"RAW_CONTROL_IDENTITY_FAIL: {fold}/seed{seed} mismatch for {key}")
            if not np.array_equal(z["horizon_step"], np.arange(1, 25, dtype=np.int64)):
                raise RuntimeError("RAW_CONTROL_IDENTITY_FAIL: horizon changed")
            if not np.array_equal(z["target_row"], z["origin_row"][:, None] + z["horizon_step"][None, :]):
                raise RuntimeError("RAW_CONTROL_IDENTITY_FAIL: target origin+horizon relation failed")
            if np.max(z["target_end_age_h"].astype(float)) >= 1819.997738:
                raise RuntimeError("DATA_FIREWALL_FAIL: Raw cache reaches Test boundary")
            metrics = bs_common.delta_metrics(z["y_true_delta_V"], z["y_pred_delta_V"], z["target_row"])
        for key, value in metrics.items():
            if key in result and isinstance(value, float) and not np.isclose(value, float(result[key]), rtol=0, atol=1e-9):
                raise RuntimeError(f"RAW_CONTROL_IDENTITY_FAIL: {fold}/seed{seed} metric mismatch {key}")
        rows.append({"run_id": f"bs_timesnet_{fold}_seed{seed}", "model": RAW, "fold": fold, "seed": seed, "status": "reused_frozen_phase_bs_cache", "parameter_count": int(result.get("parameter_count", 149046)), "peak_gpu_memory_MiB": math.nan, "total_inner_plus_refit_time_s": math.nan, "max_target_end_age_h": float(validation.target_end_age_h.max()), "input_dimension": 14, "test_accessed": False, **metrics})
    return rows


def _summary(rows: pd.DataFrame, bs_common: Any) -> pd.DataFrame:
    return bs_common.summarize_seed(rows)


def _raw_gate(candidate: pd.Series, raw: pd.Series, cfg: dict[str, Any]) -> dict[str, bool]:
    g = cfg["seed42_raw_gain_gate"]
    return {
        "formal_95pct": bool(candidate.formal_RMSE_mV <= raw.formal_RMSE_mV * g["formal_RMSE_at_most_raw_fraction"]),
        "fold3_103pct": bool(candidate.Fold3_RMSE_mV <= raw.Fold3_RMSE_mV * g["fold3_RMSE_at_most_raw_fraction"]),
        "p99_105pct": bool(candidate.P99_mV <= raw.P99_mV * g["p99_at_most_raw_fraction"]),
        "target_equal_105pct": bool(candidate.target_time_equal_RMSE_mV <= raw.target_time_equal_RMSE_mV * g["target_time_equal_at_most_raw_fraction"]),
        "fold3_target_equal_110pct": bool(candidate.Fold3_target_time_equal_RMSE_mV <= raw.Fold3_target_time_equal_RMSE_mV * g["fold3_target_time_equal_at_most_raw_fraction"]),
        "endpoint_110pct": bool(candidate.endpoint_RMSE_mV <= raw.endpoint_RMSE_mV * g["endpoint_at_most_raw_fraction"]),
    }


def _decomposition_gate(candidate: pd.Series, trend: pd.Series, cfg: dict[str, Any]) -> dict[str, bool]:
    g = cfg["seed42_decomposition_specific_gate"]
    return {
        "formal_strictly_better_than_simpletrend": bool(candidate.formal_RMSE_mV < trend.formal_RMSE_mV),
        "fold3_105pct_of_simpletrend": bool(candidate.Fold3_RMSE_mV <= trend.Fold3_RMSE_mV * g["fold3_RMSE_at_most_simpletrend_fraction"]),
        "p99_105pct_of_simpletrend": bool(candidate.P99_mV <= trend.P99_mV * g["p99_at_most_simpletrend_fraction"]),
        "endpoint_105pct_of_simpletrend": bool(candidate.endpoint_RMSE_mV <= trend.endpoint_RMSE_mV * g["endpoint_at_most_simpletrend_fraction"]),
    }


def _final_status(summary: pd.DataFrame, stage0: dict[str, Any], cfg: dict[str, Any]) -> tuple[str, list[dict[str, Any]]]:
    raw = summary.loc[(summary.model == RAW)].sort_values("seed")
    trend = summary.loc[(summary.model == "SimpleTrend_TimesNet")].sort_values("seed")
    records: list[dict[str, Any]] = []
    robust_raw: dict[str, bool] = {}
    for method in ("SimpleTrend_TimesNet", "VMDFeature_TimesNet", "EMDFeature_TimesNet"):
        candidate = summary.loc[summary.model.eq(method)].sort_values("seed")
        if len(candidate) != 3:
            robust_raw[method] = False; records.append({"method": method, "multiseed_performed": False, "robust_raw_gain": False, "reason": "seed42_gate_did_not_authorize_complete_multiseed"}); continue
        g = cfg["multiseed_raw_gain_gate"]
        raw_match = raw.set_index("seed").loc[candidate.seed]
        mean_candidate, mean_raw = candidate.mean(numeric_only=True), raw_match.mean(numeric_only=True)
        checks = {"formal_95pct": mean_candidate.formal_RMSE_mV <= mean_raw.formal_RMSE_mV * g["formal_RMSE_at_most_raw_fraction"], "paired_seeds_at_least_2": int((candidate.formal_RMSE_mV.to_numpy() < raw_match.formal_RMSE_mV.to_numpy()).sum()) >= g["minimum_paired_seeds_better_than_raw"], "no_seed_over_110pct": bool((candidate.formal_RMSE_mV.to_numpy() <= raw_match.formal_RMSE_mV.to_numpy() * g["maximum_any_seed_raw_fraction"]).all()), "fold3_103pct": mean_candidate.Fold3_RMSE_mV <= mean_raw.Fold3_RMSE_mV * g["fold3_RMSE_at_most_raw_fraction"], "p99_105pct": mean_candidate.P99_mV <= mean_raw.P99_mV * g["p99_at_most_raw_fraction"], "target_equal_105pct": mean_candidate.target_time_equal_RMSE_mV <= mean_raw.target_time_equal_RMSE_mV * g["target_time_equal_at_most_raw_fraction"], "fold3_target_equal_110pct": mean_candidate.Fold3_target_time_equal_RMSE_mV <= mean_raw.Fold3_target_time_equal_RMSE_mV * g["fold3_target_time_equal_at_most_raw_fraction"], "endpoint_110pct": mean_candidate.endpoint_RMSE_mV <= mean_raw.endpoint_RMSE_mV * g["endpoint_at_most_raw_fraction"]}
        robust_raw[method] = bool(all(checks.values()))
        records.append({"method": method, "multiseed_performed": True, "robust_raw_gain": robust_raw[method], **checks})
    specific: dict[str, bool] = {}
    for method in ("VMDFeature_TimesNet", "EMDFeature_TimesNet"):
        candidate = summary.loc[summary.model.eq(method)].sort_values("seed")
        if len(candidate) != 3 or len(trend) != 3:
            specific[method] = False; continue
        g = cfg["multiseed_decomposition_specific_gate"]
        cm, tm = candidate.mean(numeric_only=True), trend.mean(numeric_only=True)
        checks = {"formal_mean_strictly_better_than_simpletrend": cm.formal_RMSE_mV < tm.formal_RMSE_mV, "paired_seeds_at_least_2": int((candidate.formal_RMSE_mV.to_numpy() < trend.formal_RMSE_mV.to_numpy()).sum()) >= g["minimum_paired_seeds_better_than_simpletrend"], "fold3_105pct_of_simpletrend": cm.Fold3_RMSE_mV <= tm.Fold3_RMSE_mV * g["fold3_RMSE_at_most_simpletrend_fraction"], "p99_105pct_of_simpletrend": cm.P99_mV <= tm.P99_mV * g["p99_at_most_simpletrend_fraction"], "endpoint_105pct_of_simpletrend": cm.endpoint_RMSE_mV <= tm.endpoint_RMSE_mV * g["endpoint_at_most_simpletrend_fraction"]}
        specific[method] = bool(robust_raw.get(method) and all(checks.values()))
        records.append({"method": method, "decomposition_specific_advance": specific[method], **checks})
    if specific.get("VMDFeature_TimesNet") and specific.get("EMDFeature_TimesNet"):
        status = "BOTH_SPECIFIC_ADVANCE"
    elif specific.get("VMDFeature_TimesNet"):
        status = "VMD_SPECIFIC_ADVANCE"
    elif specific.get("EMDFeature_TimesNet"):
        status = "EMD_SPECIFIC_ADVANCE"
    elif any(robust_raw.values()):
        status = "GENERIC_TREND_GAIN_ONLY"
    else:
        status = "NO_FEATURE_GAIN"
    return status, records


def main() -> None:
    df_cfg, df_sha = load_df_protocol()
    bs_cfg, bs_sha = load_bs_protocol(df_cfg)
    verify_bs_source_and_adapter(df_cfg, bs_cfg)
    stage0 = json.loads((DF / "outputs" / "STAGE0_SUMMARY.json").read_text(encoding="utf-8"))
    if stage0.get("status") != "PASS":
        raise RuntimeError("STAGE0_ONLY_STOP: Stage 0 did not pass")
    common = load_bs_common_readonly()
    preflight = _synthetic_timesnet16_preflight(common, bs_cfg)
    write_json(preflight, "outputs", "TIMESNET16_IMPLEMENTATION_AUDIT.json")
    frame, firewall = read_development_prefix(df_cfg, bs_cfg)
    folds = build_frozen_folds(frame, bs_cfg, df_cfg)
    results: list[dict[str, Any]] = _raw_control(folds, common, bs_cfg, 42)
    additions = {method: _load_cache(cache_method) for method, cache_method in METHOD_TO_CACHE.items() if stage0["methods"].get(cache_method, {}).get("stage0_status") == "PASS"}
    for method, cache in additions.items():
        for fold, train, validation in folds:
            results.append(_train_one(method, fold, train, validation, 42, common, bs_cfg, cache))
    per_fold = pd.DataFrame(results).sort_values(["model", "seed", "fold"]).reset_index(drop=True)
    summary = _summary(per_fold, common)
    raw42 = summary.loc[(summary.model == RAW) & (summary.seed == 42)].iloc[0]
    trend42 = summary.loc[(summary.model == "SimpleTrend_TimesNet") & (summary.seed == 42)].iloc[0]
    seed42_gates = []
    authorise: set[str] = set()
    for method in additions:
        row = summary.loc[(summary.model == method) & (summary.seed == 42)].iloc[0]
        raw_checks = _raw_gate(row, raw42, df_cfg)
        raw_pass = bool(all(raw_checks.values()))
        specific_checks = _decomposition_gate(row, trend42, df_cfg) if method != "SimpleTrend_TimesNet" else {}
        specific_pass = bool(all(specific_checks.values())) if specific_checks else None
        if method == "SimpleTrend_TimesNet" and raw_pass:
            authorise.add(method)
        if method != "SimpleTrend_TimesNet" and raw_pass and specific_pass:
            authorise.add(method); authorise.add("SimpleTrend_TimesNet")
        seed42_gates.append({"method": method, "seed": 42, "raw_predictive_gain_pass": raw_pass, "decomposition_specific_seed42_pass": specific_pass, "additional_seeds_authorized": method in authorise, **raw_checks, **specific_checks})
    gate_frame = pd.DataFrame(seed42_gates)
    # Recompute authorization after all decomposition rows have been considered.
    if not gate_frame.empty:
        gate_frame["additional_seeds_authorized"] = gate_frame.method.isin(authorise)
    write_csv(per_fold, "outputs", "SEED42_RESULTS.csv")
    write_csv(summary, "outputs", "SEED42_SUMMARY_RESULTS.csv")
    write_csv(gate_frame, "outputs", "SEED42_GATES.csv")

    for seed in (2024, 3407):
        results.extend(_raw_control(folds, common, bs_cfg, seed))
        for method in sorted(authorise):
            for fold, train, validation in folds:
                results.append(_train_one(method, fold, train, validation, seed, common, bs_cfg, additions[method]))
    per_fold = pd.DataFrame(results).sort_values(["model", "seed", "fold"]).reset_index(drop=True)
    summary = _summary(per_fold, common)
    status, multiseed_records = _final_status(summary, stage0, df_cfg)
    write_csv(per_fold, "outputs", "PER_FOLD_RESULTS.csv")
    write_csv(summary, "outputs", "MULTISEED_RESULTS.csv")
    write_csv(pd.DataFrame(multiseed_records), "outputs", "MULTISEED_GATES.csv")
    raw_identity = {"status": "PASS", "reused_phase_bs_timesnet_caches": True, "fold_seed_count": 9, "test_accessed": False, "metrics_recomputed": True}
    write_json(raw_identity, "outputs", "RAW_CONTROL_IDENTITY.json")
    final = {"status": status, "base_head": current_git_head(), "df_protocol_sha256": df_sha, "phase_bs_protocol_sha256": bs_sha, "stage0_methods": stage0["methods"], "seed42_extra_seed_authorized_methods": sorted(authorise), "multiseed_methods": sorted(authorise), "formal_backbone": "TIMESNET_FREEZE", "backbone_search": "CLOSED", "test_accessed": False, "soh_claim": False, "rul_claim": False, "extended_context_tested": False, "model_training_performed": bool(additions)}
    write_json({"data_firewall": firewall, "raw_control_identity": raw_identity, **final}, "RUN_AUDIT.json")
    write_text("# Phase DF training status\n\nTraining follows the frozen Phase BS TimesNet architecture. See `outputs/` for seed-gated metrics; a final report is generated only after figure and artifact checks.\n", "final", "TRAINING_STATUS.md")
    import yaml
    write_text(yaml.safe_dump(final, allow_unicode=True, sort_keys=False), "final", "FINAL_DECISION.yaml")
    print(json.dumps({"status": status, "authorized": sorted(authorise), "test_accessed": False}, ensure_ascii=False))


if __name__ == "__main__":
    main()
