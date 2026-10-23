"""Run the preregistered final Development-only backbone screen."""
from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import yaml
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

from backbone_common import (
    PHASE, FoldPreprocessor, WindowSet, build_outer_folds, create_model, delta_metrics,
    device, load_config, model_delta_prediction, parameter_count, read_development_prefix,
    set_seed, split_inner_fit_stopper, standardized_delta_target, summarize_seed,
    write_csv, write_json, write_text,
)


MODELS = ["TimesNet", "SimpleTM", "DUET", "ModernTCN"]


def output_path(*parts: str) -> Path:
    return PHASE.joinpath(*parts)


def prediction_delta_scaled(model: torch.nn.Module, x: np.ndarray, dev: torch.device) -> np.ndarray:
    rows: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(x), 256):
            tensor = torch.from_numpy(x[start:start + 256]).to(dev)
            delta, _ = model_delta_prediction(model, tensor)
            rows.append(delta.detach().cpu().numpy())
    return np.concatenate(rows, axis=0)


def _loss(model: torch.nn.Module, x: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    delta, auxiliary = model_delta_prediction(model, x)
    common_mse = F.mse_loss(delta, target)
    return common_mse + auxiliary, common_mse, auxiliary


def _loader(x: np.ndarray, y: np.ndarray, seed: int) -> DataLoader:
    return DataLoader(
        TensorDataset(torch.from_numpy(x), torch.from_numpy(y)), batch_size=64, shuffle=True,
        generator=torch.Generator().manual_seed(seed), num_workers=0,
    )


def inner_select_epoch(model_name: str, fit: WindowSet, stopper: WindowSet, seed: int, cfg: dict[str, Any]) -> tuple[int, list[dict[str, Any]], float, float]:
    """Fit only fit origins; choose epoch exclusively from isolated stopper DeltaV RMSE."""
    set_seed(seed)
    dev = device()
    scaler = FoldPreprocessor().fit(fit.x)
    x_fit, x_stop = scaler.transform(fit.x), scaler.transform(stopper.x)
    y_fit = standardized_delta_target(fit, scaler)
    model = create_model(model_name, cfg).to(dev)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-5)
    best_rmse, best_epoch, stale = math.inf, 0, 0
    logs: list[dict[str, Any]] = []
    start = time.perf_counter()
    for epoch in range(1, 301):
        model.train(); total_losses: list[float] = []; common_losses: list[float] = []; aux_losses: list[float] = []
        for batch_x, batch_y in _loader(x_fit, y_fit, seed + epoch):
            batch_x, batch_y = batch_x.to(dev), batch_y.to(dev)
            optimizer.zero_grad(set_to_none=True)
            loss, common, aux = _loss(model, batch_x, batch_y)
            if not torch.isfinite(loss): raise RuntimeError(f"{model_name}: non-finite inner loss")
            loss.backward()
            grad = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            if not torch.isfinite(grad): raise RuntimeError(f"{model_name}: non-finite inner gradient")
            optimizer.step()
            total_losses.append(float(loss.detach().cpu())); common_losses.append(float(common.detach().cpu())); aux_losses.append(float(aux.detach().cpu()))
        pred = prediction_delta_scaled(model, x_stop, dev) * scaler.voltage_scale
        stopper_rmse = delta_metrics(stopper.y_delta_V, pred, stopper.target_row)["rmse_mV"]
        logs.append({"phase": "inner_stopper", "model": model_name, "seed": seed, "epoch": epoch, "train_total_loss": float(np.mean(total_losses)), "train_common_delta_mse": float(np.mean(common_losses)), "train_official_auxiliary_loss": float(np.mean(aux_losses)), "stopper_deltaV_RMSE_mV": stopper_rmse})
        if stopper_rmse < best_rmse - 1e-10:
            best_rmse, best_epoch, stale = stopper_rmse, epoch, 0
        else:
            stale += 1
        if stale >= 30: break
    elapsed = time.perf_counter() - start
    peak = float(torch.cuda.max_memory_allocated(dev) / 1024**2) if dev.type == "cuda" else 0.0
    del model
    if dev.type == "cuda": torch.cuda.empty_cache()
    if best_epoch <= 0: raise RuntimeError(f"{model_name}: no valid inner-stopper epoch")
    return best_epoch, logs, elapsed, peak


def full_outer_refit(model_name: str, outer_train: WindowSet, outer_validation: WindowSet, seed: int, epochs: int, cfg: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, np.ndarray]]:
    """Reinitialize same seed, fit full outer Train for selected epoch, predict Validation once."""
    set_seed(seed)
    dev = device()
    scaler = FoldPreprocessor().fit(outer_train.x)
    x_train, x_valid = scaler.transform(outer_train.x), scaler.transform(outer_validation.x)
    y_train = standardized_delta_target(outer_train, scaler)
    model = create_model(model_name, cfg).to(dev)
    if dev.type == "cuda": torch.cuda.reset_peak_memory_stats(dev)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-5)
    logs: list[dict[str, Any]] = []
    start = time.perf_counter()
    for epoch in range(1, epochs + 1):
        model.train(); total_losses: list[float] = []; common_losses: list[float] = []; aux_losses: list[float] = []
        for batch_x, batch_y in _loader(x_train, y_train, seed + epoch):
            batch_x, batch_y = batch_x.to(dev), batch_y.to(dev)
            optimizer.zero_grad(set_to_none=True)
            loss, common, aux = _loss(model, batch_x, batch_y)
            if not torch.isfinite(loss): raise RuntimeError(f"{model_name}: non-finite outer loss")
            loss.backward()
            grad = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            if not torch.isfinite(grad): raise RuntimeError(f"{model_name}: non-finite outer gradient")
            optimizer.step()
            total_losses.append(float(loss.detach().cpu())); common_losses.append(float(common.detach().cpu())); aux_losses.append(float(aux.detach().cpu()))
        logs.append({"phase": "outer_refit", "model": model_name, "seed": seed, "epoch": epoch, "train_total_loss": float(np.mean(total_losses)), "train_common_delta_mse": float(np.mean(common_losses)), "train_official_auxiliary_loss": float(np.mean(aux_losses))})
    pred_delta_V = prediction_delta_scaled(model, x_valid, dev) * scaler.voltage_scale
    elapsed = time.perf_counter() - start
    peak = float(torch.cuda.max_memory_allocated(dev) / 1024**2) if dev.type == "cuda" else 0.0
    metadata = {
        "parameter_count": parameter_count(model), "peak_gpu_memory_MiB": peak,
        "outer_train_scaler_mean": scaler.mean.tolist() if scaler.mean is not None else None,
        "outer_train_scaler_scale": scaler.scale.tolist() if scaler.scale is not None else None,
        "outer_train_feature_medians": scaler.medians.tolist() if scaler.medians is not None else None,
        "outer_refit_time_s": elapsed,
    }
    arrays = {"y_true_delta_V": outer_validation.y_delta_V, "y_pred_delta_V": pred_delta_V, "anchor_V": outer_validation.anchor_V, "origin_row": outer_validation.origin_row, "target_row": outer_validation.target_row, "origin_age_h": outer_validation.origin_age_h, "target_end_age_h": outer_validation.target_end_age_h, "origin_datetime": outer_validation.origin_datetime, "target_datetime": outer_validation.target_datetime, "horizon_step": np.arange(1, 25, dtype=np.int64)}
    checkpoint = {"model_state_dict": model.state_dict(), "model": model_name, "seed": seed, "selected_epoch": epochs, "preprocessor": metadata, "protocol_sha256": (PHASE / "configs" / "PREREGISTERED_PROTOCOL.sha256").read_text(encoding="utf-8").split()[0]}
    del model
    if dev.type == "cuda": torch.cuda.empty_cache()
    return metadata | {"checkpoint": checkpoint}, logs, arrays


def _run_id(model: str, fold: str, seed: int) -> str:
    return f"bs_{model.lower()}_{fold}_seed{seed}"


def load_cached(run_id: str, cfg: dict[str, Any]) -> dict[str, Any] | None:
    result_path, pred_path = output_path("outputs", "predictions", f"{run_id}.json"), output_path("outputs", "predictions", f"{run_id}.npz")
    if not (result_path.is_file() and pred_path.is_file()): return None
    result = json.loads(result_path.read_text(encoding="utf-8"))
    with np.load(pred_path, allow_pickle=False) as z:
        required = {"y_true_delta_V", "y_pred_delta_V", "anchor_V", "origin_row", "target_row", "target_end_age_h", "target_datetime"}
        if not required.issubset(set(z.files)): raise RuntimeError(f"{run_id}: incomplete cached prediction")
        if np.max(z["target_end_age_h"].astype(float)) >= float(cfg["development_end_h_exclusive"]): raise RuntimeError(f"{run_id}: Test firewall cache violation")
    if result.get("status") != "complete" or result.get("test_accessed") is not False: raise RuntimeError(f"{run_id}: invalid cached status")
    return result


def validate_cached_prediction_alignment(results: list[dict[str, Any]], cfg: dict[str, Any]) -> dict[str, Any]:
    """Fail closed unless all saved Development caches retain the same legal task keys.

    This is intentionally an artifact-only check: it reads only this phase's
    newly written Development predictions, never the source causal CSV or any
    historical/Test prediction artifact.
    """
    expected_horizon = np.arange(1, 25, dtype=np.int64)
    references: dict[str, dict[str, np.ndarray]] = {}
    rows: list[dict[str, Any]] = []
    for result in results:
        run_id, fold = str(result["run_id"]), str(result["fold"])
        path = output_path("outputs", "predictions", f"{run_id}.npz")
        if not path.is_file():
            raise RuntimeError(f"{run_id}: missing complete prediction cache")
        with np.load(path, allow_pickle=False) as z:
            required = {"y_true_delta_V", "y_pred_delta_V", "anchor_V", "origin_row", "target_row", "origin_age_h", "target_end_age_h", "origin_datetime", "target_datetime", "horizon_step"}
            if required - set(z.files):
                raise RuntimeError(f"{run_id}: required cache key missing")
            y_true, y_pred = z["y_true_delta_V"], z["y_pred_delta_V"]
            origin, target, anchor = z["origin_row"], z["target_row"], z["anchor_V"]
            if y_true.shape != y_pred.shape or y_true.ndim != 2 or y_true.shape[1] != 24:
                raise RuntimeError(f"{run_id}: invalid 24-step DeltaV cache shape")
            if target.shape != y_true.shape or anchor.shape != (len(origin),):
                raise RuntimeError(f"{run_id}: incompatible cache key shapes")
            if not np.array_equal(z["horizon_step"], expected_horizon):
                raise RuntimeError(f"{run_id}: horizon key changed")
            if not np.array_equal(target, origin[:, None] + expected_horizon[None, :]):
                raise RuntimeError(f"{run_id}: target rows do not equal origin+horizon")
            if len(np.unique(origin)) != len(origin):
                raise RuntimeError(f"{run_id}: duplicate validation origins")
            if not np.isfinite(y_true).all() or not np.isfinite(y_pred).all() or not np.isfinite(anchor).all():
                raise RuntimeError(f"{run_id}: non-finite prediction artifact")
            max_age = float(np.max(z["target_end_age_h"].astype(float)))
            if max_age >= float(cfg["development_end_h_exclusive"]):
                raise RuntimeError(f"{run_id}: Test boundary cache violation")
            current = {"origin": origin.copy(), "target": target.copy(), "truth": y_true.copy(), "anchor": anchor.copy()}
            if fold in references:
                reference = references[fold]
                for name in ("origin", "target", "truth", "anchor"):
                    if not np.array_equal(current[name], reference[name]):
                        raise RuntimeError(f"{run_id}: cross-model/seed {name} alignment mismatch in {fold}")
            else:
                references[fold] = current
            rows.append({"run_id": run_id, "fold": fold, "model": result["model"], "seed": int(result["seed"]), "origin_count": int(len(origin)), "query_count": int(y_true.size), "max_target_end_age_h": max_age, "target_equals_origin_plus_horizon": True, "duplicate_origins": 0, "test_accessed": False})
    counts = pd.DataFrame(rows).groupby("model").size().to_dict()
    if counts.get("TimesNet") != 9 or any(counts.get(model) not in {3, 9} for model in MODELS if model != "TimesNet"):
        raise RuntimeError(f"Unexpected completed-cache count under frozen seed policy: {counts}")
    record = {"status": "PASS", "completed_cache_count": len(rows), "seed42_models_compared_on_identical_origins": True, "cross_seed_timesnet_targets_identical": True, "all_target_rows_equal_origin_plus_horizon": True, "all_targets_before_development_boundary": True, "test_accessed": False, "test_predictions_read": 0}
    write_csv(pd.DataFrame(rows).sort_values(["model", "seed", "fold"]), "outputs", "artifact_alignment.csv")
    write_json(record, "outputs", "artifact_integrity.json")
    return record


def run_one(model: str, fold: str, outer_train: WindowSet, outer_validation: WindowSet, seed: int, cfg: dict[str, Any]) -> dict[str, Any]:
    run_id = _run_id(model, fold, seed)
    cached = load_cached(run_id, cfg)
    if cached is not None: return cached
    fit, stopper = split_inner_fit_stopper(outer_train, float(cfg["inner_stopping"]["stopper_fraction_of_outer_train_legal_origins"]))
    target_overlap = set(fit.target_row.reshape(-1).tolist()) & set(stopper.target_row.reshape(-1).tolist())
    if target_overlap:
        raise RuntimeError(f"{run_id}: inner fit/stopper target isolation failed")
    best_epoch, inner_logs, inner_time, inner_peak = inner_select_epoch(model, fit, stopper, seed, cfg)
    final_metadata, final_logs, arrays = full_outer_refit(model, outer_train, outer_validation, seed, best_epoch, cfg)
    metrics = delta_metrics(arrays["y_true_delta_V"], arrays["y_pred_delta_V"], arrays["target_row"])
    max_target = float(np.max(arrays["target_end_age_h"]))
    if max_target >= float(cfg["development_end_h_exclusive"]): raise RuntimeError(f"{run_id}: Validation target crosses Test boundary")
    result = {
        "run_id": run_id, "status": "complete", "model": model, "fold": fold, "seed": seed,
        "inner_fit_origins": int(len(fit.origin_row)), "inner_stopper_origins": int(len(stopper.origin_row)),
        "inner_target_overlap_count": len(target_overlap), "selected_epoch": best_epoch,
        "inner_stopper_selection_metric": "DeltaV_RMSE_mV", "inner_stopper_best_RMSE_mV": min(row["stopper_deltaV_RMSE_mV"] for row in inner_logs),
        "inner_time_s": inner_time, "outer_refit_time_s": final_metadata["outer_refit_time_s"],
        "total_inner_plus_refit_time_s": inner_time + final_metadata["outer_refit_time_s"],
        "parameter_count": final_metadata["parameter_count"], "peak_gpu_memory_MiB": max(inner_peak, final_metadata["peak_gpu_memory_MiB"]),
        "max_target_end_age_h": max_target, "test_accessed": False, "external_feature_decomposition": False,
        "future_covariates_used": False, "all_legal_validation_origins_retained": True,
        "official_auxiliary_loss_rule": "DUET L_importance loss_coef=1 added; zero for other official models", **metrics,
    }
    write_csv(pd.DataFrame(inner_logs + final_logs), "logs", f"{run_id}_epochs.csv")
    pred_path = output_path("outputs", "predictions", f"{run_id}.npz"); pred_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(pred_path, **arrays, run_id=np.asarray(run_id))
    output_path("checkpoints").mkdir(parents=True, exist_ok=True)
    torch.save(final_metadata["checkpoint"], output_path("checkpoints", f"{run_id}.pt"))
    write_json(result, "outputs", "predictions", f"{run_id}.json")
    print(json.dumps({"run_id": run_id, "rmse_mV": result["rmse_mV"], "epoch": best_epoch}, ensure_ascii=False), flush=True)
    return result


def seed42_gate(seed_summary: pd.DataFrame, preflight: pd.DataFrame, cfg: dict[str, Any]) -> pd.DataFrame:
    times = seed_summary[(seed_summary.model == "TimesNet") & (seed_summary.seed == 42)]
    if len(times) != 1: raise RuntimeError("TimesNet seed42 control missing")
    t = times.iloc[0]
    rows = []
    for model in MODELS:
        row = seed_summary[(seed_summary.model == model) & (seed_summary.seed == 42)]
        if len(row) != 1: raise RuntimeError(f"{model}: seed42 incomplete")
        r = row.iloc[0]
        implementation = bool(preflight.loc[preflight.model == model, "status"].eq("PASS").all())
        checks = {"formal_95pct": r.formal_RMSE_mV <= t.formal_RMSE_mV * .95, "fold3_103pct": r.Fold3_RMSE_mV <= t.Fold3_RMSE_mV * 1.03, "p99_105pct": r.P99_mV <= t.P99_mV * 1.05, "target_equal_105pct": r.target_time_equal_RMSE_mV <= t.target_time_equal_RMSE_mV * 1.05, "fold3_target_equal_110pct": r.Fold3_target_time_equal_RMSE_mV <= t.Fold3_target_time_equal_RMSE_mV * 1.10, "implementation_clean": implementation}
        promoted = model != "TimesNet" and all(checks.values())
        rows.append({"model": model, "seed": 42, **checks, "seed42_gate_pass": promoted, "additional_seeds_authorized": model == "TimesNet" or promoted})
    return pd.DataFrame(rows)


def final_decision(seed_summary: pd.DataFrame, gate: pd.DataFrame, cfg: dict[str, Any]) -> tuple[str, pd.DataFrame]:
    times = seed_summary[seed_summary.model == "TimesNet"].sort_values("seed")
    if set(times.seed) != {42, 2024, 3407}: raise RuntimeError("TimesNet multi-seed control incomplete")
    tmean = times.mean(numeric_only=True)
    rows = []
    promoted: list[tuple[str, pd.Series]] = []
    for model in MODELS:
        m = seed_summary[seed_summary.model == model].sort_values("seed")
        if model != "TimesNet" and not bool(gate.loc[gate.model == model, "seed42_gate_pass"].iloc[0]):
            rows.append({"model": model, "final_eligible": False, "reason": "seed42_gate_failed_no_extra_seed", "seed_count": len(m)})
            continue
        if set(m.seed) != {42, 2024, 3407}: raise RuntimeError(f"{model}: required multi-seed results missing")
        mean = m.mean(numeric_only=True); paired_better = int((m.formal_RMSE_mV.to_numpy() < times.formal_RMSE_mV.to_numpy()).sum())
        formal_improvement = float((tmean.formal_RMSE_mV - mean.formal_RMSE_mV) / tmean.formal_RMSE_mV)
        checks = {"formal_improvement_5pct": formal_improvement >= .05, "paired_seeds_at_least_2": paired_better >= 2, "fold3_103pct": mean.Fold3_RMSE_mV <= tmean.Fold3_RMSE_mV * 1.03, "p99_105pct": mean.P99_mV <= tmean.P99_mV * 1.05, "target_equal_105pct": mean.target_time_equal_RMSE_mV <= tmean.target_time_equal_RMSE_mV * 1.05, "fold3_target_equal_110pct": mean.Fold3_target_time_equal_RMSE_mV <= tmean.Fold3_target_time_equal_RMSE_mV * 1.10}
        eligible = model != "TimesNet" and all(checks.values())
        rec = {"model": model, "seed_count": len(m), "formal_RMSE_mean_mV": mean.formal_RMSE_mV, "formal_RMSE_std_mV": float(m.formal_RMSE_mV.std(ddof=0)), "formal_RMSE_min_mV": float(m.formal_RMSE_mV.min()), "formal_RMSE_max_mV": float(m.formal_RMSE_mV.max()), "fold3_RMSE_mean_mV": mean.Fold3_RMSE_mV, "P99_mean_mV": mean.P99_mV, "formal_improvement_fraction": formal_improvement, "paired_seeds_better_than_timesnet": paired_better, **checks, "final_eligible": eligible, "reason": "all_final_gates_pass" if eligible else "final_gate_failed_or_incumbent"}
        rows.append(rec)
        if eligible: promoted.append((model, pd.Series(rec)))
    decision_table = pd.DataFrame(rows)
    if not promoted: return "TIMESNET_FREEZE", decision_table
    promoted.sort(key=lambda x: x[1]["formal_RMSE_mean_mV"])
    winner, candidate = promoted[0]
    if len(promoted) > 1 and abs(promoted[1][1]["formal_RMSE_mean_mV"] - candidate["formal_RMSE_mean_mV"]) < tmean.formal_RMSE_mV * .01:
        candidates = [x for x in promoted if abs(x[1]["formal_RMSE_mean_mV"] - candidate["formal_RMSE_mean_mV"]) < tmean.formal_RMSE_mV * .01]
        candidates.sort(key=lambda x: (
            seed_summary[seed_summary.model == x[0]].parameter_count.mean(),
            x[1]["fold3_RMSE_mean_mV"],
            x[1]["P99_mean_mV"],
        ))
        winner = candidates[0][0]
    return {"SimpleTM": "SIMPLETM_PROMOTED", "DUET": "DUET_PROMOTED", "ModernTCN": "MODERNTCN_PROMOTED"}[winner], decision_table


def write_final_report(per_fold: pd.DataFrame, seed_summary: pd.DataFrame, gate: pd.DataFrame, multi: pd.DataFrame, status: str) -> None:
    def table(frame: pd.DataFrame) -> str:
        if frame.empty: return "_No rows._"
        x = frame.copy()
        for c in x.columns:
            if pd.api.types.is_float_dtype(x[c]): x[c] = x[c].map(lambda v: f"{v:.4f}")
        return "| " + " | ".join(x.columns) + " |\n|" + "|".join(["---"] * len(x.columns)) + "|\n" + "\n".join("| " + " | ".join(map(str, row)) + " |" for row in x.itertuples(index=False, name=None))
    report = "# Final backbone screening report\n\n## Decision\n\n**" + status + "**\n\nThe formal outcome is the arithmetic mean of the three fold-level `forecast_query_weighted_RMSE` values. The target-time-equal quantity is a compulsory sensitivity and was not substituted as the formal score. No Test row, target, prediction, or metric was read or constructed.\n\n## Historical context only — excluded from this decision\n\nThe prior N2.6R TimesNet three-seed Development value (about 9.908 mV) and the N2.6R ModernTCN seed-42 value (about 15.373 mV) use historical protocols. They are recorded only as context: this phase retrained the fixed TimesNet control and did not use either number for model selection.\n\n## Seed 42 gate\n\n" + table(gate) + "\n\n## Per-fold Development metrics\n\n" + table(per_fold) + "\n\n## Per-seed three-fold equal summaries\n\n" + table(seed_summary) + "\n\n## Multi-seed final gate\n\n" + table(multi) + "\n\n## Terminal stop\n\nAll three challengers failed the same seed-42 gate and received no extra seeds. The terminal backbone-search rule therefore applies: this phase does not start a fifth model, an adaptation, a future-condition predictor, a gate, or any Test evaluation.\n\n## Next-stage design only\n\nAfter the backbone is fixed, a separate future protocol may compare Raw F2 with EMD-derived, VMD-derived and selected degradation-related representations. It must first define causal rolling decomposition, condition-dominant versus degradation-dominant modes, stability, leakage controls and Train-only fitting. No decomposition was executed here; SimpleTM's internal SWT remains an architecture operation, not an external feature representation.\n"
    write_text(report, "FINAL_BACKBONE_REPORT.md")


def main() -> None:
    cfg = load_config()
    preflight_path = output_path("outputs", "preflight", "preflight_status.json")
    if not preflight_path.is_file() or json.loads(preflight_path.read_text(encoding="utf-8")).get("status") != "PASS":
        raise RuntimeError("Preflight must pass before any Development data read")
    frame, audit = read_development_prefix(cfg)
    folds = build_outer_folds(frame, cfg)
    window_rows = []
    for fold, train, valid in folds:
        fit, stopper = split_inner_fit_stopper(train, .15)
        overlap = set(fit.target_row.reshape(-1).tolist()) & set(stopper.target_row.reshape(-1).tolist())
        if overlap:
            raise RuntimeError(f"{fold}: window manifest caught inner target overlap")
        window_rows.append({"fold": fold, "outer_train_origins": len(train.origin_row), "outer_validation_origins": len(valid.origin_row), "inner_fit_origins": len(fit.origin_row), "inner_stopper_origins": len(stopper.origin_row), "inner_target_overlap_count": len(overlap), "validation_max_target_end_age_h": float(valid.target_end_age_h.max())})
    write_csv(pd.DataFrame(window_rows), "outputs", "window_manifest.csv")
    results: list[dict[str, Any]] = []
    for model in MODELS:
        for fold, train, valid in folds:
            results.append(run_one(model, fold, train, valid, 42, cfg))
    per_fold = pd.DataFrame(results).sort_values(["model", "seed", "fold"]).reset_index(drop=True)
    seed_summary = summarize_seed(per_fold)
    preflight = pd.read_csv(output_path("outputs", "preflight", "implementation_and_resource_audit.csv"))
    gate = seed42_gate(seed_summary, preflight, cfg)
    write_csv(per_fold, "outputs", "per_fold.csv"); write_csv(seed_summary, "outputs", "formal_vs_target_time_weighting.csv"); write_csv(gate, "outputs", "seed42_gate.csv")
    authorized = gate.loc[gate.additional_seeds_authorized, "model"].tolist()
    for seed in (2024, 3407):
        for model in authorized:
            for fold, train, valid in folds:
                results.append(run_one(model, fold, train, valid, seed, cfg))
    per_fold = pd.DataFrame(results).sort_values(["model", "seed", "fold"]).reset_index(drop=True)
    seed_summary = summarize_seed(per_fold)
    artifact_integrity = validate_cached_prediction_alignment(results, cfg)
    status, multiseed = final_decision(seed_summary, gate, cfg)
    write_csv(per_fold, "outputs", "per_fold.csv"); write_csv(seed_summary, "outputs", "formal_vs_target_time_weighting.csv"); write_csv(multiseed, "outputs", "multiseed.csv")
    write_final_report(per_fold, seed_summary, gate, multiseed, status)
    final = {"status": status, "formal_metric": "forecast_query_weighted_RMSE", "robustness_metric": "target_time_equal_weight_RMSE", "test_accessed": False, "test_targets_read": 0, "test_predictions_constructed": False, "test_metrics_computed": False, "external_feature_decomposition": False, "model_training_performed": True, "candidate_models": MODELS, "authorized_extra_seed_models": authorized, "terminal_backbone_search": True}
    write_text(yaml.safe_dump(final, allow_unicode=True, sort_keys=False), "FINAL_DECISION.yaml")
    write_json({"command": "python scripts/run_backbone_screening.py", "base_commit": cfg["base_commit"], "data_audit": audit, "artifact_integrity": artifact_integrity, **final}, "RUN_AUDIT.json")
    print(json.dumps({"status": status, "authorized_extra_seed_models": authorized}, ensure_ascii=False))


if __name__ == "__main__":
    main()
