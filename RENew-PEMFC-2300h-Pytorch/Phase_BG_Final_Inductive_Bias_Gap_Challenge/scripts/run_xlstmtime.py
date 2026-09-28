"""Frozen author-xLSTMTime challenge under the recovered Phase BS contract."""
from __future__ import annotations

import hashlib
import importlib
import json
import math
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset
import yaml

from legacy_protocol import PHASE, recover_windows_and_controls, write_csv, write_json, write_text


CANDIDATE = "xLSTMTime"
ROOT = PHASE / "02_xLSTMTime"
CONFIG_PATH = ROOT / "PREREGISTERED_CONFIG.yaml"
CONFIG_SHA_PATH = ROOT / "PREREGISTERED_CONFIG.sha256"
VENDOR = ROOT / "vendor" / "xLSTMTime_official"
SEEDS = (42, 2024, 3407)


def load_config() -> dict[str, Any]:
    raw = CONFIG_PATH.read_bytes()
    expected = CONFIG_SHA_PATH.read_text(encoding="utf-8").split()[0].lower()
    observed = hashlib.sha256(raw).hexdigest()
    if observed != expected:
        raise RuntimeError("xLSTMTime preregistered-config SHA256 mismatch")
    cfg = yaml.safe_load(raw)
    if cfg.get("status") != "FROZEN_BEFORE_RESOURCE_OR_DEVELOPMENT_RESULTS":
        raise RuntimeError("xLSTMTime configuration is not frozen")
    return cfg


def canonical_hashes(cfg: dict[str, Any]) -> pd.DataFrame:
    rows = []
    commit = str(cfg["source"]["commit"])
    observed_commit = subprocess.check_output(["git", "-C", str(VENDOR), "rev-parse", "HEAD"], text=True).strip()
    for relative, expected in cfg["source"]["canonical_source_sha256"].items():
        raw = subprocess.check_output(["git", "-C", str(VENDOR), "show", f"{commit}:{relative}"])
        observed = hashlib.sha256(raw).hexdigest()
        rows.append({"candidate": CANDIDATE, "repository": cfg["source"]["repository"], "expected_commit": commit, "observed_commit": observed_commit, "relative_path": relative, "expected_sha256": expected, "observed_sha256": observed, "pass": observed_commit == commit and observed == expected})
    result = pd.DataFrame(rows)
    if not result["pass"].all():
        raise RuntimeError("xLSTMTime source fingerprint mismatch")
    return result


class XLSTMTimeAdapter(nn.Module):
    """No-head author-body adapter: retain native forecast head, select V channel."""
    def __init__(self, cfg: dict[str, Any]) -> None:
        super().__init__()
        sys.path.insert(0, str(VENDOR))
        try:
            module = importlib.import_module("xlstm")
        finally:
            sys.path.remove(str(VENDOR))
        m = cfg["frozen_model_configuration"]
        args = SimpleNamespace(context_points=int(m["context_points"]), target_points=int(m["target_points"]), n2=int(m["n2"]))
        self.official_model = module.xlstm(args, enc_in=int(m["enc_in"]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self.official_model(x)
        if output.shape != (x.shape[0], 24, 14):
            raise RuntimeError(f"xLSTMTime official output contract violated: {tuple(output.shape)}")
        return output[:, :, 0]


def device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def delta_prediction(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    absolute = model(x)
    delta = absolute - x[:, -1, 0:1]
    if delta.shape != (x.shape[0], 24):
        raise RuntimeError("DeltaV adapter shape violation")
    return delta


def loader(x: np.ndarray, y: np.ndarray, seed: int) -> DataLoader:
    return DataLoader(TensorDataset(torch.from_numpy(x), torch.from_numpy(y)), batch_size=64, shuffle=True, generator=torch.Generator().manual_seed(seed), num_workers=0)


def predict(model: nn.Module, x: np.ndarray, dev: torch.device) -> np.ndarray:
    out = []
    model.eval()
    with torch.no_grad():
        for begin in range(0, len(x), 256):
            out.append(delta_prediction(model, torch.from_numpy(x[begin:begin + 256]).to(dev)).detach().cpu().numpy())
    return np.concatenate(out, axis=0)


def resource_gate(cfg: dict[str, Any], helpers: Any) -> dict[str, Any]:
    helpers.set_seed(42)
    dev = device()
    base = {
        "candidate": CANDIDATE,
        "device": str(dev),
        "synthetic_input_shape": [4, 48, 14],
        "batch_size": 64,
        "test_accessed": False,
        "development_rows_read": 0,
    }
    try:
        model = XLSTMTimeAdapter(cfg).to(dev)
    except Exception as exc:  # The author implementation itself is immutable.
        return {
            **base,
            "status": "RESOURCE_INFEASIBLE",
            "failure_stage": "author_source_import_or_construction",
            "exception_type": type(exc).__name__,
            "exception_message": str(exc),
            "source_edited": False,
            "substitute_implementation_used": False,
        }
    if dev.type == "cuda": torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats(dev)
    x = torch.randn(4, 48, 14, device=dev, requires_grad=True)
    y = torch.randn(4, 24, device=dev)
    delta = delta_prediction(model, x)
    loss = F.mse_loss(delta, y)
    loss.backward()
    core_grads = [p.grad for p in model.official_model.xlstm_stack.parameters() if p.requires_grad and p.grad is not None]
    core_gradient_finite = bool(core_grads) and all(torch.isfinite(g).all().item() for g in core_grads)
    input_gradient_finite = x.grad is not None and torch.isfinite(x.grad).all().item()
    parameter_gradients_finite = all(torch.isfinite(p.grad).all().item() for p in model.parameters() if p.requires_grad and p.grad is not None)
    model.zero_grad(set_to_none=True)
    batch_x, batch_y = torch.randn(64, 48, 14, device=dev), torch.randn(64, 24, device=dev)
    start = time.perf_counter(); batch_loss = F.mse_loss(delta_prediction(model, batch_x), batch_y); batch_loss.backward()
    if dev.type == "cuda": torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    allocated = float(torch.cuda.memory_allocated(dev) / 1024**2) if dev.type == "cuda" else 0.0
    reserved = float(torch.cuda.memory_reserved(dev) / 1024**2) if dev.type == "cuda" else 0.0
    peak = float(torch.cuda.max_memory_allocated(dev) / 1024**2) if dev.type == "cuda" else 0.0
    params = int(sum(p.numel() for p in model.parameters() if p.requires_grad))
    active_blocks = [b.__class__.__name__ for b in model.official_model.xlstm_stack.blocks]
    passed = bool(torch.isfinite(delta).all()) and bool(torch.isfinite(loss)) and bool(torch.isfinite(batch_loss)) and input_gradient_finite and parameter_gradients_finite and core_gradient_finite and active_blocks == ["mLSTMBlock"] * 3
    result = {**base, "status": "PASS" if passed else "RESOURCE_INFEASIBLE", "failure_stage": None if passed else "synthetic_forward_backward", "output_shape": list(delta.shape), "parameter_count": params, "input_gradient_finite": bool(input_gradient_finite), "parameter_gradients_finite": bool(parameter_gradients_finite), "core_xlstm_gradient_finite": bool(core_gradient_finite), "active_blocks": active_blocks, "forward_loss_finite": bool(torch.isfinite(loss)), "batch64_loss_finite": bool(torch.isfinite(batch_loss)), "batch64_step_time_s": elapsed, "cuda_allocated_MiB": allocated, "cuda_reserved_MiB": reserved, "peak_gpu_memory_MiB": peak, "source_edited": False, "substitute_implementation_used": False}
    del model
    if dev.type == "cuda": torch.cuda.empty_cache()
    return result


def fit_select_epoch(model_name: str, fit: Any, stopper: Any, seed: int, cfg: dict[str, Any], helpers: Any) -> tuple[int, list[dict[str, Any]], float, float]:
    helpers.set_seed(seed); dev = device(); scaler = helpers.FoldPreprocessor().fit(fit.x)
    x_fit, x_stop = scaler.transform(fit.x), scaler.transform(stopper.x)
    y_fit = helpers.standardized_delta_target(fit, scaler)
    model = XLSTMTimeAdapter(cfg).to(dev); opt = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=1e-5)
    best, best_epoch, stale, logs = math.inf, 0, 0, []; start = time.perf_counter()
    for epoch in range(1, 301):
        model.train(); losses = []
        for xb, yb in loader(x_fit, y_fit, seed + epoch):
            xb, yb = xb.to(dev), yb.to(dev); opt.zero_grad(set_to_none=True); loss = F.mse_loss(delta_prediction(model, xb), yb)
            if not torch.isfinite(loss): raise RuntimeError("Non-finite xLSTMTime inner loss")
            loss.backward(); grad = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            if not torch.isfinite(grad): raise RuntimeError("Non-finite xLSTMTime inner gradient")
            opt.step(); losses.append(float(loss.detach().cpu()))
        pred = predict(model, x_stop, dev) * scaler.voltage_scale
        score = helpers.delta_metrics(stopper.y_delta_V, pred, stopper.target_row)["rmse_mV"]
        logs.append({"phase": "inner_stopper", "candidate": model_name, "seed": seed, "epoch": epoch, "train_standardized_delta_mse": float(np.mean(losses)), "stopper_deltaV_RMSE_mV": score})
        if score < best - 1e-10: best, best_epoch, stale = score, epoch, 0
        else: stale += 1
        if stale >= 30: break
    elapsed = time.perf_counter() - start; peak = float(torch.cuda.max_memory_allocated(dev) / 1024**2) if dev.type == "cuda" else 0.0
    del model
    if dev.type == "cuda": torch.cuda.empty_cache()
    return best_epoch, logs, elapsed, peak


def refit_and_score(model_name: str, train: Any, valid: Any, seed: int, epochs: int, cfg: dict[str, Any], helpers: Any) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, np.ndarray]]:
    helpers.set_seed(seed); dev = device(); scaler = helpers.FoldPreprocessor().fit(train.x)
    x_train, x_valid = scaler.transform(train.x), scaler.transform(valid.x); y_train = helpers.standardized_delta_target(train, scaler)
    model = XLSTMTimeAdapter(cfg).to(dev)
    if dev.type == "cuda": torch.cuda.reset_peak_memory_stats(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=1e-5); logs = []; start = time.perf_counter()
    for epoch in range(1, epochs + 1):
        model.train(); losses = []
        for xb, yb in loader(x_train, y_train, seed + epoch):
            xb, yb = xb.to(dev), yb.to(dev); opt.zero_grad(set_to_none=True); loss = F.mse_loss(delta_prediction(model, xb), yb)
            if not torch.isfinite(loss): raise RuntimeError("Non-finite xLSTMTime outer loss")
            loss.backward(); grad = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            if not torch.isfinite(grad): raise RuntimeError("Non-finite xLSTMTime outer gradient")
            opt.step(); losses.append(float(loss.detach().cpu()))
        logs.append({"phase": "outer_refit", "candidate": model_name, "seed": seed, "epoch": epoch, "train_standardized_delta_mse": float(np.mean(losses))})
    pred = predict(model, x_valid, dev) * scaler.voltage_scale; elapsed = time.perf_counter() - start
    peak = float(torch.cuda.max_memory_allocated(dev) / 1024**2) if dev.type == "cuda" else 0.0
    metadata = {"parameter_count": int(sum(p.numel() for p in model.parameters() if p.requires_grad)), "peak_gpu_memory_MiB": peak, "outer_refit_time_s": elapsed, "scaler_mean": scaler.mean.tolist(), "scaler_scale": scaler.scale.tolist(), "scaler_median": scaler.medians.tolist(), "selected_epoch": epochs}
    arrays = {"y_true_delta_V": valid.y_delta_V, "y_pred_delta_V": pred, "anchor_V": valid.anchor_V, "origin_row": valid.origin_row, "target_row": valid.target_row, "origin_age_h": valid.origin_age_h, "target_end_age_h": valid.target_end_age_h, "origin_datetime": valid.origin_datetime, "target_datetime": valid.target_datetime, "horizon_step": np.arange(1, 25, dtype=np.int64)}
    checkpoint = {"model_state_dict": model.state_dict(), "candidate": CANDIDATE, "seed": seed, "metadata": metadata}
    del model
    if dev.type == "cuda": torch.cuda.empty_cache()
    metadata["checkpoint"] = checkpoint
    return metadata, logs, arrays


def output(*parts: str) -> Path:
    path = ROOT.joinpath(*parts); path.parent.mkdir(parents=True, exist_ok=True); return path


def run_one(fold: str, train: Any, valid: Any, seed: int, cfg: dict[str, Any], helpers: Any) -> dict[str, Any]:
    run_id = f"xlstmtime_{fold}_seed{seed}"; js, nz = output("outputs", "predictions", f"{run_id}.json"), output("outputs", "predictions", f"{run_id}.npz")
    if js.is_file() and nz.is_file():
        saved = json.loads(js.read_text(encoding="utf-8"))
        if saved.get("status") == "complete" and saved.get("test_accessed") is False: return saved
        raise RuntimeError(f"Invalid cached candidate run: {run_id}")
    fit, stopper = helpers.split_inner_fit_stopper(train, .15)
    if set(fit.target_row.reshape(-1)) & set(stopper.target_row.reshape(-1)): raise RuntimeError("Inner target isolation failed")
    selected_epoch, inner, inner_s, inner_peak = fit_select_epoch(CANDIDATE, fit, stopper, seed, cfg, helpers)
    metadata, outer, arrays = refit_and_score(CANDIDATE, train, valid, seed, selected_epoch, cfg, helpers)
    if float(arrays["target_end_age_h"].max()) >= float(cfg.get("development_end_h_exclusive", 1819.997738)): raise RuntimeError("Candidate cache crosses Development boundary")
    metrics = helpers.delta_metrics(arrays["y_true_delta_V"], arrays["y_pred_delta_V"], arrays["target_row"])
    result = {"run_id": run_id, "status": "complete", "candidate": CANDIDATE, "fold": fold, "seed": seed, "inner_fit_origins": len(fit.origin_row), "inner_stopper_origins": len(stopper.origin_row), "inner_target_overlap_count": 0, "selected_epoch": selected_epoch, "inner_stopper_metric": "DeltaV_RMSE_mV", "inner_stopper_best_RMSE_mV": min(row["stopper_deltaV_RMSE_mV"] for row in inner), "inner_time_s": inner_s, "outer_refit_time_s": metadata["outer_refit_time_s"], "total_inner_plus_refit_time_s": inner_s + metadata["outer_refit_time_s"], "parameter_count": metadata["parameter_count"], "peak_gpu_memory_MiB": max(inner_peak, metadata["peak_gpu_memory_MiB"]), "max_target_end_age_h": float(arrays["target_end_age_h"].max()), "test_accessed": False, "future_covariates_used": False, "external_feature_decomposition": False, "all_legal_validation_origins_retained": True, **metrics}
    write_csv(pd.DataFrame(inner + outer), "02_xLSTMTime", "logs", f"{run_id}_epochs.csv")
    np.savez_compressed(nz, **arrays, run_id=np.asarray(run_id))
    torch.save(metadata["checkpoint"], output("outputs", "checkpoints", f"{run_id}.pt"))
    write_json(result, "02_xLSTMTime", "outputs", "predictions", f"{run_id}.json")
    print(json.dumps({"run_id": run_id, "rmse_mV": result["rmse_mV"], "epoch": selected_epoch}, ensure_ascii=False), flush=True)
    return result


def summarize(rows: pd.DataFrame) -> pd.DataFrame:
    return rows.groupby("seed", as_index=False).agg(formal_RMSE_mV=("rmse_mV", "mean"), MAE_mV=("mae_mV", "mean"), R2=("r2", "mean"), endpoint_RMSE_mV=("endpoint_rmse_mV", "mean"), P95_mV=("p95_abs_error_mV", "mean"), P99_mV=("p99_abs_error_mV", "mean"), max_abs_error_mV=("max_abs_error_mV", "max"), target_time_equal_RMSE_mV=("target_time_equal_rmse_mV", "mean"), Fold3_RMSE_mV=("rmse_mV", lambda s: float(s[rows.loc[s.index, "fold"].eq("fold_3")].iloc[0])), Fold3_target_time_equal_RMSE_mV=("target_time_equal_rmse_mV", lambda s: float(s[rows.loc[s.index, "fold"].eq("fold_3")].iloc[0])), parameter_count=("parameter_count", "max"), peak_gpu_memory_MiB=("peak_gpu_memory_MiB", "max"), total_inner_plus_refit_time_s=("total_inner_plus_refit_time_s", "sum"))


def control_summary() -> pd.DataFrame:
    control = pd.read_csv(PHASE.parent / "Phase_BS_Final_Backbone_Screening" / "outputs" / "formal_vs_target_time_weighting.csv")
    result = control.loc[control.model.eq("TimesNet")].copy()
    if set(result.seed) != set(SEEDS): raise RuntimeError("Frozen TimesNet multi-seed control incomplete")
    return result


def seed42_gate(candidate: pd.Series, control: pd.Series) -> dict[str, Any]:
    checks = {"formal_95pct": candidate.formal_RMSE_mV <= control.formal_RMSE_mV * .95, "fold3_103pct": candidate.Fold3_RMSE_mV <= control.Fold3_RMSE_mV * 1.03, "p99_105pct": candidate.P99_mV <= control.P99_mV * 1.05, "target_equal_105pct": candidate.target_time_equal_RMSE_mV <= control.target_time_equal_RMSE_mV * 1.05, "fold3_target_equal_110pct": candidate.Fold3_target_time_equal_RMSE_mV <= control.Fold3_target_time_equal_RMSE_mV * 1.10, "protocol_compliant": True}
    return {"candidate": CANDIDATE, "seed": 42, **checks, "pass": all(checks.values()), "control": "Phase_BS_TimesNet_same_raw_F2_protocol"}


def multiseed_gate(candidate: pd.DataFrame, control: pd.DataFrame) -> dict[str, Any]:
    merged = candidate.merge(control, on="seed", suffixes=("_candidate", "_control"), validate="one_to_one")
    cmean, tmean = merged.mean(numeric_only=True), control.mean(numeric_only=True)
    improvement = (tmean.formal_RMSE_mV - cmean.formal_RMSE_mV) / tmean.formal_RMSE_mV
    checks = {"formal_improvement_5pct": improvement >= .05, "paired_seeds_at_least_2": int((merged.formal_RMSE_mV_candidate < merged.formal_RMSE_mV_control).sum()) >= 2, "fold3_103pct": cmean.Fold3_RMSE_mV <= tmean.Fold3_RMSE_mV * 1.03, "p99_105pct": cmean.P99_mV <= tmean.P99_mV * 1.05, "target_equal_105pct": cmean.target_time_equal_RMSE_mV <= tmean.target_time_equal_RMSE_mV * 1.05, "fold3_target_equal_110pct": cmean.Fold3_target_time_equal_RMSE_mV <= tmean.Fold3_target_time_equal_RMSE_mV * 1.10}
    return {"candidate": CANDIDATE, "formal_improvement_fraction": float(improvement), "paired_seeds_better_than_control": int((merged.formal_RMSE_mV_candidate < merged.formal_RMSE_mV_control).sum()), **checks, "pass": all(checks.values())}


def main() -> None:
    final_decision = PHASE / "final" / "FINAL_DECISION.yaml"
    if final_decision.is_file() and "backbone_search: CLOSED" in final_decision.read_text(encoding="utf-8"):
        raise RuntimeError("Phase BG is terminally closed; xLSTMTime training cannot be restarted")
    cfg = load_config(); legacy_cfg, helpers, frame, folds, windows, protocol = recover_windows_and_controls()
    if not bool(protocol["window_identity_pass"]): raise RuntimeError("Cannot challenge before window-identity pass")
    source = canonical_hashes(cfg); write_csv(source, "02_xLSTMTime", "SOURCE_FINGERPRINTS.csv")
    resource_path = ROOT / "RESOURCE_AUDIT.json"
    resource = json.loads(resource_path.read_text(encoding="utf-8")) if resource_path.is_file() else resource_gate(cfg, helpers)
    write_json(resource, "02_xLSTMTime", "RESOURCE_AUDIT.json")
    if resource["status"] != "PASS":
        write_json({"candidate": CANDIDATE, "final_status": "RESOURCE_INFEASIBLE", "next_candidate": "CFPT", "test_accessed": False}, "02_xLSTMTime", "CANDIDATE_DECISION.json")
        print(json.dumps({"status": "RESOURCE_INFEASIBLE"})); return
    results = [run_one(fold, train, valid, 42, cfg, helpers) for fold, train, valid in folds]
    rows = pd.DataFrame(results).sort_values(["seed", "fold"]); summary = summarize(rows); control = control_summary()
    gate = seed42_gate(summary.loc[summary.seed.eq(42)].iloc[0], control.loc[control.seed.eq(42)].iloc[0]); write_csv(rows, "02_xLSTMTime", "per_fold.csv"); write_csv(summary, "02_xLSTMTime", "seed_summary.csv"); write_json(gate, "02_xLSTMTime", "seed42_gate.json")
    if not gate["pass"]:
        decision = {"candidate": CANDIDATE, "final_status": "SEED42_FAILED", "next_candidate": "CFPT", "test_accessed": False, "extra_seeds_run": []}; write_json(decision, "02_xLSTMTime", "CANDIDATE_DECISION.json"); print(json.dumps(decision)); return
    for seed in (2024, 3407):
        results.extend(run_one(fold, train, valid, seed, cfg, helpers) for fold, train, valid in folds)
    rows = pd.DataFrame(results).sort_values(["seed", "fold"]); summary = summarize(rows); multi = multiseed_gate(summary, control)
    write_csv(rows, "02_xLSTMTime", "per_fold.csv"); write_csv(summary, "02_xLSTMTime", "seed_summary.csv"); write_json(multi, "02_xLSTMTime", "multiseed_gate.json")
    decision = {"candidate": CANDIDATE, "final_status": "PROMOTED" if multi["pass"] else "MULTISEED_FAILED", "next_candidate": None if multi["pass"] else "CFPT", "test_accessed": False, "extra_seeds_run": [2024, 3407]}; write_json(decision, "02_xLSTMTime", "CANDIDATE_DECISION.json"); print(json.dumps(decision))


if __name__ == "__main__":
    main()
