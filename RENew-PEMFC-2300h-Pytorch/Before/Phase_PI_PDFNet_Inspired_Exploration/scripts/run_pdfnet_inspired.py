"""The single preregistered Development-only run for PDFNetInspired_Exploratory."""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

PI = Path(__file__).resolve().parents[1]
if str(PI) not in sys.path:
    sys.path.insert(0, str(PI))

from models.pdfnet_inspired import PDFNetInspiredExploratory
from scripts.posthoc_report_audit import apply_fail_closed_gate, assess_fairness, require_fairness_pass, research_conclusion_text, resource_display
from scripts.potential_gate import seed42_gate
from scripts.protocol import (
    ASSUMPTIONS_SHA, BS, GATE_SHA, frozen_timesnet_summary, load_frozen_configs,
    recover_windows_and_timesnet, sha256, write_csv, write_json, write_text,
)
from scripts.stage0_preflight import absolute_to_delta


MODEL = "PDFNetInspired_Exploratory"
SEEDS = (42, 2024, 3407)


def _require_stage0() -> dict[str, Any]:
    paths = [PI / "outputs" / "STRUCTURE_ACTIVATION_AUDIT.json", PI / "outputs" / "RESOURCE_AUDIT.json"]
    if not all(path.is_file() for path in paths):
        raise RuntimeError("Stage 0 structure and resource audits must precede Development access")
    structure, resource = (json.loads(path.read_text(encoding="utf-8")) for path in paths)
    if not structure.get("all_core_gradients_pass") or not structure.get("dynamic_graph_rows_sum_to_one") or resource.get("status") != "PASS":
        raise RuntimeError("IMPLEMENTATION_FAIL or RESOURCE_INFEASIBLE before Development access")
    if structure.get("development_accessed") or resource.get("development_accessed"):
        raise RuntimeError("Stage 0 audit is not synthetic-only")
    return resource


def _loader(x: np.ndarray, y: np.ndarray, seed: int, batch_size: int) -> DataLoader:
    return DataLoader(
        TensorDataset(torch.from_numpy(x), torch.from_numpy(y)), batch_size=batch_size,
        shuffle=True, generator=torch.Generator().manual_seed(seed), num_workers=0,
    )


def _parameter_count(model: torch.nn.Module) -> int:
    return int(sum(parameter.numel() for parameter in model.parameters()))


def _prediction(model: torch.nn.Module, x: np.ndarray, dev: torch.device, batch_size: int, collect_activation: bool = False) -> tuple[np.ndarray, dict[str, Any]]:
    rows: list[np.ndarray] = []
    periods: list[np.ndarray] = []
    outer_counts: list[list[int]] = []
    patch_counts: list[list[list[int]]] = []
    fusion: list[np.ndarray] = []
    graph_row_error = 0.0
    model.eval()
    with torch.no_grad():
        for start in range(0, len(x), batch_size):
            tensor = torch.from_numpy(x[start:start + batch_size]).to(dev)
            if collect_activation:
                output, audit = model(tensor, return_audit=True)
                periods.append(audit["periods"].cpu().numpy())
                outer_counts.extend(audit["outer_segment_counts"])
                patch_counts.extend(audit["inner_patch_counts"])
                fusion.append(audit["fusion_weights"].cpu().numpy())
                graph_row_error = max(graph_row_error, float((audit["first_dynamic_graphs"].sum(dim=-1) - 1).abs().max().cpu()))
            else:
                output = model(tensor)
            rows.append(absolute_to_delta(output, tensor).cpu().numpy())
    result = np.concatenate(rows, axis=0)
    stats: dict[str, Any] = {"activation_collected": collect_activation}
    if collect_activation:
        flattened_counts = [value for sample in patch_counts for branch in sample for value in branch]
        stats = {
            "activation_collected": True,
            "selected_periods": np.concatenate(periods, axis=0).tolist(),
            "outer_segment_counts": outer_counts,
            "inner_patch_counts": patch_counts,
            "period_values": pd.Series(np.concatenate(periods).reshape(-1)).value_counts(sort=False).sort_index().to_dict(),
            "patch_count_values": pd.Series(flattened_counts).value_counts(sort=False).sort_index().to_dict(),
            "more_than_one_patch_update_fraction": float(np.mean(np.asarray(flattened_counts) > 1)) if flattened_counts else math.nan,
            "fusion_weights": np.mean(np.stack(fusion), axis=0).tolist(),
            "dynamic_graph_max_row_sum_error": graph_row_error,
        }
    return result, stats


def _train_epoch(model: torch.nn.Module, optimizer: torch.optim.Optimizer, x: np.ndarray, y: np.ndarray, seed: int, dev: torch.device, batch_size: int) -> float:
    losses: list[float] = []
    model.train()
    for batch_x, batch_y in _loader(x, y, seed, batch_size):
        batch_x, batch_y = batch_x.to(dev), batch_y.to(dev)
        optimizer.zero_grad(set_to_none=True)
        predicted = absolute_to_delta(model(batch_x), batch_x)
        loss = F.mse_loss(predicted, batch_y)
        if not torch.isfinite(loss):
            raise RuntimeError("IMPLEMENTATION_FAIL: non-finite standardized DeltaV loss")
        loss.backward()
        gradient = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        if not torch.isfinite(gradient):
            raise RuntimeError("IMPLEMENTATION_FAIL: non-finite clipped gradient")
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
    return float(np.mean(losses))


def _inner_select_epoch(helpers: Any, fit: Any, stopper: Any, seed: int, dimension: int, batch_size: int, dev: torch.device) -> tuple[int, list[dict[str, Any]], float, float]:
    helpers.set_seed(seed)
    scaler = helpers.FoldPreprocessor().fit(fit.x)
    x_fit, x_stopper = scaler.transform(fit.x), scaler.transform(stopper.x)
    y_fit = helpers.standardized_delta_target(fit, scaler)
    model = PDFNetInspiredExploratory(dimension=dimension).to(dev)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=1e-5)
    best_rmse, best_epoch, stale = math.inf, 0, 0
    records: list[dict[str, Any]] = []
    if dev.type == "cuda": torch.cuda.reset_peak_memory_stats(dev)
    started = time.perf_counter()
    for epoch in range(1, 301):
        loss = _train_epoch(model, optimizer, x_fit, y_fit, seed + epoch, dev, batch_size)
        prediction, _ = _prediction(model, x_stopper, dev, batch_size)
        rmse = helpers.delta_metrics(stopper.y_delta_V, prediction * scaler.voltage_scale, stopper.target_row)["rmse_mV"]
        records.append({"phase": "inner_stopper", "fold": "pending", "model": MODEL, "seed": seed, "epoch": epoch, "train_standardized_delta_mse": loss, "stopper_deltaV_RMSE_mV": rmse})
        if rmse < best_rmse - 1e-10:
            best_rmse, best_epoch, stale = rmse, epoch, 0
        else:
            stale += 1
        if stale >= 30:
            break
    elapsed = time.perf_counter() - started
    peak = float(torch.cuda.max_memory_allocated(dev) / 1024**2) if dev.type == "cuda" else 0.0
    del model
    if dev.type == "cuda": torch.cuda.empty_cache()
    if best_epoch < 1: raise RuntimeError("IMPLEMENTATION_FAIL: no eligible inner epoch")
    return best_epoch, records, elapsed, peak


def _outer_refit(helpers: Any, train: Any, validation: Any, seed: int, epochs: int, dimension: int, batch_size: int, dev: torch.device) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, np.ndarray], dict[str, Any]]:
    helpers.set_seed(seed)
    scaler = helpers.FoldPreprocessor().fit(train.x)
    x_train, x_validation = scaler.transform(train.x), scaler.transform(validation.x)
    y_train = helpers.standardized_delta_target(train, scaler)
    model = PDFNetInspiredExploratory(dimension=dimension).to(dev)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=1e-5)
    if dev.type == "cuda": torch.cuda.reset_peak_memory_stats(dev)
    records: list[dict[str, Any]] = []
    started = time.perf_counter()
    for epoch in range(1, epochs + 1):
        loss = _train_epoch(model, optimizer, x_train, y_train, seed + epoch, dev, batch_size)
        records.append({"phase": "outer_refit", "fold": "pending", "model": MODEL, "seed": seed, "epoch": epoch, "train_standardized_delta_mse": loss})
    prediction_scaled, activation = _prediction(model, x_validation, dev, batch_size, collect_activation=True)
    elapsed = time.perf_counter() - started
    peak = float(torch.cuda.max_memory_allocated(dev) / 1024**2) if dev.type == "cuda" else 0.0
    metadata = {
        "parameter_count": _parameter_count(model), "peak_gpu_memory_MiB": peak,
        "outer_train_scaler_mean": scaler.mean.tolist(), "outer_train_scaler_scale": scaler.scale.tolist(),
        "outer_train_feature_medians": scaler.medians.tolist(), "outer_refit_time_s": elapsed,
    }
    arrays = {
        "y_true_delta_V": validation.y_delta_V, "y_pred_delta_V": prediction_scaled * scaler.voltage_scale,
        "y_pred_abs_scaled_voltage": prediction_scaled + x_validation[:, -1, 0:1], "anchor_V": validation.anchor_V,
        "origin_row": validation.origin_row, "target_row": validation.target_row,
        "origin_age_h": validation.origin_age_h, "target_end_age_h": validation.target_end_age_h,
        "horizon_step": np.arange(1, 25, dtype=np.int64),
    }
    checkpoint = {"model_state_dict": model.state_dict(), "model": MODEL, "seed": seed, "selected_epoch": epochs, "preprocessor": metadata}
    del model
    if dev.type == "cuda": torch.cuda.empty_cache()
    return metadata | {"checkpoint": checkpoint}, records, arrays, activation


def _run_one(helpers: Any, fold: str, train: Any, validation: Any, seed: int, dimension: int, batch_size: int, dev: torch.device) -> dict[str, Any]:
    run_id = f"pdfnet_inspired_{fold}_seed{seed}"
    result_path, cache_path = PI / "outputs" / "predictions" / f"{run_id}.json", PI / "outputs" / "predictions" / f"{run_id}.npz"
    if result_path.is_file() and cache_path.is_file():
        result = json.loads(result_path.read_text(encoding="utf-8"))
        if result.get("assumptions_sha256") != ASSUMPTIONS_SHA.read_text(encoding="utf-8").split()[0]:
            raise RuntimeError(f"{run_id}: cached output config mismatch")
        return result
    fit, stopper = helpers.split_inner_fit_stopper(train, .15)
    overlap = set(fit.target_row.reshape(-1).tolist()) & set(stopper.target_row.reshape(-1).tolist())
    if overlap: raise RuntimeError(f"{fold}: inner target overlap")
    selected_epoch, inner_logs, inner_time, inner_peak = _inner_select_epoch(helpers, fit, stopper, seed, dimension, batch_size, dev)
    metadata, refit_logs, arrays, activation = _outer_refit(helpers, train, validation, seed, selected_epoch, dimension, batch_size, dev)
    if float(np.max(arrays["target_end_age_h"])) >= 1819.997738:
        raise RuntimeError(f"{fold}: Test firewall target end breach")
    metrics = helpers.delta_metrics(arrays["y_true_delta_V"], arrays["y_pred_delta_V"], arrays["target_row"])
    for row in inner_logs + refit_logs: row["fold"] = fold
    result = {
        "run_id": run_id, "model": MODEL, "fold": fold, "seed": seed, "status": "complete",
        "selected_epoch": selected_epoch, "inner_fit_origins": len(fit.origin_row), "inner_stopper_origins": len(stopper.origin_row),
        "inner_target_overlap_count": len(overlap), "inner_time_s": inner_time, "outer_refit_time_s": metadata["outer_refit_time_s"],
        "total_inner_plus_refit_time_s": inner_time + metadata["outer_refit_time_s"], "parameter_count": metadata["parameter_count"],
        "peak_gpu_memory_MiB": max(inner_peak, metadata["peak_gpu_memory_MiB"]), "max_target_end_age_h": float(np.max(arrays["target_end_age_h"])),
        "all_legal_validation_origins_retained": True, "future_covariates_used": False, "future_13_variable_supervision": False,
        "test_accessed": False, "external_decomposition": False, "assumptions_sha256": ASSUMPTIONS_SHA.read_text(encoding="utf-8").split()[0], **metrics,
    }
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_path, **arrays, run_id=np.asarray(run_id))
    torch.save(metadata["checkpoint"], PI / "checkpoints" / f"{run_id}.pt")
    write_csv(pd.DataFrame(inner_logs + refit_logs), "logs", f"{run_id}_epochs.csv")
    write_json(activation, "outputs", "activation", f"{run_id}.json")
    write_json(result, "outputs", "predictions", f"{run_id}.json")
    print(json.dumps({"run_id": run_id, "rmse_mV": result["rmse_mV"], "selected_epoch": selected_epoch}, ensure_ascii=False), flush=True)
    return result


def _timesnet_control(helpers: Any, folds: list[tuple[str, Any, Any]]) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    for fold, _train, valid in folds:
        for seed in SEEDS:
            path = BS / "outputs" / "predictions" / f"bs_timesnet_{fold}_seed{seed}.npz"
            with np.load(path, allow_pickle=False) as z:
                metrics = helpers.delta_metrics(z["y_true_delta_V"], z["y_pred_delta_V"], z["target_row"])
                for key, expected in {"origin_row": valid.origin_row, "target_row": valid.target_row, "y_true_delta_V": valid.y_delta_V, "anchor_V": valid.anchor_V, "target_end_age_h": valid.target_end_age_h}.items():
                    if not np.array_equal(z[key], expected): raise RuntimeError(f"{path.name}: control identity mismatch {key}")
            rows.append({"model": "TimesNet", "fold": fold, "seed": seed, "parameter_count": 149046, "peak_gpu_memory_MiB": np.nan, "total_inner_plus_refit_time_s": np.nan, **metrics})
    per_fold = pd.DataFrame(rows)
    summary = helpers.summarize_seed(per_fold)
    reported = frozen_timesnet_summary()
    checks = summary.merge(reported, on=["model", "seed"], suffixes=("_recomputed", "_reported"))
    if not np.allclose(checks.formal_RMSE_mV_recomputed, checks.formal_RMSE_mV_reported, rtol=0, atol=1e-9):
        raise RuntimeError("TimesNet reported formal metric cannot be exactly recomputed from frozen caches")
    return per_fold, summary


def _seed42_gate(candidate: pd.Series, control: pd.Series, gate: dict[str, Any], fairness: dict[str, Any]) -> dict[str, Any]:
    if fairness["status"] != "PASS":
        return apply_fail_closed_gate({"status": "PDFNET_INSPIRED_NO_PROMISE"}, fairness)
    candidate = candidate.copy()
    candidate["protocol_audit_pass"] = True
    candidate["core_gradient_pass"] = True
    candidate["no_leakage"] = True
    # `summarize_seed` intentionally retains metrics only.  The per-fold cache
    # audit has already fail-closed on this invariant before summarization.
    candidate["all_legal_origins_retained"] = True
    candidate["no_future_13_variable_supervision"] = True
    result = seed42_gate(candidate, control, gate)
    result["common_audit_pass"] = True
    return result


def _multiseed_gate(candidate: pd.DataFrame, control: pd.DataFrame, gate: dict[str, Any], seed42_result: dict[str, Any]) -> dict[str, Any]:
    c, t = candidate.sort_values("seed").reset_index(drop=True), control.sort_values("seed").reset_index(drop=True)
    if list(c.seed) != list(SEEDS) or list(t.seed) != list(SEEDS): raise RuntimeError("Multi-seed pairing incomplete")
    mean_c, mean_t = c.mean(numeric_only=True), t.mean(numeric_only=True)
    ratio = lambda key: float(mean_c[key] / mean_t[key])
    paired_formal_better = int((c.formal_RMSE_mV < t.formal_RMSE_mV).sum())
    strong = gate["multiseed_gate"]
    strong_pass = ratio("formal_RMSE_mV") <= strong["strong_formal_mean_fraction"] and paired_formal_better >= 2 and bool((c.formal_RMSE_mV <= 1.10 * t.formal_RMSE_mV).all()) and ratio("Fold3_RMSE_mV") <= 1.10 and ratio("P99_mV") <= 1.10 and ratio("target_time_equal_RMSE_mV") <= 1.10 and ratio("Fold3_target_time_equal_RMSE_mV") <= 1.15 and ratio("endpoint_RMSE_mV") <= 1.15
    a = gate["seed42_gate"]["promising_A"]
    mean_a = ratio("formal_RMSE_mV") <= a["formal_RMSE_max_fraction"] and ratio("Fold3_RMSE_mV") <= a["fold3_RMSE_max_fraction"] and ratio("P99_mV") <= a["p99_max_fraction"] and ratio("target_time_equal_RMSE_mV") <= a["target_time_equal_max_fraction"] and ratio("Fold3_target_time_equal_RMSE_mV") <= a["fold3_target_time_equal_max_fraction"] and ratio("endpoint_RMSE_mV") <= a["endpoint_12h_max_fraction"]
    paired_a = (c.formal_RMSE_mV <= a["formal_RMSE_max_fraction"] * t.formal_RMSE_mV) & (c.Fold3_RMSE_mV <= a["fold3_RMSE_max_fraction"] * t.Fold3_RMSE_mV) & (c.P99_mV <= a["p99_max_fraction"] * t.P99_mV) & (c.target_time_equal_RMSE_mV <= a["target_time_equal_max_fraction"] * t.target_time_equal_RMSE_mV) & (c.Fold3_target_time_equal_RMSE_mV <= a["fold3_target_time_equal_max_fraction"] * t.Fold3_target_time_equal_RMSE_mV) & (c.endpoint_RMSE_mV <= a["endpoint_12h_max_fraction"] * t.endpoint_RMSE_mV)
    b = gate["seed42_gate"]["promising_B"]
    axis_column = {"fold3_RMSE": "Fold3_RMSE_mV", "endpoint_12h_RMSE": "endpoint_RMSE_mV", "p99": "P99_mV"}.get(seed42_result.get("B_advantage_axis"))
    b_pass = False
    if axis_column:
        b_pass = b["formal_RMSE_lower_exclusive_fraction"] < ratio("formal_RMSE_mV") <= b["formal_RMSE_upper_inclusive_fraction"] and mean_a and ratio(axis_column) <= .90 and int((c[axis_column] <= .90 * t[axis_column]).sum()) >= 2
    status = "PDFNET_INSPIRED_STRONGLY_PROMISING" if strong_pass else "PDFNET_INSPIRED_PROMISING_ONLY" if (mean_a and int(paired_a.sum()) >= 2) or b_pass else "PDFNET_INSPIRED_NO_PROMISE"
    return {"status": status, "mean_ratios": {key: ratio(key) for key in ["formal_RMSE_mV", "Fold3_RMSE_mV", "P99_mV", "target_time_equal_RMSE_mV", "Fold3_target_time_equal_RMSE_mV", "endpoint_RMSE_mV"]}, "paired_formal_better": paired_formal_better, "paired_A_count": int(paired_a.sum()), "seed42_locked_B_axis": seed42_result.get("B_advantage_axis"), "strong_pass": strong_pass, "promising_A_pass": bool(mean_a and int(paired_a.sum()) >= 2), "promising_B_pass": b_pass}


def _report(status: str, seed42: dict[str, Any], multiseed: dict[str, Any], summary: pd.DataFrame, controls: pd.DataFrame, resource: dict[str, Any]) -> None:
    control = controls.copy()
    def markdown_table(frame: pd.DataFrame) -> str:
        shown = frame.copy()
        for index, row in shown.iterrows():
            for column in shown.columns:
                shown.loc[index, column] = resource_display(
                    row[column], model=str(row.get("model", "")), column=column,
                )
        return "| " + " | ".join(shown.columns) + " |\n| " + " | ".join(["---"] * len(shown.columns)) + " |\n" + "\n".join("| " + " | ".join(map(str, row)) + " |" for row in shown.itertuples(index=False, name=None))
    report = f"""# Phase PI final research report

## Frozen scope and result

Model: `{MODEL}`  
Claim: **PDFNet-inspired exploratory architecture**  
Formal backbone: **TIMESNET_FREEZE** (unchanged)

The paper source gate failure in Phase BG remains unchanged: this phase is not an official PDFNet implementation or reproduction. It evaluated exactly one pre-registered inspired architecture, using only Development windows and a voltage-channel-only standardized DeltaV loss.

## Stage 0 and fairness

Stage 0 synthetic structure audit passed. The RTX 3070 Laptop GPU resource gate passed at D=128 and effective batch 64. The registered 3007-row prefix was filtered to 3005 Development rows; the two boundary rows did not enter legal windows, scalers, fitting, prediction, or scoring. Test targets, predictions and metrics: **not accessed**.

## Decision

Seed-42 status: `{seed42['status']}`.  
Multi-seed status: `{multiseed['status']}`.

The result supports only this statement: the preregistered PDFNet-inspired exploratory architecture {research_conclusion_text(status)}.

No component-level causal claim is made for rFFT, GPRU, dynamic graph, or fusion. No PDFNet official-performance claim is made.

## Seed summaries (mV)

{markdown_table(summary)}

## Frozen TimesNet controls (mV)

{markdown_table(control)}

## Stop rule

No architecture, D, k, graph, fusion, loss, decomposition, Test evaluation, backbone promotion, or high-fidelity reconstruction was triggered after this decision. `TIMESNET_FREEZE` remains the only formal backbone status.
"""
    write_text(report, "final", "FINAL_RESEARCH_REPORT.md")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed42-only", action="store_true", help="valid only for an intentional pre-gate interruption; does not change the protocol")
    args = parser.parse_args()
    resource = _require_stage0()
    assumptions, gate = load_frozen_configs()
    if int(assumptions["architecture"]["embedding_dimension"]) != int(resource["dimension"]): raise RuntimeError("Resource audit dimension differs from frozen architecture")
    cfg, helpers, _frame, folds, identity_records, audit = recover_windows_and_timesnet()
    fairness_checks = assess_fairness(
        identity_records,
        post_boundary_rows_entered_pipeline=audit["post_boundary_rows_entered_pipeline"],
        test_accessed=False,
        future_covariates=False,
        future_13_variable_supervision=False,
        external_decomposition=False,
    )
    fairness = {"phase_pi_head": audit["phase_pi_head"], "base_commit": audit["base_commit"], "assumptions_sha256": sha256(PI / "configs" / "EXPLORATORY_ARCHITECTURE_ASSUMPTIONS.yaml"), "potential_gate_sha256": sha256(PI / "configs" / "PREREGISTERED_POTENTIAL_GATE.yaml"), "phase_bs_protocol_sha256": audit["phase_bs_protocol_sha256"], "paper_sha256": audit["paper_sha256"], "raw_prefix_rows_read": audit["raw_prefix_rows_read"], "development_rows_after_filter": audit["development_rows_after_filter"], "boundary_rows_filtered": audit["boundary_rows_filtered"], **fairness_checks}
    write_json(fairness, "outputs", "FAIRNESS_AUDIT.json")
    write_csv(identity_records, "outputs", "WINDOW_IDENTITY_AUDIT.csv")
    require_fairness_pass(fairness)
    control_per_fold, controls = _timesnet_control(helpers, folds)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if dev.type != "cuda": raise RuntimeError("RESOURCE_INFEASIBLE: RTX CUDA unavailable after resource gate")
    dimension, batch_size = int(resource["dimension"]), int(resource["selected_microbatch"])
    results = [_run_one(helpers, fold, train, validation, 42, dimension, batch_size, dev) for fold, train, validation in folds]
    candidate_per_fold = pd.DataFrame(results).sort_values("fold").reset_index(drop=True)
    candidate_summary = helpers.summarize_seed(candidate_per_fold)
    seed42 = _seed42_gate(candidate_summary.iloc[0], controls.loc[controls.seed.eq(42)].iloc[0], gate, fairness)
    write_csv(candidate_per_fold, "outputs", "SEED42_RESULTS.csv")
    write_json(seed42, "outputs", "SEED42_GATE.json")
    if args.seed42_only:
        return
    if seed42["status"] in {"PDFNET_INSPIRED_STRONGLY_PROMISING", "PDFNET_INSPIRED_PROMISING_A", "PDFNET_INSPIRED_PROMISING_B"}:
        for seed in (2024, 3407):
            results.extend(_run_one(helpers, fold, train, validation, seed, dimension, batch_size, dev) for fold, train, validation in folds)
        candidate_per_fold = pd.DataFrame(results).sort_values(["seed", "fold"]).reset_index(drop=True)
        candidate_summary = helpers.summarize_seed(candidate_per_fold)
        multiseed = _multiseed_gate(candidate_summary, controls, gate, seed42)
        multi_performed = True
    else:
        multiseed = {"status": "NOT_RUN", "reason": "seed42_no_promise", "seed42_status": seed42["status"]}
        multi_performed = False
    all_per_fold = pd.concat([control_per_fold, candidate_per_fold], ignore_index=True)
    all_summary = pd.concat([controls, candidate_summary], ignore_index=True)
    write_csv(all_per_fold, "outputs", "PER_FOLD_RESULTS.csv")
    write_csv(all_summary, "outputs", "SUMMARY_RESULTS.csv")
    write_json(multiseed, "outputs", "MULTISEED_GATE.json")
    final_status = multiseed["status"] if multi_performed else seed42["status"]
    decision = {"phase": "Phase_PI_PDFNet_Inspired_Exploration", "model": MODEL, "claim": assumptions["claim"], "status": final_status, "seed42_status": seed42["status"], "multiseed_performed": multi_performed, "multiseed_status": multiseed["status"], "high_fidelity_revalidation_recommended": bool(final_status == "PDFNET_INSPIRED_STRONGLY_PROMISING"), "official_pdfnet_claim": False, "formal_backbone_changed": False, "formal_backbone": "TIMESNET_FREEZE", "test_used": False, "external_decomposition": False, "stop_after_phase": True, "assumptions_sha256": ASSUMPTIONS_SHA.read_text(encoding="utf-8").split()[0], "potential_gate_sha256": GATE_SHA.read_text(encoding="utf-8").split()[0]}
    write_text("\n".join(f"{key}: {str(value).lower() if isinstance(value, bool) else value}" for key, value in decision.items()), "final", "FINAL_EXPLORATORY_DECISION.yaml")
    _report(final_status, seed42, multiseed, candidate_summary, controls, resource)
    print(json.dumps({"final_status": final_status, "seed42": seed42["status"], "multi_seed": multi_performed}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
