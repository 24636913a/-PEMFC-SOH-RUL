"""Run only the preregistered non-physics POT ablations M3 and M4.

M1/M2 are revalidated read-only historical controls.  Stage 0 closed the
physical branch, so no M5/M6 implementation is reached from this runner.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from pot_common import (
    BS, DF, PHASE, POTFailure, build_frozen_folds, load_inherited_protocols,
    load_protocol, safe_read_development, write_csv, write_json,
)
from pot_training import HyperParameters, formal_refit_and_predict


RESOLVED = PHASE / "configs" / "RESOLVED_OPTUNA_CONFIGS.yaml"
RESOLVED_SHA = PHASE / "configs" / "RESOLVED_OPTUNA_CONFIGS.sha256"
HISTORICAL = {
    "M1_RawF2_TimesNet": {"model": "RawF2_TimesNet", "dir": BS / "outputs" / "predictions", "stem": "bs_timesnet"},
    "M2_EMD_TimesNet": {"model": "EMDFeature_TimesNet", "dir": DF / "outputs" / "predictions", "stem": "df_emdfeature_timesnet"},
}
NEW = {"M3_Optuna_RawF2_TimesNet": "Optuna_RawF2_TimesNet", "M4_Optuna_EMD_TimesNet": "Optuna_EMDFeature_TimesNet"}


def load_resolved(protocol_sha: str) -> dict[str, Any]:
    if not RESOLVED.exists() or not RESOLVED_SHA.exists():
        raise POTFailure("RESOLVED_OPTUNA_CONFIG_MISSING")
    expected = RESOLVED_SHA.read_text(encoding="utf-8").split()[0].lower()
    if hashlib.sha256(RESOLVED.read_bytes()).hexdigest() != expected:
        raise POTFailure("RESOLVED_OPTUNA_CONFIG_HASH_MISMATCH")
    cfg = yaml.safe_load(RESOLVED.read_text(encoding="utf-8"))
    if cfg.get("status") != "FROZEN_AFTER_FOLD1_INNER_ONLY_OPTUNA_BEFORE_OUTER_EVALUATION" or cfg.get("protocol_sha256") != protocol_sha:
        raise POTFailure("RESOLVED_OPTUNA_CONFIG_PROTOCOL_MISMATCH")
    return cfg


def _archive(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as z:
        keys = ["y_true_delta_V", "y_pred_delta_V", "anchor_V", "origin_row", "target_row", "origin_age_h", "target_end_age_h", "horizon_step"]
        return {key: np.asarray(z[key]) for key in keys}


def _historical_records(folds: list[tuple[str, Any, Any]], common: Any) -> tuple[list[dict], list[dict], list[dict]]:
    metrics_rows: list[dict] = []
    prediction_rows: list[dict] = []
    audit_rows: list[dict] = []
    for pot_model, spec in HISTORICAL.items():
        for fold, _train, validation in folds:
            for seed in (42, 2024, 3407):
                path = spec["dir"] / f"{spec['stem']}_{fold}_seed{seed}.npz"
                if not path.is_file():
                    raise POTFailure(f"HISTORICAL_CONTROL_MISSING: {path.name}")
                payload = _archive(path)
                keys_equal = bool(np.array_equal(payload["origin_row"], validation.origin_row) and np.array_equal(payload["target_row"], validation.target_row) and np.allclose(payload["y_true_delta_V"], validation.y_delta_V, rtol=0, atol=1e-7))
                if not keys_equal:
                    raise POTFailure(f"HISTORICAL_CONTROL_IDENTITY_FAIL: {pot_model}/{fold}/seed{seed}")
                metrics = common.delta_metrics(payload["y_true_delta_V"], payload["y_pred_delta_V"], payload["target_row"])
                metrics_rows.append({"ablation_id": pot_model, "model": spec["model"], "fold": fold, "seed": seed, "source": "historical_frozen_revalidated", "parameter_count": 149046 if pot_model.startswith("M1") else 149304, "peak_gpu_memory_MiB": np.nan, "total_inner_plus_refit_time_s": np.nan, "selected_epoch": np.nan, "input_dimension": 14 if pot_model.startswith("M1") else 16, "physics_loss_used": False, "test_accessed": False, **metrics})
                for q in range(len(validation.origin_row)):
                    for h in range(24):
                        prediction_rows.append({"ablation_id": pot_model, "fold": fold, "seed": seed, "origin_row": int(payload["origin_row"][q]), "target_row": int(payload["target_row"][q, h]), "horizon_step": h + 1, "origin_age_h": float(payload["origin_age_h"][q]), "target_end_age_h": float(payload["target_end_age_h"][q]), "y_true_delta_V": float(payload["y_true_delta_V"][q, h]), "y_pred_delta_V": float(payload["y_pred_delta_V"][q, h]), "absolute_error_mV": float(abs(payload["y_pred_delta_V"][q, h] - payload["y_true_delta_V"][q, h]) * 1000)})
                audit_rows.append({"ablation_id": pot_model, "fold": fold, "seed": seed, "historical_npz": str(path.relative_to(PHASE.parent)).replace("\\", "/"), "origin_and_target_keys_match": True, "truth_match": True, "target_age_max_h": float(np.max(payload["target_end_age_h"])), "test_accessed": False})
    return metrics_rows, prediction_rows, audit_rows


def _prediction_rows(ablation_id: str, fold: str, seed: int, validation: Any, prediction: np.ndarray) -> list[dict]:
    rows: list[dict] = []
    for q in range(len(validation.origin_row)):
        for h in range(24):
            rows.append({"ablation_id": ablation_id, "fold": fold, "seed": seed, "origin_row": int(validation.origin_row[q]), "target_row": int(validation.target_row[q, h]), "horizon_step": h + 1, "origin_age_h": float(validation.origin_age_h[q]), "target_end_age_h": float(validation.target_end_age_h[q]), "y_true_delta_V": float(validation.y_delta_V[q, h]), "y_pred_delta_V": float(prediction[q, h]), "absolute_error_mV": float(abs(prediction[q, h] - validation.y_delta_V[q, h]) * 1000)})
    return rows


def _summary(metrics: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    numeric = ["rmse_mV", "mae_mV", "r2", "endpoint_rmse_mV", "p95_abs_error_mV", "p99_abs_error_mV", "max_abs_error_mV", "target_time_equal_rmse_mV"]
    for ablation_id, group in metrics.groupby("ablation_id", sort=False):
        for seed, by_seed in group.groupby("seed", sort=True):
            record = {"ablation_id": ablation_id, "level": "seed_equal_fold", "seed": int(seed)}
            record.update({key: float(by_seed[key].mean()) for key in numeric})
            record["fold3_rmse_mV"] = float(by_seed.loc[by_seed.fold.eq("fold_3"), "rmse_mV"].iloc[0])
            rows.append(record)
        seeds = pd.DataFrame([r for r in rows if r["ablation_id"] == ablation_id and r["level"] == "seed_equal_fold"])
        for stat in ("mean", "std", "min", "max"):
            record = {"ablation_id": ablation_id, "level": f"seed_{stat}", "seed": np.nan}
            fun = {"mean": "mean", "std": "std", "min": "min", "max": "max"}[stat]
            record.update({key: float(getattr(seeds[key], fun)()) for key in [*numeric, "fold3_rmse_mV"]})
            rows.append(record)
    return pd.DataFrame(rows)


def _gates(summary: pd.DataFrame, cfg: dict[str, Any]) -> dict[str, Any]:
    def row(model: str, level: str) -> pd.Series:
        return summary.loc[(summary.ablation_id == model) & (summary.level == level)].iloc[0]
    m2, m4 = row("M2_EMD_TimesNet", "seed_mean"), row("M4_Optuna_EMD_TimesNet", "seed_mean")
    by_seed2 = summary.loc[(summary.ablation_id == "M2_EMD_TimesNet") & (summary.level == "seed_equal_fold")].sort_values("seed")
    by_seed4 = summary.loc[(summary.ablation_id == "M4_Optuna_EMD_TimesNet") & (summary.level == "seed_equal_fold")].sort_values("seed")
    gate = cfg["optuna_only_advance_gate"]
    checks = {
        "formal_RMSE_99pct_of_M2": bool(m4.rmse_mV <= float(gate["formal_equal_fold_RMSE_at_most_fraction_of_M2"]) * m2.rmse_mV),
        "paired_seed_improvement_at_least_2": bool((by_seed4.rmse_mV.to_numpy() < by_seed2.rmse_mV.to_numpy()).sum() >= int(gate["minimum_paired_seeds_better_than_M2"])),
        "fold3_RMSE_103pct_of_M2": bool(m4.fold3_rmse_mV <= float(gate["fold3_RMSE_at_most_fraction_of_M2"]) * m2.fold3_rmse_mV),
        "P99_105pct_of_M2": bool(m4.p99_abs_error_mV <= float(gate["P99_at_most_fraction_of_M2"]) * m2.p99_abs_error_mV),
        "target_equal_105pct_of_M2": bool(m4.target_time_equal_rmse_mV <= float(gate["target_time_equal_RMSE_at_most_fraction_of_M2"]) * m2.target_time_equal_rmse_mV),
    }
    return {"checks": checks, "all_pass": bool(all(checks.values())), "paired_seed_wins": int((by_seed4.rmse_mV.to_numpy() < by_seed2.rmse_mV.to_numpy()).sum())}


def main() -> None:
    cfg, protocol_sha = load_protocol()
    stage0 = json.loads((PHASE / "outputs" / "STAGE0_PHYSICAL_FEASIBILITY_SUMMARY.json").read_text(encoding="utf-8"))
    if stage0["status"] != "PHYSICS_LOSS_NOT_ESTABLISHED":
        raise POTFailure("PHYSICS_BRANCH_STATUS_UNEXPECTED")
    resolved = load_resolved(protocol_sha)
    bs_cfg, _df_cfg, common = load_inherited_protocols(cfg)
    frame, firewall = safe_read_development(cfg, bs_cfg)
    folds = build_frozen_folds(frame, cfg, bs_cfg, common)
    rows, prediction_rows, audit_rows = _historical_records(folds, common)
    for ablation_id, method in NEW.items():
        selected = resolved["search_results"][method]
        hp = HyperParameters(**selected["hyperparameters"])
        for fold, train, validation in folds:
            for seed in cfg["formal_training"]["seeds"]:
                record, prediction = formal_refit_and_predict(method, fold, train, validation, hp, int(seed), bs_cfg, common, protocol_sha, int(cfg["formal_training"]["max_epochs"]), int(cfg["formal_training"]["patience"]), float(cfg["formal_training"]["gradient_clip_norm"]))
                rows.append({"ablation_id": ablation_id, "source": "POT_formal_refit", **record})
                prediction_rows.extend(_prediction_rows(ablation_id, fold, int(seed), validation, prediction))
    metrics = pd.DataFrame(rows).sort_values(["ablation_id", "seed", "fold"]).reset_index(drop=True)
    predictions = pd.DataFrame(prediction_rows).sort_values(["ablation_id", "seed", "fold", "origin_row", "horizon_step"]).reset_index(drop=True)
    summary = _summary(metrics)
    gates = _gates(summary, cfg)
    write_csv(metrics, "outputs", "FORMAL_ABLATION_PER_FOLD_RESULTS.csv")
    write_csv(summary, "outputs", "FORMAL_ABLATION_SEED_SUMMARY.csv")
    write_csv(predictions, "outputs", "FORMAL_ABLATION_PREDICTIONS.csv")
    write_csv(pd.DataFrame(audit_rows), "outputs", "HISTORICAL_CONTROL_COMPARABILITY_AUDIT.csv")
    write_json({"stage0_status": stage0["status"], "physics_models_M5_M6": "NOT_RUN_BY_FROZEN_STAGE0_GATE", "M1_M2_historical_controls_revalidated": True, "M3_M4_formal_refits_completed": True, "optuna_only_gate": gates, "data_firewall": firewall, "test_accessed": False}, "outputs", "FORMAL_ABLATION_AUDIT.json")
    print(json.dumps({"status": "COMPLETE", "optuna_only_gate": gates, "test_accessed": False}, ensure_ascii=False))


if __name__ == "__main__":
    main()
