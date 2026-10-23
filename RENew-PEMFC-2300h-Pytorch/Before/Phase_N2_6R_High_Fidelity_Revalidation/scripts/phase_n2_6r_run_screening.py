"""Run LSTM reproduction and seed-42 high-fidelity screening."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

from phase_n2_6r_common import (
    PHASE_ROOT,
    candidate_names,
    family_for_config,
    load_config,
    load_development_protocol,
    mean_fold_metrics,
    verify_no_test_arrays,
    write_csv,
    write_json,
)
from phase_n2_6r_models import create_model, train_model_run


def load_n26_lstm_reference(cfg: dict, fold: str, validation) -> tuple[dict, dict[str, np.ndarray]]:
    run_id = f"reproduction_LSTM_{fold}_seed42"
    root = Path(cfg["phase_n2_6_root"]) / "outputs/model_runs"
    result = json.loads((root / f"{run_id}.json").read_text(encoding="utf-8"))
    arrays = dict(np.load(root / f"{run_id}_predictions.npz", allow_pickle=False))
    if result.get("status") != "complete":
        raise RuntimeError(f"Invalid Phase N2.6 LSTM reference: {run_id}")
    if not np.array_equal(arrays["row_origin_index"], validation.row_origin_index):
        raise RuntimeError(f"Phase N2.6 LSTM origins differ in {fold}")
    if not np.array_equal(arrays["y_true"], validation.y_abs):
        raise RuntimeError(f"Phase N2.6 LSTM targets differ in {fold}")
    return result, arrays


def result_row(result: dict, fold: str, config_name: str, family: str, lstm_rmse_mV: float) -> dict:
    return {
        "row_type": "fold",
        "family": family,
        "model": family,
        "config_name": config_name,
        "seed": int(result["seed"]),
        "fold": fold,
        "rmse_V": float(result["rmse_V"]),
        "rmse_mV": float(result["rmse_mV"]),
        "mae_V": float(result["mae_V"]),
        "mae_mV": float(result["mae_mV"]),
        "r2": float(result["r2"]),
        "bias_V": float(result["bias_V"]),
        "bias_mV": float(result["bias_mV"]),
        "rmse_improvement_pct_vs_lstm": 100.0 * (lstm_rmse_mV - float(result["rmse_mV"])) / lstm_rmse_mV,
        "parameter_count": int(result["parameter_count"]),
        "peak_gpu_memory_mb": float(result["peak_gpu_memory_mb"]),
        "mean_epoch_time_s": float(result["mean_epoch_time_s"]),
        "total_training_time_s": float(result["total_training_time_s"]),
        "best_epoch": int(result["best_epoch"]),
        "epochs_ran": int(result["epochs_ran"]),
        "run_id": str(result["run_id"]),
        "implementation_fidelity_status": "PASS" if family != "LSTM" else "REFERENCE",
    }


def scaled_train_x(train_x: np.ndarray, checkpoint: dict) -> np.ndarray:
    medians = np.asarray(checkpoint["feature_medians"], dtype=np.float64)
    mean = np.asarray(checkpoint["feature_scaler_mean"], dtype=np.float64)
    scale = np.asarray(checkpoint["feature_scaler_scale"], dtype=np.float64)
    shape = train_x.shape
    flat = train_x.reshape(-1, shape[-1]).astype(np.float64)
    flat = np.where(np.isfinite(flat), flat, medians)
    return ((flat - mean) / scale).reshape(shape).astype(np.float32)


def discover_timesnet_periods(cfg: dict, config_name: str, fold: str, train_x: np.ndarray) -> list[dict]:
    checkpoint_path = PHASE_ROOT / "checkpoints" / f"screening_{config_name}_{fold}_seed42.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = create_model(config_name, 14, 24, cfg)
    model.load_state_dict(checkpoint["model_state_dict"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()
    values = scaled_train_x(train_x, checkpoint)
    loader = DataLoader(TensorDataset(torch.from_numpy(values)), batch_size=256, shuffle=False)
    amplitude_sum: np.ndarray | None = None
    sample_count = 0
    with torch.no_grad():
        for (batch_x,) in loader:
            x = batch_x.to(device)
            means = x.mean(1, keepdim=True).detach()
            normalized = x.sub(means)
            stdev = torch.sqrt(torch.var(normalized, dim=1, keepdim=True, unbiased=False) + 1e-5)
            normalized = normalized.div(stdev)
            hidden = model.official_model.enc_embedding(normalized, None)
            extended = model.official_model.predict_linear(hidden.permute(0, 2, 1)).permute(0, 2, 1)
            amplitude = torch.abs(torch.fft.rfft(extended, dim=1)).mean(dim=-1).sum(dim=0).cpu().numpy()
            amplitude_sum = amplitude if amplitude_sum is None else amplitude_sum + amplitude
            sample_count += len(x)
    if amplitude_sum is None or sample_count == 0:
        raise RuntimeError(f"No Train windows for TimesNet period discovery: {config_name} {fold}")
    average = amplitude_sum / sample_count
    average[0] = 0.0
    top_indices = np.argsort(average)[::-1][:5]
    rows = []
    for rank, frequency_index in enumerate(top_indices, start=1):
        exact_period = 72.0 / float(frequency_index)
        rows.append(
            {
                "config_name": config_name,
                "fold": fold,
                "block_index": 1,
                "rank": rank,
                "frequency_index": int(frequency_index),
                "frequency_cycles_per_step": float(frequency_index) / 72.0,
                "exact_period_steps": exact_period,
                "official_integer_period_steps": int(72 // int(frequency_index)),
                "period_hours": exact_period * 0.5,
                "fft_amplitude": float(average[frequency_index]),
                "near_19_steps": bool(abs(exact_period - 19.0) <= 1.5),
                "data_scope": "fold_train_only_first_timesblock_input",
            }
        )
    return rows


def main() -> None:
    cfg = load_config()
    fidelity = json.loads(
        (PHASE_ROOT / "outputs/implementation_audit/implementation_fidelity_audit.json").read_text(encoding="utf-8")
    )
    if fidelity.get("status") != "PASS" or not all(fidelity.get("config_status", {}).values()):
        raise RuntimeError("Implementation fidelity gate must pass before training")

    _, firewall, windows = load_development_protocol(cfg)
    write_json(firewall, PHASE_ROOT / "outputs/implementation_audit/test_firewall_audit.json")
    fold_rows: list[dict] = []
    reproduction_rows: list[dict] = []
    matched_rows: list[dict] = []
    lstm_by_fold: dict[str, tuple[dict, dict[str, np.ndarray]]] = {}

    for fold, train, validation in windows:
        reference, reference_arrays = load_n26_lstm_reference(cfg, fold, validation)
        run_id = f"reproduction_LSTM_{fold}_seed42"
        result, arrays = train_model_run(
            run_id,
            "LSTM",
            train,
            validation,
            cfg,
            42,
            {
                "stage": "lstm_reproduction",
                "fold": fold,
                "implementation_source": "Phase N2 frozen LSTM",
                "implementation_commit": "not_applicable_local_reference",
                "lookback_hours": 24,
                "horizon_hours": 12,
                "resolution_hours": 0.5,
            },
        )
        verify_no_test_arrays(arrays, cfg, run_id)
        difference = abs(float(result["rmse_mV"]) - float(reference["rmse_mV"])) / float(reference["rmse_mV"])
        reproduction_rows.append(
            {
                "fold": fold,
                "phase_n2_6_rmse_mV": float(reference["rmse_mV"]),
                "reproduced_rmse_mV": float(result["rmse_mV"]),
                "relative_difference_fraction": difference,
                "within_2pct": difference < float(cfg["selection"]["reproduction_max_difference"]),
                "parameter_count": int(result["parameter_count"]),
            }
        )
        matched_rows.append(
            {
                "fold": fold,
                "train_origin_count": len(train.x),
                "validation_origin_count": len(validation.x),
                "phase_n2_6_origins_exact_match": np.array_equal(arrays["row_origin_index"], reference_arrays["row_origin_index"]),
                "phase_n2_6_targets_exact_match": np.array_equal(arrays["y_true"], reference_arrays["y_true"]),
                "maximum_target_end_h": float(np.max(arrays["target_end_age_h"])),
                "enters_test": bool(np.any(arrays["target_end_age_h"] >= float(cfg["split"]["development_end_h"]))),
            }
        )
        lstm_by_fold[fold] = (result, arrays)

    reproduction_frame = pd.DataFrame(reproduction_rows)
    mean_ref = float(reproduction_frame["phase_n2_6_rmse_mV"].mean())
    mean_new = float(reproduction_frame["reproduced_rmse_mV"].mean())
    reproduction_frame.loc[len(reproduction_frame)] = {
        "fold": "mean_3fold",
        "phase_n2_6_rmse_mV": mean_ref,
        "reproduced_rmse_mV": mean_new,
        "relative_difference_fraction": abs(mean_new - mean_ref) / mean_ref,
        "within_2pct": abs(mean_new - mean_ref) / mean_ref < float(cfg["selection"]["reproduction_max_difference"]),
        "parameter_count": int(reproduction_rows[0]["parameter_count"]),
    }
    write_csv(reproduction_frame, PHASE_ROOT / "outputs/implementation_audit/lstm_reproduction_check.csv")
    write_csv(pd.DataFrame(matched_rows), PHASE_ROOT / "outputs/implementation_audit/matched_origin_audit.csv")
    if not bool(reproduction_frame["within_2pct"].all()):
        raise RuntimeError("LSTM reproduction failed the 2% gate")

    for fold, _, _ in windows:
        result, _ = lstm_by_fold[fold]
        fold_rows.append(result_row(result, fold, "LSTM", "LSTM", float(result["rmse_mV"])))

    run_index: dict[str, dict[str, str]] = {}
    for config_name in candidate_names(cfg):
        family = family_for_config(cfg, config_name)
        for fold, train, validation in windows:
            lstm_result, lstm_arrays = lstm_by_fold[fold]
            run_id = f"screening_{config_name}_{fold}_seed42"
            result, arrays = train_model_run(
                run_id,
                config_name,
                train,
                validation,
                cfg,
                42,
                {
                    "stage": "high_fidelity_screening",
                    "fold": fold,
                    "implementation_source": cfg["official_sources"]["timesnet" if family == "TimesNet" else "moderntcn"]["repository"],
                    "implementation_commit": cfg["official_sources"]["timesnet" if family == "TimesNet" else "moderntcn"]["commit"],
                    "implementation_fidelity_status": "PASS",
                    "lookback_hours": 24,
                    "horizon_hours": 12,
                    "resolution_hours": 0.5,
                },
            )
            verify_no_test_arrays(arrays, cfg, run_id)
            if not np.array_equal(arrays["row_origin_index"], lstm_arrays["row_origin_index"]):
                raise RuntimeError(f"Matched-Origin violation for {run_id}")
            if not np.array_equal(arrays["y_true"], lstm_arrays["y_true"]):
                raise RuntimeError(f"Matched target violation for {run_id}")
            fold_rows.append(result_row(result, fold, config_name, family, float(lstm_result["rmse_mV"])))
            run_index[f"{config_name}|42|{fold}"] = {"run_id": run_id, "source": "Phase_N2_6R"}

    fold_frame = pd.DataFrame(fold_rows)
    summary = mean_fold_metrics(fold_frame)
    lstm_mean = float(summary.loc[summary["config_name"] == "LSTM", "rmse_mV"].iloc[0])
    summary["rmse_improvement_pct_vs_lstm"] = 100.0 * (lstm_mean - summary["rmse_mV"]) / lstm_mean
    summary["implementation_fidelity_status"] = np.where(summary["family"].eq("LSTM"), "REFERENCE", "PASS")
    screening = pd.concat([fold_frame, summary], ignore_index=True, sort=False)
    write_csv(screening, PHASE_ROOT / "outputs/screening/high_fidelity_screening.csv")

    config_summary = summary.set_index("config_name")
    family_best: dict[str, str] = {}
    eligible: list[str] = []
    eligibility_details: dict[str, dict] = {}
    for family in ["TimesNet", "ModernTCN"]:
        candidates = summary.loc[summary["family"] == family].sort_values(["rmse_mV", "mae_mV", "r2"], ascending=[True, True, False])
        best_name = str(candidates.iloc[0]["config_name"])
        family_best[family] = best_name
        candidate_folds = fold_frame.loc[fold_frame["config_name"] == best_name].set_index("fold")
        lstm_folds = fold_frame.loc[fold_frame["config_name"] == "LSTM"].set_index("fold")
        improved_folds = int((candidate_folds.loc[["fold_1", "fold_2", "fold_3"], "rmse_mV"] < lstm_folds.loc[["fold_1", "fold_2", "fold_3"], "rmse_mV"]).sum())
        mean_rmse = float(config_summary.loc[best_name, "rmse_mV"])
        condition_a = mean_rmse <= 1.10 * lstm_mean
        condition_b = improved_folds >= int(cfg["selection"]["multiseed_min_improved_folds_alternative"])
        if condition_a or condition_b:
            eligible.append(best_name)
        eligibility_details[family] = {
            "best_config": best_name,
            "mean_rmse_mV": mean_rmse,
            "lstm_mean_rmse_mV": lstm_mean,
            "relative_ratio": mean_rmse / lstm_mean,
            "improved_fold_count": improved_folds,
            "condition_a_within_10pct": condition_a,
            "condition_b_at_least_two_folds": condition_b,
            "eligible_for_multiseed": condition_a or condition_b,
        }

    period_rows: list[dict] = []
    for config_name in [name for name in candidate_names(cfg) if family_for_config(cfg, name) == "TimesNet"]:
        for fold, train, _ in windows:
            period_rows.extend(discover_timesnet_periods(cfg, config_name, fold, train.x))
    write_csv(pd.DataFrame(period_rows), PHASE_ROOT / "outputs/screening/timesnet_period_discovery.csv")
    write_json(run_index, PHASE_ROOT / "outputs/frozen_config/screening_run_index.json")
    write_json(
        {
            "status": "PASS",
            "lstm_reproduction_pass": True,
            "family_best_configs": family_best,
            "eligible_for_multiseed": eligible,
            "eligibility_details": eligibility_details,
            "test_target_rows_read": 0,
            "test_predictions_constructed": False,
            "test_rmse_computed": False,
            "test_r2_computed": False,
            "test_mae_computed": False,
        },
        PHASE_ROOT / "outputs/frozen_config/screening_selection.json",
    )
    print(json.dumps({"status": "PASS", "family_best": family_best, "eligible": eligible}, ensure_ascii=False))


if __name__ == "__main__":
    main()
