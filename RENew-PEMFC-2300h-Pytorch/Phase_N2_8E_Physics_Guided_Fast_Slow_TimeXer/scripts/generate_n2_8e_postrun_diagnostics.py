"""Generate post-run N2.8E causal diagnostics without retraining or Test access."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(r"E:\沈农信电\ PEMFC-SOH+RUL\RENew-PEMFC-2300h-Pytorch\Phase_N2_8E_Physics_Guided_Fast_Slow_TimeXer")
SOURCE = ROOT / "scripts" / "phase_n2_8e_run.py"
MODELS = ("FS_TimeXer_NoPhysics", "PGFS_TimeXer_Smooth", "PGFS_TimeXer_SmoothCurrent", "PGFS_TimeXer_Full")
FOLDS = ("fold_1", "fold_2", "fold_3")


def load_phase_module():
    spec = importlib.util.spec_from_file_location("n2_8e_postrun_source", SOURCE)
    if spec is None or spec.loader is None:
        raise RuntimeError("Cannot import N2.8E source helpers")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    source = load_phase_module()
    cfg = source.load_config()
    common, frame, windows, firewall = source.load_development(cfg)
    firewall.update({
        "test_target_rows_read": 0,
        "test_predictions_constructed": False,
        "test_metrics_computed": False,
        "test_feature_diagnostics_computed": False,
        "test_performance_accessed": False,
        "postrun_diagnostics_scope": "development_validation_only",
    })
    (ROOT / "outputs" / "test_firewall_audit.json").write_text(json.dumps(firewall, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    feature_idx = [cfg["features"].index(x) for x in cfg["future_exogenous_features"]]
    density_idx = cfg["future_exogenous_features"].index(cfg["physics_loss"]["current_density_feature"])
    device = source.device()
    per_horizon, global_values = [], {model: [] for model in MODELS}
    for fold in FOLDS:
        train, valid = windows[fold]
        xtr, xva, _, _, _, _, scaler = source.prepare_fold(common, train, valid, feature_idx)
        htr, hva = train.x[:, :, feature_idx], valid.x[:, :, feature_idx]
        ztr = source.future_exogenous(frame, train, cfg)
        ridge = source.RidgeMultiHorizon(cfg["future_condition"]["ridge_alpha"]).fit(htr, ztr, scaler)
        raw_zva = ridge.predict(hva, scaler)
        zva = scaler.transform(raw_zva)
        active = (raw_zva[:, :, density_idx] > float(cfg["physics_loss"]["active_current_threshold_A_cm2"])) & (raw_zva[:, :, -1] <= float(cfg["physics_loss"]["maximum_zero_current_fraction"]))
        for label in MODELS:
            source.set_seed(42)
            model = source.StepAlignedTimeXer(cfg, fast_slow=True).to(device)
            state = torch.load(ROOT / "checkpoints" / f"{label}_{fold}_seed42.pt", map_location=device, weights_only=True)
            model.load_state_dict(state)
            model.eval()
            x = torch.from_numpy(xva).to(device)
            z = torch.from_numpy(zva).to(device).detach().requires_grad_(True)
            _, fast, _ = model(x, z, return_components=True)
            for h in range(24):
                gradients = torch.autograd.grad(fast[:, h].sum(), z, retain_graph=h < 23)[0][:, h, density_idx].detach().cpu().numpy()
                values = gradients[active[:, h]]
                if len(values) == 0:
                    row = {"model": label, "fold": fold, "horizon_step": h + 1, "horizon_hours": (h + 1) * 0.5, "active_samples": 0, "negative_fraction": np.nan, "positive_fraction": np.nan, "zero_fraction": np.nan, "mean_gradient": np.nan, "median_gradient": np.nan, "p05_gradient": np.nan, "p95_gradient": np.nan}
                else:
                    row = {"model": label, "fold": fold, "horizon_step": h + 1, "horizon_hours": (h + 1) * 0.5, "active_samples": int(len(values)), "negative_fraction": float(np.mean(values < 0.0)), "positive_fraction": float(np.mean(values > 0.0)), "zero_fraction": float(np.mean(values == 0.0)), "mean_gradient": float(np.mean(values)), "median_gradient": float(np.median(values)), "p05_gradient": float(np.quantile(values, .05)), "p95_gradient": float(np.quantile(values, .95))}
                    global_values[label].append(values)
                per_horizon.append(row)
    current = pd.DataFrame(per_horizon)
    current.to_csv(ROOT / "outputs" / "diagnostics" / "current_sensitivity_audit.csv", index=False, encoding="utf-8-sig", lineterminator="\n")
    summaries = []
    for label, chunks in global_values.items():
        values = np.concatenate(chunks)
        summaries.append({"model": label, "active_gradient_count": int(len(values)), "negative_fraction": float(np.mean(values < 0.0)), "positive_fraction": float(np.mean(values > 0.0)), "zero_fraction": float(np.mean(values == 0.0)), "mean_gradient": float(np.mean(values)), "median_gradient": float(np.median(values)), "p05_gradient": float(np.quantile(values, .05)), "p95_gradient": float(np.quantile(values, .95))})
    summary = pd.DataFrame(summaries)
    summary.to_csv(ROOT / "outputs" / "diagnostics" / "current_sensitivity_summary.csv", index=False, encoding="utf-8-sig", lineterminator="\n")
    variance_path = ROOT / "outputs" / "diagnostics" / "fast_slow_variance_audit.csv"
    variance = pd.read_csv(variance_path)
    variance["slow_variance_fraction"] = variance["slow_variance_mV2"] / (variance["fast_variance_mV2"] + variance["slow_variance_mV2"])
    variance["decomposition_status"] = np.where(variance["slow_variance_fraction"] > 0.80, "SLOW_DOMINATED_COLLAPSE", np.where(variance["slow_variance_fraction"] < 0.001, "FAST_ONLY_NEAR_ZERO_SLOW", "NON_COLLAPSED"))
    variance.to_csv(variance_path, index=False, encoding="utf-8-sig", lineterminator="\n")
    model_rows, fold_rows = [], []
    fields = {name: [] for name in ("true_delta", "pred_fast_delta", "pred_slow_delta", "pred_total_delta", "origin_age_h", "target_end_age_h", "anchor")}
    for label in MODELS:
        for fold in FOLDS:
            array = np.load(ROOT / "outputs" / "model_runs" / f"{label}_{fold}_seed42_predictions.npz", allow_pickle=False)
            count = len(array["origin_age_h"])
            model_rows.append(np.repeat(label, count)); fold_rows.append(np.repeat(fold, count))
            fields["true_delta"].append(array["y_true"] - array["anchor"][:, None])
            fields["pred_fast_delta"].append(array["fast_prediction"] - array["anchor"][:, None])
            fields["pred_slow_delta"].append(array["slow_component"])
            fields["pred_total_delta"].append(array["y_pred"] - array["anchor"][:, None])
            fields["origin_age_h"].append(array["origin_age_h"])
            fields["target_end_age_h"].append(array["target_end_age_h"])
            fields["anchor"].append(array["anchor"])
    payload = {name: np.concatenate(parts, axis=0) for name, parts in fields.items()}
    payload.update({"model": np.concatenate(model_rows), "fold": np.concatenate(fold_rows)})
    for name in ("fast_slow_component_predictions.npz", "fast_slow_components.npz"):
        np.savez_compressed(ROOT / "outputs" / "model_runs" / name, **payload)
        safe = np.load(ROOT / "outputs" / "model_runs" / name, allow_pickle=False)
        if safe["pred_total_delta"].shape[1] != 24:
            raise RuntimeError("Safe fast-slow component NPZ shape mismatch")
    print(json.dumps({"firewall": "PASS", "current_models": len(summary), "component_rows": int(len(payload["model"])), "current_summary": summary.to_dict(orient="records")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
