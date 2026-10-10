"""Phase E: fixed Top10/Raw14 and causal SG/EWMA comparison.

Only the legally allowed 3007-row prefix of the mixed N1 CSV is opened.  All
six conditions share one scaler fitted to unfiltered legal Train histories; the
only condition differences are a standardized-space Top10 mask and the frozen
causal voltage filter.  Assessment is an already-used Development segment.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import sys
import time
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader, TensorDataset

PHASE = Path(__file__).resolve().parents[1]
PROJECT = PHASE.parent
C_PHASE = PROJECT / "Phase_TFT_SingleSplit_Feature_Filter_Ablation"
D_PHASE = PROJECT / "Phase_TFT_Top12_CausalSG_Ablation"
T1R = PROJECT / "Phase_TFT_Hyperparameter_Optimization_Restart" / "outputs" / "T1R_SELECTED_CONFIGURATIONS.json"
PROTOCOL = PHASE / "configs" / "PREREGISTERED_E_COMPARISON.yaml"
PROTOCOL_SHA = PHASE / "configs" / "PREREGISTERED_E_COMPARISON.sha256"
OUT, FIG, REPORT = PHASE / "outputs", PHASE / "figures", PHASE / "reports"
sys.path.insert(0, str(C_PHASE / "scripts"))
import run_c_ablation as c


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def sha256_small(path: Path) -> str:
    """Hash only small phase-local protocol files, never the mixed causal table."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_protocol() -> tuple[dict[str, Any], str]:
    observed = sha256_small(PROTOCOL)
    expected = PROTOCOL_SHA.read_text(encoding="utf-8").split()[0]
    if observed != expected:
        raise RuntimeError("E_PROTOCOL_SHA_MISMATCH")
    cfg = yaml.safe_load(PROTOCOL.read_text(encoding="utf-8"))
    if cfg.get("status") != "FROZEN_BEFORE_SEED42_TRAINING":
        raise RuntimeError("E_PROTOCOL_NOT_FROZEN")
    return cfg, observed


def set_seed(seed: int = 42) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class CommonScaler:
    def fit(self, x: np.ndarray) -> "CommonScaler":
        flat = x.astype(np.float64).reshape(-1, x.shape[-1])
        self.mean = flat.mean(axis=0)
        self.scale = flat.std(axis=0)
        self.scale = np.where(self.scale > 1e-8, self.scale, 1.0)
        self.samples = int(flat.shape[0])
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        return ((x - self.mean) / self.scale).astype(np.float32)

    @property
    def target_scale(self) -> float:
        return float(self.scale[0])


def one_sided_sg(x: np.ndarray) -> np.ndarray:
    return c.one_sided_sg(x)


def ewma(x: np.ndarray) -> np.ndarray:
    return c.ewma(x)


def apply_filter(x: np.ndarray, name: str) -> np.ndarray:
    if name == "Raw": return x.copy()
    if name == "SG": return one_sided_sg(x)
    if name == "EWMA": return ewma(x)
    raise RuntimeError("E_UNKNOWN_FILTER")


def filter_causality_audit() -> dict[str, Any]:
    base = np.linspace(-1.0, 1.0, 48, dtype=np.float32).reshape(1, 48, 1)
    base = np.pad(base, ((0, 0), (0, 0), (0, 13)))
    mutation_start = 23
    changed = base.copy(); changed[:, mutation_start:, 0] += 8.0
    answer: dict[str, Any] = {"future_mutation_start_index": mutation_start, "test_accessed": False}
    for name, fun in (("SG", one_sided_sg), ("EWMA", ewma)):
        left, right = fun(base), fun(changed)
        answer[name] = {
            "pass": bool(np.allclose(left[:, :mutation_start, 0], right[:, :mutation_start, 0], atol=1e-7, rtol=0)
                         and np.isfinite(left).all() and left.shape == base.shape),
            "shape": list(left.shape),
        }
    answer["pass"] = bool(answer["SG"]["pass"] and answer["EWMA"]["pass"])
    return answer


def top10_scores(train: c.Windows, features: list[str]) -> tuple[pd.DataFrame, list[int]]:
    scores = c.spearman_scores(train, features).copy()
    chosen = scores.loc[scores.selected_top10, "raw_feature_index"].astype(int).tolist()
    expected = [0, 3, 13, 11, 4, 9, 6, 7, 10, 8]
    if chosen != expected:
        raise RuntimeError(f"E_TOP10_HISTORICAL_RANKING_MISMATCH:{chosen}")
    scores["rank"] = np.arange(1, len(scores) + 1)
    return scores, chosen


def prep(x: np.ndarray, scaler: CommonScaler, mask: list[int] | None, filt: str) -> np.ndarray:
    z = scaler.transform(apply_filter(x, filt))
    if mask is not None:
        z[:, :, [i for i in range(z.shape[-1]) if i not in mask]] = 0.0
    return z


def state_digest(model: torch.nn.Module) -> str:
    h = hashlib.sha256()
    for key, tensor in sorted(model.state_dict().items()):
        h.update(key.encode("utf-8")); h.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def make_model(params: dict[str, Any], device: torch.device) -> torch.nn.Module:
    mod = c.load_model_module()
    return c.make_model(mod, params, device)


def predict(model: torch.nn.Module, x: np.ndarray, batch_size: int, device: torch.device) -> np.ndarray:
    return c.predict(model, x, batch_size, device)


def fit_condition(label: str, params: dict[str, Any], train: c.Windows, validation: c.Windows,
                  scaler: CommonScaler, mask: list[int] | None, filt: str, device: torch.device) -> tuple[torch.nn.Module, dict[str, Any]]:
    set_seed(42)
    tx, vx = prep(train.x, scaler, mask, filt), prep(validation.x, scaler, mask, filt)
    ty, vy = train.y / scaler.target_scale, validation.y / scaler.target_scale
    model = make_model(params, device)
    init_sha = state_digest(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(params["learning_rate"]), weight_decay=float(params["weight_decay"]))
    loader = DataLoader(TensorDataset(torch.from_numpy(tx), torch.from_numpy(ty.astype(np.float32))),
                        batch_size=int(params["batch_size"]), shuffle=True,
                        generator=torch.Generator().manual_seed(42), num_workers=0)
    best, best_epoch, stale, best_state = math.inf, 0, 0, None
    history: list[dict[str, float]] = []; started = time.perf_counter()
    for epoch in range(1, int(params["max_epochs"]) + 1):
        model.train(); losses = []
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = torch.nn.functional.mse_loss(model.predict_delta(xb), yb)
            if not torch.isfinite(loss): raise RuntimeError("E_NONFINITE_LOSS")
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), float(params["gradient_clip_norm"])); optimizer.step()
            losses.append(float(loss.detach().cpu()))
        valid_rmse = float(np.sqrt(np.mean((predict(model, vx, int(params["batch_size"]), device) - vy) ** 2)))
        history.append({"epoch": epoch, "train_standardized_mse": float(np.mean(losses)), "validation_standardized_rmse": valid_rmse})
        if valid_rmse < best - 1e-12:
            best, best_epoch, stale = valid_rmse, epoch, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
        if stale >= int(params["patience"]): break
    if best_state is None: raise RuntimeError("E_NO_VALID_CHECKPOINT")
    model.load_state_dict(best_state)
    checkpoint = OUT / "checkpoints" / f"{label}_{'Top10' if mask is not None else 'Raw14'}_{filt}_seed42.pt"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "best_epoch": best_epoch, "protocol_sha256": ACTIVE_PROTOCOL_SHA,
                "common_target_scale_V": scaler.target_scale, "initialization_sha256": init_sha}, checkpoint)
    return model, {"best_epoch": best_epoch, "validation_standardized_RMSE": best,
                   "validation_RMSE_mV": best * scaler.target_scale * 1000,
                   "training_time_s": time.perf_counter() - started,
                   "parameter_count": int(sum(p.numel() for p in model.parameters() if p.requires_grad)),
                   "initialization_sha256": init_sha, "history": history, "checkpoint": str(checkpoint.relative_to(PHASE))}


def prediction_frame(label: str, windows: c.Windows, pred: np.ndarray, protocol_sha: str) -> pd.DataFrame:
    data = []
    for q, origin in enumerate(windows.origins):
        for h in range(24):
            data.append({"experiment_group": label, "seed": 42, "origin_row": int(origin), "target_row": int(windows.targets[q, h]),
                         "horizon_step": h + 1, "origin_age_h": float(windows.ages[q]), "y_true_delta_V": float(windows.y[q, h]),
                         "y_pred_delta_V": float(pred[q, h]), "protocol_sha256": protocol_sha})
    return pd.DataFrame(data)


def numpy_metrics(frame: pd.DataFrame) -> dict[str, float]:
    y = frame["y_true_delta_V"].to_numpy(np.float64); p = frame["y_pred_delta_V"].to_numpy(np.float64)
    e = p - y; denom = float(np.sum((y - y.mean()) ** 2))
    return {"RMSE_mV": float(np.sqrt(np.mean(e ** 2)) * 1000), "MAE_mV": float(np.mean(np.abs(e)) * 1000),
            "R2": float(1 - np.sum(e ** 2) / denom) if denom > 0 else float("nan")}


def historical_reproducibility_audit(scores: pd.DataFrame) -> None:
    c_metrics = pd.read_csv(C_PHASE / "outputs" / "C_SEED42_METRICS.csv")
    d_metrics = pd.read_csv(D_PHASE / "outputs" / "D_SEED42_METRICS.csv")
    c1 = c_metrics.loc[c_metrics.experiment_group.eq("C1")].iloc[0]
    d0 = d_metrics.loc[d_metrics.experiment_group.eq("D0")].iloc[0]
    dfeat = pd.read_csv(D_PHASE / "outputs" / "D_FEATURE_SELECTION.csv")
    same_rank = scores.loc[scores.selected_top10, "feature"].tolist() == dfeat.loc[dfeat.selected_top10, "feature"].tolist()
    text = f"""# Phase E 历史复现性只读审计\n\nC1（Top10、Raw）的记录 RMSE 为 {float(c1.RMSE_mV):.6f} mV，D0（Top10、Raw、公共未滤波 scaler）的记录 RMSE 为 {float(d0.RMSE_mV):.6f} mV，差异为 {float(d0.RMSE_mV-c1.RMSE_mV):+.6f} mV。\n\n- Top10 排名与 D 阶段一致：`{same_rank}`。\n- 两阶段均记录了相同的 3005-row Development 前缀、7:1:2 行边界、48→24 任务、Seed42 和 CanonicalStrictHistoryTFT。\n- C 采用条件各自拟合 scaler；D 采用公共未滤波 Train scaler，故 scaler 口径本身已不同。\n- 现存元数据不足以从历史 checkpoint 精确归因全部数值差异；状态为 `C1_D0_STRICT_REPRODUCIBILITY_UNRESOLVED`。\n\nPhase E 不用历史 C/D 数值选优，六组均从独立初始化重新训练，并用单一公共 scaler 比较。\n"""
    (OUT / "E_HISTORICAL_REPRODUCIBILITY_AUDIT.md").write_text(text, encoding="utf-8")


def pretrain_audit(write: bool = True) -> tuple[dict[str, Any], str, pd.DataFrame, dict[str, c.Windows], pd.DataFrame, list[int], CommonScaler, dict[str, Any]]:
    cfg, sha = load_protocol()
    # c.load_development implements only pd.read_csv(N1, nrows=3007) then the frozen age filter.
    dev = c.load_development(cfg)
    regions = {"train": (0, 2103), "validation": (2103, 2404), "assessment": (2404, 3005)}
    windows = {key: c.build_windows(dev, cfg, *bounds) for key, bounds in regions.items()}
    c.assert_isolation(windows["train"], windows["validation"], windows["assessment"])
    counts = {key: len(value.origins) for key, value in windows.items()}
    if counts != cfg["split"]["expected_legal_windows"]: raise RuntimeError("E_WINDOW_COUNT_MISMATCH")
    scores, top10 = top10_scores(windows["train"], cfg["task"]["raw_f2_feature_order"])
    scaler = CommonScaler().fit(windows["train"].x)
    d_scaler = json.loads((D_PHASE / "outputs" / "D_COMMON_SCALER.json").read_text(encoding="utf-8"))
    same_d = bool(np.allclose(scaler.mean, d_scaler["mean"], atol=1e-12, rtol=0) and np.allclose(scaler.scale, d_scaler["scale"], atol=1e-12, rtol=0))
    filter_audit = filter_causality_audit()
    # T1-R persists tunable values only; execution controls remain frozen here.
    params = {**cfg["hyperparameters"], **json.loads(T1R.read_text(encoding="utf-8"))["folds"]["fold_2"]["best_params"]}
    set_seed(42); sample_model = make_model(params, torch.device("cpu")); parameter_count = int(sum(x.numel() for x in sample_model.parameters() if x.requires_grad))
    audit = {"pass": bool(filter_audit["pass"] and same_d), "protocol_sha256": sha, "development_rows": len(dev), "read_nrows": 3007,
             "full_causal_file_hash_computed": False, "test_accessed": False, "legal_windows": counts, "target_isolation": True,
             "top10_train_only": True, "raw14_frozen": True, "filter_causality": filter_audit,
             "common_scaler_pass": same_d, "common_target_scale_pass": bool(abs(scaler.target_scale - float(d_scaler["common_target_scale_V"])) < 1e-12),
             "common_scaler_target_scale_V": scaler.target_scale, "parameter_count": parameter_count,
             "device": "cuda" if torch.cuda.is_available() else "cpu", "torch_version": torch.__version__,
             "cudnn_deterministic": True, "cudnn_benchmark": False, "all_six_configs_frozen": True}
    if not audit["pass"]: raise RuntimeError("E_PRETRAIN_AUDIT_FAILURE")
    if write:
        OUT.mkdir(exist_ok=True); FIG.mkdir(exist_ok=True); REPORT.mkdir(exist_ok=True)
        scores.to_csv(OUT / "E_FEATURE_SELECTION.csv", index=False)
        manifest = {"protocol_sha256": sha, "data_access": {"causal_features_read_nrows": 3007, "development_rows": 3005,
                    "post_read_filter": "bin_end_ophrs2_h < 1819.997738", "full_causal_file_hash_computed": False, "test_accessed": False},
                    "segments": {name: {"rows": list(pair), "n_rows": pair[1]-pair[0], "start_h": float(dev.bin_start_ophrs2_h.iloc[pair[0]]),
                    "end_h": float(dev.bin_end_ophrs2_h.iloc[pair[1]-1]), "legal_windows": counts[name]} for name, pair in regions.items()},
                    "target_sets_disjoint": True, "assessment_status": "DEVELOPMENT_EXPLORATORY_REUSED_ASSESSMENT"}
        write_json(OUT / "E_SPLIT_MANIFEST.json", manifest)
        write_json(OUT / "E_FEATURE_CONFIGURATION.json", {"raw_f2_feature_order": cfg["task"]["raw_f2_feature_order"],
                   "Top10_indices": top10, "Top10_features": scores.loc[scores.selected_top10, "feature"].tolist(),
                   "Top10_mask": "standardized_space_zero_mask", "Raw14_mask": "none", "test_accessed": False})
        write_json(OUT / "E_FILTER_CAUSALITY_AUDIT.json", filter_audit)
        write_json(OUT / "E_COMMON_SCALER.json", {"mean": scaler.mean.tolist(), "scale": scaler.scale.tolist(), "flat_samples": scaler.samples,
                   "target_scale_V": scaler.target_scale, "fit_split": "unfiltered_train_legal_windows_only", "all_conditions_share_scaler": True,
                   "matches_phase_D_common_scaler": same_d, "test_accessed": False})
        write_json(OUT / "E_PRETRAIN_AUDIT.json", audit)
        historical_reproducibility_audit(scores)
    return cfg, sha, dev, windows, scores, top10, scaler, audit


def independent_audit(sha: str) -> dict[str, Any]:
    groups = [f"E{i}" for i in range(6)]; frames = {}; rows = []
    key_ref = None; truth_ref = None
    for group in groups:
        path = OUT / f"{group}_predictions_seed42.csv"
        if not path.exists(): raise RuntimeError("E_MISSING_PREDICTION_CACHE")
        frame = pd.read_csv(path)
        need = ["experiment_group", "seed", "origin_row", "target_row", "horizon_step", "origin_age_h", "y_true_delta_V", "y_pred_delta_V", "protocol_sha256"]
        if len(frame) != 9072 or list(frame.columns[:len(need)]) != need or frame.duplicated(["origin_row", "horizon_step"]).any(): raise RuntimeError("E_CACHE_KEY_CONTRACT_FAILURE")
        if not frame.target_row.between(2404, 3004).all() or not np.isfinite(frame[["y_true_delta_V", "y_pred_delta_V"]].to_numpy()).all(): raise RuntimeError("E_CACHE_VALUE_CONTRACT_FAILURE")
        if not (frame.protocol_sha256 == sha).all() or not (frame.experiment_group == group).all() or not (frame.seed == 42).all(): raise RuntimeError("E_CACHE_IDENTITY_FAILURE")
        ordered = frame.sort_values(["origin_row", "horizon_step"]).reset_index(drop=True)
        keys = ordered[["origin_row", "target_row", "horizon_step"]].to_numpy(); truth = ordered.y_true_delta_V.to_numpy()
        if key_ref is None: key_ref, truth_ref = keys, truth
        elif not np.array_equal(keys, key_ref) or not np.array_equal(truth, truth_ref): raise RuntimeError("E_CROSS_GROUP_TRUTH_ALIGNMENT_FAILURE")
        rows.append({"experiment_group": group, **numpy_metrics(ordered), "records": len(ordered)})
        frames[group] = ordered
    reported = pd.read_csv(OUT / "E_SEED42_METRICS.csv").set_index("experiment_group")
    for row in rows:
        for name in ("RMSE_mV", "MAE_mV", "R2"):
            if not math.isclose(float(row[name]), float(reported.loc[row["experiment_group"], name]), rel_tol=0, abs_tol=1e-10): raise RuntimeError("E_INDEPENDENT_METRIC_MISMATCH")
    result = {"pass": True, "implementation": "independent_numpy_flattened_metrics", "test_accessed": False,
              "expected_records_per_group": 9072, "groups": rows, "keys_and_truth_identical_across_groups": True,
              "target_row_range": [2404, 3004]}
    write_json(OUT / "E_INDEPENDENT_METRIC_AUDIT.json", result)
    return result


def make_figures(predictions: dict[str, pd.DataFrame], metrics: pd.DataFrame) -> None:
    plt.rcParams.update({"font.size": 10, "axes.linewidth": 1.2, "axes.spines.top": False, "axes.spines.right": False, "savefig.dpi": 300})
    groups = [f"E{i}" for i in range(6)]; colors = {"E0": "#2166ac", "E1": "#1b7837", "E2": "#b2182b", "E3": "#542788", "E4": "#f46d43", "E5": "#4d4d4d"}
    labels = {"E0": "Top10 Raw", "E1": "Top10 SG", "E2": "Top10 EWMA", "E3": "Raw14 Raw", "E4": "Raw14 SG", "E5": "Raw14 EWMA"}
    value = metrics.set_index("experiment_group").loc[groups, "RMSE_mV"].to_numpy()
    fig, ax = plt.subplots(figsize=(10, 4.8)); bars = ax.bar(range(6), value, color=[colors[g] for g in groups], edgecolor="black", linewidth=.7)
    ax.set_xticks(range(6), [labels[g] for g in groups], rotation=18, ha="right"); ax.set_ylabel("Assessment RMSE (mV)"); ax.set_title("Phase E: six-group RMSE comparison")
    for b, v in zip(bars, value): ax.text(b.get_x()+b.get_width()/2, v, f"{v:.2f}", ha="center", va="bottom", fontsize=8)
    fig.tight_layout(); fig.savefig(FIG / "Fig1_Six_Group_RMSE_Comparison.png"); plt.close(fig)
    matrix = np.array([[value[0], value[1], value[2]], [value[3], value[4], value[5]]])
    fig, ax = plt.subplots(figsize=(7.0, 4.6)); im = ax.imshow(matrix, cmap="YlOrRd", aspect="auto")
    ax.set_xticks(range(3), ["Raw", "Causal SG", "Causal EWMA"]); ax.set_yticks(range(2), ["Top10", "Raw14"]); ax.set_title("RMSE feature–filter matrix (mV)")
    for i in range(2):
        for j in range(3): ax.text(j, i, f"{matrix[i,j]:.2f}", ha="center", va="center", fontsize=12, color="black")
    fig.colorbar(im, ax=ax, label="RMSE (mV)"); fig.tight_layout(); fig.savefig(FIG / "Fig2_RMSE_Feature_Filter_Matrix.png"); plt.close(fig)
    lim = max(float(max(np.abs(f.y_true_delta_V).max(), np.abs(f.y_pred_delta_V).max())) for f in predictions.values()) * 1000
    fig, axes = plt.subplots(2, 3, figsize=(12, 7.8), sharex=True, sharey=True)
    for ax, group in zip(axes.ravel(), groups):
        f = predictions[group]; ax.scatter(f.y_true_delta_V*1000, f.y_pred_delta_V*1000, s=3, alpha=.2, color=colors[group], rasterized=True)
        ax.plot([-lim, lim], [-lim, lim], "k--", lw=.8); ax.set_title(labels[group]); ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_xlabel("True ΔV (mV)"); ax.set_ylabel("Predicted ΔV (mV)")
    fig.suptitle("Assessment ΔV prediction against ground truth", y=.995); fig.tight_layout(); fig.savefig(FIG / "Fig3_Prediction_vs_Ground_Truth.png"); plt.close(fig)
    fig, ax = plt.subplots(figsize=(11, 4.8))
    for group in groups:
        f = predictions[group].assign(abs_error_mV=lambda x: np.abs(x.y_pred_delta_V-x.y_true_delta_V)*1000)
        s = f.groupby("target_row", sort=True).abs_error_mV.mean(); ax.plot(s.index, s.to_numpy(), label=labels[group], color=colors[group], lw=1.1)
    ax.set(title="Chronological assessment prediction error", xlabel="Assessment target row", ylabel="Mean absolute error (mV)"); ax.legend(ncol=3, frameon=False); fig.tight_layout(); fig.savefig(FIG / "Fig4_Chronological_Prediction_Error.png"); plt.close(fig)


def write_report(sha: str, scores: pd.DataFrame, metrics: pd.DataFrame, audit: dict[str, Any]) -> tuple[str, list[str]]:
    labels = {"E0": ("Top10", "Raw"), "E1": ("Top10", "SG(7,2)"), "E2": ("Top10", "EWMA(0.3)"), "E3": ("Raw14", "Raw"), "E4": ("Raw14", "SG(7,2)"), "E5": ("Raw14", "EWMA(0.3)")}
    table = "\n".join(f"|{r.experiment_group}|{labels[r.experiment_group][0]}|{labels[r.experiment_group][1]}|{r.RMSE_mV:.4f}|{r.MAE_mV:.4f}|{r.R2:.4f}|" for _, r in metrics.iterrows())
    best = str(metrics.loc[metrics.RMSE_mV.idxmin(), "experiment_group"]); status = {"E0":"TOP10_UNFILTERED_LOWEST_EXPLORATORY_RMSE", "E1":"TOP10_SG_LOWEST_EXPLORATORY_RMSE", "E2":"TOP10_EWMA_LOWEST_EXPLORATORY_RMSE", "E3":"RAW14_UNFILTERED_LOWEST_EXPLORATORY_RMSE", "E4":"RAW14_SG_LOWEST_EXPLORATORY_RMSE", "E5":"RAW14_EWMA_LOWEST_EXPLORATORY_RMSE"}[best]
    m = metrics.set_index("experiment_group")
    comparisons = [f"E0 vs E3 (Raw): Raw14−Top10 = {m.loc['E3','RMSE_mV']-m.loc['E0','RMSE_mV']:+.4f} mV",
                   f"E1 vs E4 (SG): Raw14−Top10 = {m.loc['E4','RMSE_mV']-m.loc['E1','RMSE_mV']:+.4f} mV",
                   f"E2 vs E5 (EWMA): Raw14−Top10 = {m.loc['E5','RMSE_mV']-m.loc['E2','RMSE_mV']:+.4f} mV",
                   f"Top10 SG−Raw = {m.loc['E1','RMSE_mV']-m.loc['E0','RMSE_mV']:+.4f} mV; EWMA−Raw = {m.loc['E2','RMSE_mV']-m.loc['E0','RMSE_mV']:+.4f} mV",
                   f"Raw14 SG−Raw = {m.loc['E4','RMSE_mV']-m.loc['E3','RMSE_mV']:+.4f} mV; EWMA−Raw = {m.loc['E5','RMSE_mV']-m.loc['E3','RMSE_mV']:+.4f} mV"]
    text = f"""# Phase E — TFT 全特征与因果滤波联合对照实验\n\n## 1. 研究目的\n\n本研究在固定的 RealHyFC Development 7:1:2 时间顺序划分上，系统比较 Top10 与全部 Raw14 输入特征，以及无滤波、单侧因果 SG(7,2) 和逐 48-step 窗口因果 EWMA(α=0.3)。目的是在公共 scaler 下检查特征数量与因果滤波是否存在探索性互补。\n\n## 2. 数据与实验协议\n\n混合 causal 表只以 `pd.read_csv(nrows=3007)` 打开，再按 `bin_end_ophrs2_h < 1819.997738` 过滤为 3005 个 Development rows。行区间为 Train [0,2103)、Validation [2103,2404)、Assessment [2404,3005)，合法窗口数为 {audit['legal_windows']}。Assessment 已在历史 C/D 阶段暴露，故状态为 `DEVELOPMENT_EXPLORATORY_REUSED_ASSESSMENT`，并非新的独立盲测。未访问 Test。\n\n## 3. TFT 模型及公共标准化\n\n六组均为独立初始化的 Seed42 CanonicalStrictHistoryTFT，固定 T1-R Fold2 超参数；模型参数量均为 {audit['parameter_count']}。公共 scaler 仅由未滤波合法 Train 历史拟合，目标 ΔV scale 为 {audit['common_scaler_target_scale_V']:.12f} V；滤波只发生在输入、公共 scaler 之前。SG 和 EWMA 的未来扰动因果性测试均通过。\n\n## 4. 特征配置与滤波\n\nTop10：{', '.join(scores.loc[scores.selected_top10, 'feature'])}。Top10 使用标准化空间零掩码，Raw14 保留所有 14 个输入通道；两者模型输入维度均保持 14，因而参数量不变。\n\n## 5. 六组结果\n\n|模型|特征|滤波|RMSE(mV)|MAE(mV)|R²|\n|---|---|---|---:|---:|---:|\n{table}\n\n## 6. 对照分析\n\n""" + "\n".join(f"- {x}" for x in comparisons) + f"""\n\nRMSE 是唯一预注册主排序指标；本轮最低 RMSE 为 **{best}**，对应状态 `{status}`。MAE 和 R²仅作辅助报告，未被用于改写 RMSE 排名。单次 Seed42 的细小差异不能作为稳定性能优势或物理机理证据。\n\n## 7. 历史数值复现性\n\n历史 C1 与 D0 的记录差异已单独审计，结论为 `C1_D0_STRICT_REPRODUCIBILITY_UNRESOLVED`。C 的条件各自 scaler 与 D/E 的公共 scaler 口径不同，且历史元数据不足以严格锁定所有执行级数值差异。Phase E 仅以 E0–E5 的同阶段、同 scaler 新训练结果排序，不跨阶段拼接改善幅度。\n\n## 8. 实验局限性\n\n本阶段只有单次时间顺序切分和 Seed42；Assessment 被历史阶段复用；滑动窗口令同一 target time 可被多个 origin 预测；没有独立 Test 泛化结论。任务是短期 48→24 ΔV 预测，而非 SOH 或 RUL 预测。\n\n## 9. 研究结论\n\n`PHASE_E_DEVELOPMENT_EXPLORATORY_ONLY`；`{status}`。本阶段至此收口，不自动搜索其他特征数、滤波参数、backbone、Optuna、额外随机种子或 Test。\n"""
    (REPORT / "E_SERIES_EXPERIMENT_REPORT.md").write_text(text, encoding="utf-8")
    return status, comparisons


def run() -> None:
    global ACTIVE_PROTOCOL_SHA
    OUT.mkdir(exist_ok=True); FIG.mkdir(exist_ok=True); REPORT.mkdir(exist_ok=True)
    cfg, sha, dev, windows, scores, top10, scaler, audit = pretrain_audit(write=True); ACTIVE_PROTOCOL_SHA = sha
    # Merge the persisted Fold-2 tuning result with the protocol's fixed training controls.
    params = {**cfg["hyperparameters"], **json.loads(T1R.read_text(encoding="utf-8"))["folds"]["fold_2"]["best_params"]}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    specs = {"E0": (top10, "Raw"), "E1": (top10, "SG"), "E2": (top10, "EWMA"), "E3": (None, "Raw"), "E4": (None, "SG"), "E5": (None, "EWMA")}
    metric_rows = []; predictions: dict[str, pd.DataFrame] = {}; init_hashes = []
    for label, (mask, filt) in specs.items():
        print(f"[Phase E] training {label}: {'Top10' if mask is not None else 'Raw14'} + {filt}", flush=True)
        model, info = fit_condition(label, params, windows["train"], windows["validation"], scaler, mask, filt, device)
        assessment = predict(model, prep(windows["assessment"].x, scaler, mask, filt), int(params["batch_size"]), device) * scaler.target_scale
        frame = prediction_frame(label, windows["assessment"], assessment, sha); path = OUT / f"{label}_predictions_seed42.csv"; frame.to_csv(path, index=False); predictions[label] = frame
        metrics = numpy_metrics(frame)
        metric_rows.append({"experiment_group": label, "seed": 42, **metrics, "best_epoch": info["best_epoch"], "training_time_s": info["training_time_s"],
                            "validation_RMSE_mV": info["validation_RMSE_mV"], "parameter_count": info["parameter_count"], "initialization_sha256": info["initialization_sha256"],
                            "features": "Top10" if mask is not None else "Raw14", "filter": filt, "checkpoint": info["checkpoint"]})
        write_json(OUT / "training_logs" / f"{label}_seed42.json", {"protocol_sha256": sha, "experiment_group": label, "fit": info, "test_accessed": False})
        init_hashes.append(info["initialization_sha256"])
    metrics = pd.DataFrame(metric_rows); metrics.to_csv(OUT / "E_SEED42_METRICS.csv", index=False)
    if len(set(init_hashes)) != 1 or not (metrics.parameter_count == metrics.parameter_count.iloc[0]).all(): raise RuntimeError("E_INITIALIZATION_OR_PARAMETER_CONTROL_FAILURE")
    independent = independent_audit(sha)
    comparison = {"protocol_sha256": sha, "real_training_completed": True, "test_accessed": False, "assessment_status": "DEVELOPMENT_EXPLORATORY_REUSED_ASSESSMENT",
                  "independent_metric_audit_pass": independent["pass"], "results": metrics.to_dict(orient="records"),
                  "best_RMSE_group": str(metrics.loc[metrics.RMSE_mV.idxmin(), "experiment_group"]), "identical_initialization_across_groups": True}
    write_json(OUT / "E_SEED42_COMPARISON.json", comparison)
    make_figures(predictions, metrics); status, comparisons = write_report(sha, scores, metrics, audit)
    write_json(OUT / "E_FINAL_STATUS.json", {"status": status, "comparisons": comparisons, "test_accessed": False, "phase_complete": True})
    print(json.dumps({"best_group": comparison["best_RMSE_group"], "status": status, "test_accessed": False}, ensure_ascii=False), flush=True)


def finalize_existing() -> None:
    """Fail-closed post-training finalization from six persisted Phase-E caches.

    This path never constructs a model, opens a checkpoint for training, or
    writes predictions. It exists solely to resume validation after an
    artifact-schema audit implementation failure.
    """
    global ACTIVE_PROTOCOL_SHA
    cfg, sha, _dev, _windows, scores, _top10, _scaler, audit = pretrain_audit(write=True); ACTIVE_PROTOCOL_SHA = sha
    metrics_path = OUT / "E_SEED42_METRICS.csv"
    if not metrics_path.exists(): raise RuntimeError("E_FINALIZATION_MISSING_METRICS")
    metrics = pd.read_csv(metrics_path)
    groups = [f"E{i}" for i in range(6)]
    if metrics.experiment_group.tolist() != groups or len(metrics) != 6: raise RuntimeError("E_FINALIZATION_GROUP_CONTRACT_FAILURE")
    predictions = {group: pd.read_csv(OUT / f"{group}_predictions_seed42.csv") for group in groups}
    independent = independent_audit(sha)
    if not independent["pass"]: raise RuntimeError("E_FINALIZATION_INDEPENDENT_AUDIT_FAILURE")
    if metrics.initialization_sha256.nunique() != 1 or metrics.parameter_count.nunique() != 1: raise RuntimeError("E_FINALIZATION_CONTROL_FAILURE")
    comparison = {"protocol_sha256": sha, "real_training_completed": True, "post_training_finalization_only": True,
                  "post_training_repair": "prediction_cache_schema_audit_restored_origin_age_h_to_required_schema",
                  "model_retraining_performed": False, "test_accessed": False,
                  "assessment_status": "DEVELOPMENT_EXPLORATORY_REUSED_ASSESSMENT", "independent_metric_audit_pass": True,
                  "results": metrics.to_dict(orient="records"), "best_RMSE_group": str(metrics.loc[metrics.RMSE_mV.idxmin(), "experiment_group"]),
                  "identical_initialization_across_groups": True}
    write_json(OUT / "E_SEED42_COMPARISON.json", comparison)
    write_json(OUT / "E_POST_TRAIN_FINALIZATION_REPAIR.json", {"status": "PASS", "cause": "independent audit omitted persisted origin_age_h column",
               "scientific_predictions_modified": False, "models_retrained": False, "independent_audit_pass": True, "test_accessed": False})
    make_figures(predictions, metrics); status, comparisons = write_report(sha, scores, metrics, audit)
    write_json(OUT / "E_FINAL_STATUS.json", {"status": status, "comparisons": comparisons, "test_accessed": False, "phase_complete": True,
               "all_six_real_trainings_completed": True, "independent_metric_audit_pass": True})
    print(json.dumps({"finalization": "PASS", "best_group": comparison["best_RMSE_group"], "status": status, "test_accessed": False}, ensure_ascii=False), flush=True)


def main() -> None:
    global ACTIVE_PROTOCOL_SHA
    parser = argparse.ArgumentParser(); group = parser.add_mutually_exclusive_group(required=True); group.add_argument("--audit", action="store_true"); group.add_argument("--run", action="store_true"); group.add_argument("--finalize-existing", action="store_true"); args = parser.parse_args()
    if args.audit:
        cfg, sha, *_ = pretrain_audit(write=True); ACTIVE_PROTOCOL_SHA = sha; print(json.dumps({"audit": "PASS", "protocol_sha256": sha}, ensure_ascii=False))
    elif args.run:
        run()
    else:
        finalize_existing()


ACTIVE_PROTOCOL_SHA = "UNSET"
if __name__ == "__main__": main()
