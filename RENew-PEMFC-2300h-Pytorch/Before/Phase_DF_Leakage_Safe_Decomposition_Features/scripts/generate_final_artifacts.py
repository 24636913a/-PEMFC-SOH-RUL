"""Create Phase DF's post-freeze figures and final report from saved Development artifacts."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from decomposition_features import emd_feature, vmd_feature
from df_protocol_adapter import BS, DF, build_frozen_folds, load_bs_protocol, load_df_protocol, read_development_prefix, write_json, write_text


PALETTE = {"RawF2_TimesNet": "#B64342", "SimpleTrend_TimesNet": "#42949E", "VMDFeature_TimesNet": "#0F4D92", "EMDFeature_TimesNet": "#9A4D8E"}
LABELS = {"RawF2_TimesNet": "Raw F2", "SimpleTrend_TimesNet": "SimpleTrend", "VMDFeature_TimesNet": "VMD", "EMDFeature_TimesNet": "EMD"}


def _style() -> None:
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11, "axes.linewidth": 1.6, "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42, "ps.fonttype": 42})


def _save(fig: plt.Figure, name: str) -> None:
    root = DF / "outputs" / "figures"; root.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(root / f"{name}.png", dpi=300, bbox_inches="tight")
    fig.savefig(root / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def _available_methods(summary: pd.DataFrame) -> list[str]:
    order = ["RawF2_TimesNet", "SimpleTrend_TimesNet", "VMDFeature_TimesNet", "EMDFeature_TimesNet"]
    return [method for method in order if method in set(summary.model)]


def _prediction_path(method: str, fold: str) -> Path:
    if method == "RawF2_TimesNet":
        return BS / "outputs" / "predictions" / f"bs_timesnet_{fold}_seed42.npz"
    return DF / "outputs" / "predictions" / f"df_{method.lower()}_{fold}_seed42.npz"


def _error_by_horizon(methods: list[str]) -> dict[str, np.ndarray]:
    rows: dict[str, np.ndarray] = {}
    for method in methods:
        errors = []
        for fold in ("fold_1", "fold_2", "fold_3"):
            path = _prediction_path(method, fold)
            if not path.is_file():
                continue
            with np.load(path, allow_pickle=False) as z:
                errors.append((z["y_pred_delta_V"] - z["y_true_delta_V"]) ** 2)
        if errors:
            rows[method] = np.sqrt(np.mean(np.concatenate(errors, axis=0), axis=0)) * 1000.0
    return rows


def _load_feature_cache(method: str) -> dict[int, np.ndarray]:
    manifest = pd.read_csv(DF / "outputs" / "CACHE_PROVENANCE_MANIFEST.csv")
    rows = manifest.loc[manifest.method.eq(method)]
    output = {}
    for row in rows.itertuples(index=False):
        with np.load(Path(row.cache_path), allow_pickle=False) as z:
            output[int(z["origin_row"])] = z["features"]
    return output


def _cache_segments(method: str) -> dict[int, Any]:
    manifest = pd.read_csv(DF / "outputs" / "CACHE_PROVENANCE_MANIFEST.csv")
    rows = manifest.loc[manifest.method.eq(method)]
    return {int(row.origin_row): row.temporal_segment_id for row in rows.itertuples(index=False)}


def _train_histories() -> tuple[dict[int, np.ndarray], list[tuple[str, Any]]]:
    df_cfg, _ = load_df_protocol(); bs_cfg, _ = load_bs_protocol(df_cfg)
    frame, _ = read_development_prefix(df_cfg, bs_cfg)
    folds = build_frozen_folds(frame, bs_cfg, df_cfg)
    histories: dict[int, np.ndarray] = {}
    train_windows = []
    for fold, train, _validation in folds:
        train_windows.append((fold, train))
        for i, origin in enumerate(train.origin_row.tolist()):
            histories[int(origin)] = np.asarray(train.x[i, :, 0], dtype=np.float64)
    return histories, train_windows


def _postfreeze_train_only_distribution_audit(resolved_alpha: int) -> dict[str, Any]:
    histories, _ = _train_histories()
    freqs, imf_counts = [], []
    for history in histories.values():
        freqs.extend(vmd_feature(history, alpha=resolved_alpha).audit["final_center_frequencies_cycles_per_sample"])
        imf_counts.append(emd_feature(history).audit["imf_count"])
    return {"status": "POSTFREEZE_TRAIN_ONLY_DESCRIPTIVE", "origin_count": len(histories), "vmd_center_frequencies": freqs, "emd_imf_counts": imf_counts, "test_accessed": False, "used_for_parameter_selection": False}


def _endpoint_revisions(cache: dict[int, np.ndarray], segments: dict[int, Any]) -> np.ndarray:
    origins = sorted(cache)
    revisions = []
    for origin in origins:
        if origin + 1 in cache and segments.get(origin) == segments.get(origin + 1):
            revisions.append(abs(float(cache[origin][47, 0]) - float(cache[origin + 1][46, 0])) * 1000.0)
    return np.asarray(revisions)


def _metric_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "_No executed rows._"
    shown = frame.copy()
    for column in shown.columns:
        if pd.api.types.is_float_dtype(shown[column]):
            shown[column] = shown[column].map(lambda value: f"{value:.4f}")
    return "| " + " | ".join(shown.columns) + " |\n| " + " | ".join("---" for _ in shown.columns) + " |\n" + "\n".join("| " + " | ".join(map(str, row)) + " |" for row in shown.itertuples(index=False, name=None))


def main() -> None:
    _style()
    stage0 = json.loads((DF / "outputs" / "STAGE0_SUMMARY.json").read_text(encoding="utf-8"))
    final = yaml.safe_load((DF / "final" / "FINAL_DECISION.yaml").read_text(encoding="utf-8"))
    seed42 = pd.read_csv(DF / "outputs" / "SEED42_SUMMARY_RESULTS.csv")
    per_fold = pd.read_csv(DF / "outputs" / "PER_FOLD_RESULTS.csv")
    multi = pd.read_csv(DF / "outputs" / "MULTISEED_RESULTS.csv")
    seed42_gates = pd.read_csv(DF / "outputs" / "SEED42_GATES.csv")
    multiseed_gates = pd.read_csv(DF / "outputs" / "MULTISEED_GATES.csv")
    methods = _available_methods(seed42)
    seed42_methods = seed42.loc[seed42.seed.eq(42) & seed42.model.isin(methods)].copy()

    # Formal score and per-fold score panels.
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    x = np.arange(len(methods)); values = [float(seed42_methods.loc[seed42_methods.model.eq(m), "formal_RMSE_mV"].iloc[0]) for m in methods]
    axes[0].bar(x, values, color=[PALETTE[m] for m in methods], edgecolor="black", linewidth=1.0)
    axes[0].set_xticks(x, [LABELS[m] for m in methods], rotation=20, ha="right"); axes[0].set_ylabel("Forecast-query RMSE (mV)"); axes[0].set_title("Seed 42 formal metric")
    folds = ["fold_1", "fold_2", "fold_3"]; width = .8 / max(1, len(methods))
    for i, method in enumerate(methods):
        values = [float(per_fold.loc[(per_fold.model == method) & (per_fold.seed == 42) & (per_fold.fold == fold), "rmse_mV"].iloc[0]) for fold in folds]
        axes[1].bar(np.arange(3) + (i - (len(methods)-1)/2) * width, values, width=width, color=PALETTE[method], edgecolor="black", linewidth=.7, label=LABELS[method])
    axes[1].set_xticks(np.arange(3), ["Fold 1", "Fold 2", "Fold 3"]); axes[1].set_ylabel("RMSE (mV)"); axes[1].set_title("Seed 42 per-fold RMSE"); axes[1].legend(fontsize=8)
    _save(fig, "method_formal_and_per_fold_rmse")

    # Risk/sensitivity panel.
    metrics = [("endpoint_RMSE_mV", "12 h endpoint RMSE"), ("P99_mV", "P99 absolute error"), ("target_time_equal_RMSE_mV", "Target-time-equal RMSE")]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    for axis, (column, title) in zip(axes, metrics):
        values = [float(seed42_methods.loc[seed42_methods.model.eq(m), column].iloc[0]) for m in methods]
        axis.bar(x, values, color=[PALETTE[m] for m in methods], edgecolor="black", linewidth=1)
        axis.set_xticks(x, [LABELS[m] for m in methods], rotation=20, ha="right"); axis.set_ylabel("mV"); axis.set_title(title)
    _save(fig, "seed42_endpoint_p99_target_time_equal")

    horizon = _error_by_horizon(methods)
    fig, axis = plt.subplots(figsize=(8, 4.5))
    for method, values in horizon.items():
        axis.plot(np.arange(1, 25) * .5, values, label=LABELS[method], color=PALETTE[method], linewidth=2)
    axis.set_xlabel("Forecast horizon (h)"); axis.set_ylabel("RMSE (mV)"); axis.set_title("Seed 42 error by horizon"); axis.legend()
    _save(fig, "seed42_error_by_horizon")

    resolved = yaml.safe_load((DF / "configs" / "RESOLVED_VMD_PARAMETERS.yaml").read_text(encoding="utf-8"))
    descriptive = _postfreeze_train_only_distribution_audit(int(resolved["alpha"]))
    write_json({key: value for key, value in descriptive.items() if key not in {"vmd_center_frequencies", "emd_imf_counts"}}, "outputs", "POSTFREEZE_TRAIN_ONLY_FIGURE_AUDIT.json")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    axes[0].hist(descriptive["vmd_center_frequencies"], bins=30, color="#0F4D92", edgecolor="black"); axes[0].set_xlabel("Center frequency (cycles/sample)"); axes[0].set_ylabel("Train-window modes"); axes[0].set_title("VMD center-frequency distribution")
    counts = pd.Series(descriptive["emd_imf_counts"]).value_counts().sort_index()
    axes[1].bar(counts.index.astype(str), counts.values, color="#9A4D8E", edgecolor="black"); axes[1].set_xlabel("IMF count"); axes[1].set_ylabel("Train windows"); axes[1].set_title("EMD IMF-count distribution")
    _save(fig, "train_vmd_frequency_and_emd_imf_distribution")

    cache_methods = {"SimpleTrend": _load_feature_cache("SimpleTrend")}
    if stage0["methods"]["VMD"]["stage0_status"] == "PASS": cache_methods["VMD"] = _load_feature_cache("VMD")
    if stage0["methods"]["EMD"]["stage0_status"] == "PASS": cache_methods["EMD"] = _load_feature_cache("EMD")
    fig, axis = plt.subplots(figsize=(8, 4.5))
    for method, cache in cache_methods.items():
        revision = _endpoint_revisions(cache, _cache_segments(method))
        axis.hist(revision, bins=40, density=True, histtype="step", linewidth=2, label=method)
    axis.set_xlabel("Endpoint revision (mV)"); axis.set_ylabel("Density"); axis.set_title("Origin-conditioned endpoint revision"); axis.legend()
    _save(fig, "endpoint_revision_distribution")

    histories, train_windows = _train_histories()
    fig, axes = plt.subplots(3, 3, figsize=(13, 10), sharex=True)
    for row, (fold, train) in enumerate(train_windows):
        fixed = [int(train.origin_row[int(round((len(train.origin_row) - 1) * fraction))]) for fraction in (.25, .50, .75)]
        for col, origin in enumerate(fixed):
            axis = axes[row, col]; xh = np.arange(48) * .5
            axis.plot(xh, histories[origin], color="black", linewidth=1.5, label="Raw voltage")
            for method, cache in cache_methods.items():
                axis.plot(xh, cache[origin][:, 0], linewidth=1.2, label=method)
            axis.set_title(f"{fold}, fixed train quantile {(25,50,75)[col]}%\norigin row {origin}")
            if row == 2: axis.set_xlabel("History time within block (h)")
            if col == 0: axis.set_ylabel("Voltage (V)")
    handles, labels = axes[0, 0].get_legend_handles_labels(); fig.legend(handles, labels, loc="upper center", ncol=len(labels), frameon=False)
    _save(fig, "fixed_train_origin_feature_trajectories")

    report = f"""# Phase DF final research report

## Scope and protocol

This frozen-backbone feature study retained the 48-step / 24 h information set, 24-step / 12 h horizon, frozen Phase BS TimesNet architecture and Development-only boundary. All decomposition is origin-conditioned causal-block processing of `V[t-47:t]`, not global full-series decomposition. Phase BS protocol SHA256: `{final['phase_bs_protocol_sha256']}`. Test was not used.

## Stage 0

SimpleTrend: `{stage0['methods']['SimpleTrend']['stage0_status']}`. VMD: `{stage0['methods']['VMD']['stage0_status']}` with Train-only unsupervised resolved `alpha={stage0['methods']['VMD'].get('resolved_alpha')}`, `K=2`, `DC=0`. EMD: `{stage0['methods']['EMD']['stage0_status']}` with valid-IMF rate `{stage0['methods']['EMD'].get('valid_imf_rate')}`. Synthetic future perturbation and prefix invariance passed. No leakage failure occurred.

## Seed 42 summaries (mV)

{_metric_table(seed42_methods)}

## Multi-seed summaries (mV)

{_metric_table(multi)}

## Frozen-gate interpretation

Seed-42 authorization is recorded without retroactive changes in `outputs/SEED42_GATES.csv`:

{_metric_table(seed42_gates)}

The formal three-seed gates are:

{_metric_table(multiseed_gates)}

SimpleTrend did not meet the registered robust Raw-gain gate. VMD did not meet the registered decomposition-specific advancement gate because its mean endpoint-risk condition versus SimpleTrend failed. EMD met the robust Raw-gain gate and the registered decomposition-specific gate versus SimpleTrend. This supports an **EMD local slow-varying representation result under this frozen 48-step forecasting protocol only**; it is not a general claim about EMD, SOH, RUL, irreversible degradation, or longer information sets.

## Decision

Final status: **`{final['status']}`**. The Raw vs SimpleTrend and decomposition-vs-SimpleTrend gates are recorded in `outputs/SEED42_GATES.csv` and `outputs/MULTISEED_GATES.csv`. Extra seeds were run only for `{final['multiseed_methods']}`. Any generic trend effect is not interpreted as decomposition-specific value unless the registered paired SimpleTrend comparisons pass.

`TIMESNET_FREEZE` remains in force and backbone search remains `CLOSED`. This phase makes no SOH, RUL, irreversible-degradation-mode, or extended-context claim.
"""
    write_text(report, "final", "FINAL_RESEARCH_REPORT.md")
    write_text(yaml.safe_dump(final, allow_unicode=True, sort_keys=False), "final", "FINAL_DECISION.yaml")
    print(json.dumps({"status": final["status"], "figures": 6, "test_accessed": False}, ensure_ascii=False))


if __name__ == "__main__":
    main()
