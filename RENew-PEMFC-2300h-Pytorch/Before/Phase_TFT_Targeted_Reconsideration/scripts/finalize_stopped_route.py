"""Deterministically close a stopped TFT route from saved seed-42 artifacts only.

This recovery exists because the initial runner completed all three frozen
Vanilla seed-42 folds but failed when serialising NumPy boolean gate values.
It neither creates a model nor reads N1/Test data; it repairs metadata from
already saved Development metrics and preserved historical references.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import yaml

from tft_common import CONFIG_SHA_PATH, PHASE, PROJECT, current_head, summarize_seed, write_csv, write_json, write_text


def file_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def markdown_table(frame: pd.DataFrame) -> str:
    columns = [str(column) for column in frame.columns]
    rows = [[str(value) for value in row] for row in frame.itertuples(index=False, name=None)]
    return "\n".join([
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
        *["| " + " | ".join(row) + " |" for row in rows],
    ])


def main() -> None:
    bs = PROJECT / "Phase_BS_Final_Backbone_Screening" / "outputs" / "per_fold.csv"
    df = PROJECT / "Phase_DF_Leakage_Safe_Decomposition_Features" / "outputs" / "MULTISEED_RESULTS.csv"
    per_fold_path = PHASE / "outputs" / "PER_FOLD_METRICS.csv"
    candidate = pd.read_csv(per_fold_path)
    if set(candidate.model) != {"LSTM"} or set(candidate.seed) != {42} or len(candidate) != 3:
        raise RuntimeError("Stopped-route finalizer only accepts complete Vanilla seed-42 artifacts")
    raw = pd.read_csv(bs).query("model == 'TimesNet' and seed in [42, 2024, 3407]").copy()
    if len(raw) != 9 or raw.test_accessed.astype(str).str.lower().eq("true").any():
        raise RuntimeError("Frozen Raw TimesNet reference is invalid")
    raw_summary = summarize_seed(raw)
    emd = pd.read_csv(df).query("model == 'EMDFeature_TimesNet' and seed in [42, 2024, 3407]").copy()
    if len(emd) != 3:
        raise RuntimeError("Frozen EMD reference is incomplete")
    normalized_raw = raw.assign(reference="TimesNet_Raw_F2", source_path=str(bs), source_sha256=file_sha(bs), metric_definition="Phase BS per-fold Development forecast-query metric")
    normalized_emd = emd.assign(reference="TimesNet_EMD", fold="three_fold_summary", source_path=str(df), source_sha256=file_sha(df), metric_definition="Phase DF historical three-fold Development metric")
    common = ["reference", "source_path", "source_sha256", "metric_definition", "model", "seed", "fold", "rmse_mV", "mae_mV", "r2", "endpoint_rmse_mV", "p95_abs_error_mV", "p99_abs_error_mV", "max_abs_error_mV", "target_time_equal_rmse_mV"]
    frozen = pd.concat([normalized_raw.reindex(columns=common), normalized_emd.reindex(columns=common)], ignore_index=True)
    write_csv(frozen, "outputs", "FROZEN_REFERENCES.csv")
    vanilla_summary = summarize_seed(candidate)
    v, r = vanilla_summary.iloc[0], raw_summary.loc[raw_summary.seed.eq(42)].iloc[0]
    viable = {
        "formal_110": bool(v.formal_RMSE_mV <= r.formal_RMSE_mV * 1.10),
        "fold3_115": bool(v.Fold3_RMSE_mV <= r.Fold3_RMSE_mV * 1.15),
        "p99_110": bool(v.P99_mV <= r.P99_mV * 1.10),
        "endpoint_115": bool(v.endpoint_RMSE_mV <= r.endpoint_RMSE_mV * 1.15),
        "target_equal_110": bool(v.target_time_equal_RMSE_mV <= r.target_time_equal_RMSE_mV * 1.10),
    }
    borderline = {
        "formal_between_110_120": bool(r.formal_RMSE_mV * 1.10 < v.formal_RMSE_mV <= r.formal_RMSE_mV * 1.20),
        "fold3_110": bool(v.Fold3_RMSE_mV <= r.Fold3_RMSE_mV * 1.10),
        "p99_105": bool(v.P99_mV <= r.P99_mV * 1.05),
        "endpoint_110": bool(v.endpoint_RMSE_mV <= r.endpoint_RMSE_mV * 1.10),
        "target_equal_105": bool(v.target_time_equal_RMSE_mV <= r.target_time_equal_RMSE_mV * 1.05),
    }
    if any(viable.values()) and all(viable.values()):
        raise RuntimeError("Route is viable; stopped-route finalizer is prohibited")
    if all(borderline.values()):
        raise RuntimeError("Route is borderline; stopped-route finalizer is prohibited")
    gate = {
        "vanilla_seed42_status": "TFT_NOT_VIABLE", "vanilla_seed42_checks": viable,
        "borderline_checks": borderline, "raw_timesnet_seed42_formal_RMSE_mV": float(r.formal_RMSE_mV),
        "raw_timesnet_seed42_formal_110_limit_mV": float(r.formal_RMSE_mV * 1.10),
        "raw_timesnet_seed42_formal_120_limit_mV": float(r.formal_RMSE_mV * 1.20),
        "vanilla_seed42_formal_RMSE_mV": float(v.formal_RMSE_mV), "final_status": "TFT_NOT_VIABLE",
        "test_accessed": False, "model_training_performed": True,
    }
    write_json(gate, "outputs", "GATE_EVALUATION.json")
    write_csv(pd.DataFrame([
        {"model": "sLSTM", "seeds": "42,2024,3407", "status": "NOT_RUN", "reason": "vanilla_seed42_TFT_NOT_VIABLE"},
        {"model": "mLSTM", "seeds": "42,2024,3407", "status": "NOT_RUN", "reason": "vanilla_seed42_TFT_NOT_VIABLE"},
        {"model": "LSTM", "seeds": "2024,3407", "status": "NOT_RUN", "reason": "vanilla_seed42_TFT_NOT_VIABLE"},
    ]), "outputs", "NOT_RUN_MANIFEST.csv")
    write_json({"post_result_metadata_recovery": True, "training_repeated": False, "data_read": "none", "test_accessed": False, "reason": "initial runner reached all saved seed42 artifacts then could not JSON-serialise NumPy boolean gate values", "metric_inputs": [str(per_fold_path), str(bs), str(df)], "protocol_sha256": CONFIG_SHA_PATH.read_text(encoding="utf-8").split()[0]}, "outputs", "RUN_METADATA_RECOVERY.json")
    report = "# Phase TFT Targeted Reconsideration — Final research report\n\n## Final status\n\n**TFT_NOT_VIABLE**\n\nThe strict-history Vanilla TFT seed=42 formal forecast-query RMSE is **{v:.6f} mV**, above the frozen Raw TimesNet seed=42 1.20× limit of **{limit:.6f} mV** (reference: {raw:.6f} mV). It therefore passes neither the viable nor the borderline gate. Per the frozen stop rule, LSTM seeds 2024/3407 and all sLSTM/mLSTM runs were not run.\n\n## Saved Vanilla seed=42 metrics\n\n{vanilla}\n\n## Frozen Raw TimesNet seed=42 reference\n\n{raw_table}\n\n## Gate evaluation\n\n```json\n{gate}\n```\n\n## Interpretation boundary\n\nThis is a Development-only dynamic voltage forecasting decision. It does not change the historical TimesNet Raw F2 incumbent, TimesNet+EMD `EMD_SPECIFIC_ADVANCE`, `TIMESNET_FREEZE`, `BACKBONE_SEARCH_CLOSED`, or the Phase HA health-representation result. No Test row, target, prediction, performance metric, polarization data or EIS was read. No EMD/VMD/wavelet, age/cycle feature, SOH target, RUL target, fast/slow state, or future actual covariate was used.\n\n## Terminal handoff\n\nThis phase closes the targeted TFT route. The next decision must be at the RealHyFC-specific mechanism level, not a new backbone search. No mechanism training is started here.\n".format(v=float(v.formal_RMSE_mV), limit=float(r.formal_RMSE_mV * 1.20), raw=float(r.formal_RMSE_mV), vanilla=markdown_table(vanilla_summary), raw_table=markdown_table(raw_summary.loc[raw_summary.seed.eq(42)]), gate=json.dumps(gate, ensure_ascii=False, indent=2))
    write_text(report, "reports", "FINAL_RESEARCH_REPORT.md")
    decision = {"phase": "Phase_TFT_Targeted_Reconsideration", "status": "TFT_NOT_VIABLE", "selected_encoder": None, "vanilla_seed42_formal_RMSE_mV": float(v.formal_RMSE_mV), "raw_timesnet_seed42_formal_RMSE_mV": float(r.formal_RMSE_mV), "test_accessed": False, "backbone_search_closed": True, "next_phase": "REALHYFC_SPECIFIC_MECHANISM", "model_training_performed": True, "additional_model_training": False, "head_at_generation": current_head()}
    (PHASE / "final").mkdir(exist_ok=True)
    (PHASE / "final" / "FINAL_DECISION.yaml").write_text(yaml.safe_dump(decision, allow_unicode=True, sort_keys=False), encoding="utf-8")


if __name__ == "__main__":
    main()
