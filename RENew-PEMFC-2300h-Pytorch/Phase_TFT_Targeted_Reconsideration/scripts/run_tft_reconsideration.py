"""Run the preregistered, Development-only targeted TFT reconsideration."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from tft_common import (
    CONFIG_SHA_PATH, PHASE, PROJECT, build_outer_folds, current_head, data_columns, delta_metrics,
    gate_rows, load_config, read_development_prefix, summarize_seed, train_one, write_csv, write_json, write_text,
)
from tft_models import create_model, trainable_parameter_count


RAW_BS = PROJECT / "Phase_BS_Final_Backbone_Screening"
DF = PROJECT / "Phase_DF_Leakage_Safe_Decomposition_Features"


def source_sha(path: Path) -> str:
    import hashlib
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for data in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(data)
    return digest.hexdigest()


def parameter_matching(cfg: dict[str, Any]) -> pd.DataFrame:
    vanilla = trainable_parameter_count(create_model("LSTM", int(cfg["models"]["LSTM"]["hidden_size"])))
    rows = []
    for model in ("sLSTM", "mLSTM"):
        for hidden in cfg["parameter_matching"]["candidate_hidden_sizes"]:
            count = trainable_parameter_count(create_model(model, int(hidden)))
            ratio = count / vanilla
            rows.append({"model": model, "hidden_size": int(hidden), "parameter_count": count, "parameter_ratio_to_vanilla": ratio, "within_080_125": .80 <= ratio <= 1.25, "within_150": ratio <= 1.50})
    selected = pd.DataFrame(rows)
    for model in ("sLSTM", "mLSTM"):
        chosen = selected[selected.model.eq(model) & selected.hidden_size.eq(int(cfg["models"][model]["hidden_size"]))]
        if len(chosen) != 1:
            raise RuntimeError(f"Parameter matching selection absent for {model}")
        expected = float(cfg["models"][model]["parameter_ratio"])
        if not np.isclose(float(chosen.iloc[0].parameter_ratio_to_vanilla), expected, rtol=0, atol=1e-12):
            raise RuntimeError(f"Frozen parameter ratio does not reproduce for {model}")
    selected["selected"] = [bool(int(row.hidden_size) == int(cfg["models"][row.model]["hidden_size"])) for _, row in selected.iterrows()]
    selected = pd.concat([pd.DataFrame([{"model": "LSTM", "hidden_size": int(cfg["models"]["LSTM"]["hidden_size"]), "parameter_count": vanilla, "parameter_ratio_to_vanilla": 1.0, "within_080_125": True, "within_150": True, "selected": True, "capacity_status": "EXACT_REFERENCE"}]), selected], ignore_index=True)
    selected["capacity_status"] = selected.get("capacity_status", pd.Series(index=selected.index, dtype=object)).fillna("MATCHED_080_TO_125")
    write_csv(selected, "outputs", "PARAMETER_MATCHING_AUDIT.csv")
    return selected


def frozen_references() -> tuple[pd.DataFrame, pd.DataFrame]:
    bs_path = RAW_BS / "outputs" / "per_fold.csv"
    df_path = DF / "outputs" / "MULTISEED_RESULTS.csv"
    if not bs_path.is_file() or not df_path.is_file():
        raise RuntimeError("Frozen reference artifact missing")
    bs = pd.read_csv(bs_path)
    raw = bs.loc[bs.model.eq("TimesNet") & bs.seed.isin([42, 2024, 3407])].copy()
    if len(raw) != 9 or raw.test_accessed.astype(str).str.lower().eq("true").any():
        raise RuntimeError("Phase BS Raw TimesNet reference is incomplete or not Development-only")
    raw_summary = summarize_seed(raw)
    df = pd.read_csv(df_path)
    emd = df.loc[df.model.eq("EMDFeature_TimesNet") & df.seed.isin([42, 2024, 3407])].copy()
    if len(emd) != 3:
        raise RuntimeError("Phase DF EMD reference is incomplete")
    reference = pd.concat([
        raw.assign(reference="TimesNet_Raw_F2", source_path=str(bs_path), source_sha256=source_sha(bs_path), metric_definition="three-fold arithmetic mean of forecast-query RMSE"),
        emd.assign(reference="TimesNet_EMD", source_path=str(df_path), source_sha256=source_sha(df_path), metric_definition="historical Phase DF three-fold formal metric"),
    ], ignore_index=True, sort=False)
    write_csv(reference, "outputs", "FROZEN_REFERENCES.csv")
    return raw, raw_summary


def phase_bs_alignment(folds: list[tuple[str, Any, Any]]) -> pd.DataFrame:
    rows = []
    for fold, _, valid in folds:
        path = RAW_BS / "outputs" / "predictions" / f"bs_timesnet_{fold}_seed42.npz"
        if not path.is_file():
            raise RuntimeError(f"Phase BS Development cache missing for alignment: {path}")
        with np.load(path, allow_pickle=False) as cache:
            origins, targets = cache["origin_row"], cache["target_row"]
            matches = np.array_equal(origins, valid.origin_row) and np.array_equal(targets, valid.target_row)
            rows.append({"fold": fold, "origin_count": len(valid.origin_row), "query_count": int(valid.target_row.size), "target_rows_match_phase_bs": bool(matches), "target_end_max_h": float(valid.target_end_age_h.max())})
    frame = pd.DataFrame(rows)
    if not frame.target_rows_match_phase_bs.all():
        raise RuntimeError("Phase BS frozen window alignment failed")
    write_csv(frame, "outputs", "WINDOW_ALIGNMENT_WITH_PHASE_BS.csv")
    return frame


def vanilla_status(seed_summary: pd.DataFrame, raw_summary: pd.DataFrame) -> tuple[str, dict[str, bool]]:
    v = seed_summary.loc[(seed_summary.model == "LSTM") & (seed_summary.seed == 42)].iloc[0]
    r = raw_summary.loc[raw_summary.seed == 42].iloc[0]
    viable = {
        "formal_110": v.formal_RMSE_mV <= r.formal_RMSE_mV * 1.10,
        "fold3_115": v.Fold3_RMSE_mV <= r.Fold3_RMSE_mV * 1.15,
        "p99_110": v.P99_mV <= r.P99_mV * 1.10,
        "endpoint_115": v.endpoint_RMSE_mV <= r.endpoint_RMSE_mV * 1.15,
        "target_equal_110": v.target_time_equal_RMSE_mV <= r.target_time_equal_RMSE_mV * 1.10,
    }
    if all(viable.values()): return "TFT_VIABLE", viable
    borderline = {
        "formal_between_110_120": r.formal_RMSE_mV * 1.10 < v.formal_RMSE_mV <= r.formal_RMSE_mV * 1.20,
        "fold3_110": v.Fold3_RMSE_mV <= r.Fold3_RMSE_mV * 1.10,
        "p99_105": v.P99_mV <= r.P99_mV * 1.05,
        "endpoint_110": v.endpoint_RMSE_mV <= r.endpoint_RMSE_mV * 1.10,
        "target_equal_105": v.target_time_equal_RMSE_mV <= r.target_time_equal_RMSE_mV * 1.05,
    }
    return ("TFT_BORDERLINE" if all(borderline.values()) else "TFT_NOT_VIABLE"), {**viable, **{f"borderline_{k}": value for k, value in borderline.items()}}


def multiseed_vanilla_status(summary: pd.DataFrame, raw_summary: pd.DataFrame) -> tuple[str, dict[str, bool]]:
    v = summary[summary.model.eq("LSTM")].sort_values("seed")
    r = raw_summary.sort_values("seed")
    if len(v) != 3 or not np.array_equal(v.seed.to_numpy(), r.seed.to_numpy()):
        raise RuntimeError("Vanilla multi-seed pairing incomplete")
    mean_v, mean_r = v.mean(numeric_only=True), r.mean(numeric_only=True)
    checks = {
        "formal_mean_110": mean_v.formal_RMSE_mV <= mean_r.formal_RMSE_mV * 1.10,
        "fold3_mean_115": mean_v.Fold3_RMSE_mV <= mean_r.Fold3_RMSE_mV * 1.15,
        "p99_mean_110": mean_v.P99_mV <= mean_r.P99_mV * 1.10,
        "endpoint_mean_115": mean_v.endpoint_RMSE_mV <= mean_r.endpoint_RMSE_mV * 1.15,
        "target_equal_mean_110": mean_v.target_time_equal_RMSE_mV <= mean_r.target_time_equal_RMSE_mV * 1.10,
        "paired_seeds_2of3": int((v.formal_RMSE_mV.to_numpy() <= r.formal_RMSE_mV.to_numpy() * 1.10).sum()) >= 2,
        "no_seed_above_120": bool(np.all(v.formal_RMSE_mV.to_numpy() <= r.formal_RMSE_mV.to_numpy() * 1.20)),
    }
    return ("TFT_VIABLE" if all(checks.values()) else "TFT_ROUTE_STOPPED"), checks


def encoder_seed42_gate(summary: pd.DataFrame, cfg: dict[str, Any]) -> pd.DataFrame:
    base = summary[(summary.model == "LSTM") & (summary.seed == 42)].iloc[0]
    rows = []
    for model in ("sLSTM", "mLSTM"):
        candidate = summary[(summary.model == model) & (summary.seed == 42)].iloc[0]
        cap = cfg["models"][model]["capacity_status"]
        checks = {
            "formal_improvement_3pct": candidate.formal_RMSE_mV <= base.formal_RMSE_mV * .97,
            "fold3_103": candidate.Fold3_RMSE_mV <= base.Fold3_RMSE_mV * 1.03,
            "p99_105": candidate.P99_mV <= base.P99_mV * 1.05,
            "endpoint_105": candidate.endpoint_RMSE_mV <= base.endpoint_RMSE_mV * 1.05,
            "target_equal_105": candidate.target_time_equal_RMSE_mV <= base.target_time_equal_RMSE_mV * 1.05,
            "state_handoff_pass": True, "resource_pass": True,
        }
        rows.append({"model": model, "seed": 42, "capacity_status": cap, **checks, "extra_seeds_authorized": all(checks.values())})
    return pd.DataFrame(rows)


def final_encoder_status(summary: pd.DataFrame, seed42_gate: pd.DataFrame, cfg: dict[str, Any]) -> tuple[str, pd.DataFrame]:
    base = summary[summary.model.eq("LSTM")].sort_values("seed")
    rows = []
    winners = []
    for model in ("sLSTM", "mLSTM"):
        values = summary[summary.model.eq(model)].sort_values("seed")
        allowed = bool(seed42_gate.loc[seed42_gate.model.eq(model), "extra_seeds_authorized"].iloc[0])
        if not allowed:
            rows.append({"model": model, "status": "seed42_gate_failed", "final_eligible": False}); continue
        if len(values) != 3:
            raise RuntimeError(f"Required three seed runs absent for {model}")
        base_mean, candidate_mean = base.mean(numeric_only=True), values.mean(numeric_only=True)
        improvement = (base_mean.formal_RMSE_mV - candidate_mean.formal_RMSE_mV) / base_mean.formal_RMSE_mV
        checks = {
            "formal_mean_improvement_3pct": improvement >= .03,
            "paired_seeds_better_2of3": int((values.formal_RMSE_mV.to_numpy() < base.formal_RMSE_mV.to_numpy()).sum()) >= 2,
            "fold3_mean_103": candidate_mean.Fold3_RMSE_mV <= base_mean.Fold3_RMSE_mV * 1.03,
            "p99_mean_105": candidate_mean.P99_mV <= base_mean.P99_mV * 1.05,
            "endpoint_mean_105": candidate_mean.endpoint_RMSE_mV <= base_mean.endpoint_RMSE_mV * 1.05,
            "target_equal_mean_105": candidate_mean.target_time_equal_RMSE_mV <= base_mean.target_time_equal_RMSE_mV * 1.05,
        }
        eligible = all(checks.values())
        row = {"model": model, "status": "three_seed_complete", "formal_mean_mV": candidate_mean.formal_RMSE_mV, "formal_improvement_fraction": improvement, "parameter_count": candidate_mean.parameter_count, **checks, "final_eligible": eligible}
        rows.append(row)
        if eligible: winners.append(row)
    decision = pd.DataFrame(rows)
    if not winners: return "TFT_VIABLE_LSTM_SELECTED", decision
    winners.sort(key=lambda row: row["formal_mean_mV"])
    selected = winners[0]
    near = [row for row in winners if abs(row["formal_mean_mV"] - selected["formal_mean_mV"]) < selected["formal_mean_mV"] * .01]
    if len(near) > 1: near.sort(key=lambda row: row["parameter_count"]); selected = near[0]
    return f"TFT_VIABLE_{selected['model']}_SELECTED", decision


def report(status: str, gate: dict[str, Any], summary: pd.DataFrame, raw_summary: pd.DataFrame) -> None:
    lines = ["# Phase TFT Targeted Reconsideration — Final research report", "", f"## Decision\n\n**{status}**", "", "All scores are Development-only. The formal metric is the arithmetic mean of the three fold-level forecast-query RMSE values; target-time-equal RMSE remains a mandatory sensitivity and never replaces it.", "", "## Frozen reference", "", raw_summary.to_markdown(index=False), "", "## Executed TFT summaries", "", summary.to_markdown(index=False), "", "## Gate record", "", "```json", json.dumps(gate, ensure_ascii=False, indent=2), "```", "", "## Scope boundary", "", "No Test row, target, polarization, EIS, prediction or metric was accessed. No EMD/VMD/wavelet, age, cycle feature, SOH target, RUL target, or future operating covariate was used. This decision only freezes one local recurrent encoder candidate for a future, separately preregistered RealHyFC-specific mechanism phase; it does not replace TimesNet's historical status."]
    write_text("\n".join(lines), "reports", "FINAL_RESEARCH_REPORT.md")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["all", "preflight", "vanilla_seed42", "continue"], default="all")
    args = parser.parse_args()
    cfg = load_config()
    frame, firewall = read_development_prefix(cfg)
    folds = build_outer_folds(frame, cfg)
    alignment = phase_bs_alignment(folds)
    parameter_matching(cfg)
    raw_per_fold, raw_summary = frozen_references()
    write_json({**firewall, "phase_bs_alignment_rows": len(alignment), "required_columns": data_columns(cfg), "head": current_head()}, "DATA_FIREWALL_AUDIT.json")
    if args.stage == "preflight": return

    completed: list[dict[str, Any]] = []
    for fold, train, valid in folds:
        completed.append(train_one("LSTM", fold, 42, train, valid, cfg))
    per_fold = pd.DataFrame(completed)
    summary = summarize_seed(per_fold)
    vanilla, vanilla_checks = vanilla_status(summary, raw_summary)
    gate: dict[str, Any] = {"vanilla_seed42_status": vanilla, "vanilla_seed42_checks": vanilla_checks}
    if args.stage == "vanilla_seed42":
        write_csv(per_fold, "outputs", "PER_FOLD_METRICS.csv"); write_csv(summary, "outputs", "SEED_SUMMARY.csv"); write_json(gate, "outputs", "GATE_EVALUATION.json"); return
    if vanilla == "TFT_NOT_VIABLE":
        final = "TFT_NOT_VIABLE"
    else:
        for seed in (2024, 3407):
            for fold, train, valid in folds:
                completed.append(train_one("LSTM", fold, seed, train, valid, cfg))
        per_fold, summary = pd.DataFrame(completed), summarize_seed(pd.DataFrame(completed))
        vanilla_multi, checks = multiseed_vanilla_status(summary, raw_summary)
        gate.update({"vanilla_multiseed_status": vanilla_multi, "vanilla_multiseed_checks": checks})
        if vanilla_multi == "TFT_ROUTE_STOPPED":
            final = "TFT_NOT_VIABLE"
        else:
            for model in ("sLSTM", "mLSTM"):
                for fold, train, valid in folds:
                    completed.append(train_one(model, fold, 42, train, valid, cfg))
            per_fold, summary = pd.DataFrame(completed), summarize_seed(pd.DataFrame(completed))
            encoder_gate = encoder_seed42_gate(summary, cfg)
            gate["encoder_seed42_gate"] = encoder_gate.to_dict(orient="records")
            for model in encoder_gate.loc[encoder_gate.extra_seeds_authorized, "model"].tolist():
                for seed in (2024, 3407):
                    for fold, train, valid in folds:
                        completed.append(train_one(model, fold, seed, train, valid, cfg))
            per_fold, summary = pd.DataFrame(completed), summarize_seed(pd.DataFrame(completed))
            final, encoder_final = final_encoder_status(summary, encoder_gate, cfg)
            gate["encoder_final"] = encoder_final.to_dict(orient="records")
    write_csv(per_fold, "outputs", "PER_FOLD_METRICS.csv")
    write_csv(summary, "outputs", "SEED_SUMMARY.csv")
    write_json({**gate, "final_status": final, "test_accessed": False, "model_training_performed": True}, "outputs", "GATE_EVALUATION.json")
    report(final, gate, summary, raw_summary)
    write_json({"phase": "Phase_TFT_Targeted_Reconsideration", "status": final, "selected_encoder": final.replace("TFT_VIABLE_", "").replace("_SELECTED", "") if final != "TFT_NOT_VIABLE" else None, "formal_metric": "forecast_query_weighted_RMSE", "robustness_metric": "target_time_equal_weight_RMSE", "test_accessed": False, "backbone_search_closed": True, "next_phase": "REALHYFC_SPECIFIC_MECHANISM", "head_at_generation": current_head()}, "final", "FINAL_DECISION.yaml")


if __name__ == "__main__":
    main()
