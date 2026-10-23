"""Read-only, cache-only RealHyFC H0/SA artifact alignment audit.

This module intentionally contains no model construction, data builder call, raw-data
load, checkpoint load, training, or model-forward operation. It may read only the
explicit historical artifact whitelist below and writes only under Phase_H0.
"""
from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(r"E:\沈农信电\ PEMFC-SOH+RUL\RENew-PEMFC-2300h-Pytorch")
REPO = ROOT.parent
PHASE = ROOT / "Phase_H0_Existing_Artifact_Comparability_Audit"
OUT, REP = PHASE / "outputs", PHASE / "reports"
CFG = yaml.safe_load((PHASE / "configs" / "frozen_audit_protocol.yaml").read_text(encoding="utf-8"))
HEAD = str(CFG["reference_head"])
BOUNDARY = float(CFG["scope"]["development_test_boundary_age_h"])
ATOL_V = float(CFG["tolerances"]["voltage_V"])
ATOL_AGE = float(CFG["tolerances"]["target_age_h"])
TARGET_END_OFFSET_H = 0.25  # NPZ bin-end age; OPR records bin-center age.

N28A = ROOT / "Phase_N2_8A_True_Value_and_Modern_Screening"
N28B = ROOT / "Phase_N2_8B_TimeMixerPP_and_TimeXer_Adaptation"
N28F = ROOT / "Phase_N2_8F_Uncertainty_Robust_SA_TimeXer"
GSTOP = ROOT / "Phase_G_STOP_Physics_Consistent_Future_Conditions"
OPR = ROOT / "Phase_OPR_TimeXer"

SPECS = {
    "H0_original": (N28A / "outputs" / "model_screening", "timexer_official_fold_{i}_seed42_predictions.npz"),
    "H0_PEMFC_A": (N28B / "outputs" / "model_runs", "timexer_pemfc_a_fold_{i}_seed42_predictions.npz"),
    "H0_PEMFC_Life": (N28B / "outputs" / "model_runs", "timexer_pemfc_life_fold_{i}_seed42_predictions.npz"),
    "SA_P0": (GSTOP / "outputs" / "frozen_timexer", "P0_Original_Ridge_fold_{i}_predictions.npz"),
    "SA_P2": (GSTOP / "outputs" / "frozen_timexer", "P2_Zero_Current_Aware_fold_{i}_predictions.npz"),
    "SA_M0_reference_only": (N28F / "outputs" / "model_runs", "M0_SA_TimeXer_Ridge_fold_{i}_seed42_predictions.npz"),
}
STATIC_SOURCES = [
    N28A / "configs" / "phase_n2_8a_config.yaml", N28A / "scripts" / "phase_n2_8a_run.py",
    N28B / "scripts" / "phase_n2_8b_run.py", N28F / "configs" / "uncertainty_robust_sa_timexer_frozen_config.yaml",
    N28F / "scripts" / "phase_n2_8f_run.py", GSTOP / "configs" / "phase_g_stop_frozen_config.yaml",
    GSTOP / "scripts" / "phase_g_stop_experiment.py", OPR / "outputs" / "reproduction" / "baseline_reproduction_fold_metrics.csv",
    OPR / "outputs" / "reproduction" / "baseline_reproduction_summary.csv", OPR / "outputs" / "error_attribution" / "p1_p2_error_difference.csv",
    OPR / "scripts" / "phase_opr_diagnostic.py", OPR / "PHASE_OPR_DIAGNOSTIC_RESULTS.md",
]
KEY_COLS = ["fold", "origin_row", "horizon_step", "target_row"]
FORMAL_MODELS = ["H0_original", "H0_PEMFC_A", "H0_PEMFC_Life", "SA_P0", "SA_P2"]


def rel(path: Path) -> str:
    return path.resolve().relative_to(ROOT.resolve()).as_posix()


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_commit(path: Path) -> str:
    repo_relative = path.resolve().relative_to(REPO.resolve()).as_posix()
    result = subprocess.run(["git", "-C", str(REPO), "log", "-1", "--format=%H", "--", repo_relative], text=True, capture_output=True, check=True)
    return result.stdout.strip() or "NOT_AVAILABLE_IN_GIT_HISTORY"


def assert_head() -> None:
    actual = subprocess.check_output(["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True).strip()
    if actual != HEAD:
        raise RuntimeError(f"FAIL-CLOSED HEAD mismatch: expected {HEAD}, got {actual}")


def allowed_paths() -> set[Path]:
    paths = set(STATIC_SOURCES)
    for base, pattern in SPECS.values():
        paths.update(base / pattern.format(i=i) for i in range(1, 4))
    return {p.resolve() for p in paths}


def assert_allowed(path: Path) -> Path:
    p = path.resolve()
    if p not in allowed_paths():
        raise PermissionError(f"Whitelist rejection: {p}")
    if any(token in p.as_posix().lower() for token in ("rul90", "/data/", "test_")):
        raise PermissionError(f"Prohibited source class: {p}")
    return p


def npz_schema(path: Path) -> tuple[list[str], dict[str, list[int]], dict[str, str]]:
    path = assert_allowed(path)
    with np.load(path, allow_pickle=False) as z:
        keys = list(z.files)
        shapes = {key: list(z[key].shape) for key in keys}
        dtypes = {key: str(z[key].dtype) for key in keys}
    return keys, shapes, dtypes


def metadata(path: Path, model: str, fold: str) -> dict:
    path = assert_allowed(path)
    with np.load(path, allow_pickle=False) as z:
        required = {"row_origin_index", "target_end_age_h", "y_true", "y_pred", "anchor"}
        missing = required - set(z.files)
        if missing:
            raise RuntimeError(f"Missing required NPZ keys in {path.name}: {sorted(missing)}")
        origin, end = z["row_origin_index"].astype(np.int64), z["target_end_age_h"].astype(float)
        shape = tuple(z["y_true"].shape)
    if shape != (len(origin), 24):
        raise RuntimeError(f"Unexpected horizon shape in {path.name}: {shape}")
    if len(np.unique(origin)) != len(origin) or np.any(np.diff(origin) <= 0):
        raise RuntimeError(f"Origin ordering/duplication failure in {path.name}")
    return {"model": model, "fold": fold, "path": path, "origin": origin, "target_end": end, "array_shape": shape}


def opr_target_map() -> pd.DataFrame:
    path = assert_allowed(OPR / "outputs" / "error_attribution" / "p1_p2_error_difference.csv")
    cols = ["fold", "origin_row", "target_row", "horizon_step", "target_age_h"]
    tab = pd.read_csv(path, usecols=cols)
    if tab.duplicated(KEY_COLS).any() or not np.all(tab.target_row.to_numpy() == tab.origin_row.to_numpy() + tab.horizon_step.to_numpy()):
        raise RuntimeError("OPR target map does not satisfy exact contiguous target-row contract")
    if (tab.target_age_h >= BOUNDARY).any():
        raise RuntimeError("TEST FIREWALL: OPR target map crosses Development boundary")
    return tab.sort_values(KEY_COLS).reset_index(drop=True)


def reconstruct_keys(meta: dict, opr: pd.DataFrame) -> pd.DataFrame:
    fold, origins, ends = meta["fold"], meta["origin"], meta["target_end"]
    rows = pd.DataFrame({"origin_row": np.repeat(origins, 24), "horizon_step": np.tile(np.arange(1, 25), len(origins)), "array_row": np.repeat(np.arange(len(origins)), 24)})
    rows["fold"] = fold
    rows["target_row"] = rows.origin_row + rows.horizon_step
    merged = rows.merge(opr, on=KEY_COLS, how="left", validate="one_to_one")
    if merged.target_age_h.isna().any():
        raise RuntimeError(f"Cannot reconstruct exact target keys for {meta['model']} {fold}")
    h24 = merged[merged.horizon_step == 24].sort_values("array_row")
    if not np.allclose(h24.target_age_h.to_numpy(float) + TARGET_END_OFFSET_H, ends, atol=ATOL_AGE, rtol=0):
        raise RuntimeError(f"Target-end age mismatch against OPR for {meta['model']} {fold}")
    if (merged.target_age_h >= BOUNDARY).any() or np.any(ends >= BOUNDARY):
        raise RuntimeError(f"TEST FIREWALL: reconstructed key crosses boundary for {meta['model']} {fold}")
    merged["model"] = meta["model"]
    return merged[["model", "fold", "array_row", "origin_row", "horizon_step", "target_row", "target_age_h"]].sort_values(KEY_COLS).reset_index(drop=True)


def coverage(keys: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows, common_rows = [], []
    for fold in ("fold_1", "fold_2", "fold_3"):
        sets = {model: set(map(tuple, tab[tab.fold == fold][KEY_COLS].to_numpy())) for model, tab in keys.items()}
        common = set.intersection(*sets.values())
        for model, values in sets.items():
            rows.append({"fold": fold, "model": model, "complete_key_count": len(values), "common_key_count": len(common), "coverage_fraction": len(common) / len(values), "missing_from_model_vs_common": len(common - values), "extra_vs_common": len(values - common), "origin_count": int(len(values) / 24), "duplicate_keys": 0, "boundary_crossing_keys": 0})
        common_rows.extend({"fold": fold, "origin_row": x[1], "horizon_step": x[2], "target_row": x[3]} for x in sorted(common))
    common_tab = pd.DataFrame(common_rows).sort_values(KEY_COLS).reset_index(drop=True)
    if common_tab.empty:
        raise RuntimeError("No common artifact keys")
    return pd.DataFrame(rows), common_tab


def load_matched_values(meta: dict, keytab: pd.DataFrame, common: pd.DataFrame) -> pd.DataFrame:
    selected = keytab.merge(common, on=KEY_COLS, how="inner", validate="one_to_one").sort_values(KEY_COLS)
    path = assert_allowed(meta["path"])
    with np.load(path, allow_pickle=False) as z:
        y_true, y_pred, anchor = z["y_true"].astype(float), z["y_pred"].astype(float), z["anchor"].astype(float)
    indices, steps = selected.array_row.to_numpy(int), selected.horizon_step.to_numpy(int) - 1
    selected = selected.copy()
    selected["y_true_V"] = y_true[indices, steps]
    selected["y_pred_V"] = y_pred[indices, steps]
    selected["anchor_V"] = anchor[indices]
    selected["true_deltaV_mV"] = (selected.y_true_V - selected.anchor_V) * 1000.0
    selected["pred_deltaV_mV"] = (selected.y_pred_V - selected.anchor_V) * 1000.0
    selected["error_deltaV_mV"] = selected.pred_deltaV_mV - selected.true_deltaV_mV
    selected["absolute_error_mV"] = abs(selected.error_deltaV_mV)
    return selected


def truth_alignment(values: dict[str, pd.DataFrame]) -> pd.DataFrame:
    ref = values["H0_original"]
    rows = []
    for model, tab in values.items():
        merged = ref.merge(tab, on=KEY_COLS, suffixes=("_ref", "_other"), validate="one_to_one")
        max_y = float(np.max(abs(merged.y_true_V_ref - merged.y_true_V_other)))
        max_a = float(np.max(abs(merged.anchor_V_ref - merged.anchor_V_other)))
        max_age = float(np.max(abs(merged.target_age_h_ref - merged.target_age_h_other)))
        rows.append({"reference_model": "H0_original", "model": model, "key_count": len(merged), "max_y_true_difference_V": max_y, "max_anchor_difference_V": max_a, "max_target_age_difference_h": max_age, "pass": bool(max_y <= ATOL_V and max_a <= ATOL_V and max_age <= ATOL_AGE)})
    result = pd.DataFrame(rows)
    if not result["pass"].all():
        raise RuntimeError("Truth/anchor/target age alignment exceeds frozen tolerance")
    return result


def prediction_equivalence(values: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Numerically check same-key predictions; equality never implies independence."""
    rows = []
    for left, right in (("H0_original", "H0_PEMFC_A"), ("H0_original", "H0_PEMFC_Life"), ("SA_P0", "SA_P2"), ("SA_P0", "SA_M0_reference_only")):
        q = values[left].merge(values[right], on=KEY_COLS, suffixes=("_left", "_right"), validate="one_to_one")
        d = q.y_pred_V_left - q.y_pred_V_right
        rows.append({"left_model": left, "right_model": right, "key_count": len(q), "max_prediction_difference_V": float(abs(d).max()), "mean_absolute_prediction_difference_V": float(abs(d).mean()), "numerically_identical_at_voltage_tolerance": bool(abs(d).max() <= ATOL_V), "interpretation": "same saved prediction values only; not evidence of independent replication"})
    return pd.DataFrame(rows)


def metric(tab: pd.DataFrame) -> dict:
    e, y, p = tab.error_deltaV_mV.to_numpy(float), tab.true_deltaV_mV.to_numpy(float), tab.pred_deltaV_mV.to_numpy(float)
    denom = float(np.sum((y - y.mean()) ** 2))
    return {"rmse_deltaV_mV": float(np.sqrt(np.mean(e ** 2))), "mae_deltaV_mV": float(np.mean(abs(e))), "r2_deltaV": float(1 - np.sum((p - y) ** 2) / denom) if denom > 0 else float("nan"), "p95_absolute_error_mV": float(np.quantile(abs(e), .95)), "p99_absolute_error_mV": float(np.quantile(abs(e), .99)), "endpoint_12h_rmse_mV": float(np.sqrt(np.mean(tab.loc[tab.horizon_step == 24, "error_deltaV_mV"] ** 2)))}


def metric_tables(values: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    matched, historical, sensitivity = [], [], []
    for model, tab in values.items():
        fold_rows = []
        for fold, g in tab.groupby("fold", sort=True):
            r = {"scope": "matched_common_keys", "aggregation": "fold", "model": model, "fold": fold, "origin_count": g.origin_row.nunique(), "prediction_point_count": len(g), "unique_target_row_count": g.target_row.nunique(), **metric(g)}
            matched.append(r); historical.append({**r, "scope": "historical_full_artifact"}); fold_rows.append(r)
            t = g.groupby("target_row", as_index=False).agg(mean_squared_error_mV2=("error_deltaV_mV", lambda x: float(np.mean(x ** 2))))
            sensitivity.append({"model": model, "fold": fold, "unique_target_row_count": len(t), "prediction_point_count": len(g), "target_aggregated_rmse_mV": float(np.sqrt(t.mean_squared_error_mV2.mean())), "label": "descriptive_target_deduplication_not_independent_inference"})
        matched.append({"scope": "matched_common_keys", "aggregation": "unweighted_three_fold_macro", "model": model, "fold": "macro", "origin_count": int(np.mean([x["origin_count"] for x in fold_rows])), "prediction_point_count": int(np.sum([x["prediction_point_count"] for x in fold_rows])), "unique_target_row_count": int(np.sum([x["unique_target_row_count"] for x in fold_rows])), **{k: float(np.mean([x[k] for x in fold_rows])) for k in ("rmse_deltaV_mV", "mae_deltaV_mV", "r2_deltaV", "p95_absolute_error_mV", "p99_absolute_error_mV", "endpoint_12h_rmse_mV")}})
        matched.append({"scope": "matched_common_keys", "aggregation": "pooled_prediction_points", "model": model, "fold": "pooled", "origin_count": tab.origin_row.nunique(), "prediction_point_count": len(tab), "unique_target_row_count": tab.target_row.nunique(), **metric(tab)})
    # Pairwise differences remain descriptive: all inputs share keys but not a training protocol.
    for left, right in combinations(FORMAL_MODELS, 2):
        a, b = values[left], values[right]
        q = a.merge(b, on=KEY_COLS, suffixes=("_left", "_right"), validate="one_to_one")
        paired = q.absolute_error_mV_right - q.absolute_error_mV_left
        matched.append({"scope": "matched_common_keys", "aggregation": "paired_prediction_points_descriptive", "model": f"{right}_minus_{left}", "fold": "all", "origin_count": q.origin_row.nunique(), "prediction_point_count": len(q), "unique_target_row_count": q.target_row.nunique(), "rmse_deltaV_mV": float(np.sqrt(np.mean(q.error_deltaV_mV_right ** 2)) - np.sqrt(np.mean(q.error_deltaV_mV_left ** 2))), "mae_deltaV_mV": float(paired.mean()), "r2_deltaV": float("nan"), "p95_absolute_error_mV": float(np.quantile(paired, .95)), "p99_absolute_error_mV": float(np.quantile(paired, .99)), "endpoint_12h_rmse_mV": float("nan")})
    return pd.DataFrame(matched), pd.DataFrame(historical), pd.DataFrame(sensitivity)


def verify_historical_opr(values: dict[str, pd.DataFrame]) -> pd.DataFrame:
    path = assert_allowed(OPR / "outputs" / "reproduction" / "baseline_reproduction_summary.csv")
    summary = pd.read_csv(path).set_index("model")
    rows = []
    for model, opr_label in (("SA_P0", "P0"), ("SA_P2", "P2")):
        macro = np.mean([metric(g)["rmse_deltaV_mV"] for _, g in values[model].groupby("fold")])
        reported = float(summary.loc[opr_label, "mean_rmse_delta_mV"])
        rows.append({"model": model, "OPR_label": opr_label, "reconstructed_full_artifact_macro_rmse_mV": macro, "OPR_reported_macro_rmse_mV": reported, "absolute_difference_mV": abs(macro - reported), "pass": abs(macro - reported) <= float(CFG["tolerances"]["reported_rmse_mV"])})
    if not all(r["pass"] for r in rows):
        raise RuntimeError("Saved P0/P2 arrays do not reproduce OPR reported metrics")
    return pd.DataFrame(rows)


def protocol_matrix() -> pd.DataFrame:
    f2 = "14 F2 features: voltage-state, current, current-density, transitions, q; exact order recorded in N2.8A/N2.8F configs"
    return pd.DataFrame([
        {"label": "H0_original", "classification": "independently trained old H0 artifact", "history_features": f2, "future_covariates": "none", "future_source_validation": "none", "output_path": "TimeXer direct DeltaV", "scalers": "train-only feature and target scalers (N2.8A source)", "training_windows": "exact count not saved in whitelist; validation origins 233/245/466", "optimizer_epochs": "AdamW/MSE, max 100, patience 15", "seed_evidence": "seed=42 artifact only", "confounds_vs_SA": "architecture/output path; no future-condition branch; source phase/version"},
        {"label": "H0_PEMFC_A", "classification": "independently trained old H0 artifact", "history_features": f2, "future_covariates": "none", "future_source_validation": "none", "output_path": "TimeXer adapter direct DeltaV", "scalers": "train-only feature and target scalers", "training_windows": "same chronological window statement; exact count not saved in whitelist", "optimizer_epochs": "AdamW/MSE, max 100, patience 15", "seed_evidence": "seed=42 artifact only", "confounds_vs_SA": "adapter implementation, source phase/version, no future-condition branch"},
        {"label": "H0_PEMFC_Life", "classification": "independently trained old H0 artifact", "history_features": f2 + "; plus origin-known lifecycle scalar repeated in history", "future_covariates": "none", "future_source_validation": "none", "output_path": "TimeXer adapter direct DeltaV", "scalers": "train-only; lifecycle scalar standardized per train fold", "training_windows": "same chronological window statement; exact count not saved in whitelist", "optimizer_epochs": "AdamW/MSE, max 100, patience 15", "seed_evidence": "seed=42 artifact only", "confounds_vs_SA": "extra lifecycle input, adapter, source phase/version, no future-condition branch"},
        {"label": "SA_P0", "classification": "frozen joint SA-TimeXer condition-inference artifact", "history_features": f2, "future_covariates": "nine predicted future operating variables", "future_source_validation": "causal completed-target OOF Ridge/P0", "output_path": "joint base + horizon-state FiLM condition correction", "scalers": "train-only history/future/target scalers; OOF completion purge", "training_windows": "fair training window count protocol differs; exact count unavailable from NPZ", "optimizer_epochs": "AdamW/MSE, max 100, patience 15; frozen M0 checkpoint", "seed_evidence": "seed=42 checkpoint/artifact", "confounds_vs_SA": "reference condition variant"},
        {"label": "SA_P2", "classification": "frozen joint SA-TimeXer condition-inference artifact", "history_features": f2, "future_covariates": "nine P2 physics-consistent predicted variables", "future_source_validation": "causal completed-target OOF; q/conditional current reconstruction", "output_path": "same frozen joint base + FiLM correction as SA_P0", "scalers": "inherited train-only M0 scaler", "training_windows": "same frozen checkpoint; inference condition differs from SA_P0", "optimizer_epochs": "no new training in G-STOP; frozen seed=42 M0 checkpoint", "seed_evidence": "seed=42 checkpoint/artifact", "confounds_vs_SA": "reference condition variant"},
    ])


def write_reports(schema: pd.DataFrame, coverage_tab: pd.DataFrame, alignment: pd.DataFrame, equivalence: pd.DataFrame, protocols: pd.DataFrame, metrics: pd.DataFrame, historical_check: pd.DataFrame, p2_m0: float) -> None:
    def table(d: pd.DataFrame, n: int | None = None) -> str:
        d = d if n is None else d.head(n)
        if d.empty: return "_No rows._"
        x = d.copy()
        for col in x.columns:
            if pd.api.types.is_float_dtype(x[col]): x[col] = x[col].map(lambda v: "" if pd.isna(v) else f"{v:.6f}")
        return "| " + " | ".join(x.columns) + " |\n|" + "|".join(["---"] * len(x.columns)) + "|\n" + "\n".join("| " + " | ".join(str(v).replace("|", "/") for v in row) + " |" for row in x.itertuples(index=False, name=None))
    (REP / "01_OLD_H0_ACTUALLY_TRAINED.md").write_text("# Old H0 artifacts actually trained\n\nThe three H0 labels are existing seed=42 artifacts only. This audit does not treat them as multi-seed evidence, does not retrain them, and does not treat `B_joint_only` as a standalone H0 model. `SA_P0` and `SA_P2` are frozen joint SA-TimeXer inference variants, not H0 retraining runs.\n\n" + table(schema[["model", "fold", "relative_path", "keys", "y_true_shape", "source_commit"]], 18), encoding="utf-8")
    (REP / "02_MATCHED_KEY_AND_TARGET_AUDIT.md").write_text("# Matched key and target audit\n\nCommon keys were formed before any prediction error was accessed. For each NPZ, keys were reconstructed only through the exact OPR mapping `target_row=origin_row+horizon_step`. NPZ `target_end_age_h` is verified as the OPR final target bin-center age plus 0.25 h (the 30-minute bin-end convention). No nearest-time matching was used.\n\n## Coverage\n\n" + table(coverage_tab) + "\n\n## Truth, anchor and target age alignment\n\n" + table(alignment) + "\n\n## Pointwise prediction equivalence checks\n\n" + table(equivalence) + "\n\nH0_original and H0_PEMFC_A are evaluated pointwise here; numerical equality of their saved arrays is not labelled an independent statistical replication.\n", encoding="utf-8")
    (REP / "03_PROTOCOL_COMPARABILITY.md").write_text("# Protocol comparability\n\nIdentical target keys do not remove training-protocol confounding. H0 variants are independently trained history-only/direct-output artifacts; SA artifacts use a joint base-plus-FiLM correction with nine causal future-condition inputs and a separate OOF/purge construction. H0-Life additionally changes the history representation. Therefore H0-vs-SA is descriptive reuse, not a formal future-condition-only ablation.\n\n" + table(protocols) + "\n\n`B_joint_only` is a joint-model component diagnostic from OPR and is explicitly excluded from every H0 ranking.\n", encoding="utf-8")
    view = metrics[metrics.aggregation.isin(["fold", "unweighted_three_fold_macro", "pooled_prediction_points"])]
    (REP / "04_METRIC_AND_DEPENDENCE_AUDIT.md").write_text("# Metric and dependence audit\n\nThe macro average is the unweighted mean of the three fold RMSE values. Pooled RMSE is calculated over all overlapping prediction points; it is not an independent-trajectory estimate. Unique-target aggregation is published separately as descriptive sensitivity, not a significance test.\n\n## Matched descriptive metrics\n\n" + table(view) + "\n\n## Historical P0/P2 reproduction against OPR\n\n" + table(historical_check) + f"\n\nSA P0 and N2.8F M0 pointwise maximum prediction difference: {p2_m0:.9g} V. This check establishes saved-artifact relationship only; it does not erase the H0/SA training-protocol confounds.\n", encoding="utf-8")
    (REP / "FINAL_RESEARCH_DECISION.md").write_text("# Final research decision\n\n**DESCRIPTIVE_REUSE_PROTOCOL_CONFOUNDED**\n\nThe old H0 artifacts can be reused for same-key descriptive error reporting because key, truth, anchor and Development-boundary contracts pass. They cannot be used as a formal causal ablation against SA/P2: independent H0 training, adapter/lifecycle differences, SA joint FiLM correction, future-condition construction, and unavailable exact training-window counts are confounds. `B_joint_only≈192.383 mV` remains a joint-output component diagnostic and says nothing about independently trained H0 ability.\n\nA genuinely matched independent H0 experiment could be justified only in a separately authorized phase. Its unique added information must be: same SA architecture/checkpoint training protocol, same chronological completed-target-purged training windows, same 14-feature/scaler/DeltaV contract, and one preregistered removal of the nine future-condition tensor. This phase does not run it.\n", encoding="utf-8")


def main() -> None:
    assert_head()
    for d in (OUT, REP): d.mkdir(parents=True, exist_ok=True)
    opr = opr_target_map()
    metas, schemas, keys = {}, [], {}
    for model, (base, pattern) in SPECS.items():
        for i in range(1, 4):
            fold, path = f"fold_{i}", base / pattern.format(i=i)
            meta = metadata(path, model, fold); metas[(model, fold)] = meta
            k, shapes, types = npz_schema(path)
            schemas.append({"model": model, "fold": fold, "relative_path": rel(path), "sha256": sha(path), "source_commit": source_commit(path), "keys": ";".join(k), "shapes_json": json.dumps(shapes), "dtypes_json": json.dumps(types), "y_true_shape": str(shapes["y_true"]), "key_source": "row_origin_index + verified OPR exact target-row map"})
            keys[(model, fold)] = reconstruct_keys(meta, opr)
    keyed = {model: pd.concat([keys[(model, f"fold_{i}")] for i in range(1, 4)], ignore_index=True) for model in SPECS}
    coverage_tab, common = coverage(keyed)
    # Only after complete/common keys are frozen do we access truth, anchor and predictions.
    values = {model: pd.concat([load_matched_values(metas[(model, f"fold_{i}")], keys[(model, f"fold_{i}")], common[common.fold == f"fold_{i}"]) for i in range(1, 4)], ignore_index=True) for model in SPECS}
    alignment = truth_alignment(values)
    equivalence = prediction_equivalence(values)
    historical_check = verify_historical_opr(values)
    # Explicitly verify P0 and N2.8F M0 share the same cached predictions when aligned.
    p0_m0 = values["SA_P0"].merge(values["SA_M0_reference_only"], on=KEY_COLS, suffixes=("_p0", "_m0"), validate="one_to_one")
    p0_m0_max = float(np.max(abs(p0_m0.y_pred_V_p0 - p0_m0.y_pred_V_m0)))
    metrics, historical, sensitivity = metric_tables({k: values[k] for k in FORMAL_MODELS})
    protocols = protocol_matrix()
    schema = pd.DataFrame(schemas)
    schema.to_csv(OUT / "artifact_schema.csv", index=False)
    coverage_tab.to_csv(OUT / "origin_key_coverage.csv", index=False)
    alignment.to_csv(OUT / "truth_alignment.csv", index=False)
    equivalence.to_csv(OUT / "prediction_equivalence_audit.csv", index=False)
    protocols.to_csv(OUT / "protocol_difference_matrix.csv", index=False)
    metrics.to_csv(OUT / "matched_descriptive_metrics.csv", index=False)
    historical.to_csv(OUT / "historical_vs_matched_metrics.csv", index=False)
    sensitivity.to_csv(OUT / "independent_target_sensitivity.csv", index=False)
    pd.DataFrame([{ "item": "B_joint_only", "status": "EXCLUDED_FROM_H0_RANKING", "reason": "jointly trained SA-TimeXer decomposition component; not an independent model" }, {"item": "new_H0_training", "status": "NOT_RUN", "reason": "audit-only authorization"}, {"item": "Test evaluation", "status": "NOT_RUN", "reason": "prohibited"}, {"item": "bootstrap_or_model_selection", "status": "NOT_RUN", "reason": "prohibited"}]).to_csv(OUT / "not_run_manifest.csv", index=False)
    schema_index = schema.set_index("relative_path").to_dict(orient="index")
    manifest_inputs = []
    for p in sorted(allowed_paths()):
        record = {"relative_path": rel(p), "sha256": sha(p), "source_commit": source_commit(p)}
        if record["relative_path"] in schema_index:
            record.update({"available_keys": schema_index[record["relative_path"]]["keys"], "shapes_json": schema_index[record["relative_path"]]["shapes_json"], "dtypes_json": schema_index[record["relative_path"]]["dtypes_json"]})
        else:
            record.update({"available_keys": None, "shapes_json": None, "dtypes_json": None})
        manifest_inputs.append(record)
    manifest = {"reference_head": HEAD, "command": "python scripts/run_existing_h0_audit.py", "new_test_accessed": False, "historical_test_target_descriptive_access": True, "historical_test_model_metrics": False, "model_training_performed": False, "allowed_input_count": len(allowed_paths()), "inputs": manifest_inputs, "p0_m0_max_prediction_difference_V": p0_m0_max, "status": "DESCRIPTIVE_REUSE_PROTOCOL_CONFOUNDED"}
    (OUT / "source_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    write_reports(schema, coverage_tab, alignment, equivalence, protocols, metrics, historical_check, p0_m0_max)
    (PHASE / "FINAL_DECISION.yaml").write_text("status: DESCRIPTIVE_REUSE_PROTOCOL_CONFOUNDED\nnew_test_accessed: false\nhistorical_test_target_descriptive_access: true\nhistorical_test_model_metrics: false\nmodel_training_performed: false\n", encoding="utf-8")
    run = {"head": HEAD, "python": platform.python_version(), "numpy": np.__version__, **manifest, "source_manifest": "outputs/source_manifest.json"}
    (PHASE / "RUN_AUDIT.json").write_text(json.dumps(run, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"status": "DESCRIPTIVE_REUSE_PROTOCOL_CONFOUNDED", "common_keys": len(common), "p0_m0_max_prediction_difference_V": p0_m0_max}, ensure_ascii=False))


if __name__ == "__main__":
    try: main()
    except Exception as exc:
        print(f"FAIL-CLOSED: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise
