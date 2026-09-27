"""Cache-only P0/P2 overlap-weighting sensitivity audit.

No raw data, Test artifacts, models, checkpoints, training, or random draws are
loaded. All input paths are frozen by config SHA256 and every key is required.
"""
from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(r"E:\沈农信电\ PEMFC-SOH+RUL\RENew-PEMFC-2300h-Pytorch")
REPO = ROOT.parent
PHASE = ROOT / "Phase_MW_Overlap_Weighting_Audit"
OUT = PHASE / "outputs"
CFG = yaml.safe_load((PHASE / "configs" / "frozen_config.yaml").read_text(encoding="utf-8"))
BOUNDARY = float(CFG["scope"]["development_boundary_age_h"])
ATOL_V = float(CFG["tolerances"]["voltage_V"])
ATOL_AGE = float(CFG["tolerances"]["target_age_h"])
ATOL_DECOMP = float(CFG["tolerances"]["decomposition_mse"])
KEY = ["fold", "origin_row", "horizon_step", "target_row"]


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def expected_paths() -> dict[Path, str]:
    return {ROOT / relative: value.lower() for relative, value in CFG["input_sha256"].items()}


def validate_starting_commit() -> str:
    """Starting commit must be an ancestor, permitting execution at current HEAD."""
    current = subprocess.check_output(["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True).strip()
    result = subprocess.run(["git", "-C", str(REPO), "merge-base", "--is-ancestor", str(CFG["starting_commit"]), current], capture_output=True)
    if result.returncode != 0:
        raise RuntimeError(f"Starting commit {CFG['starting_commit']} is not an ancestor of current HEAD {current}")
    return current


def validate_input_hashes(observed: dict[Path, str] | None = None) -> dict[Path, str]:
    expected = expected_paths(); actual = observed or {path: sha(path) for path in expected}
    if set(actual) != set(expected):
        raise RuntimeError("Input path set differs from frozen whitelist")
    bad = [str(path) for path in expected if actual[path].lower() != expected[path]]
    if bad: raise RuntimeError("Input SHA256 mismatch: " + "; ".join(bad))
    return actual


def opr_map() -> pd.DataFrame:
    path = ROOT / "Phase_OPR_TimeXer" / "outputs" / "error_attribution" / "p1_p2_error_difference.csv"
    tab = pd.read_csv(path, usecols=["fold", "origin_row", "target_row", "horizon_step", "target_age_h"])
    validate_key_frame(tab, "OPR mapping")
    return tab.sort_values(KEY).reset_index(drop=True)


def validate_key_frame(tab: pd.DataFrame, name: str) -> None:
    if set(KEY) - set(tab.columns): raise RuntimeError(f"{name}: missing key column")
    if tab.duplicated(KEY).any(): raise RuntimeError(f"{name}: duplicate key")
    if not np.all(tab.target_row.to_numpy(int) == tab.origin_row.to_numpy(int) + tab.horizon_step.to_numpy(int)):
        raise RuntimeError(f"{name}: target_row relation failure")
    if "target_age_h" in tab.columns and (tab.target_age_h.to_numpy(float) >= BOUNDARY).any():
        raise RuntimeError(f"{name}: target crosses Development boundary")


def build_keys(path: Path, model: str, fold: str, mapping: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    with np.load(path, allow_pickle=False) as z:
        need = {"row_origin_index", "target_end_age_h", "y_true", "y_pred", "anchor"}
        if need - set(z.files): raise RuntimeError(f"{path.name}: missing required NPZ keys")
        origin, end = z["row_origin_index"].astype(int), z["target_end_age_h"].astype(float)
        shapes = {k: list(z[k].shape) for k in z.files}
        if z["y_true"].shape != (len(origin), 24): raise RuntimeError(f"{path.name}: wrong y shape")
    if len(np.unique(origin)) != len(origin) or np.any(np.diff(origin) <= 0): raise RuntimeError(f"{path.name}: duplicate or unsorted origins")
    k = pd.DataFrame({"origin_row": np.repeat(origin, 24), "horizon_step": np.tile(np.arange(1, 25), len(origin)), "array_row": np.repeat(np.arange(len(origin)), 24)})
    k["fold"] = fold; k["target_row"] = k.origin_row + k.horizon_step
    k = k.merge(mapping, on=KEY, how="left", validate="one_to_one")
    if k.target_age_h.isna().any(): raise RuntimeError(f"{path.name}: keys absent from OPR mapping")
    final = k[k.horizon_step == 24].sort_values("array_row")
    if not np.allclose(final.target_age_h.to_numpy(float) + .25, end, atol=ATOL_AGE, rtol=0): raise RuntimeError(f"{path.name}: target-end convention mismatch")
    validate_key_frame(k, path.name)
    return k.assign(model=model), {"keys": list(shapes), "shapes": shapes}


def validate_complete_alignment(p0: pd.DataFrame, p2: pd.DataFrame, expected: pd.DataFrame) -> None:
    validate_key_frame(p0, "P0"); validate_key_frame(p2, "P2"); validate_key_frame(expected, "expected")
    p0s, p2s, exps = set(map(tuple, p0[KEY].to_numpy())), set(map(tuple, p2[KEY].to_numpy())), set(map(tuple, expected[KEY].to_numpy()))
    if p0s != p2s or p0s != exps: raise RuntimeError("P0/P2/expected keys are not an identical complete set")


def load_values(path: Path, keys: pd.DataFrame) -> pd.DataFrame:
    with np.load(path, allow_pickle=False) as z:
        yt, yp, anchor = z["y_true"].astype(float), z["y_pred"].astype(float), z["anchor"].astype(float)
    q = keys.sort_values(KEY).copy(); ix, h = q.array_row.to_numpy(int), q.horizon_step.to_numpy(int) - 1
    q["y_true_V"] = yt[ix, h]; q["y_pred_V"] = yp[ix, h]; q["anchor_V"] = anchor[ix]
    q["error_mV"] = (q.y_pred_V - q.y_true_V) * 1000.0
    q["squared_error_mV2"] = q.error_mV ** 2
    return q


def validate_truth_anchor(p0: pd.DataFrame, p2: pd.DataFrame) -> None:
    q = p0.merge(p2, on=KEY, suffixes=("_p0", "_p2"), validate="one_to_one")
    if np.max(abs(q.y_true_V_p0 - q.y_true_V_p2)) > ATOL_V: raise RuntimeError("P0/P2 true values misaligned")
    if np.max(abs(q.anchor_V_p0 - q.anchor_V_p2)) > ATOL_V: raise RuntimeError("P0/P2 anchors misaligned")
    if np.max(abs(q.target_age_h_p0 - q.target_age_h_p2)) > ATOL_AGE: raise RuntimeError("P0/P2 target ages misaligned")


def rmse(mse: float) -> float: return float(np.sqrt(mse))


def fold_metrics(joined: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for fold, g in joined.groupby("fold", sort=True):
        target = g.groupby("target_row", as_index=False).agg(p0_mse=("squared_error_mV2_p0", "mean"), p2_mse=("squared_error_mV2_p2", "mean"))
        for label, p0m, p2m, n in (("forecast_query_weighted_RMSE", g.squared_error_mV2_p0.mean(), g.squared_error_mV2_p2.mean(), len(g)), ("target_time_equal_weight_RMSE", target.p0_mse.mean(), target.p2_mse.mean(), len(target))):
            rows.append({"fold": fold, "evaluation_unit": label, "unit_count": n, "p0_mse_mV2": float(p0m), "p2_mse_mV2": float(p2m), "p2_minus_p0_delta_mse_mV2": float(p2m - p0m), "p0_RMSE_mV": rmse(p0m), "p2_RMSE_mV": rmse(p2m), "p2_minus_p0_delta_RMSE_mV": rmse(p2m) - rmse(p0m)})
    out = pd.DataFrame(rows)
    for unit, g in out.groupby("evaluation_unit", sort=False):
        rows.append({"fold": "three_fold_macro", "evaluation_unit": unit, "unit_count": int(g.unit_count.sum()), **{c: float(g[c].mean()) for c in ("p0_mse_mV2", "p2_mse_mV2", "p2_minus_p0_delta_mse_mV2", "p0_RMSE_mV", "p2_RMSE_mV", "p2_minus_p0_delta_RMSE_mV")}})
    return pd.DataFrame(rows)


def segment_ids(tab: pd.DataFrame) -> pd.DataFrame:
    targets = tab[["fold", "target_row"]].drop_duplicates().sort_values(["fold", "target_row"]).copy()
    targets["new_segment"] = targets.groupby("fold").target_row.diff().fillna(1).ne(1)
    targets["segment_index"] = targets.groupby("fold").new_segment.cumsum().astype(int)
    targets["time_segment"] = targets.fold + "_segment_" + targets.segment_index.astype(str)
    return targets[["fold", "target_row", "time_segment"]]


def contribution_tables(joined: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    d = joined.copy(); d["d_mV2"] = d.squared_error_mV2_p2 - d.squared_error_mV2_p0
    d["multiplicity"] = d.groupby(["fold", "target_row"])["d_mV2"].transform("size")
    d = d.merge(segment_ids(d), on=["fold", "target_row"], validate="many_to_one")
    rows = []
    for fold, g in d.groupby("fold", sort=True):
        n_query, n_target = len(g), g.target_row.nunique()
        for m, x in g.groupby("multiplicity", sort=True):
            rows.append({"fold": fold, "multiplicity": int(m), "query_point_count": len(x), "target_time_count": x.target_row.nunique(), "forecast_query_weighted_delta_MSE_contribution_mV2": float(x.d_mV2.sum() / n_query), "target_time_equal_weight_delta_MSE_contribution_mV2": float((x.d_mV2 / x.multiplicity).sum() / n_target)})
    multi = pd.DataFrame(rows)
    def pieces(group_cols: list[str], name_cols: list[str]) -> pd.DataFrame:
        out = []
        for fold, g in d.groupby("fold", sort=True):
            n_query, n_target = len(g), g.target_row.nunique()
            for key, x in g.groupby(group_cols, sort=True):
                key = key if isinstance(key, tuple) else (key,)
                rec = {"fold": fold, **dict(zip(name_cols, key)), "query_point_count": len(x), "target_time_count": x.target_row.nunique(), "forecast_query_weighted_delta_MSE_contribution_mV2": float(x.d_mV2.sum() / n_query), "target_time_equal_weight_delta_MSE_contribution_mV2": float((x.d_mV2 / x.multiplicity).sum() / n_target)}
                out.append(rec)
        return pd.DataFrame(out)
    horizon = pieces(["horizon_step"], ["horizon_step"])
    segments = pieces(["time_segment"], ["time_segment"])
    validate_decomposition(d, multi, horizon, segments)
    return multi, horizon, segments


def validate_decomposition(d: pd.DataFrame, multi: pd.DataFrame, horizon: pd.DataFrame, segments: pd.DataFrame) -> None:
    for fold, g in d.groupby("fold", sort=True):
        direct_q = float(g.d_mV2.mean()); direct_t = float(g.groupby("target_row").d_mV2.mean().mean())
        for table, name in ((multi, "multiplicity"), (horizon, "horizon"), (segments, "segment")):
            x = table[table.fold == fold]
            q = float(x.forecast_query_weighted_delta_MSE_contribution_mV2.sum()); t = float(x.target_time_equal_weight_delta_MSE_contribution_mV2.sum())
            if abs(q - direct_q) > ATOL_DECOMP or abs(t - direct_t) > ATOL_DECOMP: raise RuntimeError(f"{name} DeltaMSE decomposition does not sum for {fold}")


def write_report(metrics: pd.DataFrame, multi: pd.DataFrame, horizon: pd.DataFrame, segments: pd.DataFrame, reversal: bool) -> None:
    def md(d: pd.DataFrame, n: int | None = None) -> str:
        d = d if n is None else d.head(n)
        if d.empty: return "_No rows._"
        x = d.copy()
        for c in x.columns:
            if pd.api.types.is_float_dtype(x[c]): x[c] = x[c].map(lambda v: f"{v:.6f}")
        return "| " + " | ".join(x.columns) + " |\n|" + "|".join(["---"] * len(x.columns)) + "|\n" + "\n".join("| " + " | ".join(map(str, row)) + " |" for row in x.itertuples(index=False, name=None))
    macro = metrics[metrics.fold == "three_fold_macro"]
    major_h = horizon.assign(abs_q=abs(horizon.forecast_query_weighted_delta_MSE_contribution_mV2), abs_t=abs(horizon.target_time_equal_weight_delta_MSE_contribution_mV2)).sort_values(["fold", "abs_q"], ascending=[True, False]).groupby("fold").head(5)
    major_s = segments.assign(abs_q=abs(segments.forecast_query_weighted_delta_MSE_contribution_mV2), abs_t=abs(segments.target_time_equal_weight_delta_MSE_contribution_mV2)).sort_values(["fold", "abs_q"], ascending=[True, False]).groupby("fold").head(5)
    status = "SIGN_REVERSAL_CONFIRMED" if reversal else "SIGN_REVERSAL_NOT_REPRODUCED"
    text = f"# Overlap weighting audit\n\n**{status}**\n\n`forecast_query_weighted_RMSE` remains the formal historical metric. `target_time_equal_weight_RMSE` is a mandatory robustness sensitivity only; neither result selects a model. All P2−P0 attribution is in additive MSE space, not summed RMSE differences. No target times are treated as independent samples.\n\n## Three-fold macro metrics\n\n{md(macro)}\n\n## Per-fold metrics\n\n{md(metrics[metrics.fold != 'three_fold_macro'])}\n\n## Multiplicity contributions\n\n{md(multi)}\n\n## Main horizon contributions\n\n{md(major_h)}\n\n## Main pre-existing contiguous target-row segments\n\n{md(major_s)}\n\nThe historical H0 audit tagged `historical_full_artifact` only because coverage was 100%; this phase verifies the same condition for P0/P2. H0's old script used a strict historical-HEAD reproduction guard. This new script intentionally instead requires the specified start commit to be an ancestor of the current HEAD, while pinning every historical input SHA256.\n"
    (PHASE / "AUDIT_REPORT.md").write_text(text, encoding="utf-8")


def main() -> None:
    current = validate_starting_commit(); hashes = validate_input_hashes(); OUT.mkdir(parents=True, exist_ok=True)
    mapping = opr_map(); p0keys, p2keys, schema = [], [], []
    for i in range(1, 4):
        fold = f"fold_{i}"
        p0path = ROOT / f"Phase_G_STOP_Physics_Consistent_Future_Conditions/outputs/frozen_timexer/P0_Original_Ridge_{fold}_predictions.npz"
        p2path = ROOT / f"Phase_G_STOP_Physics_Consistent_Future_Conditions/outputs/frozen_timexer/P2_Zero_Current_Aware_{fold}_predictions.npz"
        a, sa = build_keys(p0path, "P0", fold, mapping); b, sb = build_keys(p2path, "P2", fold, mapping)
        expected = mapping[mapping.fold == fold]
        validate_complete_alignment(a, b, expected)
        p0keys.append(a); p2keys.append(b)
        schema += [{"model": "P0", "fold": fold, "path": str(p0path.relative_to(ROOT)), "keys": ";".join(sa["keys"]), "shapes_json": json.dumps(sa["shapes"])}, {"model": "P2", "fold": fold, "path": str(p2path.relative_to(ROOT)), "keys": ";".join(sb["keys"]), "shapes_json": json.dumps(sb["shapes"])}]
    p0k, p2k = pd.concat(p0keys, ignore_index=True), pd.concat(p2keys, ignore_index=True)
    p0 = load_values(ROOT / "Phase_G_STOP_Physics_Consistent_Future_Conditions/outputs/frozen_timexer/P0_Original_Ridge_fold_1_predictions.npz", p0k[p0k.fold == "fold_1"])
    p2 = load_values(ROOT / "Phase_G_STOP_Physics_Consistent_Future_Conditions/outputs/frozen_timexer/P2_Zero_Current_Aware_fold_1_predictions.npz", p2k[p2k.fold == "fold_1"])
    for i in (2, 3):
        fold = f"fold_{i}"; p0 = pd.concat([p0, load_values(ROOT / f"Phase_G_STOP_Physics_Consistent_Future_Conditions/outputs/frozen_timexer/P0_Original_Ridge_{fold}_predictions.npz", p0k[p0k.fold == fold])], ignore_index=True); p2 = pd.concat([p2, load_values(ROOT / f"Phase_G_STOP_Physics_Consistent_Future_Conditions/outputs/frozen_timexer/P2_Zero_Current_Aware_{fold}_predictions.npz", p2k[p2k.fold == fold])], ignore_index=True)
    validate_truth_anchor(p0, p2)
    joined = p0.merge(p2, on=KEY, suffixes=("_p0", "_p2"), validate="one_to_one")
    metrics = fold_metrics(joined); multi, horizon, segments = contribution_tables(joined)
    macro = metrics[metrics.fold == "three_fold_macro"].set_index("evaluation_unit")
    q = float(macro.loc["forecast_query_weighted_RMSE", "p2_minus_p0_delta_mse_mV2"]); t = float(macro.loc["target_time_equal_weight_RMSE", "p2_minus_p0_delta_mse_mV2"])
    reversal = q < 0 < t
    pd.DataFrame(schema).to_csv(OUT / "artifact_schema.csv", index=False)
    metrics.to_csv(OUT / "per_fold_metrics.csv", index=False); multi.to_csv(OUT / "multiplicity_contributions.csv", index=False); horizon.to_csv(OUT / "horizon_contributions.csv", index=False); segments.to_csv(OUT / "time_segment_contributions.csv", index=False)
    manifest = {"starting_commit": CFG["starting_commit"], "current_head": current, "input_sha256": {str(p.relative_to(ROOT)).replace('\\', '/'): h for p, h in hashes.items()}, "new_test_accessed": False, "model_training_performed": False, "random_process": "none; seed_not_applicable", "status": "SIGN_REVERSAL_CONFIRMED" if reversal else "SIGN_REVERSAL_NOT_REPRODUCED"}
    (OUT / "source_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(metrics, multi, horizon, segments, reversal)
    final = manifest["status"]
    (PHASE / "FINAL_DECISION.yaml").write_text(f"status: {final}\nformal_metric: forecast_query_weighted_RMSE\nrobustness_metric: target_time_equal_weight_RMSE\nnew_test_accessed: false\nmodel_training_performed: false\n", encoding="utf-8")
    (PHASE / "run_record.json").write_text(json.dumps({"command": "python scripts/run_overlap_weighting_audit.py", **manifest, "macro_query_delta_mse_mV2": q, "macro_target_equal_delta_mse_mV2": t}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"status": final, "query_delta_mse_mV2": q, "target_equal_delta_mse_mV2": t}, ensure_ascii=False))


if __name__ == "__main__":
    try: main()
    except Exception as exc:
        print(f"FAIL-CLOSED: {type(exc).__name__}: {exc}", file=sys.stderr); raise
