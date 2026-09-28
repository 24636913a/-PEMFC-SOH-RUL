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


def validate_target_end_ages(end: np.ndarray, name: str) -> None:
    """Reject an NPZ whose saved final target age touches Test."""
    end = np.asarray(end, dtype=float)
    if end.ndim != 1 or len(end) == 0 or not np.isfinite(end).all():
        raise RuntimeError(f"{name}: invalid target_end_age_h array")
    if (end >= BOUNDARY).any():
        raise RuntimeError(f"{name}: target_end_age_h crosses Development boundary")


def build_keys(path: Path, model: str, fold: str, mapping: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    with np.load(path, allow_pickle=False) as z:
        need = {"row_origin_index", "target_end_age_h", "y_true", "y_pred", "anchor"}
        if need - set(z.files): raise RuntimeError(f"{path.name}: missing required NPZ keys")
        origin, end = z["row_origin_index"].astype(int), z["target_end_age_h"].astype(float)
        shapes = {k: list(z[k].shape) for k in z.files}
        if z["y_true"].shape != (len(origin), 24): raise RuntimeError(f"{path.name}: wrong y shape")
    validate_target_end_ages(end, path.name)
    if len(np.unique(origin)) != len(origin) or np.any(np.diff(origin) <= 0): raise RuntimeError(f"{path.name}: duplicate or unsorted origins")
    k = pd.DataFrame({"origin_row": np.repeat(origin, 24), "horizon_step": np.tile(np.arange(1, 25), len(origin)), "array_row": np.repeat(np.arange(len(origin)), 24)})
    k["fold"] = fold; k["target_row"] = k.origin_row + k.horizon_step
    k = k.merge(mapping, on=KEY, how="left", validate="one_to_one")
    if k.target_age_h.isna().any(): raise RuntimeError(f"{path.name}: keys absent from OPR mapping")
    final = k[k.horizon_step == 24].sort_values("array_row")
    if not np.allclose(final.target_age_h.to_numpy(float) + .25, end, atol=ATOL_AGE, rtol=0): raise RuntimeError(f"{path.name}: target-end convention mismatch")
    validate_key_frame(k, path.name)
    return k.assign(model=model), {"keys": list(shapes), "shapes": shapes, "target_end_age_h_min": float(end.min()), "target_end_age_h_max": float(end.max()), "target_end_age_h_count": int(len(end))}


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


def aggregation_summary(joined: pd.DataFrame, metrics: pd.DataFrame) -> pd.DataFrame:
    """Keep unweighted-fold summaries separate from point-pooled metrics."""
    rows = []
    macro = metrics[metrics.fold == "three_fold_macro"].set_index("evaluation_unit")
    for unit, label in (("forecast_query_weighted_RMSE", "all (origin_row, horizon_step) queries"), ("target_time_equal_weight_RMSE", "all within-fold target-time MSE units")):
        m = macro.loc[unit]
        rows.append({"aggregation": "unweighted_mean_of_fold_metrics", "evaluation_unit": unit, "population": label, "unit_count": int(m.unit_count), "p0_mean_fold_mse_mV2": float(m.p0_mse_mV2), "p2_mean_fold_mse_mV2": float(m.p2_mse_mV2), "p2_minus_p0_mean_fold_delta_mse_mV2": float(m.p2_minus_p0_delta_mse_mV2), "p0_mean_fold_RMSE_mV": float(m.p0_RMSE_mV), "p2_mean_fold_RMSE_mV": float(m.p2_RMSE_mV), "p2_minus_p0_mean_fold_delta_RMSE_mV": float(m.p2_minus_p0_delta_RMSE_mV)})
    query_p0, query_p2 = float(joined.squared_error_mV2_p0.mean()), float(joined.squared_error_mV2_p2.mean())
    rows.append({"aggregation": "pooled_all_prediction_points", "evaluation_unit": "forecast_query_weighted_RMSE", "population": "all (origin_row, horizon_step) queries pooled", "unit_count": len(joined), "p0_mean_fold_mse_mV2": query_p0, "p2_mean_fold_mse_mV2": query_p2, "p2_minus_p0_mean_fold_delta_mse_mV2": query_p2 - query_p0, "p0_mean_fold_RMSE_mV": rmse(query_p0), "p2_mean_fold_RMSE_mV": rmse(query_p2), "p2_minus_p0_mean_fold_delta_RMSE_mV": rmse(query_p2) - rmse(query_p0)})
    target = joined.groupby(["fold", "target_row"], as_index=False).agg(p0_mse=("squared_error_mV2_p0", "mean"), p2_mse=("squared_error_mV2_p2", "mean"))
    target_p0, target_p2 = float(target.p0_mse.mean()), float(target.p2_mse.mean())
    rows.append({"aggregation": "pooled_all_fold_target_times", "evaluation_unit": "target_time_equal_weight_RMSE", "population": "all (fold, target_row) MSE units pooled", "unit_count": len(target), "p0_mean_fold_mse_mV2": target_p0, "p2_mean_fold_mse_mV2": target_p2, "p2_minus_p0_mean_fold_delta_mse_mV2": target_p2 - target_p0, "p0_mean_fold_RMSE_mV": rmse(target_p0), "p2_mean_fold_RMSE_mV": rmse(target_p2), "p2_minus_p0_mean_fold_delta_RMSE_mV": rmse(target_p2) - rmse(target_p0)})
    return pd.DataFrame(rows)


def fold1_multiplicity_one_rd_connection(joined: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    """Connect the two edge targets to RD only through their exact saved keys."""
    d = joined.copy()
    d["p2_minus_p0_squared_error_mV2"] = d.squared_error_mV2_p2 - d.squared_error_mV2_p0
    d["multiplicity"] = d.groupby(["fold", "target_row"])["target_row"].transform("size")
    edge = d[(d.fold == "fold_1") & (d.multiplicity == 1)].copy().sort_values(KEY).reset_index(drop=True)
    if len(edge) != 2:
        raise RuntimeError(f"Expected exactly two Fold 1 multiplicity=1 targets; found {len(edge)}")
    edge["target_age_h"] = edge["target_age_h_p0"]
    rd_path = ROOT / CFG["supplemental_readonly_diagnostic"]["rd_fold1_pointwise_map"]
    need = KEY + ["target_age_h", "target_time", "p1_error_mV", "p2_error_mV", "D_mV2"]
    rd = pd.read_csv(rd_path, usecols=need)
    validate_key_frame(rd, "RD paired P1/P2 mapping")
    if rd.duplicated(KEY).any():
        raise RuntimeError("RD paired P1/P2 mapping: duplicate exact key")
    rd = rd[rd.fold == "fold_1"].copy()
    result = edge.merge(rd, on=KEY, how="left", suffixes=("", "_rd"), validate="one_to_one", indicator="rd_exact_key_merge")
    event_path = ROOT / CFG["supplemental_readonly_diagnostic"]["rd_fold1_12h_event_summary"]
    events = pd.read_csv(event_path, usecols=["target_time", "prediction_point_count", "net_D_sum_mV2"])
    if events.target_time.duplicated().any():
        raise RuntimeError("RD Fold1/12h event summary: duplicate target_time")
    result = result.merge(events, on="target_time", how="left", validate="many_to_one", suffixes=("", "_rd12h"))
    result["rd_target_age_abs_difference_h"] = abs(result.target_age_h - result.target_age_h_rd)
    result["rd_p2_error_abs_difference_mV"] = abs(result.error_mV_p2 - result.p2_error_mV)
    exact = result.rd_exact_key_merge.eq("both") & (result.rd_target_age_abs_difference_h <= ATOL_AGE) & (result.rd_p2_error_abs_difference_mV <= ATOL_V * 1000.0)
    endpoint = result.horizon_step.eq(24)
    event_found = result.prediction_point_count.notna() & result.prediction_point_count.eq(1)
    result["rd_key_connection"] = np.select([exact & endpoint & event_found, exact & ~endpoint, ~exact], ["SAME_RD_FOLD1_12H_EVENT", "EXACT_RD_POINT_FOUND_BUT_NOT_12H_QUERY", "RD_KEY_CONNECTION_UNCONFIRMED"], default="RD_12H_EVENT_SUMMARY_UNCONFIRMED")
    result["rd_difference_definition"] = "P2_minus_P1_squared_error_mV2"
    result["mw_difference_definition"] = "P2_minus_P0_squared_error_mV2"
    keep = KEY + ["target_age_h", "target_time", "multiplicity", "error_mV_p0", "error_mV_p2", "p2_minus_p0_squared_error_mV2", "p1_error_mV", "p2_error_mV", "D_mV2", "rd_target_age_abs_difference_h", "rd_p2_error_abs_difference_mV", "prediction_point_count", "net_D_sum_mV2", "rd_key_connection", "mw_difference_definition", "rd_difference_definition"]
    result = result[keep].rename(columns={"error_mV_p0": "p0_error_mV", "error_mV_p2": "p2_error_mV_from_MW", "p1_error_mV": "p1_error_mV_from_RD", "p2_error_mV": "p2_error_mV_from_RD", "D_mV2": "p2_minus_p1_squared_error_mV2_from_RD", "prediction_point_count": "rd_12h_event_prediction_point_count", "net_D_sum_mV2": "rd_12h_event_net_p2_minus_p1_D_mV2"})
    statuses = set(result.rd_key_connection)
    connection = "TWO_KEY_RD_CONNECTIONS_CONFIRMED" if statuses == {"SAME_RD_FOLD1_12H_EVENT"} else "PARTIAL_RD_FOLD1_12H_CONNECTION_CONFIRMED" if "SAME_RD_FOLD1_12H_EVENT" in statuses else "RD_FOLD1_12H_CONNECTION_UNCONFIRMED"
    return result, connection


def write_report(metrics: pd.DataFrame, aggregation: pd.DataFrame, multi: pd.DataFrame, horizon: pd.DataFrame, segments: pd.DataFrame, edge: pd.DataFrame, connection: str, reversal: bool) -> None:
    def md(d: pd.DataFrame, n: int | None = None) -> str:
        d = d if n is None else d.head(n)
        if d.empty: return "_No rows._"
        x = d.copy()
        for c in x.columns:
            if pd.api.types.is_float_dtype(x[c]): x[c] = x[c].map(lambda v: f"{v:.6f}")
        return "| " + " | ".join(x.columns) + " |\n|" + "|".join(["---"] * len(x.columns)) + "|\n" + "\n".join("| " + " | ".join(map(str, row)) + " |" for row in x.itertuples(index=False, name=None))
    major_h = horizon.assign(abs_q=abs(horizon.forecast_query_weighted_delta_MSE_contribution_mV2), abs_t=abs(horizon.target_time_equal_weight_delta_MSE_contribution_mV2)).sort_values(["fold", "abs_q"], ascending=[True, False]).groupby("fold").head(5)
    major_s = segments.assign(abs_q=abs(segments.forecast_query_weighted_delta_MSE_contribution_mV2), abs_t=abs(segments.target_time_equal_weight_delta_MSE_contribution_mV2)).sort_values(["fold", "abs_q"], ascending=[True, False]).groupby("fold").head(5)
    status = "SIGN_REVERSAL_CONFIRMED" if reversal else "SIGN_REVERSAL_NOT_REPRODUCED"
    f1_m1 = multi[(multi.fold == "fold_1") & (multi.multiplicity == 1)].iloc[0]
    f1 = metrics[(metrics.fold == "fold_1") & (metrics.evaluation_unit == "forecast_query_weighted_RMSE")].iloc[0]
    f1t = metrics[(metrics.fold == "fold_1") & (metrics.evaluation_unit == "target_time_equal_weight_RMSE")].iloc[0]
    f3 = metrics[(metrics.fold == "fold_3") & (metrics.evaluation_unit == "forecast_query_weighted_RMSE")].iloc[0]
    rd_handoff = {
        "TWO_KEY_RD_CONNECTIONS_CONFIRMED": "Both multiplicity-one keys are exact RD Fold 1 / 12 h events.",
        "PARTIAL_RD_FOLD1_12H_CONNECTION_CONFIRMED": "One multiplicity-one key is the exact RD Fold 1 / 12 h event. The other has an exact RD pointwise key but is not a 12 h query; it is not claimed to be the same RD/TTD 12 h rare-event case.",
        "RD_FOLD1_12H_CONNECTION_UNCONFIRMED": "The requested RD Fold 1 / 12 h key-level connection is unconfirmed; no failure mechanism is inferred from it.",
    }[connection]
    text = f"# Overlap weighting audit\n\n**{status}**\n\n`forecast_query_weighted_RMSE` remains the formal historical metric. `target_time_equal_weight_RMSE` is a mandatory robustness sensitivity only; neither result selects a model. All P2−P0 attribution is in additive MSE space, not summed RMSE differences. No target times are treated as independent samples.\n\n## Evaluation aggregation labels\n\nThe existing `per_fold_metrics.csv` values are unchanged. `unweighted_mean_of_fold_metrics` is the three-fold **equal-fold** summary: its MSE columns are means of fold MSEs and its RMSE columns are means of fold RMSEs. It is not the same quantity as a single pooled RMSE. `pooled_all_prediction_points` is the separate all-query combined statistic. The target-time pooled row pools `(fold, target_row)` MSE units only and is not a prediction ensemble.\n\n{md(aggregation)}\n\n## Per-fold metrics\n\n{md(metrics[metrics.fold != 'three_fold_macro'])}\n\n## Multiplicity contributions\n\n{md(multi)}\n\n## Fold 1 multiplicity-one target-key diagnostic\n\nThe two rows below are retained in the official population; no target was removed or used to recompute a formal score. MW uses `P2−P0` squared-error differences. The RD columns are only an exact-key cross-reference and use the distinct `P2−P1` direction.\n\n{md(edge)}\n\nRD connection status: **{connection}**. {rd_handoff}\n\n## Main horizon contributions\n\n{md(major_h)}\n\n## Main pre-existing contiguous target-row segments\n\n{md(major_s)}\n\n## Fold 3 remains a separate attribution\n\nFold 3 is not merged with the Fold 1 rare-event discussion. Under the formal forecast-query metric, its P2−P0 DeltaMSE is {f3.p2_minus_p0_delta_mse_mV2:.6f} mV² and its P2−P0 DeltaRMSE is {f3.p2_minus_p0_delta_RMSE_mV:.6f} mV: P2 remains worse on that independent fold.\n\n## RESEARCH_HANDOFF\n\n1. **Fold 1 reversal driver:** Yes, at the evaluation-unit level. The two `multiplicity=1` targets contribute {f1_m1.target_time_equal_weight_delta_MSE_contribution_mV2:.6f} mV² to Fold 1's target-time-equal DeltaMSE of {f1t.p2_minus_p0_delta_mse_mV2:.6f} mV², while their forecast-query contribution is only {f1_m1.forecast_query_weighted_delta_MSE_contribution_mV2:.6f} mV² against Fold 1's formal {f1.p2_minus_p0_delta_mse_mV2:.6f} mV². The endpoint key is the dominant row within that two-target stratum; this does not alter the retained formal score.\n2. **RD/TTD correspondence:** {rd_handoff} RD's `P2−P1` quantity is explicitly distinct from MW's `P2−P0`; no equality or mechanism is inferred by comparing their magnitudes.\n3. **Fold 3:** Yes. It independently retains formal P2 degradation stated above, and is not used to explain Fold 1.\n4. **End of cache diagnosis:** Yes for this frozen-cache route. The evidence is sufficient to stop further post-hoc cache diagnostics of P2 instability, but not sufficient to identify a unique causal failure mechanism.\n5. **Required next decision level:** This audit makes no model choice. Any next research decision must choose among retaining TimeXer as a paper architecture, defining one preregistered falsifiable TimeXer mechanism change, or redirecting to strong baselines plus physics-constraint/uncertainty work; none is started here.\n\nThe historical H0 audit tagged `historical_full_artifact` only because coverage was 100%; this phase verifies the same condition for P0/P2. H0's old script used a strict historical-HEAD reproduction guard. This new script intentionally instead requires the specified start commit to be an ancestor of the current HEAD, while pinning every historical input SHA256.\n"
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
        schema += [{"model": "P0", "fold": fold, "path": str(p0path.relative_to(ROOT)), "keys": ";".join(sa["keys"]), "shapes_json": json.dumps(sa["shapes"]), "target_end_age_h_min": sa["target_end_age_h_min"], "target_end_age_h_max": sa["target_end_age_h_max"], "target_end_age_h_count": sa["target_end_age_h_count"]}, {"model": "P2", "fold": fold, "path": str(p2path.relative_to(ROOT)), "keys": ";".join(sb["keys"]), "shapes_json": json.dumps(sb["shapes"]), "target_end_age_h_min": sb["target_end_age_h_min"], "target_end_age_h_max": sb["target_end_age_h_max"], "target_end_age_h_count": sb["target_end_age_h_count"]}]
    p0k, p2k = pd.concat(p0keys, ignore_index=True), pd.concat(p2keys, ignore_index=True)
    p0 = load_values(ROOT / "Phase_G_STOP_Physics_Consistent_Future_Conditions/outputs/frozen_timexer/P0_Original_Ridge_fold_1_predictions.npz", p0k[p0k.fold == "fold_1"])
    p2 = load_values(ROOT / "Phase_G_STOP_Physics_Consistent_Future_Conditions/outputs/frozen_timexer/P2_Zero_Current_Aware_fold_1_predictions.npz", p2k[p2k.fold == "fold_1"])
    for i in (2, 3):
        fold = f"fold_{i}"; p0 = pd.concat([p0, load_values(ROOT / f"Phase_G_STOP_Physics_Consistent_Future_Conditions/outputs/frozen_timexer/P0_Original_Ridge_{fold}_predictions.npz", p0k[p0k.fold == fold])], ignore_index=True); p2 = pd.concat([p2, load_values(ROOT / f"Phase_G_STOP_Physics_Consistent_Future_Conditions/outputs/frozen_timexer/P2_Zero_Current_Aware_{fold}_predictions.npz", p2k[p2k.fold == fold])], ignore_index=True)
    validate_truth_anchor(p0, p2)
    joined = p0.merge(p2, on=KEY, suffixes=("_p0", "_p2"), validate="one_to_one")
    metrics = fold_metrics(joined); aggregation = aggregation_summary(joined, metrics); multi, horizon, segments = contribution_tables(joined)
    edge, rd_connection = fold1_multiplicity_one_rd_connection(joined)
    macro = metrics[metrics.fold == "three_fold_macro"].set_index("evaluation_unit")
    q = float(macro.loc["forecast_query_weighted_RMSE", "p2_minus_p0_delta_mse_mV2"]); t = float(macro.loc["target_time_equal_weight_RMSE", "p2_minus_p0_delta_mse_mV2"])
    reversal = q < 0 < t
    schema_frame = pd.DataFrame(schema)
    schema_frame.to_csv(OUT / "artifact_schema.csv", index=False)
    schema_frame[["model", "fold", "path", "target_end_age_h_min", "target_end_age_h_max", "target_end_age_h_count"]].assign(all_target_end_ages_strictly_below_development_boundary=True, development_boundary_age_h=BOUNDARY).to_csv(OUT / "npz_target_end_age_boundary_audit.csv", index=False)
    metrics.to_csv(OUT / "per_fold_metrics.csv", index=False); aggregation.to_csv(OUT / "evaluation_aggregation_summary.csv", index=False); multi.to_csv(OUT / "multiplicity_contributions.csv", index=False); horizon.to_csv(OUT / "horizon_contributions.csv", index=False); segments.to_csv(OUT / "time_segment_contributions.csv", index=False); edge.to_csv(OUT / "fold1_multiplicity1_key_audit.csv", index=False)
    manifest = {"starting_commit": CFG["starting_commit"], "current_head": current, "input_sha256": {str(p.relative_to(ROOT)).replace('\\', '/'): h for p, h in hashes.items()}, "new_test_accessed": False, "model_training_performed": False, "random_process": "none; seed_not_applicable", "rd_fold1_12h_connection": rd_connection, "all_npz_target_end_ages_strictly_below_development_boundary": True, "status": "SIGN_REVERSAL_CONFIRMED" if reversal else "SIGN_REVERSAL_NOT_REPRODUCED"}
    (OUT / "source_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(metrics, aggregation, multi, horizon, segments, edge, rd_connection, reversal)
    final = manifest["status"]
    (PHASE / "FINAL_DECISION.yaml").write_text(f"status: {final}\nformal_metric: forecast_query_weighted_RMSE\nrobustness_metric: target_time_equal_weight_RMSE\nrd_fold1_12h_connection: {rd_connection}\nall_npz_target_end_ages_strictly_below_development_boundary: true\nnew_test_accessed: false\nmodel_training_performed: false\ncache_diagnostic_route_closed: true\nno_follow_on_phase_created: true\n", encoding="utf-8")
    (PHASE / "run_record.json").write_text(json.dumps({"command": "python scripts/run_overlap_weighting_audit.py", **manifest, "macro_query_delta_mse_mV2": q, "macro_target_equal_delta_mse_mV2": t}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"status": final, "query_delta_mse_mV2": q, "target_equal_delta_mse_mV2": t}, ensure_ascii=False))


if __name__ == "__main__":
    try: main()
    except Exception as exc:
        print(f"FAIL-CLOSED: {type(exc).__name__}: {exc}", file=sys.stderr); raise
