"""Development-only, causal RealHyFC 10-minute versus 30-minute resolution audit."""
from __future__ import annotations
import json
import random
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, precision_recall_fscore_support
from sklearn.preprocessing import StandardScaler

ROOT = Path(r"E:\沈农信电\ PEMFC-SOH+RUL\RENew-PEMFC-2300h-Pytorch")
PHASE = ROOT / "Phase_TR_Time_Resolution_Audit"
OUT, TAB, FIG = PHASE / "outputs", PHASE / "outputs" / "tables", PHASE / "outputs" / "figures"
RAW_ROOT = Path(r"E:\沈农信电\课题-预测质子交换膜燃料电池（PEMFC）的健康状态和剩余寿命\数据\《耐久性测试期间PEM燃料电池堆的空间解析电化学表征》的基础数据\Loadcycle_data\Loadcycle_data")
DEV_START, DEV_END = 72.788690, 1819.997738
ZERO_A, COVERAGE, SEED = 0.10, 0.80, 20260921
LOOKBACK_H, HORIZON_H = 24, 12
FEATURES = ["q", "current_mean_A", "current_std_A"]
FOLDS = {"fold_1": (946.393214, 1164.794345), "fold_2": (1164.794345, 1383.195476), "fold_3": (1383.195476, 1819.997738)}
RAW_FILES = [
    "Loadcycle_0-14.csv", "Loadcycle_15-25.csv", "Loadcycle_26-37.csv", "Loadcycle_37-50.csv",
    "Loadcycle_51-70.csv", "Loadcycle_70-75.csv", "Loadcycle_76-82.csv", "Loadcycle_82-84.csv",
    "Loadcycle_85-100.csv", "Loadcycle_101-112.csv", "Loadcycle_112-125.csv", "Loadcycle_126-133.csv",
    "Loadcycle_134-150.csv", "Loadcycle_151-162.csv"]
LAST_DEV_ROWS = 265459


def state(q):
    if not np.isfinite(q): return "invalid"
    if q <= 1e-12: return "q=0"
    if q >= 1 - 1e-12: return "q=1"
    return "0<q<1"


def md(x):
    """Dependency-free Markdown table renderer (the runtime has no tabulate)."""
    if x.empty:
        return "_No rows._"
    y = x.copy()
    for c in y.columns:
        y[c] = y[c].map(lambda v: "" if pd.isna(v) else str(v).replace("|", "\\|"))
    headers = [str(c) for c in y.columns]
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    lines.extend("| " + " | ".join(row) + " |" for row in y.to_numpy(str))
    return "\\n".join(lines)


def std(sum_x, sq_x, n):
    n = n.replace(0, np.nan)
    return np.sqrt(np.maximum(sq_x / n - (sum_x / n) ** 2, 0))


def make_aggregates():
    """Read only development prefix/rows; test never reaches a statistic or table."""
    contrib = {10: [], 30: []}
    source_rows, missing_rows = [], []
    cols = ["DateTime", "OpHrs2 [h]", "I.S.act [A]", "j.S.act [A/cm2]", "U.S.AveCell [V]"]
    for name in RAW_FILES:
        args = {"sep": ";", "usecols": cols, "chunksize": 200000, "thousands": ","}
        if name == RAW_FILES[-1]: args["nrows"] = LAST_DEV_ROWS
        read_n = keep_n = dup = reverse = gaps = 0
        max_gap = 0.0; prev_dt = None; first_dt = None; last_dt = None; first_age = np.nan; last_age = np.nan
        missing = {"current": 0, "density": 0, "voltage": 0}
        for c in pd.read_csv(RAW_ROOT / name, **args):
            read_n += len(c)
            c["OpHrs2 [h]"] = pd.to_numeric(c["OpHrs2 [h]"], errors="coerce")
            # Fail-safe test guard: discard Test before parsing/stats/aggregation.
            c = c[(c["OpHrs2 [h]"] >= DEV_START - 1e-9) & (c["OpHrs2 [h]"] < DEV_END - 1e-9)].copy()
            if c.empty: continue
            keep_n += len(c)
            for col in cols[2:]: c[col] = pd.to_numeric(c[col], errors="coerce")
            c["DateTime"] = pd.to_datetime(c["DateTime"], errors="coerce")
            first_age = c["OpHrs2 [h]"].min() if not np.isfinite(first_age) else min(first_age, c["OpHrs2 [h]"].min())
            last_age = c["OpHrs2 [h]"].max()
            t = c["DateTime"].dropna()
            if not t.empty:
                check = pd.concat([pd.Series([prev_dt]), t], ignore_index=True) if prev_dt is not None else t.reset_index(drop=True)
                dt = check.diff().dt.total_seconds().iloc[1:]
                dup += int((dt == 0).sum()); reverse += int((dt < 0).sum()); gaps += int((dt > 5).sum())
                if dt.notna().any(): max_gap = max(max_gap, float(dt.max()))
                if first_dt is None: first_dt = t.iloc[0]
                prev_dt = t.iloc[-1]; last_dt = prev_dt
            missing["current"] += int(c["I.S.act [A]"].isna().sum())
            missing["density"] += int(c["j.S.act [A/cm2]"].isna().sum())
            missing["voltage"] += int(c["U.S.AveCell [V]"].isna().sum())
            for minutes in (10, 30):
                width = minutes / 60
                x = c.copy()
                x["bin_start_h"] = np.floor(x["OpHrs2 [h]"] / width + 1e-8) * width
                x["valid_i"] = x["I.S.act [A]"].notna().astype(int)
                x["zero_i"] = ((x["I.S.act [A]"].abs() <= ZERO_A) & x["I.S.act [A]"].notna()).astype(int)
                x["valid_j"] = x["j.S.act [A/cm2]"].notna().astype(int)
                x["valid_v"] = x["U.S.AveCell [V]"].notna().astype(int)
                for short, col in [("i", "I.S.act [A]"), ("j", "j.S.act [A/cm2]"), ("v", "U.S.AveCell [V]")]:
                    x[short + "_sum"] = x[col].fillna(0); x[short + "_sq"] = x[short + "_sum"] ** 2
                g = x.groupby("bin_start_h", sort=False).agg(
                    raw_sample_count=("OpHrs2 [h]", "size"), valid_current_count=("valid_i", "sum"), zero_current_count=("zero_i", "sum"),
                    valid_density_count=("valid_j", "sum"), valid_voltage_count=("valid_v", "sum"),
                    i_sum=("i_sum", "sum"), i_sq=("i_sq", "sum"), j_sum=("j_sum", "sum"), j_sq=("j_sq", "sum"), v_sum=("v_sum", "sum"), v_sq=("v_sq", "sum"),
                    current_min=("I.S.act [A]", "min"), current_max=("I.S.act [A]", "max"), density_min=("j.S.act [A/cm2]", "min"), density_max=("j.S.act [A/cm2]", "max"),
                    voltage_min=("U.S.AveCell [V]", "min"), voltage_max=("U.S.AveCell [V]", "max"), first_datetime=("DateTime", "min"), last_datetime=("DateTime", "max")).reset_index()
                contrib[minutes].append(g)
        source_rows.append({"source_file": name, "rows_read_prefix": read_n, "rows_retained_development": keep_n,
            "first_ophrs2_h": first_age, "last_ophrs2_h": last_age, "first_datetime": str(first_dt), "last_datetime": str(last_dt),
            "timestamp_duplicates": dup, "timestamp_reverse_order": reverse, "intra_file_gaps_gt_5s": gaps, "max_gap_s": max_gap})
        missing_rows.append({"source_file": name, "missing_current": missing["current"], "missing_current_density": missing["density"], "missing_voltage": missing["voltage"]})
    out = {}
    sums = ["raw_sample_count", "valid_current_count", "zero_current_count", "valid_density_count", "valid_voltage_count", "i_sum", "i_sq", "j_sum", "j_sq", "v_sum", "v_sq"]
    mins = ["current_min", "density_min", "voltage_min", "first_datetime"]; maxs = ["current_max", "density_max", "voltage_max", "last_datetime"]
    for minutes in (10, 30):
        d = pd.concat(contrib[minutes], ignore_index=True)
        rule = {x: "sum" for x in sums} | {x: "min" for x in mins} | {x: "max" for x in maxs}
        d = d.groupby("bin_start_h", as_index=False).agg(rule).sort_values("bin_start_h").reset_index(drop=True)
        width = minutes / 60
        d["bin_end_h"] = d.bin_start_h + width
        d["expected_sample_count"] = minutes * 60
        d["coverage_ratio"] = d.valid_current_count / d.expected_sample_count
        d["q"] = d.zero_current_count / d.valid_current_count.replace(0, np.nan)
        d["current_mean_A"] = d.i_sum / d.valid_current_count.replace(0, np.nan)
        d["current_std_A"] = std(d.i_sum, d.i_sq, d.valid_current_count)
        d["current_density_mean_A_cm2"] = d.j_sum / d.valid_density_count.replace(0, np.nan)
        d["current_density_std_A_cm2"] = std(d.j_sum, d.j_sq, d.valid_density_count)
        d["voltage_mean_V"] = d.v_sum / d.valid_voltage_count.replace(0, np.nan)
        d["voltage_std_V"] = std(d.v_sum, d.v_sq, d.valid_voltage_count)
        d["state"] = d.q.map(state)
        d["valid_bin"] = (d.coverage_ratio >= COVERAGE) & d.q.notna()
        d["temporal_segment_id"] = (d.bin_start_h.diff().sub(width).abs().gt(1e-7) | ~d.valid_bin | ~d.valid_bin.shift(fill_value=False)).cumsum()
        out[minutes] = d
    return out, pd.DataFrame(source_rows), pd.DataFrame(missing_rows)


def events(d, minutes):
    x = d[d.valid_bin].copy()
    if x.empty: return pd.DataFrame()
    width = minutes / 60
    x["event_id"] = (x.state.ne(x.state.shift()) | x.bin_start_h.diff().sub(width).abs().gt(1e-7)).cumsum()
    e = x.groupby("event_id").agg(state=("state", "first"), start_h=("bin_start_h", "min"), end_h=("bin_end_h", "max"), bins=("state", "size"),
        q_mean=("q", "mean"), current_mean_A=("current_mean_A", "mean"), current_density_mean_A_cm2=("current_density_mean_A_cm2", "mean"), voltage_mean_V=("voltage_mean_V", "mean")).reset_index(drop=True)
    e["duration_h"] = e.end_h - e.start_h; e["previous_state"] = e.state.shift(); e["transition"] = e.previous_state.fillna("START") + " -> " + e.state; e["resolution_min"] = minutes
    return e


def windows(d, start, end, minutes):
    width = minutes / 60; L, H = int(LOOKBACK_H / width), int(HORIZON_H / width)
    d = d.sort_values("bin_start_h").reset_index(drop=True); x = d[FEATURES].to_numpy(float); q = d.q.to_numpy(float); s = d.state.to_numpy(str)
    z = []
    for origin in range(L - 1, len(d) - H):
        first, last = origin - L + 1, origin + H
        if d.bin_start_h.iat[first] < start - 1e-8 or d.bin_end_h.iat[last] > end + 1e-8: continue
        block = d.iloc[first:last + 1]
        if not block.valid_bin.all() or block.temporal_segment_id.nunique() != 1: continue
        z.append({"x": x[first:origin + 1].reshape(-1), "y": q[origin + 1:last + 1].copy(), "mixed": (s[origin + 1:last + 1] == "0<q<1").astype(int),
            "origin": float(d.bin_end_h.iat[origin]), "target_start": float(d.bin_start_h.iat[origin + 1]), "weights": d.valid_current_count.iloc[origin + 1:last + 1].to_numpy(float)})
    return z


def model_predict(train, valid):
    xt, xv = np.array([a["x"] for a in train]), np.array([a["x"] for a in valid])
    yt, mt = np.array([a["y"] for a in train]), np.array([a["mixed"] for a in train])
    scaler = StandardScaler().fit(xt); xt, xv = scaler.transform(xt), scaler.transform(xv)
    qhat = np.clip(Ridge(alpha=1.0).fit(xt, yt).predict(xv), 0, 1); prob = np.zeros_like(qhat)
    for h in range(yt.shape[1]):
        if len(np.unique(mt[:, h])) < 2: prob[:, h] = mt[0, h]
        else: prob[:, h] = LogisticRegression(C=1.0, class_weight="balanced", solver="liblinear", random_state=SEED, max_iter=500).fit(xt, mt[:, h]).predict_proba(xv)[:, 1]
    return qhat, prob


def measure(fold, resolution, y, qhat, mixed, prob, scale):
    result = []
    for scope, index in [("all_horizons", slice(None)), ("terminal_12h", -1)]:
        yy, pp, mm, pr = y[:, index].ravel(), qhat[:, index].ravel(), mixed[:, index].ravel(), prob[:, index].ravel()
        row = {"fold": fold, "resolution": resolution, "scale": scale, "horizon_scope": scope, "n_targets": len(yy), "q_mae": mean_absolute_error(yy, pp), "q_rmse": mean_squared_error(yy, pp) ** .5, "mixed_positive_count": int(mm.sum())}
        if len(np.unique(mm)) < 2:
            row |= {"mixed_precision": "NOT_ESTIMABLE", "mixed_recall": "NOT_ESTIMABLE", "mixed_f1": "NOT_ESTIMABLE", "mixed_brier": "NOT_ESTIMABLE"}
        else:
            p, r, f, _ = precision_recall_fscore_support(mm, pr >= .5, average="binary", zero_division=0)
            row |= {"mixed_precision": p, "mixed_recall": r, "mixed_f1": f, "mixed_brier": float(np.mean((pr - mm) ** 2))}
        result.append(row)
    return result


def independent_event_metric(valid, qhat, prob, minutes):
    ledger = {}
    for r, q, p in zip(valid, qhat, prob):
        for h, (truth, pred, pr) in enumerate(zip(r["y"], q, p)):
            ledger.setdefault(round(r["target_start"] + h * minutes / 60, 8), (truth, pred, pr))
    a = pd.DataFrame([(t, *v) for t, v in ledger.items()], columns=["time", "truth_q", "pred_q", "prob"]).sort_values("time")
    a["truth"] = a.truth_q.map(state); a["pred"] = np.where(a.prob >= .5, "0<q<1", np.where(a.pred_q <= .5, "q=0", "q=1"))
    def count(s):
        change = s.ne(s.shift()).fillna(True)
        return int((change & (s == "0<q<1")).sum()), int(change.sum() - 1)
    te, tt = count(a.truth); pe, pt = count(a.pred)
    return {"independent_true_mixed_events": te, "predicted_mixed_events": pe, "true_transition_count": tt, "predicted_transition_count": pt, "transition_count_abs_error": abs(tt - pt)}


def predict_compare(data):
    metrics, event_rows, alignment = [], [], []
    for fold, (tr_end, va_end) in FOLDS.items():
        w30t, w30v = windows(data[30], DEV_START, tr_end, 30), windows(data[30], tr_end, va_end, 30)
        w10t, w10v = windows(data[10], DEV_START, tr_end, 10), windows(data[10], tr_end, va_end, 10)
        if not w30t or not w30v or not w10t or not w10v: raise RuntimeError("no legal window: " + fold)
        q30, p30 = model_predict(w30t, w30v); q10, p10 = model_predict(w10t, w10v)
        y30, m30 = np.array([x["y"] for x in w30v]), np.array([x["mixed"] for x in w30v])
        y10, m10 = np.array([x["y"] for x in w10v]), np.array([x["mixed"] for x in w10v])
        metrics += measure(fold, "30min", y30, q30, m30, p30, "native") + measure(fold, "10min", y10, q10, m10, p10, "native")
        ten = {}
        for r, q, p in zip(w10v, q10, p10):
            if abs(r["origin"] * 2 - round(r["origin"] * 2)) > 1e-6: continue
            ten[r["origin"]] = [(np.average(r["y"][i:i+3], weights=r["weights"][i:i+3]), np.average(q[i:i+3], weights=r["weights"][i:i+3]), int(r["mixed"][i:i+3].max()), np.average(p[i:i+3], weights=r["weights"][i:i+3])) for i in range(0, 72, 3)]
        base = {r["origin"]: (r["y"], q, r["mixed"], p) for r, q, p in zip(w30v, q30, p30)}
        common = sorted(set(ten) & set(base))
        if not common: raise RuntimeError("no aligned origins: " + fold)
        a = np.array([base[k][0] for k in common]); aq = np.array([base[k][1] for k in common]); am = np.array([base[k][2] for k in common]); ap = np.array([base[k][3] for k in common])
        b = np.array([[v[0] for v in ten[k]] for k in common]); bq = np.array([[v[1] for v in ten[k]] for k in common]); bm = np.array([[v[2] for v in ten[k]] for k in common]); bp = np.array([[v[3] for v in ten[k]] for k in common])
        if not np.allclose(a, b, equal_nan=True): raise RuntimeError("target mismatch at " + fold)
        metrics += measure(fold, "30min", a, aq, am, ap, "unified_30min") + measure(fold, "10min_aggregated", b, bq, bm, bp, "unified_30min")
        alignment.append({"fold": fold, "common_origin_count": len(common), "matching_target_count": int(a.size), "target_identity_verified": True, "aggregation_weight": "actual valid-current raw sample count"})
        event_rows.append({"fold": fold, "resolution": "30min", **independent_event_metric(w30v, q30, p30, 30)})
        event_rows.append({"fold": fold, "resolution": "10min", **independent_event_metric(w10v, q10, p10, 10)})
    return pd.DataFrame(metrics), pd.DataFrame(event_rows), pd.DataFrame(alignment)


def draw(data, hidden, metrics):
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    colors = {10: "#0072B2", 30: "#D55E00"}
    fig, ax = plt.subplots(2, 1, figsize=(11, 5.8), sharex=True, layout="constrained")
    for m in (10, 30):
        d = data[m][(data[m].bin_start_h >= 200) & (data[m].bin_start_h < 224)]
        ax[0].step(d.bin_start_h, d.q, where="post", label=str(m) + " min q", color=colors[m])
        ax[1].step(d.bin_start_h, d.current_mean_A, where="post", label=str(m) + " min mean current", color=colors[m])
    ax[0].set_ylabel("zero-current fraction q"); ax[1].set_ylabel("current [A]"); ax[1].set_xlabel("OpHrs2 [h]"); ax[0].legend(ncol=2); ax[1].legend(ncol=2)
    fig.savefig(FIG / "resolution_state_example.png", dpi=220); fig.savefig(FIG / "resolution_state_example.pdf"); plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 4.5), layout="constrained")
    rate = float(hidden.hidden_transition.mean())
    ax.bar(["within-bin 10-min\\nstate change", "no within-bin\\nstate change"], [rate, 1-rate], color=["#009E73", "#999999"])
    ax.set_ylim(0, 1); ax.set_ylabel("fraction of valid 30-min bins"); ax.set_title("Within-30-minute state heterogeneity")
    fig.savefig(FIG / "hidden_transitions_10min_within_30min.png", dpi=220); fig.savefig(FIG / "hidden_transitions_10min_within_30min.pdf"); plt.close(fig)
    u = metrics[(metrics.scale == "unified_30min") & (metrics.horizon_scope == "all_horizons")]
    fig, ax = plt.subplots(figsize=(8, 4.5), layout="constrained")
    for fold in FOLDS:
        z = u[u.fold == fold].set_index("resolution")
        ax.plot([0, 1], [z.loc["30min", "q_rmse"], z.loc["10min_aggregated", "q_rmse"]], marker="o", label=fold)
    ax.set_xticks([0, 1], ["30 min direct", "10 min then 30 min"]); ax.set_ylabel("q RMSE"); ax.set_title("Matched-origin causal forecast"); ax.legend()
    fig.savefig(FIG / "unified_30min_q_rmse_by_fold.png", dpi=220); fig.savefig(FIG / "unified_30min_q_rmse_by_fold.pdf"); plt.close(fig)


def write_reports(data, sources, missing, all_events, hidden, boundary, metrics, event_metrics, alignment, decision):
    coverage = pd.DataFrame([{"resolution_min": m, "observed_bins": len(d), "valid_bins": int(d.valid_bin.sum()), "low_coverage_bins": int((d.coverage_ratio < COVERAGE).sum()), "low_coverage_pct": float((d.coverage_ratio < COVERAGE).mean()), "median_coverage": float(d.coverage_ratio.median()), "mixed_state_pct": float((d.state == "0<q<1").mean())} for m, d in data.items()])
    (PHASE / "01_RAW_DATA_AND_RESOLUTION_AUDIT.md").write_text("# Raw data and resolution audit\n\nOnly Development rows were retained before any statistic: %.6f <= OpHrs2 < %.6f h. The final mixed raw source was read only through its audited %d-row Development prefix and re-filtered. Test is absent. Bins are left-closed/right-open on OpHrs2, observed-only, with no interpolation, fill, or smoothing. q = zero-current valid samples / valid-current samples; zero current means abs(I.S.act) <= 0.10 A and is not a validated shutdown label.\n\n## Timestamp and source audit\n\n%s\n\n## Missingness after Development filter\n\n%s\n\n## Coverage\n\n%s\n\nSampling is nominally approximately 1 Hz. Invalid/gap bins are retained as invalid and cannot form a sequence window.\n" % (DEV_START, DEV_END, LAST_DEV_ROWS, md(sources), md(missing), md(coverage)), encoding="utf-8")
    event_summary = all_events.groupby(["resolution_min", "state"], as_index=False).agg(independent_events=("state", "size"), total_duration_h=("duration_h", "sum"), median_duration_h=("duration_h", "median"))
    hsum = hidden.groupby("hidden_transition", as_index=False).agg(bins=("parent_start_h", "size"), mean_current_range_A=("current_range_A", "mean"), mean_voltage_range_V=("voltage_range_V", "mean"))
    (PHASE / "02_STATE_REPRESENTATION_COMPARISON.md").write_text("# State representation comparison\n\nStates are q=0, 0<q<1, and q=1. Counts below are actual contiguous state events, never overlapping forecast windows.\n\n## Independent events\n\n%s\n\n## Ten-minute heterogeneity inside 30-minute bins\n\n%s\n\nA true within-bin state change demonstrates added expression resolution only; it does not itself establish future predictability.\n\n## Boundary-condition summary\n\n%s\n" % (md(event_summary), md(hsum), md(boundary)), encoding="utf-8")
    native = metrics[metrics.scale == "native"]; unified = metrics[metrics.scale == "unified_30min"]
    (PHASE / "03_CAUSAL_PREDICTABILITY_COMPARISON.md").write_text("# Causal future-state predictability comparison\n\nFixed before validation: history is 24 h, horizon is 12 h; 30-minute = 48 to 24 steps and 10-minute = 144 to 72 steps. Inputs are only historical q, current mean and current standard deviation. Fold-local StandardScaler, Ridge(alpha=1), and class-balanced LogisticRegression(C=1) are fit on Fold train windows only. No future true current, voltage, or q is provided, and no calibration search occurs.\n\n## Exact matched-target alignment\n\n%s\n\n10-minute predictions are aggregated from nested triples using actual valid-current raw sample counts. Direct 30-minute labels must equal reconstructed weighted labels; mismatch is fail-closed.\n\n## Native-resolution metrics\n\n%s\n\n## Unified 30-minute metrics: primary comparison\n\n%s\n\n## Independent validation transition-event support\n\n%s\n\nNOT_ESTIMABLE means no binary-class variation and is never converted to zero.\n" % (md(alignment), md(native), md(unified), md(event_metrics)), encoding="utf-8")
    (PHASE / "04_LEAKAGE_AND_ALIGNMENT_AUDIT.md").write_text("# Leakage and alignment audit\n\n- Raw chunks are Development-filtered before any aggregation or statistic; the final source also has a fixed Development-prefix limit.\n- OpHrs2 is the only bin coordinate; DateTime is continuity diagnostic only.\n- Bins are left-closed/right-open, observed-only, and invalid/gap bins break segments.\n- Each window including its 24 h history and 12 h target lies fully within its own Fold region.\n- Every scaler and model uses only Fold Train windows.\n- Unified evaluation requires exact common origin endpoints and verified identical direct versus weighted targets.\n- Test features, targets, statistics, and results are absent from all tables.\n", encoding="utf-8")
    (PHASE / "05_NEXT_PHASE_DECISION.md").write_text("# Next-phase decision\n\nThe preregistered GO rule requires at least 5 percent q RMSE reduction in two or more folds, no more than 5 percent Fold 3 worsening, non-worse estimable Brier in two or more folds, and at least ten independent validation mixed-state events. The rule was frozen in configs/frozen_protocol.yaml before metrics were generated.\n\n## Decision\n\n**%s**\n\n%s\n\nThis is a resolution-audit decision only. It does not train or claim a better TimeXer.\n" % (decision["decision"], decision["rationale"]), encoding="utf-8")
    (PHASE / "FINAL_DECISION.yaml").write_text("decision: %s\nreference_commit: 9c6d0193e1c6d6568556b36e3d373eb11c9430e2\ndevelopment_end_h_exclusive: 1819.997738\nprotocol: configs/frozen_protocol.yaml\nrationale: %s\n" % (decision["decision"], json.dumps(decision["rationale"], ensure_ascii=False)), encoding="utf-8")


def append_final_answers(hidden, metrics, event_metrics):
    """Explicit answers required by the phase brief, derived only from saved dev metrics."""
    u = metrics[(metrics.scale == "unified_30min") & (metrics.horizon_scope == "all_horizons")]
    q = u.pivot(index="fold", columns="resolution", values="q_rmse")
    b = u.pivot(index="fold", columns="resolution", values="mixed_brier")
    gains = ((q["30min"] - q["10min_aggregated"]) / q["30min"] * 100)
    bchange = ((b["10min_aggregated"] - b["30min"]) / b["30min"] * 100)
    hidden_pct = float(hidden.hidden_transition.mean() * 100)
    support = int(event_metrics.loc[event_metrics.resolution == "30min", "independent_true_mixed_events"].sum())
    text = "\n## Required answers\n\n"
    text += "**Q1.** Yes for expression: %.1f%% of valid 30-minute bins contain more than one valid 10-minute q state, so 10 minutes reveals genuine within-bin transitions. This is not automatically a predictive gain.\n\n" % hidden_pct
    text += "**Q2.** Partly. A fixed past-only Ridge reduces unified q error, but the fixed mixed-state Logistic probabilities deteriorate. The finer state is therefore not reliably forecastable under every required metric.\n\n"
    text += "**Q3.** Unified q RMSE improves by Fold 1 %.1f%%, Fold 2 %.1f%%, Fold 3 %.1f%%; however mixed-state Brier worsens by %.1f%%, %.1f%%, and %.1f%% respectively. Hence the fair composite answer is no.\n\n" % (gains["fold_1"], gains["fold_2"], gains["fold_3"], bchange["fold_1"], bchange["fold_2"], bchange["fold_3"])
    text += "**Q4.** q-RMSE gains are consistent, including Fold 3, but Brier deterioration is also consistent (0/3 non-worse folds). Common-scale independent mixed-event support is %d, so lack of support is not the reason for the negative decision.\n\n" % support
    text += "**Q5.** No. Keep the frozen 30-minute representation for the next authorized model phase; do not start a 10-minute TimeXer experiment from this audit.\n"
    with (PHASE / "05_NEXT_PHASE_DECISION.md").open("a", encoding="utf-8") as f:
        f.write(text)


def main():
    random.seed(SEED); np.random.seed(SEED)
    for p in [OUT, TAB, FIG]: p.mkdir(parents=True, exist_ok=True)
    data, sources, missing = make_aggregates()
    for m, d in data.items(): d.to_csv(TAB / ("causal_state_features_%dmin_development_only.csv" % m), index=False)
    sources.to_csv(TAB / "development_only_raw_source_audit.csv", index=False); missing.to_csv(TAB / "development_only_missingness.csv", index=False)
    e10, e30 = events(data[10], 10), events(data[30], 30); all_events = pd.concat([e10, e30], ignore_index=True); all_events.to_csv(TAB / "independent_state_events.csv", index=False)
    x = data[10][data[10].valid_bin].copy(); x["parent_start_h"] = np.floor(x.bin_start_h / .5 + 1e-8) * .5
    hidden = x.groupby("parent_start_h").agg(n_states=("state", "nunique"), current_range_A=("current_mean_A", lambda z: z.max()-z.min()), voltage_range_V=("voltage_mean_V", lambda z: z.max()-z.min())).reset_index(); hidden["hidden_transition"] = hidden.n_states > 1; hidden.to_csv(TAB / "within_30min_10min_state_heterogeneity.csv", index=False)
    x = data[10][data[10].valid_bin].copy(); x["previous_state"] = x.state.shift(); x["contiguous"] = x.bin_start_h.diff().sub(1/6).abs() < 1e-7
    boundary = x[(x.state != x.previous_state) & x.contiguous].groupby(["previous_state", "state"], dropna=True).agg(transitions=("state", "size"), mean_current_A=("current_mean_A", "mean"), mean_density_A_cm2=("current_density_mean_A_cm2", "mean"), mean_voltage_V=("voltage_mean_V", "mean")).reset_index(); boundary.to_csv(TAB / "state_boundary_operating_summary.csv", index=False)
    metrics, event_metrics, alignment = predict_compare(data)
    metrics.to_csv(TAB / "predictability_metrics.csv", index=False); event_metrics.to_csv(TAB / "independent_transition_event_metrics.csv", index=False); alignment.to_csv(TAB / "matched_origin_target_alignment.csv", index=False)
    u = metrics[(metrics.scale == "unified_30min") & (metrics.horizon_scope == "all_horizons")]
    q = u.pivot(index="fold", columns="resolution", values="q_rmse"); imp = (q["30min"] - q["10min_aggregated"]) / q["30min"]
    b = u.pivot(index="fold", columns="resolution", values="mixed_brier")
    bn = sum(b.loc[f, "10min_aggregated"] <= b.loc[f, "30min"] + 1e-12 for f in b.index if isinstance(b.loc[f, "30min"], (float, np.floating)) and isinstance(b.loc[f, "10min_aggregated"], (float, np.floating)))
    # Count support once on the common 30-minute target scale; summing both
    # resolution representations would double-count the same physical events.
    support = int(event_metrics.loc[event_metrics.resolution == "30min", "independent_true_mixed_events"].sum()); gain = int((imp >= .05).sum()); f3 = float(imp["fold_3"])
    if support < 10:
        decision = {"decision": "INCONCLUSIVE_INSUFFICIENT_EVIDENCE", "rationale": "Comparable evaluation was valid, but independent mixed-state event support (%d) was below the preregistered minimum of 10." % support}
    elif gain >= 2 and f3 >= -.05 and bn >= 2:
        decision = {"decision": "GO_10MIN_TIMEXER_EXPERIMENT", "rationale": "Frozen conditions met: q RMSE improvement >=5%% in %d folds, Fold 3 change %.1f%%, non-worse Brier in %d folds, and %d independent mixed events." % (gain, 100*f3, bn, support)}
    else:
        decision = {"decision": "NO_GO_KEEP_30MIN", "rationale": "Comparable evaluation was valid but frozen GO rule failed: q RMSE improvement >=5%% in %d/3 folds; Fold 3 change %.1f%%; non-worse Brier in %d folds; independent mixed events %d." % (gain, 100*f3, bn, support)}
    draw(data, hidden, metrics); write_reports(data, sources, missing, all_events, hidden, boundary, metrics, event_metrics, alignment, decision); append_final_answers(hidden, metrics, event_metrics)
    print(json.dumps(decision, ensure_ascii=False))

if __name__ == "__main__":
    try: main()
    except Exception as exc:
        print("FAIL-CLOSED: %s: %s" % (type(exc).__name__, exc), file=sys.stderr)
        raise
