"""Predeclared CPU-only event attribution and strictly causal precursor checks.

--freeze-rules reads TRAIN observations only and must precede new GPU training.
--run reads immutable completed predictions/checkpoints, never optimizes a model.
Physical episode grouping is an operational definition, not hardware shutdown truth.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

sys.dont_write_bytecode = True

import numpy as np
import pandas as pd

STUDY = Path(__file__).resolve().parents[1]
OLD = STUDY.parent / "RealHyFC_Raw1s_Dynamic_Study"
OUT = STUDY / "outputs"
DIAG = OUT / "diagnostics"
THRESHOLD_V = 0.038477648942
ZERO_A = 0.1
WINDOWS = (30, 60, 120)
FEATURES = ("current_ptp_A", "set_current_ptp_A", "temperature_max_ptp_C",
            "pressure_max_ptp_kPag")
RUNS = {"A1": ("LSTM", 30), "A2": ("F2", 30),
        "A3": ("LSTM", 120), "A4": ("F2", 120),
        "B1": ("F2", 30), "B2": ("F2", 120)}


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(2 ** 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                    encoding="utf-8", newline="\n")


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def cache() -> tuple[dict, dict]:
    """Historical arrays are opened mode=r; no historical module is executed."""
    cfg = read_json(OLD / "outputs/frozen_protocol.json")
    n = read_json(OLD / "outputs/cache_schema.json")["rows"]
    dtypes = {"x": "float32", "time_ms": "int64", "age_h": "float64",
              "segment": "int32", "valid": "uint8", "voltage_V": "float64",
              "file_id": "uint16", "source_row": "int32"}
    arrays = {key: np.memmap(OLD / f"cache/{key}.bin", mode="r", dtype=dtype,
                            shape=(n, 24) if key == "x" else (n,))
              for key, dtype in dtypes.items()}
    return cfg, arrays


def bounds(cfg: dict, n: int) -> dict[str, tuple[int, int]]:
    seams = [s["after_raw_row"] for s in cfg["seams"]]
    return {"TRAIN": (0, seams[0]), "VALIDATION": (seams[0], seams[1])}


def legal_grid(arrays: dict, lo: int, hi: int, anchor: int) -> np.ndarray:
    """A fixed 60s timestamp lattice, independent of voltages and model errors.

    Both the 329-record raw information support and all 120 future targets
    remain inside the selected partition and one recorded continuous segment.
    This is a diagnostic grid, never a replacement for model evaluation origins.
    """
    time = arrays["time_ms"]
    rem = (np.asarray(time[lo:hi]) - anchor) % 60000
    rows = np.flatnonzero(np.minimum(rem, 60000 - rem) <= 50) + lo
    rows = rows[(rows >= lo + 328) & (rows + 120 < hi)]
    clean = arrays["segment"][rows - 328] == arrays["segment"][rows + 120]
    rows = rows[clean]
    invalid = np.concatenate(([0], np.cumsum(np.asarray(arrays["valid"][lo:hi]) == 0)))
    rows = rows[(invalid[rows + 121 - lo] - invalid[rows - 299 - lo]) == 0]
    rows = rows[np.abs(time[rows + 120] - time[rows] - 120000) <= 100]
    assert len(rows) == len(np.unique(rows)) and np.all(np.diff(rows) > 0)
    return rows.astype(np.int64)


def causal_features(arrays: dict, rows: np.ndarray, seconds: int) -> np.ndarray:
    """Only rows [origin-seconds, origin], never future labels or commands."""
    rows = np.asarray(rows, np.int64)
    values = np.empty((len(rows), 5), dtype=np.float64)
    for start in range(0, len(rows), 1024):
        r = rows[start:start + 1024]
        index = r[:, None] + np.arange(-seconds, 1)
        x = np.asarray(arrays["x"][index])
        values[start:start + len(r), 0] = np.ptp(x[:, :, 5], axis=1)
        values[start:start + len(r), 1] = np.ptp(x[:, :, 14], axis=1)
        values[start:start + len(r), 2] = np.max(np.ptp(x[:, :, 16:20], axis=1), axis=1)
        values[start:start + len(r), 3] = np.max(np.ptp(x[:, :, 20:22], axis=1), axis=1)
        values[start:start + len(r), 4] = np.any(np.diff(x[:, :, 22], axis=1) != 0, axis=1)
    assert np.isfinite(values).all()
    return values


def freeze_rules() -> dict:
    path = OUT / "diagnostic_rules.json"
    assert not path.exists(), "Refuse to replace preregistered diagnostic rules"
    cfg, arrays = cache()
    train_hi = bounds(cfg, len(arrays["time_ms"]))["TRAIN"][1]
    train_rows = legal_grid(arrays, 0, train_hi, cfg["sampling"]["anchor_ms"])
    fitted = {}
    for seconds in WINDOWS:
        values = causal_features(arrays, train_rows, seconds)
        fitted[str(seconds)] = {name: float(np.quantile(values[:, i], .995))
                                for i, name in enumerate(FEATURES)}
    result = {
        "status": "FROZEN_BEFORE_NEW_GPU_TRAINING", "version": 1,
        "old_protocol_sha256": sha(OLD / "outputs/frozen_protocol.json"),
        "voltage_trigger_absolute_1s_delta_V": THRESHOLD_V,
        "episode_rule": "Within same recorded segment and partition, consecutive trigger DateTime gap <=120000ms is one operational voltage episode. No hardware independence is asserted.",
        "zero_current_rule": "abs(actual current)<=0.1A is an observed low/zero-current proxy, never a hardware shutdown label",
        "classification_window_seconds": [-30, 30],
        "classification_priority": ["loaded_to_low_or_zero", "low_or_zero_to_loaded",
                                    "current_or_command_change", "temperature_or_pressure_change",
                                    "no_obvious_control_change"],
        "classification_definition": {
            "transition": "Any crossing of the 0.1A observed current boundary within +/-30s of episode first trigger; down crossing takes priority over up if both occur.",
            "current_or_command_change": "No threshold crossing; past/around-first-trigger 60s actual or set-current range exceeds TRAIN-fitted 60s threshold, or recorded operation_point changes.",
            "temperature_or_pressure_change": "No prior class; 60s range of any recorded temperature or inlet pressure exceeds TRAIN-fitted 60s threshold.",
            "no_obvious_control_change": "No above rule fires; not proof of instrument error or physical causality.",
            "ordinary_queries": "Non-event queries separately split by observed origin/target abs(I)<=0.1A and mixed-state interval; ordinary loaded and stable-zero states are not separate voltage episodes."},
        "precursor_windows_seconds": list(WINDOWS),
        "precursor_quantile": .995,
        "train_only_fitted_thresholds": fitted,
        "train_threshold_fit_origins": len(train_rows),
        "train_threshold_origin_sha256": hashlib.sha256(train_rows.tobytes()).hexdigest(),
        "precursor_flag": "OR of the four range > their TRAIN-only 99.5% thresholds, or recorded operation_point changed in the causal window. Equality is not exceedance.",
        "grid_predictability_test": "Fixed 60s chronological diagnostic grid; past W seconds only predicts whether any voltage trigger occurs in next W recorded seconds for W=30,60,120. TRAIN thresholds are unchanged on VALIDATION; report precision/recall/FPR and class base rates, not deterministic advance control knowledge.",
        "event_alignment_offsets_seconds": [-120, -60, -30, 0, 30],
        "event_alignment_past_window_seconds": 30,
        "event_alignment_interpretation": "Strictly before (<0), first recorded trigger (0), and only after (>0) are reported separately. Future-time alignment is retrospective diagnostic evidence, not a prediction input.",
        "largest_error_fraction": .01,
        "largest_error_count": "ceil(0.01 * all origin-times-lead queries); ties use deterministic flat query index",
        "query_to_episode_assignment": "For event-labeled queries, attach the most recent actual voltage-trigger episode since that origin; normal queries remain event_id=0. Each query is counted once; an event can be reached by multiple overlapping forecasts.",
        "top_contribution_denominator": "All Validation query squared error, retaining every origin and every lead",
        "hook_selection": "TRAIN first three and VALIDATION first three chronological episodes whose offsets [-120,-60,-30,0,30,60,120] all admit legal 329-record raw support; never selected by error. Full-model forward uses CPU only and at most 6 episodes x 7 offsets per F2 checkpoint.",
        "hook_offsets_seconds": [-120, -60, -30, 0, 30, 60, 120],
        "hook_limitations": "Report native post-LSTM encoder and causal-CNN hidden representation RMS, cosine and change. They share VSN inputs and nonlinear downstream decoding; these are not separately identifiable branch voltage contributions or causal attributions.",
        "no_future_information": True, "holdout_data_used_to_fit_rules": False,
        "GPU_calls_by_diagnostics": 0, "optimizer_steps_by_diagnostics": 0,
        "historically_exposed_1873h_2150h_used_to_define_rules": False,
        "AI_assistance_disclosure": "AI-assisted implementation; deterministic scripts, source records and saved model outputs are the empirical authority."
    }
    write_json(path, result)
    DIAG.mkdir(parents=True, exist_ok=True)
    np.save(DIAG / "precursor_train_threshold_origins.npy", train_rows)
    return result


def precursor_flags(values: np.ndarray, seconds: int, rules: dict) -> np.ndarray:
    cut = rules["train_only_fitted_thresholds"][str(seconds)]
    return np.any(values[:, :4] > np.array([cut[name] for name in FEATURES]), axis=1) | (values[:, 4] > 0)


def revise_rules_before_main_freeze() -> dict:
    """Explicit semantic correction authorized before any main freeze or GPU run."""
    assert not (OUT / "frozen_protocol.json").exists(), "Main freeze already exists"
    assert not any((STUDY / "runs").glob("*/training_state.json")), "A model run already exists"
    path = OUT / "diagnostic_rules.json"
    old = read_json(path)
    assert old["version"] == 1
    previous_sha = sha(path)
    new = dict(old)
    new["version"] = 2
    new["classification_priority"] = ["loaded_to_low_or_zero", "low_or_zero_to_loaded",
                                      "current_or_command_change", "temperature_or_pressure_change",
                                      "no_above_threshold_observed_control_change"]
    new["classification_definition"] = dict(old["classification_definition"])
    new["classification_definition"]["current_or_command_change"] = (
        "No threshold crossing; any consecutive recorded actual-current or set-current change >0.1A in +/-30s, or recorded operation_point changes. The 0.1A rule is inherited from the existing causal input flag, not fitted on Validation.")
    new["classification_definition"].pop("no_obvious_control_change")
    new["classification_definition"]["no_above_threshold_observed_control_change"] = (
        "No above rule fires; does not mean physical controls were unchanged and is not proof of instrument error or physical causality.")
    new["pre_GPU_rule_revision"] = {
        "superseded_rules_sha256": previous_sha,
        "reason": "Correct an overstrong category name and ensure ordinary observed current/setpoint changes are described rather than missed by the extreme 99.5% range threshold.",
        "main_protocol_frozen": False, "new_GPU_trainings_started": 0,
        "validation_outcomes_consulted": False,
        "unchanged": "Every precursor threshold, event trigger, quiet-gap episode rule, sampler, model, loss, forecast target and gate diagnostic selection."}
    write_json(DIAG / "pre_GPU_rule_revision.json", {"superseded_rules": old,
               "revision": new["pre_GPU_rule_revision"]})
    write_json(path, new)
    return new


def confusion(label: np.ndarray, flag: np.ndarray) -> dict:
    tp = int(np.sum(label & flag)); fp = int(np.sum(~label & flag))
    fn = int(np.sum(label & ~flag)); tn = int(np.sum(~label & ~flag))
    return {"origins": len(label), "positive_origins": int(label.sum()),
            "positive_rate": float(np.mean(label)) if len(label) else None,
            "flag_rate": float(np.mean(flag)) if len(label) else None,
            "TP": tp, "FP": fp, "FN": fn, "TN": tn,
            "precision": tp / (tp + fp) if tp + fp else None,
            "recall": tp / (tp + fn) if tp + fn else None,
            "false_positive_rate": fp / (fp + tn) if fp + tn else None}


def precursor_study(arrays: dict, cfg: dict, rules: dict, events: pd.DataFrame) -> dict:
    records = []; alignment = []; threshold = rules["voltage_trigger_absolute_1s_delta_V"]
    limits = bounds(cfg, len(arrays["time_ms"]))
    for partition, (lo, hi) in limits.items():
        origins = legal_grid(arrays, lo, hi, cfg["sampling"]["anchor_ms"])
        v = arrays["voltage_V"]
        for seconds in WINDOWS:
            values = causal_features(arrays, origins, seconds)
            flag = precursor_flags(values, seconds, rules)
            rows = origins[:, None] + np.arange(seconds + 1)
            label = np.any(np.abs(np.diff(v[rows], axis=1)) > threshold, axis=1)
            records.append({"partition": partition, "past_window_seconds": seconds,
                            "future_seconds": seconds, **confusion(label, flag)})
            detail = pd.DataFrame({"origin_row": origins, "origin_time_ms": arrays["time_ms"][origins],
                                   "future_voltage_event": label, "causal_precursor_flag": flag})
            for i, name in enumerate((*FEATURES, "operation_point_changed")):
                detail[name] = values[:, i]
            detail.to_csv(DIAG / f"precursor_{partition}_W{seconds}.csv", index=False, lineterminator="\n")
        part = events[events.partition == partition]
        for offset in rules["event_alignment_offsets_seconds"]:
            rows = part.first_row.to_numpy(np.int64) + offset
            eligible = (rows - 30 >= lo) & (rows < hi)
            if len(rows):
                safe = np.clip(rows, 30, len(arrays["time_ms"]) - 1)
                eligible &= arrays["segment"][safe - 30] == arrays["segment"][safe]
                eligible &= arrays["segment"][safe] == part.segment_id.to_numpy()
                eligible &= np.abs(arrays["time_ms"][safe] - part.first_ms.to_numpy() - offset * 1000) <= 100
            selected = rows[eligible]
            values = causal_features(arrays, selected, 30)
            flags = precursor_flags(values, 30, rules)
            for event_id, row, features, flag in zip(part.event_id.to_numpy()[eligible], selected, values, flags):
                alignment.append({"event_id": int(event_id), "partition": partition,
                                  "first_trigger_offset_seconds": offset,
                                  "observable_row": int(row), "observable_time_ms": int(arrays["time_ms"][row]),
                                  "timing": "BEFORE" if offset < 0 else "AT" if offset == 0 else "AFTER",
                                  "causal_past30_precursor_flag": bool(flag),
                                  **dict(zip((*FEATURES, "operation_point_changed"), features.tolist()))})
    pd.DataFrame(records).to_csv(DIAG / "precursor_fixed_grid_summary.csv", index=False, lineterminator="\n")
    frame = pd.DataFrame(alignment)
    frame.to_csv(DIAG / "precursor_event_alignment.csv", index=False, lineterminator="\n")
    rates = frame.groupby(["partition", "first_trigger_offset_seconds", "timing"]).agg(
        events=("event_id", "count"), clue_fraction=("causal_past30_precursor_flag", "mean")).reset_index()
    rates.to_csv(DIAG / "precursor_event_alignment_summary.csv", index=False, lineterminator="\n")
    result = {"status": "COMPLETE", "rules_sha256": sha(OUT / "diagnostic_rules.json"),
              "fixed_grid": records, "event_alignment": rates.to_dict("records"),
              "limits": ["Associations at fixed horizons are not proof of deterministic event time predictability.",
                         "Overlapping diagnostic contexts and episodes are not independent experiment repeats.",
                         "The rule includes high past variability and operation-point changes; it may identify an ongoing event rather than a novel unseen transition.",
                         "No forecast model, data sampler, threshold or checkpoint is changed by these findings."]}
    write_json(DIAG / "precursor_study.json", result)
    return result


def classify_events(arrays: dict, rules: dict, events: pd.DataFrame) -> pd.DataFrame:
    records = []; cut = rules["train_only_fitted_thresholds"]["60"]
    limits = bounds(read_json(OLD / "outputs/frozen_protocol.json"), len(arrays["time_ms"]))
    for rec in events.to_dict("records"):
        row = int(rec["first_row"])
        part_lo, part_hi = limits[rec["partition"]]
        lo, hi = max(part_lo, row - 30), min(part_hi, row + 31)
        good = arrays["segment"][lo:hi] == int(rec["segment_id"])
        x = np.asarray(arrays["x"][lo:hi])[good]
        zero = np.abs(x[:, 5]) <= ZERO_A
        down = bool(np.any(~zero[:-1] & zero[1:])); up = bool(np.any(zero[:-1] & ~zero[1:]))
        spans = np.ptp(x, axis=0)
        actual = float(spans[5]); command = float(spans[14])
        temperature = float(np.max(spans[16:20])); pressure = float(np.max(spans[20:22]))
        mode_change = bool(np.any(np.diff(x[:, 22]) != 0))
        actual_change = bool(np.any(np.abs(np.diff(x[:, 5])) > .1))
        command_change = bool(np.any(np.abs(np.diff(x[:, 14])) > .1))
        if down:
            category = "loaded_to_low_or_zero"
        elif up:
            category = "low_or_zero_to_loaded"
        elif actual_change or command_change or mode_change:
            category = "current_or_command_change"
        elif temperature > cut["temperature_max_ptp_C"] or pressure > cut["pressure_max_ptp_kPag"]:
            category = "temperature_or_pressure_change"
        else:
            category = "no_above_threshold_observed_control_change"
        records.append({**rec, "category": category, "current_downcross_in_pm30s": down,
                        "current_upcross_in_pm30s": up, "current_ptp_A": actual,
                        "set_current_ptp_A": command, "temperature_max_ptp_C": temperature,
                        "pressure_max_ptp_kPag": pressure, "operation_point_changed": mode_change,
                        "actual_current_adjacent_change_gt_0p1A": actual_change,
                        "set_current_adjacent_change_gt_0p1A": command_change,
                        "first_trigger_current_A": float(arrays["x"][row, 5]),
                        "first_trigger_age_h": float(arrays["age_h"][row]),
                        "classification_is_retrospective_not_model_input": True})
    result = pd.DataFrame(records)
    result.to_csv(DIAG / "classified_events.csv", index=False, lineterminator="\n")
    return result


def available_runs() -> list[tuple[str, str, int, Path]]:
    runs = []
    for name, (model, horizon) in RUNS.items():
        path = STUDY / f"runs/{name}_{model}_H{horizon}"
        if (path / "predictions_VALIDATION_60s.npz").exists():
            state_path = path / "training_state.json"
            assert state_path.exists() and read_json(state_path)["status"] == "COMPLETE", f"Refuse incomplete run {name}"
            runs.append((name, model, horizon, path))
    for horizon in (30, 120):
        for path in sorted((STUDY / "runs").glob(f"*Persistence*H{horizon}*")):
            if (path / "predictions_VALIDATION_60s.npz").exists():
                runs.append((f"Persistence_H{horizon}", "Persistence", horizon, path))
    return runs


def query_metrics(truth: np.ndarray, error: np.ndarray) -> dict:
    if not len(error):
        return {"queries": 0, "RMSE_V": None, "MAE_V": None, "R2_delta": None,
                "P95_abs_V": None, "P99_abs_V": None, "SSE_V2": 0.0}
    sse = float(np.sum(error ** 2)); denominator = float(np.sum((truth - truth.mean()) ** 2))
    return {"queries": int(error.size), "RMSE_V": math.sqrt(sse / len(error)),
            "MAE_V": float(np.mean(np.abs(error))),
            "R2_delta": 1 - sse / denominator if denominator else None,
            "P95_abs_V": float(np.quantile(np.abs(error), .95)),
            "P99_abs_V": float(np.quantile(np.abs(error), .99)), "SSE_V2": sse}


def attribute_queries(arrays: dict, rules: dict, events: pd.DataFrame,
                      runs: list[tuple[str, str, int, Path]]) -> dict:
    """Keep all queries; operational event IDs only aggregate explanatory results."""
    n = len(arrays["voltage_V"])
    voltage = arrays["voltage_V"]
    first_val = events[events.partition == "VALIDATION"]
    val_lo = int(first_val.first_row.min()) - 120 if len(first_val) else 0
    # The row-to-ID map exists only at actual frozen voltage triggers.
    trigger_id = np.zeros(n, np.int32)
    for event in events.itertuples():
        lo, hi = int(event.first_row), int(event.last_row)
        rows = np.arange(lo, hi + 1)
        rows = rows[(rows > 0) & (arrays["segment"][rows] == arrays["segment"][rows - 1])]
        rows = rows[np.abs(voltage[rows] - voltage[rows - 1]) > THRESHOLD_V]
        trigger_id[rows] = int(event.event_id)
    lookup = events.set_index("event_id").category.to_dict()
    summaries = []; groups = []; conditions = []; leads = []; top_rows = []; categories = []; contribution = []
    source = []
    for name, model, horizon, run in runs:
        print(json.dumps({"stage": "CPU_QUERY_DIAGNOSTIC_START", "run": name,
                          "horizon_seconds": horizon}, ensure_ascii=False), flush=True)
        path = run / "predictions_VALIDATION_60s.npz"
        with np.load(path, allow_pickle=False) as data:
            origins = data["origins"]; targets = data["targets"]
            origin_time_ms = data["origin_time_ms"]
            target_time_ms = data["target_time_ms"]
            with np.load(OUT / f"origins_H{horizon}.npz", allow_pickle=False) as frozen_origins:
                assert np.array_equal(origins, frozen_origins["VALIDATION"]), "Unexpected Validation origin selection"
            truth = data["truth_delta_V"].astype(np.float64)
            prediction = data["prediction_delta_V"].astype(np.float64)
            event = data["event_query"].astype(bool)
            assert targets.shape == truth.shape == prediction.shape == (len(origins), horizon)
            assert np.array_equal(targets, origins[:, None] + np.arange(1, horizon + 1))
            assert np.array_equal(target_time_ms, arrays["time_ms"][targets])
            assert np.array_equal(origin_time_ms, arrays["time_ms"][origins])
            assert np.max(np.abs(truth - (voltage[targets] - voltage[origins, None]))) < 1e-12
            actual_event = np.maximum.accumulate(np.abs(np.diff(np.column_stack(
                (voltage[origins], voltage[targets])), axis=1)) > THRESHOLD_V, axis=1)
            assert np.array_equal(event, actual_event)
            episode = np.maximum.accumulate(trigger_id[targets], axis=1)
            episode[~event] = 0
            assert np.all(episode[event] > 0), "Catalog must contain every actual Validation trigger"
            error = (prediction - truth).ravel(); flat_truth = truth.ravel()
            flat_id = episode.ravel(); sse = error ** 2
            k = int(math.ceil(len(error) * .01))
            # Stable ordering makes equal squared errors reproducible.
            ordered = np.argsort(-sse, kind="stable")
            top = ordered[:k]
            all_sse = float(sse.sum())
            cumulative = np.cumsum(sse[ordered])
            for fraction in (0.001, .005, .01, .02, .05, .1, .2, .5, 1.0):
                count = min(len(error), int(math.ceil(len(error) * fraction)))
                contribution.append({"run": name, "model": model, "horizon_seconds": horizon,
                                     "largest_error_query_fraction": fraction, "queries": count,
                                     "cumulative_SSE_fraction": float(cumulative[count - 1] / all_sse) if all_sse else 0.0})
            summary = {"run": name, "model": model, "horizon_seconds": horizon,
                       **query_metrics(flat_truth, error), "top_1pct_queries": k,
                       "top_1pct_SSE_V2": float(sse[top].sum()),
                       "top_1pct_SSE_fraction": float(sse[top].sum() / all_sse) if all_sse else 0.0,
                       "independent_operational_episodes_reached": int(len(np.unique(flat_id[flat_id > 0]))),
                       "normal_SSE_fraction": float(sse[flat_id == 0].sum() / all_sse) if all_sse else 0.0}
            summaries.append(summary)
            query_frame = pd.DataFrame({"event_id": flat_id, "truth": flat_truth, "error": error,
                                        "origin": np.repeat(origins, horizon), "target": targets.ravel()})
            for event_id, group in query_frame.groupby("event_id", sort=True):
                e = group.error.to_numpy(); y = group.truth.to_numpy()
                part = {"run": name, "model": model, "horizon_seconds": horizon,
                        "event_id": int(event_id), "category": lookup.get(int(event_id), "ordinary_no_voltage_trigger"),
                        **query_metrics(y, e),
                        "all_SSE_fraction": float(np.sum(e ** 2) / all_sse) if all_sse else 0.0,
                        "unique_origins": int(group.origin.nunique()),
                        "unique_target_seconds": int(group.target.nunique())}
                groups.append(part)
            query_frame["category"] = query_frame.event_id.map(lookup).fillna("ordinary_no_voltage_trigger")
            for category, group in query_frame.groupby("category", sort=True):
                categories.append({"run": name, "model": model, "horizon_seconds": horizon,
                                   "category": category, "episodes": int(group.event_id[group.event_id > 0].nunique()),
                                   **query_metrics(group.truth.to_numpy(), group.error.to_numpy())})
            for column in range(horizon):
                leads.append({"run": name, "model": model, "horizon_seconds": horizon,
                              "lead_seconds": column + 1,
                              **query_metrics(truth[:, column], (prediction - truth)[:, column])})
            origin_zero = np.abs(arrays["x"][origins, 5]) <= ZERO_A
            target_zero = np.abs(arrays["x"][targets, 5]) <= ZERO_A
            all_zero = np.logical_and.accumulate(np.column_stack((origin_zero, target_zero)), axis=1)[:, 1:]
            all_loaded = np.logical_and.accumulate(np.column_stack((~origin_zero, ~target_zero)), axis=1)[:, 1:]
            for condition, take in (("origin_loaded", np.repeat(~origin_zero, horizon)),
                                    ("origin_low_or_zero", np.repeat(origin_zero, horizon)),
                                    ("target_loaded", (~target_zero).ravel()),
                                    ("target_low_or_zero", target_zero.ravel()),
                                    ("ordinary_stable_low_or_zero_interval", ((~event) & all_zero).ravel()),
                                    ("ordinary_loaded_interval", ((~event) & all_loaded).ravel()),
                                    ("ordinary_mixed_current_interval", ((~event) & ~all_zero & ~all_loaded).ravel()),
                                    ("event_query", event.ravel()), ("ordinary_query", (~event).ravel())):
                conditions.append({"run": name, "model": model, "horizon_seconds": horizon,
                                   "condition": condition, **query_metrics(flat_truth[take], error[take])})
            for rank, index in enumerate(top):
                i, lead = divmod(int(index), horizon)
                top_rows.append({"run": name, "model": model, "horizon_seconds": horizon,
                                 "rank": rank + 1, "origin_row": int(origins[i]), "lead_seconds": lead + 1,
                                 "target_row": int(targets[i, lead]),
                                 "origin_time_ms": int(origin_time_ms[i]),
                                 "target_time_ms": int(target_time_ms[i, lead]),
                                 "event_id": int(flat_id[index]), "error_V": float(error[index]),
                                 "squared_error_V2": float(sse[index])})
            source.append({"run": name, "predictions_sha256": sha(path),
                           "queries": int(truth.size), "origins": len(origins)})
            print(json.dumps({"stage": "CPU_QUERY_DIAGNOSTIC_COMPLETE", "run": name,
                              "queries": int(truth.size),
                              "top_1pct_SSE_fraction": summary["top_1pct_SSE_fraction"]},
                             ensure_ascii=False), flush=True)
    for filename, records in (("extreme_error_summary.csv", summaries),
                              ("error_by_independent_episode.csv", groups),
                              ("error_by_current_condition.csv", conditions),
                              ("error_by_event_category.csv", categories),
                              ("error_contribution_curve.csv", contribution),
                              ("error_by_forecast_lead.csv", leads),
                              ("top_1pct_prediction_queries.csv", top_rows)):
        pd.DataFrame(records).to_csv(DIAG / filename, index=False, lineterminator="\n")
    result = {"status": "COMPLETE", "all_queries_retained": True,
              "rules_sha256": sha(OUT / "diagnostic_rules.json"),
              "event_catalog_sha256": sha(OUT / "event_catalog.csv"),
              "sources": source, "all_query_summaries": summaries,
              "limits": ["Operational episodes merged by a preregistered quiet gap are not experimentally proven independent physical causes.",
                         "At H120 the same target second may be queried from overlapping origins. Contributions use all origin x lead queries, not independent repeats.",
                         "A query following more than one episode is attached once to its latest voltage-trigger episode; this assignment is bookkeeping, not causal attribution.",
                         "Top 1% is a diagnostic rank on all Validation errors and never removes any prediction from the main metrics."]}
    write_json(DIAG / "extreme_error_diagnostics.json", result)
    return result


def run() -> None:
    rules = read_json(OUT / "diagnostic_rules.json")
    DIAG.mkdir(parents=True, exist_ok=True)
    cfg, arrays = cache()
    assert sha(OLD / "outputs/frozen_protocol.json") == rules["old_protocol_sha256"]
    events = pd.read_csv(OUT / "event_catalog.csv")
    assert set(events.partition.unique()).issubset({"TRAIN", "VALIDATION"})
    events = classify_events(arrays, rules, events)
    precursor = precursor_study(arrays, cfg, rules, events)
    result = attribute_queries(arrays, rules, events, available_runs())
    print(json.dumps({"stage": "CPU_EVENT_DIAGNOSTICS", "runs": len(result["sources"]),
                      "episodes": len(events), "precursor_rows": len(precursor["fixed_grid"]),
                      "GPU_calls": 0, "optimizer_steps": 0}, ensure_ascii=False), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--freeze-rules", action="store_true")
    group.add_argument("--revise-before-main-freeze", action="store_true")
    group.add_argument("--run", action="store_true")
    args = parser.parse_args()
    if args.freeze_rules or args.revise_before_main_freeze:
        result = freeze_rules() if args.freeze_rules else revise_rules_before_main_freeze()
        print(json.dumps({"stage": "FREEZE_TRAIN_ONLY_DIAGNOSTIC_RULES",
                          "train_origins": result["train_threshold_fit_origins"],
                          "rules_sha256": sha(OUT / "diagnostic_rules.json")}, ensure_ascii=False), flush=True)
    else:
        run()


if __name__ == "__main__":
    main()
