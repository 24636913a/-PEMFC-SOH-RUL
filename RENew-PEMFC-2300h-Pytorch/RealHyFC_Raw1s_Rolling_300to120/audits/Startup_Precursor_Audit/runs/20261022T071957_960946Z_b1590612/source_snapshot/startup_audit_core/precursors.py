"""Strictly causal fixed signals; Train-only thresholds; all legal timestamps."""
import numpy as np
import pandas as pd
from .data_io import SIGNALS, col, date, dump, match_past, partitions, sha

FEATURES = (('coolant_in_120', 'coolant_in', 120), ('anode_p_1', 'anode_p', 1),
            ('anode_p_30', 'anode_p', 30), ('anode_p_120', 'anode_p', 120),
            ('cathode_p_30', 'cathode_p', 30), ('coolant_out_120', 'coolant_out', 120))


def ratio(a, b):
    return float(a/b) if b else None


def build_features(arrays, run, cfg, log):
    """Endpoint rises at real DateTime offsets. Never use future values."""
    n = len(arrays['time_ns']); ts = arrays['time_ns']
    path = run/'cache/causal_features.bin'
    features = np.memmap(path, mode='w+', dtype='float64', shape=(n, len(FEATURES)))
    eligible = np.memmap(run/'cache/past120_legal.bin', mode='w+', dtype='uint8', shape=(n,))
    for part, (lo, hi) in partitions(n).items():
        for start in range(lo, hi, cfg['chunk_rows']):
            rows = np.arange(start, min(hi, start+cfg['chunk_rows']))
            previous, ok = match_past(arrays, ts[rows]-120_000_000_000, rows, cfg)
            eligible[rows] = ok & (previous >= lo)
            for j, (_, signal, seconds) in enumerate(FEATURES):
                previous, found = match_past(arrays, ts[rows]-int(seconds*1e9), rows, cfg)
                values = col(arrays, SIGNALS[signal])
                delta = np.asarray(values[rows]-values[previous])
                delta[~(found & (previous >= lo))] = np.nan
                features[rows, j] = delta
        log(f'{part}因果前兆特征已提取：只用起点及以前的真实记录')
    features.flush(); eligible.flush()
    lo, hi = partitions(n)['TRAIN']; cuts = []; fit = []
    for j, (name, signal, seconds) in enumerate(FEATURES):
        values = np.asarray(features[lo:hi, j])
        valid = np.asarray(eligible[lo:hi], bool) & np.isfinite(values)
        cut = max(0.0, float(np.quantile(values[valid], cfg['causal_feature_quantile']))) if valid.any() else np.nan
        cuts.append(cut)
        fit.append(dict(feature=name, sensor=SIGNALS[signal], lookback_seconds=seconds,
                        threshold=cut, finite_train_origins=int(valid.sum()),
                        formula='x(t)-x(asof(t-lookback)); both timestamps <= t',
                        comparator='strict > max(0, Train Q.995); equality is not alarm'))
    dump(run/'frozen_precursor_rules.json', dict(fit_partition='TRAIN', signals=fit,
         combinations=dict(thermal_OR_pressure='coolant_in_120 OR anode_p_30 OR cathode_p_30',
                           thermal_AND_pressure='coolant_in_120 AND (anode_p_30 OR cathode_p_30)'),
         no_label_used_for_threshold_fitting=True, warning_evaluation_stride=1,
         causal_features_sha256=sha(path), causal_feature_cache=dict(path='cache/causal_features.bin', dtype='float64', shape=[n, len(FEATURES)]),
         history_mask=dict(path='cache/past120_legal.bin', dtype='uint8', shape=[n], sha256=sha(run/'cache/past120_legal.bin')),
         no_holdout_rule_selection=True))
    return features, eligible, np.asarray(cuts)


def event_snapshots(arrays, events, targets, cfg, run):
    recovery = dict(targets['LOW_ZERO_RECOVERY']); records = []; onset = []
    for event in events.itertuples(index=False):
        if pd.isna(event.first_positive_row):
            continue
        anchor = int(event.first_positive_row)
        for offset in cfg['precursor_seconds']:
            desired = int(arrays['time_ns'][anchor])-int(offset*1e9)
            matched, ok = match_past(arrays, np.asarray([desired]), np.asarray([anchor]), cfg)
            row = int(matched[0]); supported = bool(ok[0])
            for name, field in SIGNALS.items():
                record = dict(event_id=int(event.event_id), partition=event.partition, low_zero_recovery_proxy=int(event.event_id) in recovery,
                              signal=name, field=field, seconds_before=offset, event_DateTime=date(arrays, anchor),
                              requested_DateTime=str(np.datetime64(desired, 'ns')), matched=supported,
                              observed_DateTime=date(arrays, row) if supported else None,
                              observed_global_row=row if supported else None,
                              observed_file_id=int(arrays['file_id'][row]) if supported else None,
                              observed_source_row=int(arrays['source_row'][row]) if supported else None,
                              actual_seconds_before=(int(arrays['time_ns'][anchor])-int(arrays['time_ns'][row]))/1e9 if supported else None,
                              value=float(col(arrays, field)[row]) if supported else np.nan,
                              interpretation='event-time snapshot, NOT advance input' if offset == 0 else 'past-only snapshot')
                for seconds in (30, 120):
                    if not supported:
                        record.update({f'past{seconds}_delta': np.nan, f'past{seconds}_slope_per_s': np.nan, f'past{seconds}_range': np.nan})
                        continue
                    previous, found = match_past(arrays, np.asarray([int(arrays['time_ns'][row])-int(seconds*1e9)]), np.asarray([row]), cfg)
                    left = int(previous[0]); values = np.asarray(col(arrays, field)[left:row+1])
                    times = (np.asarray(arrays['time_ns'][left:row+1])-int(arrays['time_ns'][left]))/1e9
                    good = bool(found[0]) and len(values) >= 2 and np.isfinite(values).all()
                    denominator = np.sum((times-times.mean())**2)
                    record[f'past{seconds}_delta'] = float(values[-1]-values[0]) if good else np.nan
                    record[f'past{seconds}_slope_per_s'] = float(np.dot(times-times.mean(), values-values.mean())/denominator) if good and denominator else np.nan
                    record[f'past{seconds}_range'] = float(np.ptp(values)) if good else np.nan
                records.append(record)
    cols = ['event_id', 'partition', 'low_zero_recovery_proxy', 'signal', 'seconds_before', 'value']
    table = pd.DataFrame(records) if records else pd.DataFrame(columns=cols)
    table.to_csv(run/'precursor_features.csv', index=False)
    if len(table):
        table.groupby(['partition', 'low_zero_recovery_proxy', 'signal', 'seconds_before']).value.agg(['count', 'mean', 'std', 'median', 'min', 'max']).reset_index().to_csv(run/'temperature_pressure_plot_data.csv', index=False)
    return table


def flags_and_support(feature_values, cuts):
    valid = np.isfinite(feature_values) & np.isfinite(cuts)[None, :]
    singles = feature_values > cuts[None, :]
    joint = valid[:, 0] & valid[:, 2] & valid[:, 4]
    either = singles[:, 0] | singles[:, 2] | singles[:, 4]
    both = singles[:, 0] & (singles[:, 2] | singles[:, 4])
    return np.column_stack((singles, either, both)), np.column_stack((valid, joint, joint))


def observable_onsets(arrays, events, features, past_legal, cuts, cfg, run):
    """First registered signal within fixed past context, not a physical onset."""
    records = []
    for event in events.itertuples(index=False):
        if pd.isna(event.first_positive_row):
            continue
        anchor = int(event.first_positive_row); t = int(arrays['time_ns'][anchor])
        left = int(np.searchsorted(arrays['time_ns'], t-int(cfg['context_seconds']*1e9)))
        origins = np.arange(left, anchor)
        same = (arrays['segment'][origins] == arrays['segment'][anchor]) & np.asarray(past_legal[origins], bool)
        flags, valid = flags_and_support(np.asarray(features[origins]), cuts)
        for j, name in enumerate(cfg['rule_names']):
            use = same & valid[:, j]; hits = np.flatnonzero(use & flags[:, j])
            first = int(origins[hits[0]]) if len(hits) else None
            first_eligible = np.flatnonzero(use)
            records.append(dict(event_id=event.event_id, partition=event.partition, rule=name,
                                observable_before_event=first is not None,
                                first_observed_DateTime=date(arrays, first),
                                first_observed_lead_seconds=(t-int(arrays['time_ns'][first]))/1e9 if first is not None else np.nan,
                                left_censored_onset=bool(len(hits) and len(first_eligible) and hits[0] == first_eligible[0]),
                                limitation='first signal in fixed past context only; onset earlier than context is unknown'))
    pd.DataFrame(records).to_csv(run/'precursor_onsets.csv', index=False)


def alarm_episodes(rows, labels, arrays, cfg):
    if not len(rows):
        return 0, 0
    times = arrays['time_ns'][rows]
    split = np.r_[True, (np.diff(times) > cfg['warning_alarm_merge_seconds']*1e9) | (np.diff(arrays['segment'][rows]) != 0)]
    starts = np.flatnonzero(split)
    any_hit = np.logical_or.reduceat(labels, starts)
    return len(starts), int((~any_hit).sum())


def evaluate(arrays, events, targets, features, past_legal, cuts, cfg, run, log):
    ts = arrays['time_ns']; n = len(ts); rows_out = []; event_out = []; normal_out = []
    for part, (lo, hi) in partitions(n).items():
        origins = np.arange(lo, hi)
        values = np.asarray(features[lo:hi]); alarms, signal_ok = flags_and_support(values, cuts)
        base = np.asarray(past_legal[lo:hi], dtype=bool)
        # Ordinary controls are outside EVERY voltage episode +/-120s,
        # including negative events, rather than only no-recovery labels.
        ep = events[events.partition == part]
        starts = np.asarray([int(ts[int(r)])-120_000_000_000 for r in ep.first_row], dtype=np.int64)
        ends = np.asarray([int(ts[int(r)])+120_000_000_000 for r in ep.last_row], dtype=np.int64)
        if len(starts):
            index = np.searchsorted(starts, ts[lo:hi], side='right')-1
            covered = (index >= 0) & (ts[lo:hi] <= np.maximum.accumulate(ends)[np.maximum(index, 0)])
        else:
            covered = np.zeros(hi-lo, dtype=bool)
        for j, (name, _, _) in enumerate(FEATURES):
            for state, mask in (('ordinary_outside_all_event_contexts', ~covered), ('event_contexts_retrospective', covered)):
                use = base & signal_ok[:, j] & mask
                x = values[use, j]
                normal_out.append(dict(partition=part, feature=name, group=state, observations=len(x),
                                       fraction_above_train_threshold=ratio(int(np.sum(x > cuts[j])), len(x)),
                                       median=float(np.median(x)) if len(x) else None,
                                       p95=float(np.quantile(x, .95)) if len(x) else None))
        for horizon in cfg['prediction_horizons']:
            future, future_ok = match_past(arrays, ts[origins]+int(horizon*1e9), origins, cfg)
            legal = base & future_ok & (future < hi) & (future > origins)
            for scope, target_list in targets.items():
                selected = [(event_id, row) for event_id, row in target_list if lo <= row < hi]
                erows = np.asarray([r for _, r in selected], dtype=np.int64)
                etimes = ts[erows]
                next_event = np.searchsorted(etimes, ts[lo:hi], side='right')
                y = np.zeros(hi-lo, dtype=bool)
                candidate = next_event < len(etimes)
                if len(etimes):
                    y[candidate] = etimes[next_event[candidate]] <= ts[origins[candidate]]+int(horizon*1e9)
                for j, rule in enumerate(cfg['rule_names']):
                    use = legal & signal_ok[:, j]
                    flag = alarms[:, j] & use
                    tp = int(np.sum(flag & y)); fp = int(np.sum(flag & ~y))
                    fn = int(np.sum(use & ~alarms[:, j] & y)); tn = int(np.sum(use & ~alarms[:, j] & ~y))
                    alarm_rows = origins[flag]
                    alarm_count, false_alarm_count = alarm_episodes(alarm_rows, y[flag], arrays, cfg)
                    trial_count = detected = exact_count = exact_detected = 0
                    exposure_hours = float(np.sum(ts[origins[use]]-ts[np.maximum(origins[use]-1, 0)]))/3.6e12
                    leadtimes = []
                    for event_id, erow in selected:
                        etime = int(ts[erow])
                        a = max(lo, int(np.searchsorted(ts, etime-int(horizon*1e9), side='left')))
                        b = min(hi, int(np.searchsorted(ts, etime, side='left')))
                        eligible = bool(np.any(use[a-lo:b-lo]))
                        hit_rows = np.flatnonzero(flag[a-lo:b-lo])+a
                        hit = bool(len(hit_rows)); first_lead = (etime-int(ts[int(hit_rows[0])]))/1e9 if hit else np.nan
                        exact, found = match_past(arrays, np.asarray([etime-int(horizon*1e9)]), np.asarray([erow]), cfg)
                        exact_row = int(exact[0])
                        exact_eligible = bool(found[0]) and lo <= exact_row < hi and bool(base[exact_row-lo] and signal_ok[exact_row-lo, j])
                        # Approx H-second advance snapshot is separate from
                        # the future-within-H query label (jitter may make lead H+.001s).
                        exact_hit = exact_eligible and bool(alarms[exact_row-lo, j])
                        trial_count += eligible; detected += eligible and hit
                        exact_count += exact_eligible; exact_detected += exact_hit
                        if hit:
                            leadtimes.append(first_lead)
                        event_out.append(dict(partition=part, target_scope=scope, horizon_seconds=horizon, rule=rule,
                                              event_id=event_id, event_row=erow, event_DateTime=date(arrays, erow),
                                              eligible_any_origin=eligible, detected_within_horizon=hit,
                                              earliest_alarm_lead_seconds=first_lead, exact_H_eligible=exact_eligible,
                                              exact_H_detected=exact_hit, exact_origin_row=exact_row if exact_eligible else None,
                                              exact_actual_lead_seconds=(etime-int(ts[exact_row]))/1e9 if exact_eligible else None))
                    rows_out.append(dict(partition=part, target_scope=scope, rule=rule, horizon_seconds=horizon,
                                         eligible_origins=int(use.sum()), TP=tp, FP=fp, FN=fn, TN=tn,
                                         precision=ratio(tp, tp+fp), query_recall=ratio(tp, tp+fn), false_positive_rate=ratio(fp, fp+tn),
                                         event_base_rate=ratio(tp+fn, int(use.sum())), target_operational_events=len(selected),
                                         effective_event_trials=trial_count, event_coverage=ratio(trial_count, len(selected)),
                                         detected_events=detected, event_recall=ratio(detected, trial_count),
                                         exact_H_eligible_events=exact_count, exact_H_detected_events=exact_detected,
                                         exact_H_event_recall=ratio(exact_detected, exact_count),
                                         earliest_alarm_lead_median_s=float(np.median(leadtimes)) if leadtimes else None,
                                         alarm_episodes=alarm_count, false_alarm_episodes=false_alarm_count,
                                         valid_exposure_hours=exposure_hours,
                                         false_alarm_episodes_per_valid_hour=ratio(false_alarm_count, exposure_hours),
                                         independence='operational event IDs only; overlapping query rows are NOT independent trials'))
            log(f'{part}未来{horizon}秒预警固定规则已评价：全部合法逐秒起点')
    metrics = pd.DataFrame(rows_out)
    metrics.to_csv(run/'precursor_evaluation.csv', index=False)
    pd.DataFrame(event_out).to_csv(run/'precursor_event_detection.csv', index=False)
    pd.DataFrame(normal_out).to_csv(run/'ordinary_signal_controls.csv', index=False)
    # Collapse recall by same event identities, never count horizons as new subjects.
    dump(run/'precursor_interpretation.json', dict(definition='within next H seconds is not exactly H seconds in advance',
         exact_H_comparison='separate causal snapshot before each event at requested H seconds; actual DateTime lag retained',
         false_positive_rate='FP/(FP+TN)', precision='TP/(TP+FP)',
         overlapping_queries_are_independent=False, horizons_are_independent_events=False,
         statistical_significance_claimed=False, holdout='descriptive fixed-rule evaluation; no retuning'))
    return metrics
