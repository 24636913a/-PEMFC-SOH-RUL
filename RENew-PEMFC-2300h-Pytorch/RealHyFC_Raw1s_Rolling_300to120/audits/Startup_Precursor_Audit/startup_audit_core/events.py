"""Operational voltage episodes, current-state proxies, and restart timelines."""
import csv
import json
import numpy as np
import pandas as pd
from .data_io import COLUMNS, INDEX, SIGNALS, col, date, dump, partitions, sha, match_past


def detect(arrays, run, cfg, log):
    n = len(arrays['time_ns']); parts = partitions(n)
    train = np.asarray(arrays['delta_V'][:parts['TRAIN'][1]])
    finite = np.abs(train[np.isfinite(train)])
    threshold = float(np.quantile(finite, cfg['event_quantile']))
    if not np.isfinite(threshold) or threshold <= 0:
        raise RuntimeError('Train voltage threshold undefined/nonpositive; do not tune on later events')
    if cfg['event_threshold_mode'] == 'HISTORICAL_FIXED':
        threshold = cfg['historical_threshold_V']
    elif cfg['event_threshold_mode'] != 'TRAIN_Q995':
        raise ValueError('Unsupported preregistered event threshold mode')
    dump(run/'voltage_threshold.json', dict(threshold_V=threshold, fit_partition='TRAIN',
         mode=cfg['event_threshold_mode'], historical_reference_V=cfg['historical_threshold_V'],
         train_adjacent_pairs=len(finite), quantile=cfg['event_quantile'], comparator='strict >',
         merge_quiet_seconds=cfg['merge_quiet_seconds']))
    records = []; active = None; event_id = 0; reference = []
    names = [p['path'].split('/')[-1] for p in json.loads((run/'source_manifest.json').read_text(encoding='utf-8'))['source_files']]
    with (run/'voltage_trigger_records.csv').open('w', newline='', encoding='utf-8') as stream:
        fields = ('event_id', 'partition', 'global_row', 'source_file', 'source_row', 'DateTime', 'delta_V', 'direction')
        writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader()
        for part, (lo, hi) in parts.items():
            active = None; total_triggers = legacy_triggers = 0
            for begin in range(lo, hi, cfg['chunk_rows']):
                end = min(hi, begin+cfg['chunk_rows'])
                delta = np.asarray(arrays['delta_V'][begin:end])
                valid = np.isfinite(delta)
                if begin == lo:
                    valid[0] = False
                legacy_triggers += int(np.sum(valid & (np.abs(delta) > cfg['historical_threshold_V'])))
                hits = np.flatnonzero(valid & (np.abs(delta) > threshold))+begin
                hits = hits[hits > lo]  # No adjacent label crossing a partition boundary.
                for row in hits:
                    row = int(row); t = int(arrays['time_ns'][row]); d = float(arrays['delta_V'][row])
                    if active is None or arrays['segment'][row] != active['segment'] or t-active['_last_ns'] > cfg['merge_quiet_seconds']*1e9:
                        event_id += 1
                        active = dict(event_id=event_id, partition=part, segment=int(arrays['segment'][row]),
                                      first_row=row, last_row=row, first_DateTime=date(arrays, row), last_DateTime=date(arrays, row),
                                      first_positive_row=None, first_negative_row=None, positive_triggers=0, negative_triggers=0,
                                      max_positive_delta_V=None, min_negative_delta_V=None, _last_ns=t)
                        active.update(source_file=names[int(arrays['file_id'][row])], first_source_row=int(arrays['source_row'][row]))
                        records.append(active)
                    active.update(last_row=row, last_DateTime=date(arrays, row), _last_ns=t)
                    sign = 'positive' if d > 0 else 'negative'
                    active[sign+'_triggers'] += 1
                    if active['first_'+sign+'_row'] is None:
                        active['first_'+sign+'_row'] = row
                    if d > 0:
                        old = active['max_positive_delta_V']
                        active['max_positive_delta_V'] = d if old is None else max(old, d)
                    else:
                        old = active['min_negative_delta_V']
                        active['min_negative_delta_V'] = d if old is None else min(old, d)
                    writer.writerow(dict(event_id=event_id, partition=part, global_row=row,
                                         source_file=names[int(arrays['file_id'][row])], source_row=int(arrays['source_row'][row]),
                                         DateTime=date(arrays, row), delta_V=d, direction=sign))
                    total_triggers += 1
            reference.append(dict(partition=part, registered_triggers=total_triggers,
                                  historical_threshold_triggers=legacy_triggers,
                                  interpretation='threshold comparison only, not selection'))
            log(f'{part}触发记录与120秒静默间隔事件段已建立')
    for event in records:
        event.pop('_last_ns', None)
        event['direction'] = 'mixed' if event['positive_triggers'] and event['negative_triggers'] else ('positive' if event['positive_triggers'] else 'negative')
    columns = ['event_id', 'partition', 'segment', 'first_row', 'last_row', 'first_DateTime', 'last_DateTime',
               'first_positive_row', 'first_negative_row', 'positive_triggers', 'negative_triggers',
               'max_positive_delta_V', 'min_negative_delta_V', 'direction']
    columns += ['source_file', 'first_source_row']
    events = pd.DataFrame(records, columns=columns)
    events.to_csv(run/'event_catalog.csv', index=False)
    pd.DataFrame(reference).to_csv(run/'historical_threshold_comparison.csv', index=False)
    return events, threshold


def spells(mask, segments):
    """Inclusive intervals, censored at source/time/voltage gaps."""
    mask = np.asarray(mask, dtype=bool)
    seam = np.r_[True, segments[1:] != segments[:-1]]
    starts = np.flatnonzero(mask & (seam | ~np.r_[False, mask[:-1]]))
    ends = np.flatnonzero(mask & (np.r_[seam[1:], True] | ~np.r_[mask[1:], False]))
    if len(starts) != len(ends):
        raise RuntimeError('State interval alignment failure')
    return starts, ends


def context_indices(arrays, row, seconds):
    ts = arrays['time_ns']; segment = arrays['segment'][row]
    left = int(np.searchsorted(ts, int(ts[row])-seconds*1_000_000_000))
    right = int(np.searchsorted(ts, int(ts[row])+seconds*1_000_000_000, side='right'))
    index = np.arange(left, right)
    return index[arrays['segment'][index] == segment]


def classify(arrays, events, cfg, run):
    records = []; transitions = []; current = col(arrays, 'I.S.act [A]'); command = col(arrays, 'I.S.set [A]')
    recovery_rows = []; positive_rows = []
    for event in events.itertuples(index=False):
        row = int(event.first_positive_row) if pd.notna(event.first_positive_row) else int(event.first_row)
        index = context_indices(arrays, row, cfg['context_seconds'])
        x = arrays['values'][index]; i = x[:, INDEX['I.S.act [A]']]
        finite_pairs = np.isfinite(i[1:]) & np.isfinite(i[:-1])
        zero = np.abs(i) <= cfg['zero_current_A']
        down = bool(np.any(finite_pairs & ~zero[:-1] & zero[1:]))
        up = bool(np.any(finite_pairs & zero[:-1] & ~zero[1:]))
        c = x[:, INDEX['I.S.set [A]']]
        loaded_step = bool(np.any(finite_pairs & (np.abs(np.diff(i)) >= 5) & ~zero[:-1] & ~zero[1:]))
        low_recovery = (pd.notna(event.first_positive_row) and row > 0 and
                        arrays['segment'][row] == arrays['segment'][row-1] and
                        np.isfinite(current[row-1:row+1]).all() and np.isfinite(command[row-1:row+1]).all() and
                        np.all(np.abs(current[row-1:row+1]) <= cfg['zero_current_A']) and
                        np.all(np.abs(command[row-1:row+1]) <= cfg['zero_current_A']))
        if pd.notna(event.first_positive_row):
            positive_rows.append((int(event.event_id), row))
        if low_recovery:
            recovery_rows.append((int(event.event_id), row))
        fields = dict(event_id=event.event_id, partition=event.partition, anchor_row=row, anchor_DateTime=date(arrays, row),
                      A_loaded_to_zero_proxy=down, B_zero_to_loaded_proxy=up, C_low_zero_voltage_recovery_proxy=bool(low_recovery),
                      D_loaded_current_step_proxy=loaded_step, context_records=len(index),
                      context_before_seconds=(int(arrays['time_ns'][row])-int(arrays['time_ns'][index[0]]))/1e9,
                      context_after_seconds=(int(arrays['time_ns'][index[-1]])-int(arrays['time_ns'][row]))/1e9,
                      current_missing_records=int((~np.isfinite(i)).sum()), evidence='observational_proxy_not_hardware_cause')
        accompanied = False
        for name, field in SIGNALS.items():
            values = x[:, INDEX[field]]; v = values[np.isfinite(values)]
            value_range = float(np.ptp(v)) if len(v) else np.nan
            fields[name+'_range'] = value_range
            # Descriptive project cutoffs, NOT claimed instrument precision.
            accompanied |= bool(len(v) and value_range > (1.0 if 'p' in name and name.endswith('_p') else .1))
        fields['E_temperature_pressure_association_proxy'] = accompanied
        fields['F_not_explained_by_registered_proxies'] = not (down or up or low_recovery or loaded_step or accompanied)
        records.append(fields)
        for field in ('I.S.act [A]', 'I.S.set [A]', 'I.S.set.internal [A]', 'operation_point',
                      *SIGNALS.values(), 'F.Si.H.A [nl/min]', 'F.Si.Air.C [nl/min]'):
            values = np.asarray(x[:, INDEX[field]])
            changed = np.flatnonzero(np.isfinite(values[1:]) & np.isfinite(values[:-1]) & (values[1:] != values[:-1]))+1
            lags = (arrays['time_ns'][index[changed]]-int(arrays['time_ns'][row]))/1e9
            for kind, mask in (('nearest_strictly_before', lags < 0), ('at_event', lags == 0), ('first_strictly_after', lags > 0)):
                candidate = np.flatnonzero(mask)
                j = int(candidate[np.argmin(np.abs(lags[candidate]))]) if len(candidate) else None
                k = int(changed[j]) if j is not None else None
                transitions.append(dict(event_id=event.event_id, variable=field, timing=kind,
                                        change_DateTime=date(arrays, int(index[k])) if k is not None else None,
                                        change_global_row=int(index[k]) if k is not None else None,
                                        lag_seconds=float(lags[j]) if j is not None else np.nan,
                                        step_delta=float(values[k]-values[k-1]) if k is not None else np.nan,
                                        interpretation='recorded value change only; not physical onset or validated precursor'))
        view = pd.DataFrame(np.asarray(x), columns=COLUMNS)
        view.insert(0, 'event_id', int(event.event_id)); view.insert(1, 'global_row', index)
        view.insert(2, 'elapsed_seconds', (arrays['time_ns'][index]-int(arrays['time_ns'][row]))/1e9)
        view.insert(3, 'DateTime', np.asarray(arrays['time_ns'][index]).astype('datetime64[ns]').astype(str))
        view.insert(4, 'file_id', np.asarray(arrays['file_id'][index])); view.insert(5, 'source_row', np.asarray(arrays['source_row'][index]))
        view.to_csv(run/'event_contexts.csv', mode='a', header=not (run/'event_contexts.csv').exists(), index=False)
    fields = ['event_id', 'partition', 'anchor_row', 'anchor_DateTime', 'C_low_zero_voltage_recovery_proxy']
    table = pd.DataFrame(records) if records else pd.DataFrame(columns=fields)
    table.to_csv(run/'event_classification.csv', index=False)
    pd.DataFrame(transitions).to_csv(run/'event_transition_lags.csv', index=False)
    summary = []
    for part in partitions(len(arrays['time_ns'])):
        subset = table[table.partition == part]
        for key in table:
            if key.startswith(('A_', 'B_', 'C_', 'D_', 'E_', 'F_')):
                summary.append(dict(partition=part, category=key, events=len(subset), count=int(subset[key].sum()),
                                    proportion=float(subset[key].mean()) if len(subset) else None, mutually_exclusive=False))
    pd.DataFrame(summary).to_csv(run/'event_category_statistics.csv', index=False)
    return table, dict(ALL_POSITIVE=positive_rows, LOW_ZERO_RECOVERY=recovery_rows)


def startup_timelines(arrays, events, threshold, cfg, run):
    ts = arrays['time_ns']; seg = arrays['segment']; n = len(ts)
    command = col(arrays, 'I.S.set [A]'); current = col(arrays, 'I.S.act [A]')
    starts, ends = spells(np.isfinite(command) & (np.abs(command) <= cfg['zero_current_A']), seg)
    load_starts, load_ends = spells(np.isfinite(current) & (current >= cfg['loaded_current_A']), seg)
    load_starts = load_starts[(ts[load_ends]-ts[load_starts]) >= cfg['loaded_sustain_seconds']*1e9]
    records = []; all_spells = []
    event_first = events.first_row.to_numpy(dtype=np.int64)
    for spell_id, (first, last) in enumerate(zip(starts, ends), 1):
        first, last = int(first), int(last)
        command_begin_observed = first > 0 and seg[first-1] == seg[first] and np.isfinite(command[first-1]) and abs(command[first-1]) > cfg['zero_current_A']
        command_end_observed = last+1 < n and seg[last+1] == seg[last] and np.isfinite(command[last+1]) and command[last+1] > cfg['zero_current_A']
        observed_end = last+1 if command_end_observed else last
        all_spells.append(dict(zero_command_spell_id=spell_id, first_row=first, last_zero_row=last,
                               start_DateTime=date(arrays, first), end_DateTime=date(arrays, observed_end),
                               start_observed=bool(command_begin_observed), end_observed=bool(command_end_observed),
                               stop_seconds=(int(ts[observed_end])-int(ts[first]))/1e9))
        d = np.asarray(arrays['delta_V'][first:last+1])
        valid = np.isfinite(d) & (d > threshold)
        if not valid.any():
            continue
        # Description only: strongest positive trigger INSIDE this zero command
        # spell, ties earliest. Never used as a warning feature or warning target.
        anchor = first + int(np.argmax(np.where(valid, d, -np.inf)))
        actual_zero = np.flatnonzero(np.isfinite(current[first:anchor+1]) & (np.abs(current[first:anchor+1]) <= cfg['zero_current_A']))
        zero_row = first+int(actual_zero[0]) if len(actual_zero) else None
        cmd_row = last+1 if command_end_observed else None
        loaded_candidates = load_starts[np.searchsorted(load_starts, anchor+1):]
        loaded_row = int(loaded_candidates[0]) if len(loaded_candidates) and seg[int(loaded_candidates[0])] == seg[anchor] else None
        if loaded_row is not None and cmd_row is not None and ts[loaded_row]-ts[cmd_row] > 300e9:
            loaded_row = None
        if loaded_row is not None and ts[loaded_row]-ts[anchor] > 8*3600e9:
            loaded_row = None
        end_search = min(n, int(np.searchsorted(ts, int(ts[cmd_row if cmd_row is not None else anchor])+300_000_000_000, side='right')))
        v = np.asarray(current[anchor:end_search]); candidates = np.flatnonzero(np.isfinite(v) & (np.abs(v) > cfg['zero_current_A']))
        nonzero = anchor+int(candidates[0]) if len(candidates) else None
        if nonzero is not None and seg[nonzero] != seg[anchor]:
            nonzero = None
        duration = (int(ts[cmd_row])-int(ts[first]))/1e9 if cmd_row is not None else (int(ts[last])-int(ts[first]))/1e9
        comparable = bool(command_begin_observed and command_end_observed and anchor > first and
                          np.isfinite(current[anchor-1:anchor+1]).all() and
                          np.all(np.abs(current[anchor-1:anchor+1]) <= cfg['zero_current_A']))
        event_pos = int(np.searchsorted(event_first, anchor, side='right')-1)
        event_id = int(events.iloc[event_pos].event_id) if event_pos >= 0 and anchor <= int(events.iloc[event_pos].last_row) else None
        part = next(name for name, (lo, hi) in partitions(n).items() if lo <= anchor < hi)
        record = dict(zero_command_spell_id=spell_id, event_id=event_id, partition=part, first_row=first, last_zero_row=last,
                      command_zero_start_DateTime=date(arrays, first), command_start_observed=bool(command_begin_observed),
                      actual_zero_first_DateTime=date(arrays, zero_row), recovery_row=anchor, recovery_DateTime=date(arrays, anchor),
                      recovery_file_id=int(arrays['file_id'][anchor]), recovery_source_row=int(arrays['source_row'][anchor]),
                      actual_zero_first_global_row=zero_row, first_positive_command_global_row=cmd_row,
                      actual_first_nonzero_global_row=nonzero, actual_sustained_load_global_row=loaded_row,
                      recovery_delta_V=float(arrays['delta_V'][anchor]), first_positive_command_DateTime=date(arrays, cmd_row),
                      actual_first_nonzero_DateTime=date(arrays, nonzero), actual_sustained_load_DateTime=date(arrays, loaded_row),
                      stop_seconds=duration, stop_group='long_ge_3600s' if duration >= cfg['long_stop_seconds'] else 'short_lt_3600s',
                      comparable=comparable, right_censored=not bool(command_end_observed),
                      command_lag_seconds=(int(ts[cmd_row])-int(ts[anchor]))/1e9 if cmd_row is not None else np.nan,
                      actual_load_lag_seconds=(int(ts[loaded_row])-int(ts[anchor]))/1e9 if loaded_row is not None else np.nan,
                      actual_nonzero_lag_seconds=(int(ts[nonzero])-int(ts[anchor]))/1e9 if nonzero is not None else np.nan,
                      voltage_at_zero_command_V=float(col(arrays, 'mean_voltage_V')[first]),
                      voltage_immediately_before_recovery_V=float(col(arrays, 'mean_voltage_V')[anchor-1]) if anchor else np.nan,
                      voltage_at_recovery_V=float(col(arrays, 'mean_voltage_V')[anchor]),
                      anchor_selection='largest positive trigger in zero-command spell; retrospective only')
        before = np.asarray(col(arrays, 'mean_voltage_V')[first:anchor+1])
        finite_voltage = before[np.isfinite(before)]
        record['minimum_voltage_before_recovery_V'] = float(finite_voltage.min()) if len(finite_voltage) else np.nan
        later, supported = match_past(arrays, np.asarray([int(ts[anchor])+120_000_000_000]), np.asarray([anchor]), cfg)
        record['voltage_120s_after_recovery_V'] = float(col(arrays, 'mean_voltage_V')[int(later[0])]) if supported[0] else np.nan
        for field, key in (('I.S.set.internal [A]', 'internal_command'), ('F.Si.H.A [nl/min]', 'hydrogen_flow'), ('F.Si.Air.C [nl/min]', 'air_flow'),
                           *[(field, key) for key, field in SIGNALS.items()]):
            local = context_indices(arrays, anchor, cfg['context_seconds'])
            values = np.asarray(col(arrays, field)[local]); changes = np.flatnonzero(np.isfinite(values[1:]) & np.isfinite(values[:-1]) & (values[1:] != values[:-1]))+1
            if len(changes):
                nearest = int(local[changes[np.argmin(np.abs(ts[local[changes]]-ts[anchor]))]])
                record[key+'_nearest_change_lag_seconds'] = (int(ts[nearest])-int(ts[anchor]))/1e9
                record[key+'_nearest_change_DateTime'] = date(arrays, nearest)
            else:
                record[key+'_nearest_change_lag_seconds'] = np.nan
                record[key+'_nearest_change_DateTime'] = None
        descent = np.asarray(arrays['delta_V'][first:anchor+1]); negative = np.flatnonzero(np.isfinite(descent) & (descent < -threshold))
        record['first_voltage_down_trigger_DateTime'] = date(arrays, first+int(negative[0])) if len(negative) else None
        records.append(record)
    pd.DataFrame(all_spells, columns=['zero_command_spell_id', 'first_row', 'last_zero_row', 'start_DateTime', 'end_DateTime',
                                    'start_observed', 'end_observed', 'stop_seconds']).to_csv(run/'zero_command_spells.csv', index=False)
    cols = ['comparable', 'partition', 'stop_group', 'recovery_row', 'command_lag_seconds', 'actual_load_lag_seconds', 'actual_nonzero_lag_seconds']
    timeline = pd.DataFrame(records) if records else pd.DataFrame(columns=cols)
    timeline.to_csv(run/'startup_timeline.csv', index=False)
    summaries = []
    qualified = timeline.loc[timeline.comparable.fillna(False).astype(bool)]
    for (part, group), table in qualified.groupby(['partition', 'stop_group']):
        for kind in ('command_lag_seconds', 'actual_load_lag_seconds', 'actual_nonzero_lag_seconds'):
            values = table[kind].dropna().to_numpy()
            summaries.append(dict(partition=part, stop_group=group, lag_type=kind, comparable_spells=len(table),
                                  observed_lags=len(values), missing_lags=len(table)-len(values),
                                  minimum=np.min(values) if len(values) else None, median=np.median(values) if len(values) else None,
                                  p05=np.quantile(values, .05) if len(values) else None, p95=np.quantile(values, .95) if len(values) else None,
                                  maximum=np.max(values) if len(values) else None,
                                  near65_fraction=np.mean(np.abs(values-65) <= cfg['near_65_tolerance_seconds']) if len(values) else None))
    pd.DataFrame(summaries).to_csv(run/'restart_lag_distribution.csv', index=False)
    qualified[(qualified.command_lag_seconds-65).abs() > cfg['near_65_tolerance_seconds']].to_csv(run/'restart_65_counterexamples.csv', index=False)
    return timeline


def cycle_mapping(arrays, events, timeline, research, cfg, run):
    """Read historical RCR only. Never discover cycles from voltage or extrapolate."""
    root = research/'Phase_RCR_Raw_Resolution_Cycle_Reconstruction_Audit'
    source = root/'outputs/RAW_CYCLE_CATALOG.csv'
    cycles = pd.read_csv(source)
    start = pd.to_datetime(cycles.cycle_start).to_numpy(dtype='datetime64[ns]').astype(np.int64)
    end = pd.to_datetime(cycles.cycle_end).to_numpy(dtype='datetime64[ns]').astype(np.int64)
    if np.any(np.diff(start) <= 0):
        raise RuntimeError('RCR catalog is not chronological')
    rows = []
    for event in events.itertuples(index=False):
        t = int(arrays['time_ns'][int(event.first_row)])
        j = int(np.searchsorted(start, t, side='right')-1)
        ok = j >= 0 and t < end[j]
        rows.append(dict(event_id=event.event_id, rcr_cycle_id=int(cycles.iloc[j].raw_cycle_id) if ok else None,
                         rcr_complete_cycle=bool(str(cycles.iloc[j].complete_cycle).lower() == 'true') if ok else None,
                         phase_seconds=(t-int(start[j]))/1e9 if ok else None,
                         status='historical_RCR_covered' if ok else 'NO_RCR_COVERAGE_NO_EXTRAPOLATION'))
    pd.DataFrame(rows).to_csv(run/'event_cycle_mapping.csv', index=False)
    # Calibrate plan phase ONLY from the earliest RCR complete interval and
    # recorded set-current spells, not from voltage recovery time.
    complete = np.flatnonzero(cycles.complete_cycle.astype(str).str.lower().eq('true').to_numpy())
    templates = []; output = []
    all_spells = pd.read_csv(run/'zero_command_spells.csv')
    if len(complete) and len(all_spells):
        calibration = int(complete[0])
        for row in all_spells.itertuples(index=False):
            t = int(arrays['time_ns'][int(row.first_row)])
            if start[calibration] <= t < end[calibration] and row.start_observed and row.end_observed:
                minutes = np.asarray(cfg['descriptive_plan_stop_minutes'])
                k = int(np.argmin(np.abs(minutes*60-row.stop_seconds)))
                templates.append(dict(template_id=len(templates)+1, phase_seconds=(t-int(start[calibration]))/1e9,
                                      nominal_minutes=int(minutes[k]), observed_zero_command_seconds=float(row.stop_seconds),
                                      duration_match=bool(abs(float(minutes[k])*60-row.stop_seconds) <= cfg['plan_duration_match_tolerance_seconds']),
                                      calibration_cycle_id=int(cycles.iloc[calibration].raw_cycle_id)))
        for row in timeline.itertuples(index=False):
            t = int(arrays['time_ns'][int(row.first_row)])
            j = int(np.searchsorted(start, t, side='right')-1)
            if j < 0 or t >= end[j] or not templates:
                output.append(dict(zero_command_spell_id=row.zero_command_spell_id, status='NO_RCR_OR_CALIBRATION_SUPPORT'))
                continue
            candidates = sorted(templates, key=lambda x: abs(t-(int(start[j])+x['phase_seconds']*1e9)))
            best = candidates[0]; expected = int(start[j])+int(best['phase_seconds']*1e9)
            output.append(dict(zero_command_spell_id=row.zero_command_spell_id, rcr_cycle_id=int(cycles.iloc[j].raw_cycle_id),
                               template_id=best['template_id'], template_nominal_stop_minutes=best['nominal_minutes'],
                               nominal_duration_mapping_qualified=best['duration_match'],
                               nominal_template_command_start=str(np.datetime64(expected, 'ns')),
                               actual_command_minus_template_seconds=(t-expected)/1e9,
                               recovery_minus_template_restart_seconds=(int(arrays['time_ns'][int(row.recovery_row)])-expected)/1e9-best['nominal_minutes']*60,
                               stop_duration_minus_nominal_seconds=row.stop_seconds-best['nominal_minutes']*60,
                               phase_mapping_qualified=bool(abs(t-expected) <= 300e9),
                               status='RETROSPECTIVE_TEMPLATE_CANDIDATE_NOT_VERIFIED_PLAN', future_plan_release_time_known=False))
    pd.DataFrame(output).to_csv(run/'planned_vs_observed_timeline.csv', index=False)
    if output and 'template_nominal_stop_minutes' in pd.DataFrame(output):
        linked = pd.DataFrame(output).merge(timeline, on='zero_command_spell_id', how='left', suffixes=('_mapping', ''))
        linked.to_csv(run/'planned_stop_voltage_comparison.csv', index=False)
    dump(run/'cycle_mapping_identity.json', dict(RCR_catalog_sha256=sha(source),
         RCR_template_sha256=sha(root/'outputs/CYCLE_TEMPLATE.json'),
         historical_complete_cycles=int(len(complete)), calibration_template=templates,
         template_source='earliest historical RCR complete interval, set-current spells; not controller schedule',
         limitation='RCR has limited Development coverage; no new late-life cycle discovery; future plan publication time unknown'))
