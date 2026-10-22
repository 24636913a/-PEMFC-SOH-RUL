"""Bounded cache slices only; identical feature definitions for events and controls."""
import math
import numpy as np
import pandas as pd
from .io_contract import VARIABLES, PHASES

METRICS = ('max_abs_step', 'abs_net_change', 'range')


def context(arrays, anchor, limits, cfg):
    lo, hi = limits
    radius = math.ceil(cfg['phase_seconds']/cfg['instant_dt_min_seconds'])+3
    rows = np.arange(max(lo, anchor-radius), min(hi, anchor+radius+1))
    elapsed = (np.asarray(arrays['time_ns'][rows])-int(arrays['time_ns'][anchor]))/1e9
    same = arrays['segment'][rows] == arrays['segment'][anchor]
    keep = same & (np.abs(elapsed) <= cfg['phase_seconds'])
    return rows[keep], elapsed[keep]


def phase_features(arrays, positions, anchor, limits, cfg, identity):
    rows, elapsed = context(arrays, anchor, limits, cfg)
    records = []
    for phase in PHASES:
        if phase == 'instant':
            index = np.asarray([anchor-1, anchor], dtype=np.int64)
            legal = anchor > limits[0] and arrays['segment'][anchor-1] == arrays['segment'][anchor]
            dt = (int(arrays['time_ns'][anchor])-int(arrays['time_ns'][anchor-1]))/1e9 if legal else np.nan
            legal = legal and cfg['instant_dt_min_seconds'] <= dt <= cfg['instant_dt_max_seconds']
            if not legal:
                index = np.array([], dtype=np.int64)
            times = np.asarray([-dt, 0.0]) if legal else np.array([])
            coverage = 1.0 if legal else 0.0
        else:
            select = elapsed < 0 if phase == 'before' else elapsed > 0
            index, times = rows[select], elapsed[select]
            # Extent and sample count both matter; no event/anchor sample in these phases.
            extent = (times[-1]-times[0]+1.0)/cfg['phase_seconds'] if len(times) >= 2 else 0.0
            count = len(times)/cfg['phase_seconds']
            coverage = min(1.0, extent, count)
            legal = coverage >= cfg['phase_minimum_coverage_fraction']
            if len(times) >= 2:
                legal = legal and bool(np.all((np.diff(times) >= cfg['instant_dt_min_seconds']) &
                                             (np.diff(times) <= cfg['instant_dt_max_seconds'])))
        for variable, (column, family, unit) in VARIABLES.items():
            record = dict(identity, variable=variable, source_column=column, family=family, unit=unit,
                          phase=phase, samples=len(index), coverage_fraction=coverage,
                          finite_fraction=0.0, support='MISSING_COLUMN',
                          max_abs_step=np.nan, signed_net_change=np.nan, abs_net_change=np.nan,
                          range=np.nan, slope_per_second=np.nan, mean_level=np.nan,
                          signed_instant_step=np.nan, largest_step_lag_seconds=np.nan,
                          first_numerical_change_lag_seconds=np.nan, last_numerical_change_lag_seconds=np.nan)
            if column not in positions:
                records.append(record); continue
            values = np.asarray(arrays['values'][index, positions[column]], dtype=float)
            finite = np.isfinite(values)
            fraction = float(finite.mean()) if len(values) else 0.0
            record['finite_fraction'] = fraction
            if not legal or len(values) < 2:
                record['support'] = 'INSUFFICIENT_PHASE_COVERAGE'
            elif fraction < cfg['minimum_finite_fraction'] or not (finite[0] and finite[-1]):
                record['support'] = 'MISSING_VALUES_OR_ENDPOINT'
            else:
                differences = np.diff(values)
                pairs = finite[:-1] & finite[1:]
                if not pairs.any():
                    record['support'] = 'NO_ADJACENT_FINITE_PAIR'
                    records.append(record); continue
                valid_steps = np.abs(differences[pairs])
                best = np.flatnonzero(pairs)[int(np.argmax(valid_steps))]+1
                x, y = times[finite], values[finite]
                centered = x-x.mean(); denominator = np.dot(centered, centered)
                net = float(values[-1]-values[0])
                changed = np.flatnonzero(pairs & (np.abs(differences) > 0))+1
                record.update(support='OK', max_abs_step=float(valid_steps.max()),
                              signed_net_change=net, abs_net_change=abs(net),
                              range=float(np.ptp(y)), mean_level=float(y.mean()),
                              slope_per_second=float(np.dot(centered, y-y.mean())/denominator) if denominator > 0 else np.nan,
                              signed_instant_step=net if phase == 'instant' else np.nan,
                              largest_step_lag_seconds=float(times[best]),
                              first_numerical_change_lag_seconds=float(times[changed[0]]) if len(changed) else np.nan,
                              last_numerical_change_lag_seconds=float(times[changed[-1]]) if len(changed) else np.nan)
            records.append(record)
    return records


def choose_controls(arrays, events, partition, limits, cfg):
    """Calendar-grid selection independent of current/temp/pressure/flow amplitudes."""
    lo, hi = limits; ts = arrays['time_ns']; grid_ns = int(cfg['control_grid_seconds']*1e9)
    first = ((int(ts[lo])+grid_ns-1)//grid_ns)*grid_ns
    grid = np.arange(first, int(ts[hi-1])+1, grid_ns, dtype=np.int64)
    right = np.clip(np.searchsorted(ts, grid), lo, hi-1)
    left = np.clip(right-1, lo, hi-1)
    chosen = np.where(np.abs(ts[left]-grid) <= np.abs(ts[right]-grid), left, right)
    matched = np.abs(ts[chosen]-grid) <= cfg['control_match_tolerance_seconds']*1e9
    chosen = np.unique(chosen[matched])
    all_matched = len(chosen)
    # All directions within THIS partition. Later event identities cannot select Train controls.
    margin = int(cfg['control_exclusion_margin_seconds']*1e9)
    intervals = sorted((int(ts[int(e.first_row)])-margin, int(ts[int(e.last_row)])+margin)
                       for e in events.loc[events.partition.eq(partition)].itertuples())
    merged = []
    for start, stop in intervals:
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], stop)
        else:
            merged.append([start, stop])
    starts, stops = np.asarray(merged, dtype=np.int64).T
    where = np.searchsorted(starts, ts[chosen], side='right')-1
    excluded = (where >= 0) & (ts[chosen] <= stops[np.maximum(where, 0)])
    chosen = chosen[~excluded]
    outside = len(chosen)
    # Legal full support: same continuity segment at both endpoints and original partition.
    radius = math.ceil(cfg['phase_seconds']/cfg['instant_dt_min_seconds'])+3
    valid = []
    for row in chosen:
        indices, elapsed = context(arrays, int(row), limits, cfg)
        if len(indices) < 2 or row <= lo:
            continue
        if elapsed[0] <= -cfg['phase_seconds']+1.1 and elapsed[-1] >= cfg['phase_seconds']-1.1:
            valid.append(int(row))
    cap = cfg['max_controls'][partition]
    selection = np.asarray(valid, dtype=np.int64)
    if len(selection) > cap:
        selection = selection[np.linspace(0, len(selection)-1, cap, dtype=np.int64)]
    audit = dict(partition=partition, calendar_grid_points=len(grid), matched_unique=all_matched,
                 outside_frozen_event_contexts=outside, legal_contexts=len(valid), selected=len(selection),
                 selection_rule='uniform rank in time; no feature/voltage amplitude selection',
                 maximum_context_rows_each_side=radius)
    return selection, audit


def stream_features(anchors, arrays, positions, bounds, cfg, path, progress):
    """anchors: iterable of dictionaries with id, partition, anchor_row."""
    batch = []; written = 0
    for anchor in anchors:
        batch.extend(phase_features(arrays, positions, int(anchor['anchor_row']),
                                    bounds[anchor['partition']], cfg, anchor))
        if len(batch) >= cfg['batch_anchors']*len(VARIABLES)*len(PHASES):
            pd.DataFrame(batch).to_csv(path, mode='a', index=False, header=not path.exists())
            written += len(batch); batch.clear(); progress(f'{path.name}: {written} phase rows saved')
    if batch:
        pd.DataFrame(batch).to_csv(path, mode='a', index=False, header=not path.exists())
    if not path.exists():
        pd.DataFrame(columns=['id', 'partition', 'anchor_row', 'variable', 'phase', 'support',
                              *METRICS]).to_csv(path, index=False)
    return path
