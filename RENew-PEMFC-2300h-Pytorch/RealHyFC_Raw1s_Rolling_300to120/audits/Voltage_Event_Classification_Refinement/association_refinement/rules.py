"""Train-control-only cutoffs and three-valued observational labels (never causes)."""
import numpy as np
import pandas as pd
from .io_contract import VARIABLES, PHASES, FAMILIES, clean
from .features import METRICS


def registered_metrics(phase):
    return ('max_abs_step',) if phase == 'instant' else METRICS


def fit_train_rules(path, cfg):
    # Read the bounded control FEATURE table, never raw measurements in full.
    samples = {(v, p, m): [] for v in VARIABLES for p in PHASES for m in registered_metrics(p)}
    for frame in pd.read_csv(path, chunksize=cfg['fit_csv_chunksize']):
        if not frame.partition.eq('TRAIN').all():
            raise RuntimeError('Threshold fitting file must contain TRAIN controls only')
        for (variable, phase), group in frame.loc[frame.support.eq('OK')].groupby(['variable', 'phase']):
            for metric in registered_metrics(phase):
                values = group[metric].to_numpy(float)
                samples[(variable, phase, metric)].extend(values[np.isfinite(values)].tolist())
    floors = {}
    for variable in VARIABLES:
        values = np.asarray(samples[(variable, 'instant', 'max_abs_step')])
        positive = values[values > 0]
        floor = (float(np.quantile(positive, cfg['positive_step_quantile']))*cfg['step_floor_multiplier']
                 if len(positive) >= cfg['minimum_positive_train_steps'] else None)
        floors[variable] = dict(value=floor, positive_train_increments=len(positive),
                                identity='numerical-change proxy; NOT instrument resolution')
    rules = {}
    for (variable, phase, metric), values in samples.items():
        data = np.asarray(values, dtype=float)
        q = float(np.quantile(data, cfg['background_quantile'])) if len(data) else None
        floor = floors[variable]['value']
        status = 'OK'
        if len(data) < cfg['minimum_train_control_observations']:
            status = 'TOO_FEW_TRAIN_CONTROLS'
        elif q is None or q <= 0 or np.ptp(data) == 0:
            status = 'CONSTANT_OR_ZERO_TRAIN_BACKGROUND'
        cut = max(q, floor or 0.0) if status == 'OK' else None
        rules[f'{variable}|{phase}|{metric}'] = dict(status=status, n=len(data), train_q99=q,
                                                   cutoff=cut, strict_comparison='value > cutoff')
    return clean(dict(fitted_partition='TRAIN', selected_from='ordinary calendar-grid controls outside frozen events',
                      quantile=cfg['background_quantile'], numerical_floors=floors, rules=rules,
                      multiple_comparisons='OR of metrics; no claimed 1% family false-positive guarantee'))


def tri_any(values):
    """False needs ALL member tests observed false; missing is never absence."""
    values = list(values)
    if any(v is True for v in values):
        return True
    if values and all(v is False for v in values):
        return False
    return None


def assess(row, frozen):
    variable, phase = row['variable'], row['phase']
    tests = {}; reasons = []
    for metric in registered_metrics(phase):
        rule = frozen['rules'][f'{variable}|{phase}|{metric}']
        value = row[metric]
        usable = row['support'] == 'OK' and rule['status'] == 'OK' and np.isfinite(value)
        tests[metric] = bool(value > rule['cutoff']) if usable else None
        if not usable:
            reasons.append(metric+':'+(row['support'] if row['support'] != 'OK' else rule['status']))
    unusual = tri_any(tests.values())
    floor = frozen['numerical_floors'][variable]['value']
    observed = row['support'] == 'OK'
    material = bool(row['max_abs_step'] > floor) if observed and floor is not None else None
    raw = bool(row['max_abs_step'] > 0) if observed else None
    return dict(row, raw_numerical_change=raw, material_step=material,
                background_unusual=unusual, uncalibrated_or_missing=';'.join(reasons),
                triggered_metrics=';'.join(k for k, v in tests.items() if v is True))


def event_classification(event, phases, arrays, positions):
    lookup = {(p['variable'], p['phase']): p for p in phases}
    output = dict(event)
    tags = []; family_values = []
    for family in FAMILIES:
        variables = [v for v, spec in VARIABLES.items() if spec[1] == family]
        states = {}
        for phase in PHASES:
            state = tri_any(lookup[(v, phase)]['background_unusual'] for v in variables)
            states[phase] = state; family_values.append(state)
            output[f'{phase}_{family}'] = state
            if state is True:
                tags.append(f'{phase.upper()}_{family}_BACKGROUND_UNUSUAL')
        after_only = states['after'] is True and states['before'] is False and states['instant'] is False
        output[f'after_only_{family}'] = after_only if states['after'] is not None and all(
            states[p] is not None for p in ('before', 'instant')) else None
        if after_only:
            tags.append(f'AFTER_ONLY_{family}_NOT_A_PRECURSOR')
    combined = tri_any(family_values)
    output['no_specific_association_in_observed_fields'] = not combined if combined is not None else None
    output['has_unsupported_or_uncalibrated_field_phase'] = any(v is None for v in family_values)
    output['physical_cause_status'] = 'UNDETERMINED'
    output['observational_tags'] = ';'.join(tags) if tags else ('NO_BACKGROUND_UNUSUAL_ASSOCIATION' if combined is False else 'INSUFFICIENT_SUPPORTED_TESTS')
    current = lookup[('actual_current', 'instant')]
    row = int(event['anchor_row']); lo = int(event['partition_first_row'])
    valid = row > lo and arrays['segment'][row-1] == arrays['segment'][row]
    dvoltage = float(arrays['values'][row, positions['mean_voltage_V']]-arrays['values'][row-1, positions['mean_voltage_V']]) if valid else np.nan
    output['instant_voltage_delta_V'] = dvoltage
    output['instant_current_delta_A'] = current['signed_instant_step']
    output['instant_load_voltage_opposite_sign_compatibility'] = (bool(dvoltage*current['signed_instant_step'] < 0)
        if current['material_step'] is True and np.isfinite(dvoltage) else None)
    # Compatibility describes sign only; not proof of electrical-load causation.
    return output
