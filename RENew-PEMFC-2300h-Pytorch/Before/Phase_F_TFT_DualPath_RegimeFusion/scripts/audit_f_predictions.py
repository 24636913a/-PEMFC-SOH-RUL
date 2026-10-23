"""Independent persisted-CSV metrics, key/truth and error-concentration audit.

Does not import the training script, Phase E metric helper, models or N1.
Uses math.fsum scalar reductions independently of training's numpy_metrics.
"""
from __future__ import annotations
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

PHASE = Path(__file__).resolve().parents[1]
OUT = PHASE / 'outputs'
KEYS = ['origin_row', 'horizon_step', 'target_row']


def metrics_from_records(frame):
    y = frame.y_true_delta_V.tolist()
    p = frame.y_pred_delta_V.tolist()
    n = len(y)
    error = [b - a for a, b in zip(y, p)]
    sse = math.fsum(x*x for x in error)
    mean_y = math.fsum(y) / n
    sst = math.fsum((x-mean_y)**2 for x in y)
    return {'RMSE_mV': math.sqrt(sse / n) * 1000,
            'MAE_mV': math.fsum(abs(x) for x in error) / n * 1000,
            'R2': 1 - sse / sst}


def validate_frame(frame, group, split, sha, reference):
    expected = {'validation': (205, 2103, 2404), 'assessment': (378, 2404, 3005)}[split]
    origins, start, end = expected
    if len(frame) != origins * 24 or frame.origin_row.nunique() != origins:
        raise RuntimeError('PREDICTION_RECORD_COUNT_FAILURE')
    if frame.duplicated(['origin_row', 'horizon_step']).any():
        raise RuntimeError('PREDICTION_DUPLICATE_KEY')
    if not ((frame.experiment_group == group).all() and (frame.seed == 42).all()
            and (frame.split == split).all() and (frame.protocol_sha256 == sha).all()):
        raise RuntimeError('PREDICTION_IDENTITY_FAILURE')
    if not (frame.target_row.between(start, end-1).all()
            and (frame.target_row == frame.origin_row + frame.horizon_step).all()
            and frame.horizon_step.between(1, 24).all()
            and np.isfinite(frame[['y_true_delta_V', 'y_pred_delta_V', 'target_age_h']].to_numpy()).all()
            and (frame.target_age_h < 1819.997738).all()):
        raise RuntimeError('PREDICTION_VALUE_OR_BOUNDARY_FAILURE')
    ordered = frame.sort_values(KEYS).reset_index(drop=True)
    ref = reference.sort_values(KEYS).reset_index(drop=True)
    if not (np.array_equal(ordered[KEYS].to_numpy(), ref[KEYS].to_numpy())
            and np.array_equal(ordered.y_true_delta_V.to_numpy(), ref.y_true_delta_V.to_numpy())):
        raise RuntimeError('PREDICTION_KEY_OR_TRUTH_MISMATCH')
    if not ordered.groupby('origin_row').horizon_step.apply(lambda s: s.tolist() == list(range(1, 25))).all():
        # Every origin must have the complete 1..24 set (sort order above).
        raise RuntimeError('PREDICTION_HORIZON_SET_FAILURE')
    return ordered


def diagnostics(frame, group, split):
    f = frame.copy()
    f['abs_error_mV'] = abs(f.y_pred_delta_V - f.y_true_delta_V) * 1000
    f['squared_error_mV2'] = f.abs_error_mV**2
    total = f.squared_error_mV2.sum()
    target = f.groupby('target_row', sort=True).agg(SSE_mV2=('squared_error_mV2', 'sum'),
                multiplicity=('squared_error_mV2', 'size'), mean_squared_error_mV2=('squared_error_mV2', 'mean'),
                mean_absolute_error_mV=('abs_error_mV', 'mean'), target_age_h=('target_age_h', 'first')).reset_index()
    target['SSE_fraction'] = target.SSE_mV2 / total
    target['experiment_group'], target['split'] = group, split
    target.to_csv(OUT / f'metrics/{group}_{split}_TARGET_SSE.csv', index=False)
    horizon = f.groupby('horizon_step').squared_error_mV2.mean().pow(.5).reset_index(name='RMSE_mV')
    horizon['experiment_group'], horizon['split'] = group, split
    horizon.to_csv(OUT / f'metrics/{group}_{split}_HORIZON_RMSE.csv', index=False)
    top = target.sort_values('SSE_mV2', ascending=False).iloc[0]
    return {'experiment_group': group, 'split': split,
        'endpoint_12h_RMSE_mV': math.sqrt(f.loc[f.horizon_step == 24, 'squared_error_mV2'].mean()),
        'P95_absolute_error_mV': float(f.abs_error_mV.quantile(.95)),
        'P99_absolute_error_mV': float(f.abs_error_mV.quantile(.99)), 'max_absolute_error_mV': float(f.abs_error_mV.max()),
        'target_time_equal_weight_RMSE_mV': math.sqrt(target.mean_squared_error_mV2.mean()),
        'max_SSE_target_row': int(top.target_row), 'max_SSE_target_multiplicity': int(top.multiplicity),
        'max_target_SSE_fraction': float(top.SSE_fraction),
        'top5_target_SSE_fraction': float(target.nlargest(5, 'SSE_mV2').SSE_fraction.sum()),
        'independent_target_rows': len(target)}


def run():
    proto = PHASE / 'configs/F0_F2_EXECUTION_PROTOCOL.yaml'
    sha = hashlib.sha256(proto.read_bytes()).hexdigest()
    if sha != proto.with_suffix('.sha256').read_text().split()[0]:
        raise RuntimeError('AUDIT_PROTOCOL_IDENTITY_FAILURE')
    freeze = json.loads((OUT / 'audits/F_ALL_CHECKPOINTS_FROZEN.json').read_text())
    if not freeze['pass'] or freeze['protocol_sha256'] != sha or set(freeze['checkpoints']) != {'F0', 'F1', 'F2'}:
        raise RuntimeError('ALL_CHECKPOINTS_NOT_FROZEN')
    reported = pd.read_csv(OUT / 'metrics/F_SEED42_METRICS.csv', float_precision='round_trip')
    authority = pd.read_csv(OUT / 'audits/F_PREDICTION_KEY_TRUTH_AUTHORITY.csv', float_precision='round_trip')
    if len(reported) != 6 or reported.duplicated(['experiment_group', 'split']).any():
        raise RuntimeError('INCOMPLETE_REPORTED_METRICS')
    metrics, diag = [], []
    for split in ('validation', 'assessment'):
        reference = authority.loc[authority.split == split]
        for group in ('F0', 'F1', 'F2'):
            path = OUT / f'predictions/{group}_{split}_seed42.csv'
            f = validate_frame(pd.read_csv(path, float_precision='round_trip'), group, split, sha, reference)
            measured = metrics_from_records(f)
            row = reported.loc[(reported.experiment_group == group) & (reported.split == split)].iloc[0]
            for key, value in measured.items():
                if not math.isclose(value, float(row[key]), rel_tol=0, abs_tol=1e-10):
                    raise RuntimeError(f'INDEPENDENT_METRIC_MISMATCH:{group}:{split}:{key}')
            metrics.append({'experiment_group': group, 'split': split, **measured, 'records': len(f),
                            'origins': f.origin_row.nunique(), 'max_metric_difference': max(abs(measured[k] - row[k]) for k in measured)})
            diag.append(diagnostics(f, group, split))
    pd.DataFrame(metrics).to_csv(OUT / 'metrics/F_INDEPENDENT_METRICS.csv', index=False)
    pd.DataFrame(diag).to_csv(OUT / 'metrics/F_ERROR_CONCENTRATION.csv', index=False)
    gate_rows = []
    for split in ('validation', 'assessment'):
        f = pd.read_csv(OUT / f'predictions/F2_{split}_GATES.csv')
        g = f.gate.to_numpy()
        if len(g) != {'validation': 205, 'assessment': 378}[split] or not np.isfinite(g).all():
            raise RuntimeError('GATE_CACHE_INCOMPLETE')
        gate_rows.append({'split': split, 'mean': float(g.mean()), 'std': float(g.std()), 'min': float(g.min()),
            'max': float(g.max()), 'P5': float(np.quantile(g, .05)), 'P50': float(np.quantile(g, .5)),
            'P95': float(np.quantile(g, .95)), 'fraction_near_0_or_1': float(np.mean((g < .05) | (g > .95))),
            'fraction_within_0p01_of_half': float(np.mean(abs(g-.5) <= .01))})
        for column in f.columns.difference(['split', 'origin_row', 'gate']):
            # Prespecified quartile display; never used to select architecture.
            bucket = pd.qcut(f[column], 4, duplicates='drop')
            result = f.groupby(bucket, observed=True).gate.agg(['size', 'mean', 'std', 'min', 'max'])
            result.to_csv(OUT / f'metrics/F2_{split}_GATE_BY_{column}.csv')
    pd.DataFrame(gate_rows).to_csv(OUT / 'metrics/F2_GATE_SUMMARY.csv', index=False)
    record = {'pass': True, 'protocol_sha256': sha, 'implementation': 'independent_math_fsum_from_roundtrip_CSV',
              'metric_tolerance': 1e-10, 'key_truth_equal': True, 'all_horizons_complete': True,
              'assessment_records_per_group': 9072, 'assessment_origins': 378, 'test_accessed': False, 'results': metrics}
    (OUT / 'audits/F_INDEPENDENT_PREDICTION_AUDIT.json').write_text(json.dumps(record, indent=2) + '\n', encoding='utf-8')
    print('F_INDEPENDENT_PREDICTION_AUDIT_PASS', flush=True)


if __name__ == '__main__':
    run()
