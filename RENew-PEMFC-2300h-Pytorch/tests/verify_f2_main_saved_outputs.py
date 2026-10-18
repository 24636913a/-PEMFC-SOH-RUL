"""Independently verify output-only replay against the original frozen results.

Run from the research root: python -B tests/verify_f2_main_saved_outputs.py
"""
from pathlib import Path
import hashlib
import json
import math
import sys

import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = ROOT / 'RealHyFC_F2_Main_Outputs'
OLD = OUTPUTS / '20261017T151243_601888Z_evaluate_seedall'


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_csv(path):
    return pd.read_csv(path, float_precision='round_trip')


def verify(run):
    config = json.loads((run / 'FROZEN_RUN_CONFIGURATION.json').read_text(encoding='utf-8'))
    acceptance = json.loads((run / 'INTEGRATION_ACCEPTANCE.json').read_text(encoding='utf-8'))
    assert acceptance['status'] == 'PASS' and not acceptance['new_training_executed']
    assert not acceptance['optuna_executed']
    assert config['main_program_sha256'] == sha256(ROOT / 'RealHyFC_F2_Main.py')
    before = read_csv(OLD / 'F2_PARTITION_METRICS.csv')
    after = read_csv(run / 'F2_PARTITION_METRICS.csv')
    counts = 0
    for _, row in after.iterrows():
        previous = before.loc[(before.seed == row.seed) & (before.partition == row.partition)].iloc[0]
        for key in ['RMSE_mV', 'MAE_mV', 'R2', 'origins', 'queries', 'unique_target_times']:
            assert row[key] == previous[key], (row.seed, row.partition, key)
        counts += 3
    manifest = read_csv(run / 'ARTIFACT_MANIFEST.csv')
    for _, item in manifest.iterrows():
        assert sha256(run / item.path) == item.sha256, item.path
    metric_checks = []
    for item in acceptance['seeds']:
        seed = item['seed']; folder = run / f'seed{seed}'
        assert item['optimizer_steps'] == 0 and item['all_four_partitions_pointwise_exact']
        assert sha256(Path(item['checkpoint'])) == item['checkpoint_sha256']
        assert sha256(folder / 'best_model.pth') == item['checkpoint_sha256']
        for partition in ['TRAIN', 'VALIDATION', 'HISTORICAL_ASSESSMENT', 'LATE_LIFE']:
            pd.testing.assert_frame_equal(read_csv(folder / f'predictions/{partition}.csv'),
                                          read_csv(OLD / f'seed{seed}/predictions/{partition}.csv'), check_exact=True)
        points = read_csv(folder / 'predictions.csv')
        assert len(points) == 4920 and points.partition.eq('VALIDATION').all()
        assert points.origin_row.nunique() == 205
        errors = points.y_pred_delta_V.to_numpy() - points.y_true_delta_V.to_numpy()
        truth = points.y_true_delta_V.to_numpy()
        sse = math.fsum(float(x) ** 2 for x in errors)
        mean = math.fsum(map(float, truth)) / len(truth)
        measured = [math.sqrt(sse / len(errors)) * 1000,
                    math.fsum(abs(float(x)) for x in errors) / len(errors) * 1000,
                    1 - sse / math.fsum((float(x) - mean) ** 2 for x in truth)]
        official = after.loc[(after.seed == seed) & (after.partition == 'VALIDATION')].iloc[0]
        assert measured == [official.RMSE_mV, official.MAE_mV, official.R2]
        curve = read_csv(folder / 'prediction_vs_true_plot_data.csv')
        pd.testing.assert_frame_equal(curve, points.loc[points.horizon_step == 24].sort_values(
            'target_row').reset_index(drop=True), check_exact=True)
        assert len(curve) == 205 and curve.target_row.is_unique
        scatter = read_csv(folder / 'scatter_true_vs_pred_plot_data.csv')
        pd.testing.assert_frame_equal(scatter, points, check_exact=True)
        history = read_csv(folder / 'training_history.csv')
        historical = read_csv(ROOT / f'Phase_H_Determinism_MultiSeed_Stability/outputs/training_logs/H-F2-Seed{seed}/epochs.csv')
        pd.testing.assert_frame_equal(history, historical, check_exact=True)
        for name in ['training_history', 'prediction_vs_true', 'scatter_true_vs_pred']:
            with Image.open(folder / (name + '.png')) as image:
                image.verify()
            with Image.open(folder / (name + '.png')) as image:
                assert min(image.size) >= 1200 and min(image.info['dpi']) >= 299
        metric_checks.append(dict(seed=seed, RMSE_mV=measured[0], MAE_mV=measured[1], R2=measured[2]))
    summary = json.loads((run / 'evaluation_results.json').read_text(encoding='utf-8'))
    assert summary['partition'] == 'VALIDATION' and summary['seeds'] == config['seeds']
    assert summary['RMSE_V'] == math.fsum(x['RMSE_mV'] for x in metric_checks) / len(metric_checks) / 1000
    assert summary['MAE_V'] == math.fsum(x['MAE_mV'] for x in metric_checks) / len(metric_checks) / 1000
    assert summary['R2'] == math.fsum(x['R2'] for x in metric_checks) / len(metric_checks)
    return dict(run=run.name, status='PASS', metrics_exactly_equal=counts,
                per_seed=metric_checks, all_partition_predictions_exact=True,
                original_weights_unchanged=True, copied_weights_byte_exact=True,
                figures_verified=True, fixed_step=24, plot_points_per_seed=205,
                formal_queries_per_seed=4920, no_training=True, artifacts=len(manifest))


if __name__ == '__main__':
    if hasattr(sys.stdout, 'reconfigure'): sys.stdout.reconfigure(encoding='utf-8')
    runs = [p for p in OUTPUTS.glob('*_F2_evaluate_seed*')
            if (p / 'INTEGRATION_ACCEPTANCE.json').exists() and not (p / 'FAILURE.txt').exists()]
    assert len(runs) >= 2, 'Expected default seed42 and all-seed completed runs'
    results = [verify(run) for run in sorted(runs)]
    receipt = OUTPUTS / 'FC2_OUTPUT_ACCEPTANCE.json'
    receipt.write_text(json.dumps(dict(status='PASS', runs=results), ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print('PASS:', len(results), 'runs;', sum(row['metrics_exactly_equal'] for row in results),
          'frozen metric values exactly equal; predictions and weights exact')
