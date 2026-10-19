"""Independent six-seed output audit. No training or additional GPU inference."""
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
SEEDS = [42, 103, 193, 202, 210, 212]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def csv(path):
    return pd.read_csv(path, float_precision='round_trip')


def metrics(points):
    truth = points.y_true_delta_V.to_numpy(float)
    error = points.y_pred_delta_V.to_numpy(float) - truth
    sse = math.fsum(float(x) ** 2 for x in error)
    mean = math.fsum(map(float, truth)) / len(truth)
    return dict(RMSE_mV=math.sqrt(sse / len(truth)) * 1000,
                MAE_mV=math.fsum(abs(float(x)) for x in error) / len(truth) * 1000,
                R2=1 - sse / math.fsum((float(x) - mean) ** 2 for x in truth))


if __name__ == '__main__':
    if hasattr(sys.stdout, 'reconfigure'): sys.stdout.reconfigure(encoding='utf-8')
    ledger = read(OUTPUTS / 'SIX_SEED_EXECUTION.json')
    assert ledger['status'] == 'PASS' and ledger['seeds'] == SEEDS
    assert ledger['program_sha256'] == sha(ROOT / 'RealHyFC_F2_Main.py')
    assert [row['seed_argument'] for row in ledger['records']] == [*map(str, SEEDS), 'all']
    assert [row['seed'] for row in ledger['invalid_seeds']] == ['52', '62']
    for row in ledger['invalid_seeds']:
        assert row['returncode'] == 2 and 'invalid choice' in row['stderr'] and not row['stdout']
    assets = read(ROOT / 'RealHyFC_F2_Main_Seed_Assets.json')
    assert assets['seeds'] == SEEDS
    sources = {row['seed']: row for row in assets['models']}
    checks = []; singles = {}; all_results = {}; metric_differences = []; table = []
    for execution in ledger['records']:
        assert execution['returncode'] == 0 and not execution['stderr'] and len(execution['runs']) == 1
        run = OUTPUTS / execution['runs'][0]
        config = read(run / 'FROZEN_RUN_CONFIGURATION.json')
        expected = SEEDS if execution['seed_argument'] == 'all' else [int(execution['seed_argument'])]
        assert config['seeds'] == expected and config['supported_seeds'] == SEEDS
        assert config['displayed_evaluation_partition'] == 'VALIDATION'
        assert config['formal_metric_horizons'] == list(range(1, 25))
        assert config['main_program_sha256'] == ledger['program_sha256']
        assert config['seed_assets_sha256'] == sha(ROOT / 'RealHyFC_F2_Main_Seed_Assets.json')
        acceptance = read(run / 'INTEGRATION_ACCEPTANCE.json')
        assert acceptance['status'] == 'PASS' and not acceptance['new_training_executed'] and not acceptance['optuna_executed']
        assert [row['seed'] for row in acceptance['seeds']] == expected
        for row in csv(run / 'ARTIFACT_MANIFEST.csv').itertuples():
            assert sha(run / row.path) == row.sha256, row.path
        formal = csv(run / 'F2_PARTITION_METRICS.csv')
        assert formal.seed.unique().tolist() == expected
        assert len(formal) == len(expected) * 4
        validation_metrics = []
        for seed in expected:
            asset = sources[seed]; folder = run / f'seed{seed}'
            audit = next(row for row in acceptance['seeds'] if row['seed'] == seed)
            assert audit['optimizer_steps'] == 0 and not audit['training_performed']
            assert audit['best_epoch'] == asset['best_epoch']
            assert audit['checkpoint_sha256'] == asset['checkpoint_sha256']
            assert sha(ROOT.parent / asset['checkpoint']) == sha(folder / 'best_model.pth') == asset['checkpoint_sha256']
            assert sha(ROOT.parent / asset['reference_prediction']) == asset['prediction_sha256']
            assert audit['pointwise_exact'] and audit['queries'] == 4920
            assert audit['partition_reference_checks']['VALIDATION'].startswith('POINTWISE_EXACT')
            assert audit['all_four_partitions_pointwise_exact'] == (seed == 42)
            points = csv(folder / 'predictions.csv')
            pd.testing.assert_frame_equal(points, csv(folder / 'predictions/VALIDATION.csv'), check_exact=True)
            reference = csv(ROOT.parent / asset['reference_prediction'])
            assert len(points) == 4920 and points.origin_row.nunique() == 205 and points.partition.eq('VALIDATION').all()
            for key in ['origin_row', 'target_row', 'horizon_step', 'y_true_delta_V', 'y_pred_delta_V']:
                assert np.array_equal(points[key], reference[key]), (seed, key)
            measured = metrics(points)
            ref_metrics = metrics(reference)
            assert measured == ref_metrics
            for key, value in measured.items():
                difference = value - asset['historical_summary_metrics'][key]
                assert abs(difference) <= 4 * math.ulp(value), (seed, key, difference)
                metric_differences.append(dict(seed=seed, metric=key, difference=difference,
                                               explanation='same exact predictions; floating reduction order only'))
            validation_metrics.append(measured)
            for partition in ['TRAIN', 'VALIDATION', 'HISTORICAL_ASSESSMENT', 'LATE_LIFE']:
                part = csv(folder / f'predictions/{partition}.csv')
                actual = metrics(part)
                recorded = formal.loc[(formal.seed == seed) & (formal.partition == partition)].iloc[0]
                assert all(recorded[key] == value for key, value in actual.items())
                if seed == 42:
                    old = csv(ROOT / f'Phase_SCI_RealHyFC_Full_Chronological_Evaluation/outputs/predictions/{partition}_F2_seed42.csv')
                    for key in ['origin_row', 'target_row', 'horizon_step', 'y_true_delta_V', 'y_pred_delta_V', 'origin_raw_voltage_V', 'predicted_voltage_V']:
                        assert np.array_equal(part[key], old[key]), key
            pd.testing.assert_frame_equal(csv(folder / 'training_history.csv'), csv(ROOT.parent / asset['training_history']), check_exact=True)
            plot = csv(folder / 'prediction_vs_true_plot_data.csv')
            pd.testing.assert_frame_equal(plot, points.loc[points.horizon_step == 24].sort_values(
                'target_row').reset_index(drop=True), check_exact=True)
            pd.testing.assert_frame_equal(csv(folder / 'scatter_true_vs_pred_plot_data.csv'), points, check_exact=True)
            assert len(plot) == 205 and plot.target_row.is_unique
            for name in ['training_history', 'prediction_vs_true', 'scatter_true_vs_pred']:
                with Image.open(folder / (name + '.png')) as image:
                    image.verify()
                with Image.open(folder / (name + '.png')) as image:
                    assert min(image.size) >= 1200 and min(image.info['dpi']) >= 299
            preprocessing = read(folder / 'preprocess_info.json')
            assert preprocessing['checkpoint_sha256'] == asset['checkpoint_sha256']
            assert preprocessing['evaluation_partition'] == 'VALIDATION'
            if execution['seed_argument'] == 'all':
                all_results[seed] = folder
            else:
                singles[seed] = folder
                table.append(dict(seed=str(seed), RMSE_V=measured['RMSE_mV'] / 1000,
                                  MAE_V=measured['MAE_mV'] / 1000, R2=measured['R2']))
        summary = read(run / 'evaluation_results.json')
        assert summary['seeds'] == expected and summary['partition'] == 'VALIDATION'
        assert summary['queries_per_seed'] == [4920] * len(expected)
        assert summary['RMSE_V'] == math.fsum(row['RMSE_mV'] for row in validation_metrics) / len(expected) / 1000
        assert summary['MAE_V'] == math.fsum(row['MAE_mV'] for row in validation_metrics) / len(expected) / 1000
        assert summary['R2'] == math.fsum(row['R2'] for row in validation_metrics) / len(expected)
        summary_csv = csv(run / 'evaluation_results.csv')
        assert str(summary_csv.seeds.iloc[0]) == ','.join(map(str, expected))
        assert execution['stdout'] == f'RMSE: {summary["RMSE_V"]:.6f} V\nMAE : {summary["MAE_V"]:.6f} V\nR²  : {summary["R2"]:.4f}\n'
        if execution['seed_argument'] == 'all':
            assert summary['aggregation'] == 'equal mean of per-seed metrics; no ensemble'
            table.append(dict(seed='all', RMSE_V=summary['RMSE_V'], MAE_V=summary['MAE_V'], R2=summary['R2']))
        checks.append(dict(run=run.name, seeds=expected, status='PASS', validation_queries_per_seed=4920))
    assert sorted(singles) == sorted(all_results) == sorted(SEEDS)
    for seed in SEEDS:
        for partition in ['TRAIN', 'VALIDATION', 'HISTORICAL_ASSESSMENT', 'LATE_LIFE']:
            pd.testing.assert_frame_equal(csv(singles[seed] / f'predictions/{partition}.csv'),
                                          csv(all_results[seed] / f'predictions/{partition}.csv'), check_exact=True)
    for relative, digest in ledger['history_before'].items():
        assert sha(ROOT.parent / relative) == digest, ('HISTORY_CHANGED', relative)
    pd.DataFrame(table).to_csv(OUTPUTS / 'SIX_SEED_VALIDATION_SUMMARY.csv', index=False, float_format='%.17g')
    result = dict(status='PASS', seeds=SEEDS, commands=checks, unit_tests=10, validation_predictions_bit_exact=True,
                  single_vs_all_predictions_bit_exact=True, arithmetic_mean_no_ensemble=True,
                  legacy_seeds_rejected=True, historical_files_unchanged=len(ledger['history_before']),
                  no_training=True, historical_summary_reduction_precision=metric_differences)
    (OUTPUTS / 'SIX_SEED_ACCEPTANCE.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(pd.DataFrame(table).to_string(index=False))
    print('PASS: 7 real commands, 12 seed instances, 59040 Validation queries exactly replayed; all figures verified')
