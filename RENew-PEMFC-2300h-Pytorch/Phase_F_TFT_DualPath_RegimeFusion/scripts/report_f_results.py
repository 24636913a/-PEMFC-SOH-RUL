"""Report measured, independently audited results; never changes models."""
import json
from pathlib import Path
import pandas as pd

PHASE = Path(__file__).resolve().parents[1]
OUT, REPORT = PHASE / 'outputs', PHASE / 'reports'
LIMITS = ('PHASE_F_ASSESSMENT_INHERITED_SELECTION_BIAS; C1_D0_E0_REPRODUCIBILITY_UNRESOLVED; '
          'PHASE_F_TFT_ARCHITECTURE_EXPLORATION_ONLY; TIMESNET_FREEZE; BACKBONE_SEARCH_CLOSED')


def generate_reports(cfg, sha):
    audit = json.loads((OUT / 'audits/F_INDEPENDENT_PREDICTION_AUDIT.json').read_text())
    if not audit['pass'] or audit['protocol_sha256'] != sha:
        raise RuntimeError('REPORT_REQUIRES_INDEPENDENT_AUDIT_PASS')
    REPORT.mkdir(exist_ok=True)
    m = pd.read_csv(OUT / 'metrics/F_SEED42_METRICS.csv').set_index(['experiment_group', 'split'])
    d = pd.read_csv(OUT / 'metrics/F_ERROR_CONCENTRATION.csv').set_index(['experiment_group', 'split'])
    gates = pd.read_csv(OUT / 'metrics/F2_GATE_SUMMARY.csv')
    lines = []
    for g in ('F0', 'F1', 'F2'):
        v, a = m.loc[(g, 'validation')], m.loc[(g, 'assessment')]
        lines.append(f'|{g}|{v.RMSE_mV:.6f}|{a.RMSE_mV:.6f}|{v.MAE_mV:.6f}|{a.MAE_mV:.6f}|{v.R2:.6f}|{a.R2:.6f}|{int(a.parameter_count)}|{int(a.best_epoch)}|{a.training_time_s:.2f}|')
    table = '|Model|Validation RMSE mV|Assessment RMSE mV|Val MAE mV|Assess MAE mV|Val R²|Assess R²|Params|Best epoch|Training s|\n|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n' + '\n'.join(lines) + '\n\n'
    comparisons = []
    same_direction_gain = []
    for before, after in (('F0', 'F1'), ('F1', 'F2'), ('F0', 'F2')):
        differences = {s: float(m.loc[(after, s), 'RMSE_mV'] - m.loc[(before, s), 'RMSE_mV']) for s in ('validation', 'assessment')}
        gain = all(v < 0 for v in differences.values())
        same_direction_gain.append(gain)
        comparisons.append({'comparison': f'{after} minus {before}', 'RMSE_difference_mV': differences,
            'assessment_relative_reduction_percent': float((1-m.loc[(after, 'assessment'), 'RMSE_mV']/m.loc[(before, 'assessment'), 'RMSE_mV'])*100),
            'parameter_increase': int(m.loc[(after, 'assessment'), 'parameter_count']-m.loc[(before, 'assessment'), 'parameter_count']),
            'training_time_ratio': float(m.loc[(after, 'assessment'), 'training_time_s']/m.loc[(before, 'assessment'), 'training_time_s']),
            'both_splits_improved': gain})
    text = '\n'.join(f"- {x['comparison']}: Validation {x['RMSE_difference_mV']['validation']:+.6f} mV; Assessment {x['RMSE_difference_mV']['assessment']:+.6f} mV; Assessment relative reduction {x['assessment_relative_reduction_percent']:+.3f}%; parameters +{x['parameter_increase']}; total training-time ratio {x['training_time_ratio']:.3f}." for x in comparisons)
    gain_status = ('SINGLE_SEED_DEVELOPMENT_STRUCTURAL_GAIN_OBSERVED' if any(same_direction_gain)
                   else 'PHASE_F_ARCHITECTURE_GAIN_NOT_ESTABLISHED')
    summary = {'status': 'PHASE_F_F0_F1_F2_DEVELOPMENT_EXPLORATION_COMPLETED', 'architecture_evidence': gain_status,
               'protocol_sha256': sha, 'base_commit': cfg['base_commit'], 'seed': 42,
               'independent_metric_audit_pass': True, 'all_checkpoints_reloaded': True, 'test_accessed': False,
               'comparisons': comparisons, 'interpretation_limits': LIMITS,
               'further_research': 'Human review of same-stage Validation/Assessment evidence only; no automatic search or training.'}
    (OUT / 'F_FINAL_EXPERIMENT_SUMMARY.json').write_text(json.dumps(summary, indent=2) + '\n', encoding='utf-8')
    concentration = []
    for g in ('F0', 'F1', 'F2'):
        r = d.loc[(g, 'assessment')]
        concentration.append(f"|{g}|{r.endpoint_12h_RMSE_mV:.6f}|{r.P95_absolute_error_mV:.6f}|{r.P99_absolute_error_mV:.6f}|{r.max_absolute_error_mV:.6f}|{r.target_time_equal_weight_RMSE_mV:.6f}|{int(r.max_SSE_target_row)}|{int(r.max_SSE_target_multiplicity)}|{r.max_target_SSE_fraction:.3%}|{r.top5_target_SSE_fraction:.3%}|")
    ctable = '|Model|12h RMSE mV|P95 mV|P99 mV|Max mV|Target-equal RMSE mV|Max-SSE row|Multiplicity|Max-row SSE share|Top5 SSE share|\n|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n' + '\n'.join(concentration) + '\n\n'
    common = f'Protocol SHA256: `{sha}`. Base: `{cfg["base_commit"]}`. Seed42 only. Development exploratory, not independent Test.\n\n{LIMITS}.\n\n'
    (REPORT / 'F_TRAINING_REPORT.md').write_text('# Phase F Formal Training Report\n\n' + common +
        '## Frozen protocol and input\n\nExactly 3007 prefix rows were read, then 3005 Development rows retained. Train [0,2103), Validation [2103,2404), Assessment [2404,3005). Legal windows: 1380/205/378. Metadata eligibility alone was counted for Assessment before training; its model arrays/targets were constructed only after all three best checkpoints froze.\n\n'
        'Phase E/C window, Top10, CommonScaler, per-window EWMA and zero-mask functions were imported without modifying historical files. Shared unfiltered-Train scaler, target scale, fixed inputs and first shuffled batch were hashed. Native parameters/buffers and F1/F2 branch initialization matched exactly. Seed42 was reset independently for each group. Training dropout random-number paths need not match.\n\n'
        'AdamW/MSE, batch32, max450, patience40, clip1, frozen LR/weight-decay. Validation standardized RMSE alone selected epoch using strict improvement >1e-12; Assessment never selected epochs. Train MSE is the mean of batch losses, matching Phase E. Each best checkpoint includes identities, scale, initialization and fixed Validation predictions and passed exact reload parity. All 3 froze before Assessment. See per-epoch CSV, run JSON, environment and source-code manifest.\n\n' + table +
        '## Technical verification\n\nIndependent scalar math.fsum recomputation from round-trip CSVs agrees within 1e-10. All keys, true ΔV, horizon sets, target rows and Development boundary passed. 9072 queries/group on Assessment, 4920/group on Validation. Checkpoints are retained locally and subject to repository file-size rules at delivery.\n\n'
        '## Historical E2 comparison\n\n' + f"F0 Validation difference from historical E2 6.064596 mV: {m.loc[('F0','validation'),'RMSE_mV']-6.064596:+.6f} mV; Assessment difference from 7.785821 mV: {m.loc[('F0','assessment'),'RMSE_mV']-7.785821:+.6f} mV. Current protocol/input/checkpoint identities are self-consistent; no unverified explanation of cross-stage differences is asserted.\n", encoding='utf-8')
    (REPORT / 'F_ABLATION_ANALYSIS.md').write_text('# Phase F Architecture Ablation\n\n' + common + table + text + '\n\n' +
        f'`{gain_status}`. Directional consistency must involve both Validation and Assessment; lowest Assessment alone does not prove an architecture. Added 8896 convolution parameters and 57 gate parameters, training-time/mean-epoch/memory ledgers are explicit confounders. A single seed does not establish stability or causal mechanism.\n\n'
        '## Gate behavior\n\n' + gates.to_csv(index=False) + '\nThe context-conditioned gate distributions are recorded by predefined quartiles of the five masked/standardized historical regime means. Thresholds near 0/1 (0.05/0.95) and half (±0.01) are descriptive diagnostics only, never model-selection rules. Gate correlation is not electrochemical mechanism evidence.\n', encoding='utf-8')
    (REPORT / 'F_ERROR_CONCENTRATION_AUDIT.md').write_text('# Phase F Error Concentration\n\n' + common + ctable +
        'Target-time-equal RMSE first averages squared errors within each target, then weights target times equally; it does not average predictions into an ensemble. Formal primary RMSE remains forecast-query-weighted. All outliers remain in primary scores. Target SSE contributions and horizon error curves are saved for both splits. Assessment has 9072 records from 378 overlapping origins, not 9072 independent observations; no significance tests or anomaly-driven retuning were performed.\n', encoding='utf-8')
    (REPORT / 'F_FINAL_EXPERIMENT_SUMMARY.md').write_text('# Phase F Final Summary\n\n' + common + table + text + '\n\n' + ctable +
        f"`{summary['status']}`\n\n`{gain_status}`\n\nNo Test, SOH/RUL, new feature search, loss search, extra seeds, Optuna or architecture revision. Model science is limited to same-stage exploratory structure effects. Further research requires explicit human review; this execution stops here.\n", encoding='utf-8')
    readme = PHASE / 'README.md'
    original = readme.read_text(encoding='utf-8')
    original = original.replace('No formal RealHyFC training has been run; no formal training checkpoint, prediction cache, or Assessment result has been generated.',
        'The explicitly authorized Seed42 F0/F1/F2 Development training has now completed. Best checkpoints, Validation and Assessment caches, independent metric audits and reports are saved. Formal Test remains prohibited.')
    original = original.replace('No training, Optuna, Test reading, backbone search, feature-count search, EWMA tuning, SOH, or RUL work is authorized by this phase.',
        'Only the explicitly frozen F0/F1/F2 Seed42 Development training was authorized by the subsequent user instruction. No additional training, Optuna, Test reading, backbone search, feature-count search, EWMA tuning, SOH, or RUL work is authorized.')
    original += f'\n## Completed formal execution\n\n`{summary["status"]}`\n\n`{gain_status}`\n\nRun: `python -B scripts/run_f_training.py --run` (one-shot identity-guarded; completed runs cannot retrain). Independent cache verification: `python -B scripts/audit_f_predictions.py`. Results: `reports/F_FINAL_EXPERIMENT_SUMMARY.md`. Checkpoints: `outputs/checkpoints`. All three best checkpoints were frozen and reload-checked before unified Assessment. Test accessed: NO.\n'
    readme.write_text(original, encoding='utf-8')
