"""One formal Seed42 run per frozen F0/F1/F2, Development prefix only.

Phase E/C are imported read-only for window/scaler/filter/selection semantics.
Assessment arrays are not built until all three checkpoints pass reload.
An interrupted training group is never silently restarted.
"""
from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import math
import platform
import subprocess
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import pytorch_forecasting as pf
import torch
import yaml
from torch.utils.data import DataLoader, TensorDataset

PHASE = Path(__file__).resolve().parents[1]
PROJECT = PHASE.parent
REPO = PROJECT.parent
sys.path.insert(0, str(PHASE))
sys.path.insert(0, str(PROJECT / 'Phase_TFT_FullFeature_CausalFilter_Comparison' / 'scripts'))
import run_e_comparison as e
from models import DualPathStrictHistoryTFT, RegimeAwareDualPathTFT
from canonical_tft import CanonicalStrictHistoryTFT, count_parameters
from pytorch_forecasting.models.temporal_fusion_transformer import TemporalFusionTransformer
import pytorch_forecasting.models.temporal_fusion_transformer.sub_modules as sub_modules

OUT = PHASE / 'outputs'
PROTOCOL = PHASE / 'configs/F0_F2_EXECUTION_PROTOCOL.yaml'
SHA_FILE = PROTOCOL.with_suffix('.sha256')
GROUPS = ('F0', 'F1', 'F2')


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.writing')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n', encoding='utf-8')
    temporary.replace(path)


def file_hash(path):
    path = Path(path).resolve()
    if path.name == 'causal_features_30min.csv':
        raise RuntimeError('FULL_CAUSAL_FILE_HASH_PROHIBITED')
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def array_hash(x):
    a = np.ascontiguousarray(x)
    h = hashlib.sha256(str((a.shape, str(a.dtype))).encode())
    h.update(a.tobytes())
    return h.hexdigest()


def load_protocol():
    digest = file_hash(PROTOCOL)
    if SHA_FILE.exists():
        if SHA_FILE.read_text(encoding='utf-8').split()[0] != digest:
            raise RuntimeError('F_PROTOCOL_HASH_CHANGED')
    else:
        # Freeze BEFORE any formal model result. Never silently amend a hash.
        if (OUT / 'audits/F_EXECUTION_STATE.json').exists():
            raise RuntimeError('PROTOCOL_SHA_MISSING_FOR_EXISTING_RUN')
        SHA_FILE.write_text(digest + '  F0_F2_EXECUTION_PROTOCOL.yaml\n', encoding='utf-8')
    return yaml.safe_load(PROTOCOL.read_text(encoding='utf-8')), digest


def code_identity():
    files = list((PHASE / 'models').glob('*.py')) + list((PHASE / 'scripts').glob('*.py'))
    files += list((PHASE / 'tests').glob('*.py')) + list((PHASE / 'tests_execution').glob('*.py'))
    files += [Path(e.__file__), Path(e.c.__file__), PROJECT / 'Phase_TFT_Original_Baseline/models/canonical_tft.py']
    records = {str(p.relative_to(REPO)).replace('\\', '/'): file_hash(p) for p in sorted(set(files))}
    digest = hashlib.sha256(json.dumps(records, sort_keys=True).encode()).hexdigest()
    return {'files': records, 'manifest_sha256': digest}


def make_model(group, cfg, device):
    p = cfg['hyperparameters']
    kwargs = dict(hidden_size=p['hidden_size'], hidden_continuous_size=p['hidden_continuous_size'],
                  attention_head_size=4, lstm_layers=1, dropout=p['dropout'])
    classes = dict(F0=CanonicalStrictHistoryTFT, F1=DualPathStrictHistoryTFT, F2=RegimeAwareDualPathTFT)
    model = classes[group](**kwargs).to(device)
    if count_parameters(model) != cfg['architecture']['parameter_counts'][group]:
        raise RuntimeError('F_PARAMETER_COUNT_CHANGED')
    return model


def planned_origins(dev, cfg, start, end):
    """Metadata-only assessment eligibility; no Assessment x/y arrays yet."""
    good = dev.bin_valid_for_windows.astype(bool).to_numpy()
    seg = dev.temporal_segment_id.to_numpy()
    age = dev.bin_end_ophrs2_h.to_numpy(float)
    finite = np.isfinite(dev[cfg['task']['raw_f2_feature_order']].to_numpy(float)).all(axis=1)
    origins = []
    for origin in range(start, end - 24):
        first, last = origin - 47, origin + 24
        if first < 0 or last >= end:
            continue
        block = slice(first, last + 1)
        if (good[block].all() and finite[block].all() and (seg[block] == seg[origin]).all()
                and np.allclose(np.diff(age[block]), .5, atol=1e-6, rtol=0)):
            origins.append(origin)
    return np.asarray(origins)


def prepare(cfg, sha):
    e_cfg, e_sha = e.load_protocol()
    if e_cfg['task']['raw_f2_feature_order'] != cfg['task']['raw_f2_feature_order']:
        raise RuntimeError('F_PHASE_E_FEATURE_SCHEMA_MISMATCH')
    for key in ('learning_rate', 'weight_decay', 'batch_size', 'max_epochs', 'patience',
                'gradient_clip_norm', 'hidden_size', 'hidden_continuous_size', 'dropout'):
        if e_cfg['hyperparameters'][key] != cfg['hyperparameters'][key]:
            raise RuntimeError('F_PHASE_E_TRAINING_CONTRACT_MISMATCH')
    dev = e.c.load_development(cfg)  # exactly nrows=3007 -> filter -> 3005
    train = e.c.build_windows(dev, cfg, 0, 2103)
    val = e.c.build_windows(dev, cfg, 2103, 2404)
    planned = planned_origins(dev, cfg, 2404, 3005)
    counts = dict(train=len(train.origins), validation=len(val.origins), assessment=len(planned))
    if counts != cfg['split']['expected_legal_windows']:
        raise RuntimeError('F_WINDOW_COUNT_MISMATCH')
    if set(train.targets.ravel()) & set(val.targets.ravel()):
        raise RuntimeError('TARGET_ROW_OVERLAP')
    scores, top10 = e.top10_scores(train, cfg['task']['raw_f2_feature_order'])
    if top10 != cfg['preprocessing']['Top10_indices']:
        raise RuntimeError('F_TOP10_MISMATCH')
    scaler = e.CommonScaler().fit(train.x)
    historical = json.loads((e.OUT / 'E_COMMON_SCALER.json').read_text(encoding='utf-8'))
    if not (np.allclose(scaler.mean, historical['mean'], atol=1e-12, rtol=0)
            and np.allclose(scaler.scale, historical['scale'], atol=1e-12, rtol=0)
            and abs(scaler.target_scale - historical['target_scale_V']) < 1e-12):
        raise RuntimeError('F_PHASE_E_SCALER_MISMATCH')
    tx, vx = e.prep(train.x, scaler, top10, 'EWMA'), e.prep(val.x, scaler, top10, 'EWMA')
    if not e.filter_causality_audit()['EWMA']['pass']:
        raise RuntimeError('F_EWMA_CAUSALITY_FAILURE')
    prefix = dev.to_csv(index=False, header=True, lineterminator='\n', float_format='%.17g').encode('utf-8')
    fingerprints = {'development_prefix_fingerprint_sha256': hashlib.sha256(prefix).hexdigest(),
                    'train_raw_x_sha256': array_hash(train.x), 'train_y_sha256': array_hash(train.y),
                    'train_preprocessed_x_sha256': array_hash(tx), 'validation_x_sha256': array_hash(vx),
                    'scaler_sha256': array_hash(np.stack([scaler.mean, scaler.scale]))}
    write_json(OUT / 'audits/F_DATA_ACCESS_AUDIT.json', {
        'protocol_sha256': sha, 'read_nrows': 3007, 'development_rows': 3005,
        'development_filter': 'bin_end_ophrs2_h < 1819.997738', 'full_causal_file_hash_computed': False,
        'test_accessed': False, 'assessment_pretrain_access': 'metadata eligibility count only; no model x/y arrays or predictions',
        'fingerprint_type': 'DEVELOPMENT_LOGICAL_PREFIX_ONLY', 'serialization': {'float_format': '%.17g',
        'encoding': 'utf-8', 'line_terminator': 'LF', 'index': False, 'header': True, 'column_order': list(dev.columns)},
        'phase_E_protocol_sha256': e_sha, 'legal_windows': counts, **fingerprints})
    scores.to_csv(OUT / 'audits/F_FEATURE_SELECTION.csv', index=False)
    write_json(OUT / 'audits/F_COMMON_SCALER.json', {'mean': scaler.mean.tolist(), 'scale': scaler.scale.tolist(),
        'target_scale_V': scaler.target_scale, 'flat_train_samples': scaler.samples,
        'fit': 'unfiltered_train_legal_history_only', 'matches_Phase_E': True, 'Top10_indices': top10, **fingerprints})
    return dev, train, val, planned, scaler, top10, tx, vx, fingerprints


def initialization_audit(cfg):
    models = {}
    for group in GROUPS:
        e.set_seed(42)
        models[group] = make_model(group, cfg, torch.device('cpu'))
    base = models['F0'].state_dict()
    for group in ('F1', 'F2'):
        if not all(torch.equal(v, models[group].state_dict()[k]) for k, v in base.items()):
            raise RuntimeError('F_SHARED_NATIVE_INITIALIZATION_MISMATCH')
    if not all(torch.equal(v, models['F2'].state_dict()[k]) for k, v in models['F1'].state_dict().items()):
        raise RuntimeError('F_SHARED_BRANCH_INITIALIZATION_MISMATCH')
    with torch.no_grad():
        gate = models['F2'].regime_weight(torch.zeros(2, 48, 15))
    if not torch.equal(gate, torch.full_like(gate, .5)):
        raise RuntimeError('F_GATE_INITIALIZATION_MISMATCH')
    result = {'seed': 42, 'shared_native_equal': True, 'shared_branch_equal': True, 'g_equals_half': True,
              'model_initialization_sha256': {k: e.state_digest(m) for k, m in models.items()},
              'shared_native_sha256': e.state_digest(models['F0'])}
    write_json(OUT / 'audits/F_FORMAL_INITIALIZATION_AUDIT.json', result)
    return result


def checkpoint_reload(group, cfg, sha, identity, device, vx, expected=None):
    path = OUT / f'checkpoints/{group}_best_seed42.pt'
    checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    if (checkpoint['protocol_sha256'] != sha or checkpoint['code_identity_sha256'] != identity
            or checkpoint['model_group'] != group or checkpoint['seed'] != 42
            or checkpoint['parameter_count'] != cfg['architecture']['parameter_counts'][group]):
        raise RuntimeError('F_CHECKPOINT_IDENTITY_FAILURE')
    e.set_seed(42)
    model = make_model(group, cfg, device)
    model.load_state_dict(checkpoint['state_dict'], strict=True)
    prediction = e.predict(model, vx[:32], 32, device)
    saved = checkpoint['validation_fixed_prediction'].numpy()
    if not np.array_equal(prediction, saved) or (expected is not None and not np.array_equal(prediction, expected)):
        raise RuntimeError('F_CHECKPOINT_RELOAD_PARITY_FAILURE')
    return model, {'pass': True, 'checkpoint': str(path.relative_to(PHASE)), 'file_sha256': file_hash(path),
                   'max_reload_difference': float(np.max(np.abs(prediction - saved)))}


def fit_one(group, cfg, sha, code_sha, fingerprints, init, train, tx, vx, val, scaler, device, state):
    p = cfg['hyperparameters']
    entry = state['groups'].get(group)
    if entry is not None:
        if entry['status'] == 'COMPLETED_CHECKPOINT_FROZEN':
            _, audit = checkpoint_reload(group, cfg, sha, code_sha, device, vx)
            if audit['file_sha256'] != entry['checkpoint_reload']['file_sha256']:
                raise RuntimeError('COMPLETED_CHECKPOINT_CHANGED')
            return entry['training']
        raise RuntimeError(f'INTERRUPTED_GROUP_REQUIRES_TECHNICAL_REVIEW_NO_RESTART:{group}')
    e.set_seed(42)
    model = make_model(group, cfg, device)
    if e.state_digest(model) != init['model_initialization_sha256'][group]:
        raise RuntimeError('F_RUN_INITIALIZATION_MISMATCH')
    optimizer = torch.optim.AdamW(model.parameters(), lr=p['learning_rate'], weight_decay=p['weight_decay'])
    ty, vy = train.y / scaler.target_scale, val.y / scaler.target_scale
    loader = DataLoader(TensorDataset(torch.from_numpy(tx), torch.from_numpy(ty.astype(np.float32)),
                        torch.arange(len(tx))), batch_size=p['batch_size'], shuffle=True,
                        generator=torch.Generator().manual_seed(42), num_workers=0)
    state['groups'][group] = {'status': 'TRAINING', 'started_epoch': 0}
    write_json(OUT / 'audits/F_EXECUTION_STATE.json', state)
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)
    history = []
    best, best_epoch, stale = math.inf, 0, 0
    started = time.perf_counter()
    best_path = OUT / f'checkpoints/{group}_best_seed42.pt'
    best_path.parent.mkdir(parents=True, exist_ok=True)
    for epoch in range(1, p['max_epochs'] + 1):
        epoch_started = time.perf_counter()
        model.train()
        losses = []
        for batch, (xb, yb, ids) in enumerate(loader):
            if epoch == 1 and batch == 0:
                first = {'origins': train.origins[ids.numpy()].tolist(), 'x_sha256': array_hash(xb.numpy()),
                         'y_sha256': array_hash(yb.numpy())}
                if state.get('first_batch') is None:
                    state['first_batch'] = first
                elif state['first_batch'] != first:
                    raise RuntimeError('F_FIRST_BATCH_ORDER_OR_HASH_MISMATCH')
                write_json(OUT / f'audits/{group}_FIRST_BATCH.json', first)
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = torch.nn.functional.mse_loss(model.predict_delta(xb), yb)
            if not torch.isfinite(loss):
                raise RuntimeError('F_NONFINITE_LOSS')
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), p['gradient_clip_norm'])
            if not torch.isfinite(norm):
                raise RuntimeError('F_NONFINITE_GRADIENT')
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        valid_pred = e.predict(model, vx, p['batch_size'], device)
        rmse = float(np.sqrt(np.mean((valid_pred - vy) ** 2)))
        if not math.isfinite(rmse):
            raise RuntimeError('F_NONFINITE_VALIDATION')
        improved = rmse < best - 1e-12
        if improved:
            best, best_epoch, stale = rmse, epoch, 0
            torch.save({'state_dict': {k: v.detach().cpu() for k, v in model.state_dict().items()},
                        'model_group': group, 'seed': 42, 'parameter_count': count_parameters(model),
                        'protocol_sha256': sha, 'code_identity_sha256': code_sha,
                        'best_epoch': best_epoch, 'target_scale_V': scaler.target_scale,
                        'initialization_sha256': init['model_initialization_sha256'][group],
                        'validation_fixed_prediction': torch.from_numpy(valid_pred[:32].copy()),
                        'data_fingerprints': fingerprints}, best_path)
        else:
            stale += 1
        history.append({'epoch': epoch, 'train_standardized_MSE': float(np.mean(losses)),
                        'validation_standardized_RMSE': rmse, 'validation_RMSE_mV': rmse * scaler.target_scale * 1000,
                        'best_epoch': best_epoch, 'stale_epochs': stale, 'epoch_time_s': time.perf_counter() - epoch_started})
        pd.DataFrame(history).to_csv(OUT / f'training_logs/{group}_epochs_seed42.csv', index=False)
        state['groups'][group]['last_epoch'] = epoch
        state['groups'][group]['best_epoch'] = best_epoch
        write_json(OUT / 'audits/F_EXECUTION_STATE.json', state)
        message = f'{group} epoch={epoch} validation_RMSE_mV={rmse * scaler.target_scale * 1000:.6f} best_epoch={best_epoch} stale={stale}'
        print(message, flush=True)
        with (OUT / 'training_logs/F_REAL_TRAINING_CONSOLE.log').open('a', encoding='utf-8') as log:
            log.write(message + '\n')
        if stale >= p['patience']:
            break
    info = {'run_id': f'{state["run_id"]}_{group}_seed42', 'model': group, 'seed': 42,
            'git_commit': state['git_commit'], 'protocol_sha256': sha, 'code_identity_sha256': code_sha,
            'parameter_count': count_parameters(model), 'initialization_sha256': init['model_initialization_sha256'][group],
            'fingerprints': fingerprints, 'best_epoch': best_epoch, 'stop_epoch': epoch,
            'patience_early_stopped': stale >= p['patience'], 'validation_standardized_RMSE': best,
            'training_time_s': time.perf_counter() - started, 'mean_epoch_time_s': float(np.mean([r['epoch_time_s'] for r in history])),
            'peak_GPU_memory_bytes': int(torch.cuda.max_memory_allocated(device)) if device.type == 'cuda' else 0}
    del model, optimizer
    _, reload = checkpoint_reload(group, cfg, sha, code_sha, device, vx)
    state['groups'][group] = {'status': 'COMPLETED_CHECKPOINT_FROZEN', 'training': info, 'checkpoint_reload': reload}
    write_json(OUT / f'training_logs/{group}_run_seed42.json', info)
    write_json(OUT / 'audits/F_EXECUTION_STATE.json', state)
    return info


def run():
    cfg, sha = load_protocol()
    head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip()
    if head != cfg['base_commit']:
        raise RuntimeError('F_BASE_COMMIT_CHANGED')
    for folder in ('audits', 'checkpoints', 'training_logs', 'predictions', 'metrics'):
        (OUT / folder).mkdir(parents=True, exist_ok=True)
    identity = code_identity()
    torch.set_num_threads(cfg['environment']['torch_cpu_threads'])
    env = cfg['environment']
    if (platform.python_version() != env['python'] or str(torch.__version__) != env['torch']
            or str(pf.__version__) != env['pytorch_forecasting']):
        raise RuntimeError('F_ENVIRONMENT_MISMATCH')
    for path, expected in ((inspect.getsourcefile(TemporalFusionTransformer), env['tft_sha256']),
                           (inspect.getsourcefile(sub_modules.InterpretableMultiHeadAttention), env['sub_modules_sha256'])):
        if file_hash(path) != expected:
            raise RuntimeError('F_UPSTREAM_SOURCE_IDENTITY_MISMATCH')
    subprocess.run([sys.executable, '-B', str(PHASE / 'scripts/audit_f_model_implementation.py')], cwd=PHASE, check=True)
    execution_tests = subprocess.run([sys.executable, '-B', '-m', 'unittest', 'discover', '-s', 'tests_execution', '-v'],
                                    cwd=PHASE, text=True, capture_output=True)
    write_json(OUT / 'audits/F_EXECUTION_PLUMBING_TESTS.json', {'returncode': execution_tests.returncode,
               'stdout': execution_tests.stdout, 'stderr': execution_tests.stderr, 'synthetic_only': True,
               'test_accessed': False, 'formal_training': False})
    if execution_tests.returncode != 0:
        raise RuntimeError('F_EXECUTION_PLUMBING_TESTS_FAILED')
    write_json(OUT / 'audits/F_EXECUTION_CODE_IDENTITY.json', identity)
    e.set_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    write_json(OUT / 'audits/F_ENVIRONMENT.json', {'python': platform.python_version(), 'torch': str(torch.__version__),
        'pytorch_forecasting': pf.__version__, 'GPU': torch.cuda.get_device_name(0) if device.type == 'cuda' else None,
        'device': str(device), 'CUDA': torch.version.cuda, 'cuDNN': torch.backends.cudnn.version(),
        'cudnn_deterministic': torch.backends.cudnn.deterministic, 'cudnn_benchmark': torch.backends.cudnn.benchmark,
        'torch_deterministic_algorithms': torch.are_deterministic_algorithms_enabled(), 'CPU_threads': torch.get_num_threads()})
    init = initialization_audit(cfg)
    dev, train, val, planned, scaler, top10, tx, vx, fingerprints = prepare(cfg, sha)
    state_path = OUT / 'audits/F_EXECUTION_STATE.json'
    if state_path.exists():
        state = json.loads(state_path.read_text(encoding='utf-8'))
        if state['protocol_sha256'] != sha or state['code_identity_sha256'] != identity['manifest_sha256'] or state['fingerprints'] != fingerprints:
            raise RuntimeError('F_EXISTING_RUN_IDENTITY_MISMATCH')
        if state['status'] in ('COMPLETED', 'PHASE_F_EXECUTION_BLOCKED'):
            raise RuntimeError('F_TERMINAL_RUN_NO_AUTOMATIC_REEXECUTION')
    else:
        state = {'status': 'TRAINING', 'run_id': time.strftime('PhaseF_%Y%m%d_%H%M%S'), 'git_commit': head,
                 'protocol_sha256': sha, 'code_identity_sha256': identity['manifest_sha256'], 'fingerprints': fingerprints, 'groups': {}}
    for group in GROUPS:
        fit_one(group, cfg, sha, identity['manifest_sha256'], fingerprints, init, train, tx, vx, val, scaler, device, state)
    reloads = {g: checkpoint_reload(g, cfg, sha, identity['manifest_sha256'], device, vx)[1] for g in GROUPS}
    write_json(OUT / 'audits/F_ALL_CHECKPOINTS_FROZEN.json', {'pass': True, 'protocol_sha256': sha, 'checkpoints': reloads,
               'assessment_predictions_generated_before_freeze': False})
    state['status'] = 'ALL_CHECKPOINTS_FROZEN'
    write_json(state_path, state)
    # First construction of Assessment inputs/targets, after all freezes.
    assess = e.c.build_windows(dev, cfg, 2404, 3005)
    if not np.array_equal(assess.origins, planned):
        raise RuntimeError('F_ASSESSMENT_KEY_PLAN_MISMATCH')
    e.c.assert_isolation(train, val, assess)
    ax = e.prep(assess.x, scaler, top10, 'EWMA')
    metrics = []
    for group in GROUPS:
        model, _ = checkpoint_reload(group, cfg, sha, identity['manifest_sha256'], device, vx)
        for split, windows, x in (('validation', val, vx), ('assessment', assess, ax)):
            pred = e.predict(model, x, 32, device) * scaler.target_scale
            frame = e.prediction_frame(group, windows, pred, sha)
            frame['split'] = split
            frame['target_age_h'] = dev.bin_end_ophrs2_h.to_numpy()[frame.target_row.to_numpy()]
            frame.to_csv(OUT / f'predictions/{group}_{split}_seed42.csv', index=False, float_format='%.17g')
            metrics.append({'experiment_group': group, 'split': split, 'seed': 42, **e.numpy_metrics(frame),
                            **state['groups'][group]['training']})
        if group == 'F2':
            gates = []
            handle = model.regime_gate.register_forward_hook(lambda _m, _a, out: gates.extend(out.detach().cpu().reshape(-1).tolist()))
            try:
                for split, windows, x in (('validation', val, vx), ('assessment', assess, ax)):
                    gates.clear()
                    e.predict(model, x, 32, device)
                    context = model.regime_indices
                    gf = pd.DataFrame({'split': split, 'origin_row': windows.origins, 'gate': gates})
                    for idx in context:
                        gf[cfg['task']['raw_f2_feature_order'][idx]] = x[:, :, idx].mean(axis=1)
                    gf.to_csv(OUT / f'predictions/F2_{split}_GATES.csv', index=False, float_format='%.17g')
            finally:
                handle.remove()
        del model
    table = pd.DataFrame(metrics)
    table.drop(columns=['fingerprints']).to_csv(OUT / 'metrics/F_SEED42_METRICS.csv', index=False)
    authority = []
    for split, windows in (('validation', val), ('assessment', assess)):
        rows = e.prediction_frame('FROZEN_TRUTH', windows, windows.y, sha)
        rows['split'] = split
        authority.append(rows[['split', 'origin_row', 'target_row', 'horizon_step', 'y_true_delta_V']])
    pd.concat(authority).to_csv(OUT / 'audits/F_PREDICTION_KEY_TRUTH_AUTHORITY.csv', index=False, float_format='%.17g')
    subprocess.run([sys.executable, '-B', str(PHASE / 'scripts/audit_f_predictions.py')], cwd=PHASE, check=True)
    subprocess.run([sys.executable, '-B', str(PHASE / 'scripts/plot_f_results.py')], cwd=PHASE, check=True)
    from report_f_results import generate_reports
    generate_reports(cfg, sha)
    state['status'] = 'COMPLETED'
    write_json(state_path, state)
    print('PHASE_F_F0_F1_F2_DEVELOPMENT_EXPLORATION_COMPLETED', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', action='store_true', required=True)
    parser.parse_args()
    try:
        run()
    except Exception as exc:
        record = {'status': 'PHASE_F_EXECUTION_BLOCKED', 'exception': repr(exc), 'traceback': traceback.format_exc(),
                  'test_accessed': False, 'automatic_retraining_allowed': False}
        write_json(OUT / 'audits/F_TECHNICAL_FAILURE.json', record)
        print(json.dumps(record, ensure_ascii=False), flush=True)
        raise
