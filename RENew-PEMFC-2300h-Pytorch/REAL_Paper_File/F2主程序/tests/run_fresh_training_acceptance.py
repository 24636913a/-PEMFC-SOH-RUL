"""Full GPU acceptance: Seed202 compare first, then gated six-seed compare.

Both passes execute the unmodified 450-epoch/patience40 training program in
real IPython kernels. No old weights, reduced epochs, resumes or cached results.
"""
from pathlib import Path
import ast
import ctypes
import hashlib
import json
import math
import queue
import re
import subprocess
import sys
import time
import types

from jupyter_client import KernelManager
import nbformat
import numpy as np
import pandas as pd

FOLDER = Path(__file__).resolve().parents[1]
ROOT = FOLDER.parents[1]
REPO = ROOT.parent
SCRIPT = FOLDER / 'F2_Train_Interactive.py'
OUTPUTS = FOLDER / 'F2_Training_Outputs'
RECEIPTS = FOLDER / 'fresh_training_acceptance'
SEEDS = [42, 103, 193, 202, 210, 212]


def sha(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''): result.update(block)
    return result.hexdigest()


def read(path): return json.loads(Path(path).read_text(encoding='utf-8'))
def csv(path): return pd.read_csv(path, float_precision='round_trip')


def save_report(report):
    path = RECEIPTS / 'ACCEPTANCE.json'
    temporary = path.with_suffix('.json.writing')
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8', newline='\n')
    temporary.replace(path)


def make_notebook(seed):
    notebook = nbformat.v4.new_notebook()
    for part in re.split(r'^# %%', SCRIPT.read_text(encoding='utf-8'), flags=re.MULTILINE)[1:]:
        label, _, content = part.partition('\n')
        notebook.cells.append(nbformat.v4.new_code_cell(content.strip()))
    first = notebook.cells[0]
    assert first.source.count('SEED = 202') == 1
    first.source = first.source.replace('SEED = 202', f'SEED = {seed!r}', 1)
    assert 'MAX_EPOCHS = 450' in first.source and 'PATIENCE = 40' in first.source
    assert 'FEATURE_MODE = "compare"' in first.source
    notebook.metadata.kernelspec = dict(name='python3', display_name='RealHyFC Python 3.11 CUDA', language='python')
    notebook.metadata.language_info = dict(name='python', version='3.11.9')
    return notebook


def independently_verify_run(run, expected_seeds):
    acceptance = read(run / 'TRAINING_ACCEPTANCE.json')
    protocol = read(run / 'PROTOCOL_BEFORE_TRAINING.json')
    assert acceptance['status'] == 'PASS' and acceptance['seeds'] == expected_seeds
    assert acceptance['feature_modes'] == ['top10', 'top11'] and acceptance['fresh_models'] == len(expected_seeds) * 2
    assert protocol['code_sha256'] == sha(SCRIPT) and protocol['max_epochs'] == 450 and protocol['patience'] == 40
    assert [(r['start_row_inclusive'], r['end_row_exclusive']) for r in protocol['partitions']] == [(0, 2690), (2690, 3074), (3074, 3843)]
    assert [r['legal_origins'] for r in protocol['partitions']] == [1750, 218, 527]
    assert protocol['masks']['top11'] == protocol['masks']['top10'] + [5]
    module = types.ModuleType('new_training_independent_audit'); module.__file__ = str(SCRIPT)
    sys.modules[module.__name__] = module
    prefix = SCRIPT.read_text(encoding='utf-8').split('# %% 4. 每次Run All')[0]
    exec(compile(prefix, str(SCRIPT), 'exec'), module.__dict__)
    module.configure_environment()
    frame = pd.read_csv(module.SOURCE)
    train, _ = module.build_windows(frame, 0, 2690)
    holdout, _ = module.build_windows(frame, 3074, 3843)
    scaler = module.TrainScaler().fit(train.x_raw)
    saved_scaler = read(run / 'PREPROCESSOR.json')
    assert np.array_equal(scaler.mean, saved_scaler['mean']) and np.array_equal(scaler.scale, saved_scaler['scale'])
    rows = []; updates = 0
    for seed in expected_seeds:
        initializations, histories = [], []
        for mode in ['top10', 'top11']:
            folder = run / f'seed{seed}/{mode}'
            ledger = read(folder / 'TRAINING_LEDGER.json')
            assert ledger['status'] == 'TRAINED' and ledger['training_from_scratch'] and not ledger['historical_checkpoint_loaded']
            assert ledger['optimizer_steps'] == ledger['backward_calls'] > 0 and ledger['optimizer_initial_state_empty']
            assert ledger['initialization_sha256'] != ledger['first_update_sha256'] != ledger['final_state_sha256']
            assert ledger['stop_epoch'] <= 450
            history = csv(folder / 'training_history.csv'); histories.append(history)
            assert history.epoch.tolist() == list(range(1, len(history) + 1)) and len(history) == ledger['stop_epoch']
            assert ledger['optimizer_steps'] == len(history) * math.ceil(1750 / 32)
            best = math.inf; best_epoch = 0; stale = 0
            for record in history.itertuples():
                value = math.sqrt(record.validation_standardized_MSE)
                if value < best - 1e-12: best, best_epoch, stale = value, record.epoch, 0
                else: stale += 1
                assert record.best_epoch == best_epoch and record.stale_epochs == stale
            assert ledger['best_epoch'] == best_epoch
            if len(history) < 450: assert stale == 40
            checkpoint_file = folder / 'best_model.pt'
            assert sha(checkpoint_file) == ledger['best_checkpoint_sha256']
            checkpoint = module.torch.load(checkpoint_file, map_location='cpu', weights_only=True)
            assert checkpoint['run_id'] == run.name and checkpoint['training_from_scratch'] and not checkpoint['historical_checkpoint_loaded']
            assert checkpoint['best_epoch'] == best_epoch and checkpoint['mask'] == protocol['masks'][mode]
            initializations.append(ledger['initialization_sha256']); updates += ledger['optimizer_steps']
            points = csv(folder / 'predictions_TEST.csv')
            assert len(points) == 12648 and points.origin_row.nunique() == 527
            assert np.array_equal(points.origin_row, np.repeat(holdout.origins, 24))
            assert np.array_equal(points.target_row, holdout.targets.ravel())
            assert np.array_equal(points.y_true_delta_V, holdout.y.ravel().astype('float64'))
            # Recompute from the NEW checkpoint using independently rebuilt inputs.
            model = module.initialize_model(seed); model.load_state_dict(checkpoint['state_dict'], strict=True); model.eval()
            replay = module.predict_standardized(model, module.masked_inputs(holdout.x_raw, scaler, mode), 32) * scaler.target_scale
            assert np.array_equal(replay.ravel().astype('float64'), points.y_pred_delta_V), ('NEW_CHECKPOINT_REPLAY', seed, mode)
            del model, checkpoint; module.torch.cuda.empty_cache()
            error = points.y_pred_delta_V.to_numpy() - points.y_true_delta_V.to_numpy()
            truth = points.y_true_delta_V.to_numpy(); n = len(error)
            sse = math.fsum(float(x) ** 2 for x in error); mean = math.fsum(map(float, truth)) / n
            row = dict(seed=seed, feature_mode=mode, RMSE_V=math.sqrt(sse / n),
                MAE_V=math.fsum(abs(float(x)) for x in error) / n,
                R2=1 - sse / math.fsum((float(x) - mean) ** 2 for x in truth))
            rows.append(row)
            for figure in ['training_history.png', 'prediction_vs_true.png', 'scatter_true_vs_pred.png', 'scatter_colored_by_load.png']:
                from PIL import Image
                with Image.open(folder / figure) as image: image.verify()
                with Image.open(folder / figure) as image: assert min(image.info['dpi']) >= 299
        assert initializations[0] == initializations[1]
        common = min(map(len, histories))
        assert np.array_equal(histories[0].sampler_order_sha256.iloc[:common], histories[1].sampler_order_sha256.iloc[:common])
    recorded = csv(run / 'HOLDOUT_PER_SEED_METRICS.csv')
    for row in rows:
        actual = recorded.loc[(recorded.seed == row['seed']) & (recorded.feature_mode == row['feature_mode'])].iloc[0]
        assert all(actual[key] == row[key] for key in ['RMSE_V', 'MAE_V', 'R2'])
    means = csv(run / 'HOLDOUT_MEAN_METRICS.csv')
    for mode in ['top10', 'top11']:
        records = [row for row in rows if row['feature_mode'] == mode]
        stored = means.loc[means.feature_mode == mode].iloc[0]
        for key in ['RMSE_V', 'MAE_V', 'R2']: assert stored[key] == math.fsum(row[key] for row in records) / len(records)
    for record in csv(run / 'ARTIFACT_MANIFEST.csv').itertuples(): assert sha(run / record.path) == record.sha256
    return dict(status='PASS', seeds=expected_seeds, fresh_models=len(rows), optimizer_steps=updates,
                new_best_checkpoint_replay_exact=True, initializations_and_sampler_paired=True,
                history_and_best_epoch_recomputed=True, metrics_independently_recomputed=True)


def execute_cohort(seed, report):
    notebook = make_notebook(seed)
    before = set(OUTPUTS.iterdir()) if OUTPUTS.exists() else set()
    file = RECEIPTS / ('Seed202_fresh.executed.ipynb' if seed == 202 else 'All_fresh.executed.ipynb')
    km = KernelManager(kernel_name='python3')
    km.kernel_spec.argv = [sys.executable, '-B', '-m', 'ipykernel_launcher', '-f', '{connection_file}']
    km.start_kernel(cwd=str(FOLDER)); client = km.client(); client.start_channels()
    display_locations = {}; displayed = 0; refreshed = 0
    last_save = time.monotonic()
    try:
        client.wait_for_ready(timeout=60)
        for cell_index, cell in enumerate(notebook.cells):
            msg_id = client.execute(cell.source, store_history=True, stop_on_error=True)
            cell.outputs = []; deadline = time.monotonic() + 24 * 3600
            while True:
                if time.monotonic() > deadline: raise TimeoutError('Full training cell exceeded 24h')
                try: message = client.get_iopub_msg(timeout=2)
                except queue.Empty: continue
                if message['parent_header'].get('msg_id') != msg_id: continue
                kind, content = message['msg_type'], message['content']
                if kind == 'execute_input': cell.execution_count = content['execution_count']
                elif kind in ['display_data', 'execute_result', 'update_display_data']:
                    display_id = content.get('transient', {}).get('display_id')
                    if kind == 'update_display_data':
                        if display_id in display_locations:
                            old_cell, index = display_locations[display_id]
                            old_cell.outputs[index].data = content['data']; old_cell.outputs[index].metadata = content.get('metadata', {})
                        refreshed += 1
                    else:
                        fields = dict(data=content['data'], metadata=content.get('metadata', {}))
                        if kind == 'execute_result': fields['execution_count'] = content['execution_count']
                        cell.outputs.append(nbformat.v4.new_output(kind, **fields))
                        if display_id: display_locations[display_id] = (cell, len(cell.outputs) - 1)
                        if 'image/png' in content['data']: displayed += 1
                elif kind == 'stream':
                    text = content['text']
                    if cell.outputs and cell.outputs[-1].output_type == 'stream' and cell.outputs[-1].name == content['name']:
                        cell.outputs[-1].text += text
                    else: cell.outputs.append(nbformat.v4.new_output('stream', name=content['name'], text=text))
                    for line in text.splitlines():
                        match = re.search(r'Epoch (\d+)/450', line)
                        if not match or int(match.group(1)) == 1 or int(match.group(1)) % 10 == 0:
                            print(str(seed) + ' | ' + line, flush=True)
                    report['last_progress'] = dict(cohort=seed, cell=cell_index, text=text[-1000:])
                elif kind == 'error':
                    cell.outputs.append(nbformat.v4.new_output('error', ename=content['ename'], evalue=content['evalue'], traceback=content['traceback']))
                    raise RuntimeError('\n'.join(content['traceback']))
                elif kind == 'status' and content['execution_state'] == 'idle': break
                if time.monotonic() - last_save > 25:
                    nbformat.write(notebook, file); save_report(report); last_save = time.monotonic()
            nbformat.write(notebook, file); save_report(report)
            print('CELL_COMPLETE', seed, cell_index, flush=True)
    finally:
        nbformat.write(notebook, file); client.stop_channels(); km.shutdown_kernel(now=True)
    created = [path for path in set(OUTPUTS.iterdir()) - before if path.is_dir()]
    assert len(created) == 1
    expected = [202] if seed == 202 else SEEDS
    verification = independently_verify_run(created[0], expected)
    assert displayed >= 3 + 7 * len(expected) and refreshed > 0
    return dict(cohort=seed, run=created[0].relative_to(FOLDER).as_posix(), notebook=file.name,
                displayed_PNG=displayed, real_loss_curve_refreshes=refreshed, verification=verification)


if __name__ == '__main__':
    if hasattr(sys.stdout, 'reconfigure'): sys.stdout.reconfigure(encoding='utf-8')
    RECEIPTS.mkdir(exist_ok=True)
    assert not (RECEIPTS / 'ACCEPTANCE.json').exists(), 'Inspect the previous run rather than overwrite training evidence'
    report = dict(status='RUNNING_SEED202_GATE', base_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip(),
        program_sha256=sha(SCRIPT), python=sys.executable, max_epochs=450, patience=40,
        expected_total_fresh_trainings=14, cohorts=[], all_started_after_seed202_pass=False,
        no_historical_checkpoints=True, VSCode_GUI_clicked=False, display_evidence='real ipykernel display_data/update_display_data')
    # Protect the actual RealHyFC research assets; unrelated old datasets are never accessed.
    names=subprocess.check_output(['git','ls-files','-z'],cwd=REPO).decode('utf-8').rstrip('\0').split('\0')
    eligible=[name for name in names if name.startswith('RENew-PEMFC-2300h-Pytorch/') and (REPO/name).is_file()]
    protected={}; large_file_stats={}
    for index,name in enumerate(eligible):
        path=REPO/name
        if path.stat().st_size <= 64*1024*1024: protected[name]=sha(path)
        else: large_file_stats[name]=dict(bytes=path.stat().st_size,mtime_ns=path.stat().st_mtime_ns)
        if index%5000==0: print('HISTORY_PROTECTION',index,'/',len(eligible),flush=True)
    for path in [FOLDER/'__pycache__', ROOT/'RealHyFC_F2_Main_Outputs/20261019T040926_981777Z_F2_interactive_seed202']:
        if path.exists():
            protected.update({p.relative_to(REPO).as_posix():sha(p) for p in path.rglob('*') if p.is_file()})
    (RECEIPTS/'HISTORY_BEFORE.json').write_text(json.dumps(protected,ensure_ascii=False,indent=2)+'\n',encoding='utf-8',newline='\n')
    save_report(report)
    ctypes.windll.kernel32.SetThreadExecutionState(0x80000003)
    try:
        report['cohorts'].append(execute_cohort(202,report))
        report.update(status='SEED202_GATE_PASSED_RUNNING_ALL',all_started_after_seed202_pass=True)
        save_report(report); print('SEED202_FULL_TRAINING_GATE_PASSED_STARTING_ALL',flush=True)
        report['cohorts'].append(execute_cohort('all',report))
        for name,digest in protected.items(): assert sha(REPO/name)==digest,('HISTORY_CHANGED',name)
        for name,stat in large_file_stats.items():
            actual=(REPO/name).stat()
            assert actual.st_size==stat['bytes'] and actual.st_mtime_ns==stat['mtime_ns'],('UNUSED_LARGE_FILE_CHANGED',name)
        assert sha(SCRIPT)==report['program_sha256']
        report.update(status='PASS',historical_files_unchanged=len(protected),unused_large_files_stat_unchanged=len(large_file_stats),actual_fresh_trainings=14,
                      no_deletion_or_archival=True,finished_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()))
    except Exception as error:
        report.update(status='FAILURE_PRESERVED',error=repr(error)); raise
    finally:
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)
        save_report(report)
    print('ALL_FRESH_TRAINING_ACCEPTANCE_PASSED',flush=True)
