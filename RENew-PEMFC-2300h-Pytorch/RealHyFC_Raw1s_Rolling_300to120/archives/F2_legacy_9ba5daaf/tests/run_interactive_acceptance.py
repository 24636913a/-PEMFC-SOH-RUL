"""Execute real # %% cells in Python 3.11 ipykernel; retain actual image outputs.

No VS Code GUI automation is claimed. VS Code uses the same IPython image/png
display protocol; executed notebooks are a directly viewable display receipt.
"""
from pathlib import Path
import hashlib
import json
import math
import queue
import re
import subprocess
import sys
import time

from jupyter_client import KernelManager
import nbformat
import numpy as np
import pandas as pd

FOLDER = Path(__file__).resolve().parents[1]
ROOT = FOLDER.parents[1]
OUTPUTS = ROOT / 'RealHyFC_F2_Main_Outputs'
SCRIPT = FOLDER / 'F2_VSCode_Interactive.py'
RECEIPTS = FOLDER / 'interactive_acceptance'
SEEDS = [42, 103, 193, 202, 210, 212]


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for data in iter(lambda: stream.read(1024 * 1024), b''): digest.update(data)
    return digest.hexdigest()


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def csv(path):
    return pd.read_csv(path, float_precision='round_trip')


def notebook_from_script():
    notebook = nbformat.v4.new_notebook()
    for part in re.split(r'^# %%', SCRIPT.read_text(encoding='utf-8'), flags=re.MULTILINE)[1:]:
        label, _, content = part.partition('\n')
        if '[markdown]' in label:
            text = '\n'.join(re.sub(r'^# ?', '', line) for line in content.splitlines()).strip()
            notebook.cells.append(nbformat.v4.new_markdown_cell(text))
        else:
            notebook.cells.append(nbformat.v4.new_code_cell(content.strip()))
    notebook.metadata.kernelspec = dict(display_name='Python 3.11 (RealHyFC CUDA)', language='python', name='python3')
    notebook.metadata.language_info = dict(name='python', version='3.11.9')
    return notebook


def execute_notebook(seed):
    notebook = notebook_from_script()
    first_code = next(cell for cell in notebook.cells if cell.cell_type == 'code')
    assert first_code.source.count('SEED = 202') == 1
    first_code.source = first_code.source.replace('SEED = 202', f'SEED = {seed!r}', 1)
    before = set(OUTPUTS.iterdir())
    km = KernelManager(kernel_name='python3')
    km.kernel_spec.argv = [sys.executable, '-B', '-m', 'ipykernel_launcher', '-f', '{connection_file}']
    km.start_kernel(cwd=str(FOLDER))
    client = km.client()
    client.start_channels()
    images, event_types = 0, []
    receipt = RECEIPTS / ('Seed202.executed.ipynb' if seed == 202 else 'SixSeeds.executed.ipynb')
    try:
        client.wait_for_ready(timeout=60)
        for cell_index, cell in enumerate(notebook.cells):
            if cell.cell_type != 'code': continue
            message_id = client.execute(cell.source, store_history=True, stop_on_error=True)
            cell.outputs = []
            deadline = time.monotonic() + 300
            while True:
                if time.monotonic() > deadline: raise TimeoutError(f'Kernel cell {cell_index} timed out')
                try: message = client.get_iopub_msg(timeout=2)
                except queue.Empty: continue
                if message['parent_header'].get('msg_id') != message_id: continue
                kind, content = message['msg_type'], message['content']
                if kind == 'execute_input': cell.execution_count = content['execution_count']
                elif kind in ['display_data', 'execute_result']:
                    fields = dict(data=content['data'], metadata=content.get('metadata', {}))
                    if kind == 'execute_result': fields['execution_count'] = content['execution_count']
                    cell.outputs.append(nbformat.v4.new_output(kind, **fields))
                    if 'image/png' in content['data']: images += 1
                    event_types.append(kind)
                elif kind == 'stream':
                    cell.outputs.append(nbformat.v4.new_output('stream', name=content['name'], text=content['text']))
                elif kind == 'error':
                    cell.outputs.append(nbformat.v4.new_output('error', ename=content['ename'], evalue=content['evalue'], traceback=content['traceback']))
                    raise RuntimeError('\n'.join(content['traceback']))
                elif kind == 'status' and content['execution_state'] == 'idle': break
            nbformat.write(notebook, receipt)
            print('CELL_PASS', seed, cell_index, 'actual_png_displays', images, flush=True)
        client.execute('pass', store_history=False)
    finally:
        nbformat.write(notebook, receipt)
        client.stop_channels()
        km.shutdown_kernel(now=True)
    created = [path for path in set(OUTPUTS.iterdir()) - before if path.is_dir()]
    assert len(created) == 1
    run = created[0]
    expected = [202] if seed == 202 else SEEDS
    acceptance = read(run / 'INTERACTIVE_ACCEPTANCE.json')
    assert acceptance['status'] == 'PASS' and acceptance['seeds'] == expected
    assert acceptance['ipython_active'] and images == acceptance['requested_image_displays'] == 5 + 3 * len(expected)
    assert not acceptance['new_training'] and not acceptance['new_Optuna_search']
    for item in csv(run / 'ARTIFACT_MANIFEST.csv').itertuples(): assert sha(run / item.path) == item.sha256
    for name in acceptance['images']: assert (run / name).is_file()
    original_cli = read(OUTPUTS / 'SIX_SEED_EXECUTION.json')
    old_all = OUTPUTS / next(row['runs'][0] for row in original_cli['records'] if row['seed_argument'] == 'all')
    official = csv(run / 'F2_PARTITION_METRICS.csv')
    for active in expected:
        for partition in ['TRAIN', 'VALIDATION', 'HISTORICAL_ASSESSMENT', 'LATE_LIFE']:
            actual = csv(run / f'seed{active}/predictions/{partition}.csv')
            reference = csv(old_all / f'seed{active}/predictions/{partition}.csv')
            pd.testing.assert_frame_equal(actual, reference, check_exact=True)
        audit = next(row for row in acceptance['seed_audits'] if row['seed'] == active)
        assert audit['pointwise_exact'] and audit['optimizer_steps'] == 0
        assert audit['state_unchanged'] and not audit['training_performed']
    old_metrics = csv(old_all / 'F2_PARTITION_METRICS.csv')
    for row in official.itertuples():
        reference = old_metrics.loc[(old_metrics.seed == row.seed) & (old_metrics.partition == row.partition)].iloc[0]
        assert [row.RMSE_mV, row.MAE_mV, row.R2] == [reference.RMSE_mV, reference.MAE_mV, reference.R2]
    validation = official.loc[official.partition == 'VALIDATION']
    summary = read(run / 'evaluation_results.json')
    assert summary['RMSE_V'] == math.fsum(validation.RMSE_mV) / len(expected) / 1000
    assert summary['MAE_V'] == math.fsum(validation.MAE_mV) / len(expected) / 1000
    assert summary['R2'] == math.fsum(validation.R2) / len(expected)
    return dict(seed=seed, run=run.relative_to(ROOT).as_posix(), status='PASS',
                real_ipykernel=True, actual_image_png_displays=images, executed_code_cells=9,
                all_partition_predictions_and_metrics_exact=True, summary=summary,
                notebook=receipt.relative_to(FOLDER).as_posix())


if __name__ == '__main__':
    if hasattr(sys.stdout, 'reconfigure'): sys.stdout.reconfigure(encoding='utf-8')
    RECEIPTS.mkdir(exist_ok=False)
    # Save a clean companion notebook; the .py remains the single authoring source.
    nbformat.write(notebook_from_script(), FOLDER / 'F2_VSCode_Interactive.ipynb')
    history = [path for path in OUTPUTS.rglob('*') if path.is_file()]
    history.extend([FOLDER / 'RealHyFC_F2_Main.py', ROOT / 'RealHyFC_F2_Main.py'])
    if (FOLDER / 'test_cli_arguments.py').is_file(): history.append(FOLDER / 'test_cli_arguments.py')
    assets = read(FOLDER / 'RealHyFC_F2_Main_Seed_Assets.json')
    for row in assets['models']:
        history.extend(ROOT.parent / row[key] for key in ['checkpoint', 'reference_prediction', 'training_history', 'protocol_source'])
    history.extend((ROOT / 'Phase_TFT_Hyperparameter_Optimization_Restart/outputs').glob('*.db'))
    before = {str(path): sha(path) for path in set(history)}
    report = dict(status='RUNNING', python=sys.executable, requested=[202, 'all'], runs=[],
                  VS_Code_GUI_clicked=False, display_validation='Real IPython image/png outputs; open executed notebooks in VS Code')
    report_file = RECEIPTS / 'ACCEPTANCE.json'
    try:
        for seed in [202, 'all']:
            report['runs'].append(execute_notebook(seed))
            report_file.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8', newline='\n')
        # Explicitly confirm the original command-line interface still returns only three metrics.
        directories = set(OUTPUTS.iterdir())
        cli = subprocess.run([sys.executable, '-B', str(FOLDER / 'RealHyFC_F2_Main.py'), '--seed', '202'],
                             cwd=FOLDER, capture_output=True, encoding='utf-8')
        assert cli.returncode == 0 and not cli.stderr and len(cli.stdout.strip().splitlines()) == 3
        created = [path for path in set(OUTPUTS.iterdir()) - directories if path.is_dir()]
        assert len(created) == 1
        cli_summary = read(created[0] / 'evaluation_results.json')
        assert cli_summary == report['runs'][0]['summary']
        report['CLI'] = dict(status='PASS', stdout=cli.stdout, run=created[0].relative_to(ROOT).as_posix())
        for filename, digest in before.items(): assert sha(Path(filename)) == digest, ('HISTORY_MODIFIED', filename)
        report.update(status='PASS', historical_files_unchanged=len(before),
                      model_and_original_programs_modified=False, new_training=False, new_search=False)
    except Exception as error:
        report.update(status='FAILURE_PRESERVED', error=repr(error))
        raise
    finally:
        report_file.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8', newline='\n')
    print('INTERACTIVE_KERNEL_AND_CLI_ACCEPTANCE_PASS', flush=True)
