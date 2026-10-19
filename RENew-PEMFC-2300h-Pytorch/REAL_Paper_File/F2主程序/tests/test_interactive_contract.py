"""Static UI contracts plus actual saved IPython display receipts; no inference."""
import ast
import importlib.util
import json
from pathlib import Path
import sys
import unittest

import nbformat

FOLDER = Path(__file__).resolve().parents[1]
SCRIPT = FOLDER / 'F2_VSCode_Interactive.py'
spec = importlib.util.spec_from_file_location('interactive_receipt_tools', FOLDER / 'tests/run_interactive_acceptance.py')
tools = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = tools
spec.loader.exec_module(tools)


class InteractiveContracts(unittest.TestCase):
    def test_all_nine_code_cells_present(self):
        notebook = tools.notebook_from_script()
        self.assertEqual(sum(cell.cell_type == 'code' for cell in notebook.cells), 9)
        self.assertEqual(sum(cell.cell_type == 'markdown' for cell in notebook.cells), 1)
        first = next(cell for cell in notebook.cells if cell.cell_type == 'code')
        self.assertEqual(first.source.count('SEED = 202'), 1)

    def test_notebook_is_exact_script_companion(self):
        actual = nbformat.read(FOLDER / 'F2_VSCode_Interactive.ipynb', as_version=4)
        expected = tools.notebook_from_script()
        self.assertEqual([(c.cell_type, c.source) for c in actual.cells],
                         [(c.cell_type, c.source) for c in expected.cells])
        self.assertTrue(all(not cell.outputs for cell in actual.cells if cell.cell_type == 'code'))

    def test_no_training_or_search_entry_calls(self):
        tree = ast.parse(SCRIPT.read_text(encoding='utf-8'))
        forbidden = {'backward', 'optimizer_step', 'train_new_f2', 'optimize', 'create_study', 'load_study', 'minimize'}
        called = {getattr(node.func, 'attr', getattr(node.func, 'id', '')) for node in ast.walk(tree)
                  if isinstance(node, ast.Call)}
        self.assertFalse(called & forbidden)
        self.assertNotIn('main', called)
        self.assertIn('predict_partitions', called)
        self.assertIn('evaluation_summary', called)

    def test_sqlite_access_is_read_only(self):
        text = SCRIPT.read_text(encoding='utf-8')
        self.assertIn("'?mode=ro'", text)
        self.assertNotIn('import optuna', text)

    def test_saved_kernels_displayed_real_images(self):
        report = json.loads((FOLDER / 'interactive_acceptance/ACCEPTANCE.json').read_text(encoding='utf-8'))
        self.assertEqual(report['status'], 'PASS')
        self.assertFalse(report['new_training'])
        self.assertFalse(report['new_search'])
        self.assertFalse(report['VS_Code_GUI_clicked'])
        for item, expected_count in zip(report['runs'], [8, 23]):
            notebook = nbformat.read(FOLDER / item['notebook'], as_version=4)
            png_count = sum('image/png' in output.get('data', {}) for cell in notebook.cells
                            if cell.cell_type == 'code' for output in cell.outputs)
            self.assertEqual(png_count, expected_count)
            self.assertTrue(all(cell.execution_count is not None for cell in notebook.cells if cell.cell_type == 'code'))
            self.assertTrue(item['all_partition_predictions_and_metrics_exact'])

    def test_cli_result_matches_interactive_seed202(self):
        report = json.loads((FOLDER / 'interactive_acceptance/ACCEPTANCE.json').read_text(encoding='utf-8'))
        self.assertEqual(report['CLI']['status'], 'PASS')
        self.assertEqual(len(report['CLI']['stdout'].strip().splitlines()), 3)
        self.assertIn('RMSE: 0.003912 V', report['CLI']['stdout'])
        self.assertGreater(report['historical_files_unchanged'], 0)


if __name__ == '__main__': unittest.main()
