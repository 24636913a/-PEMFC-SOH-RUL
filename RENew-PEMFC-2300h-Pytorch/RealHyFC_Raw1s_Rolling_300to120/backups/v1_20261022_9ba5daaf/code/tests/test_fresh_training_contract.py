"""Scientific contracts for the new independent 70:10:20 training entry."""
import ast
from pathlib import Path
import sys
import types
import unittest

import numpy as np
import pandas as pd

FOLDER = Path(__file__).resolve().parents[1]
SOURCE = FOLDER / 'F2_Train_Interactive.py'
module = types.ModuleType('fresh_f2_contracts')
module.__file__ = str(SOURCE)
sys.modules[module.__name__] = module
prefix = SOURCE.read_text(encoding='utf-8').split('# %% 4. 每次Run All')[0]
exec(compile(prefix, str(SOURCE), 'exec'), module.__dict__)


class FreshTrainingContracts(unittest.TestCase):
    def test_exact_single_split(self):
        self.assertEqual(module.partitions(3843), {'TRAIN': (0, 2690), 'VALIDATION': (2690, 3074), 'TEST': (3074, 3843)})

    def test_original_model_definitions(self):
        self.assertTrue(module.verify_network_definitions()['original_F2_classes_exact'])
        model = module.RegimeAwareDualPathTFT(hidden_size=64, hidden_continuous_size=32,
            attention_head_size=4, lstm_layers=1, dropout=.1)
        self.assertEqual(sum(p.numel() for p in model.parameters() if p.requires_grad), 332436)

    def test_feature_difference_only_channel_five(self):
        x = np.random.default_rng(1).normal(size=(8, 48, 14)).astype('float32')
        scaler = module.TrainScaler().fit(x)
        ten = module.masked_inputs(x, scaler, 'top10')
        eleven = module.masked_inputs(x, scaler, 'top11')
        others = [i for i in range(14) if i != 5]
        np.testing.assert_array_equal(ten[:, :, others], eleven[:, :, others])
        self.assertTrue(np.all(ten[:, :, 5] == 0))
        self.assertGreater(np.count_nonzero(eleven[:, :, 5]), 0)
        self.assertEqual(module.MASKS['top11'], module.MASKS['top10'] + [5])

    def test_no_future_features_in_decoder(self):
        import torch
        x = torch.randn(3, 48, 14)
        decoder = module.CanonicalStrictHistoryTFT.build_input(x)['decoder_cont']
        self.assertTrue(bool((decoder[:, :, :14] == 0).all()))
        self.assertTrue(torch.equal(decoder[0, :, -1], torch.arange(1, 25) / 24))

    def test_causal_ewma(self):
        x = np.ones((1, 48, 14), np.float32); altered = x.copy(); altered[:, 20:, 0] = 9
        np.testing.assert_array_equal(module.causal_ewma(x)[:, :20], module.causal_ewma(altered)[:, :20])

    def test_scaler_fits_train_only(self):
        x = np.arange(8 * 48 * 14, dtype=np.float32).reshape(8, 48, 14)
        scaler = module.TrainScaler().fit(x)
        expected = x.astype('float64').reshape(-1, 14)
        np.testing.assert_array_equal(scaler.mean, expected.mean(0))
        np.testing.assert_array_equal(scaler.scale, expected.std(0))

    def test_window_boundaries_and_causality(self):
        frame = pd.DataFrame({name: np.linspace(.5, .8, 220) for name in module.FEATURES})
        frame['bin_end_ophrs2_h'] = np.arange(220) * .5
        frame['bin_valid_for_windows'] = 1; frame['temporal_segment_id'] = 1
        before, _ = module.build_windows(frame, 100, 200)
        frame.loc[101:, 'actual_current_A_mean'] += 100
        after, _ = module.build_windows(frame, 100, 200)
        np.testing.assert_array_equal(before.x_raw[0], after.x_raw[0])
        self.assertEqual(before.origins[0], 100)
        self.assertEqual(before.targets.max(), 199)

    def test_no_old_python_import_or_optuna(self):
        tree = ast.parse(SOURCE.read_text(encoding='utf-8'))
        imported = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import): imported.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom): imported.append(node.module)
        self.assertFalse(any('RealHyFC' in name or 'optuna' in name or 'importlib' in name for name in imported))
        calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                 and node.func.attr == 'load']
        torch_loads = [node for node in calls if isinstance(node.func.value, ast.Name) and node.func.value.id == 'torch']
        self.assertEqual(len(torch_loads), 1)  # this run's own best checkpoint for final inference only

    def test_real_training_operations_present(self):
        tree = ast.parse(SOURCE.read_text(encoding='utf-8'))
        train = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'train_fresh')
        calls = {node.func.attr for node in ast.walk(train) if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)}
        self.assertIn('backward', calls); self.assertIn('step', calls)
        self.assertNotIn('load', calls); self.assertNotIn('load_state_dict', calls)


if __name__ == '__main__': unittest.main()
