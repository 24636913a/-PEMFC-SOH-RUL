"""Synthetic semantic tests supplied for USER execution; Codex did not run them."""
from pathlib import Path
import sys
import unittest
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from association_refinement.io_contract import read_json, VARIABLES
from association_refinement.features import phase_features
from association_refinement.rules import tri_any, assess


class TestTemporalSemantics(unittest.TestCase):
    def setUp(self):
        self.cfg = read_json(Path(__file__).resolve().parents[1]/'configs/PROTOCOL.json')
        self.positions = {spec[0]: i for i, spec in enumerate(VARIABLES.values())}
        self.arrays = dict(time_ns=np.arange(500, dtype=np.int64)*1_000_000_000,
                           segment=np.zeros(500, dtype=np.int32), values=np.zeros((500, 9)))

    def features(self):
        return phase_features(self.arrays, self.positions, 250, (0, 500), self.cfg,
                              dict(id=1, partition='TRAIN', anchor_row=250))

    def test_anchor_step_cannot_enter_before_or_after_steps(self):
        self.arrays['values'][250:, 0] = 20
        rows = {r['phase']: r for r in self.features() if r['variable'] == 'actual_current'}
        self.assertEqual(rows['before']['max_abs_step'], 0)
        self.assertEqual(rows['instant']['max_abs_step'], 20)
        self.assertEqual(rows['after']['max_abs_step'], 0)

    def test_after_change_is_never_a_before_change(self):
        self.arrays['values'][270:, 0] = 30
        rows = {r['phase']: r for r in self.features() if r['variable'] == 'actual_current'}
        self.assertEqual(rows['before']['max_abs_step'], 0)
        self.assertEqual(rows['instant']['max_abs_step'], 0)
        self.assertEqual(rows['after']['max_abs_step'], 30)
        self.assertEqual(rows['after']['largest_step_lag_seconds'], 20)

    def test_segment_break_cannot_create_instant_step(self):
        self.arrays['segment'][250:] = 1
        rows = [r for r in self.features() if r['phase'] == 'instant']
        self.assertTrue(all(r['support'] != 'OK' for r in rows))

    def test_missing_never_means_no_association(self):
        self.assertIsNone(tri_any([False, None]))
        self.assertIs(tri_any([True, None]), True)
        self.assertIs(tri_any([False, False]), False)

    def test_strict_cutoff_and_unavailable_background(self):
        frozen = dict(rules={'actual_current|instant|max_abs_step': dict(status='OK', cutoff=5.0)},
                      numerical_floors={'actual_current': dict(value=1.0)})
        row = dict(variable='actual_current', phase='instant', support='OK', max_abs_step=5.0)
        self.assertIs(assess(row, frozen)['background_unusual'], False)
        frozen['rules']['actual_current|instant|max_abs_step']['status'] = 'TOO_FEW_TRAIN_CONTROLS'
        self.assertIsNone(assess(row, frozen)['background_unusual'])


if __name__ == '__main__':
    unittest.main()
