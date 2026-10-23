import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("cprf_h1_test", ROOT / "scripts" / "run_cprf_h1.py")
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class H1ContractTests(unittest.TestCase):
    def test_fingerprint_is_suffix_invariant_and_prefix_sensitive(self):
        columns = MODULE.CAUSAL_COLUMNS
        data = []
        for i in range(3007):
            row = {column: 1.0 for column in columns}
            row["bin_start_ophrs2_h"] = float(i) * .5
            row["bin_end_ophrs2_h"] = float(i + 1) * .5
            row["bin_valid_for_windows"] = True
            row["temporal_segment_id"] = 1
            data.append(row)
        frame = pd.DataFrame(data)
        frame.loc[:3004, "bin_end_ophrs2_h"] = np.arange(3005) * .5 + .5
        frame.loc[3005:, "bin_end_ophrs2_h"] = MODULE.DEVELOPMENT_BOUNDARY + np.arange(2)
        first = frame.loc[frame.bin_end_ophrs2_h < MODULE.DEVELOPMENT_BOUNDARY, columns]
        digest_a, _ = MODULE.development_prefix_fingerprint(first)
        suffix_changed = frame.copy(); suffix_changed.loc[3006, "cell_voltage_mean"] = 999.0
        digest_b, _ = MODULE.development_prefix_fingerprint(suffix_changed.loc[suffix_changed.bin_end_ophrs2_h < MODULE.DEVELOPMENT_BOUNDARY, columns])
        prefix_changed = frame.copy(); prefix_changed.loc[100, "cell_voltage_mean"] = 999.0
        digest_c, _ = MODULE.development_prefix_fingerprint(prefix_changed.loc[prefix_changed.bin_end_ophrs2_h < MODULE.DEVELOPMENT_BOUNDARY, columns])
        self.assertEqual(digest_a, digest_b)
        self.assertNotEqual(digest_a, digest_c)

    def test_h1_contiguity_and_target_measurement_boundary(self):
        sample = pd.DataFrame({"origin_raw_cycle_id": [11], "target_raw_cycle_id": [12], "input_end_ophrs2_h": [200.0], "origin_p1s_measurement_end_ophrs2_h": [200.1], "target_p1s_measurement_start_ophrs2_h": [200.6]})
        self.assertTrue((sample.target_raw_cycle_id == sample.origin_raw_cycle_id + 1).all())
        self.assertTrue((sample.input_end_ophrs2_h <= sample.origin_p1s_measurement_end_ophrs2_h).all())
        self.assertTrue((sample.input_end_ophrs2_h < sample.target_p1s_measurement_start_ophrs2_h).all())

    def test_markdown_table_has_no_optional_tabulate_dependency(self):
        table = MODULE.md_table(pd.DataFrame({"a": [1.0], "b": ["x"]}), ["a", "b"])
        self.assertIn("| a | b |", table)


if __name__ == "__main__":
    unittest.main()
