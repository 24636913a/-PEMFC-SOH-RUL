"""Regression tests for the CSHSE Development-prefix firewall."""
from __future__ import annotations

import ast
import importlib.util
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


PHASE = Path(__file__).resolve().parents[1]
SCRIPT = PHASE / "scripts" / "run_cshse.py"


def load_runner():
    spec = importlib.util.spec_from_file_location("cshse_prefix_test_runner", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class PrefixFingerprintFirewallTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.runner = load_runner()

    def synthetic_prefix(self) -> pd.DataFrame:
        runner = self.runner
        n = runner.DEVELOPMENT_PREFIX_NROWS
        values: dict[str, object] = {}
        for col in runner.CAUSAL_SOURCE_COLUMNS:
            if col == "bin_start_ophrs2_h":
                values[col] = np.arange(n, dtype=float) * .5
            elif col == "bin_end_ophrs2_h":
                values[col] = np.r_[np.arange(3005, dtype=float) * .5 + .5, [1820.0, 1820.5]]
            elif col == "bin_valid_for_windows":
                values[col] = np.ones(n, dtype=bool)
            elif col == "temporal_segment_id":
                values[col] = np.ones(n, dtype=int)
            else:
                values[col] = np.linspace(.1, .2, n)
        return pd.DataFrame(values, columns=runner.CAUSAL_SOURCE_COLUMNS)

    def test_post_prefix_mutation_is_invariant(self):
        runner = self.runner
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            prefix = self.synthetic_prefix()
            a = root / "synthetic_A.csv"; b = root / "synthetic_B.csv"
            prefix.to_csv(a, index=False)
            suffix = prefix.iloc[:4].copy(); suffix["cell_voltage_mean"] += 999.0
            pd.concat([prefix, suffix], ignore_index=True).to_csv(b, index=False)
            dev_a = runner.read_causal_development_prefix(a)
            dev_b = runner.read_causal_development_prefix(b)
            self.assertTrue(dev_a.equals(dev_b))
            self.assertEqual(runner.development_prefix_fingerprint(dev_a)[0], runner.development_prefix_fingerprint(dev_b)[0])

    def test_development_mutation_changes_fingerprint(self):
        runner = self.runner
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = self.synthetic_prefix(); changed = source.copy(); changed.loc[17, "cell_voltage_mean"] += .001
            a = root / "synthetic_A.csv"; c = root / "synthetic_C.csv"
            source.to_csv(a, index=False); changed.to_csv(c, index=False)
            fp_a = runner.development_prefix_fingerprint(runner.read_causal_development_prefix(a))[0]
            fp_c = runner.development_prefix_fingerprint(runner.read_causal_development_prefix(c))[0]
            self.assertNotEqual(fp_a, fp_c)

    def test_causal_reader_is_bounded_and_hash_is_prohibited(self):
        runner = self.runner
        source = SCRIPT.read_text(encoding="utf-8")
        self.assertEqual(runner.DEVELOPMENT_PREFIX_NROWS, 3007)
        tree = ast.parse(source)
        full_hash_calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                           and node.func.id == "sha256" and node.args and isinstance(node.args[0], ast.Name) and node.args[0].id == "CAUSAL"]
        read_bytes_calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                            and node.func.attr == "read_bytes" and isinstance(node.func.value, ast.Name) and node.func.value.id == "CAUSAL"]
        self.assertFalse(full_hash_calls)
        self.assertFalse(read_bytes_calls)
        functions = {node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)}
        reader = functions["read_causal_development_prefix"]
        calls = [node for node in ast.walk(reader) if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "read_csv"]
        self.assertEqual(len(calls), 1)
        nrows = [keyword.value for keyword in calls[0].keywords if keyword.arg == "nrows"]
        self.assertEqual(len(nrows), 1)
        self.assertIsInstance(nrows[0], ast.Name)
        self.assertEqual(nrows[0].id, "DEVELOPMENT_PREFIX_NROWS")


if __name__ == "__main__":
    unittest.main()
