from __future__ import annotations

import ast
import sys
import unittest
from pathlib import Path

PHASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PHASE / "scripts"))
import run_optuna_search as ros  # noqa: E402


class OptunaInnerFirewallTests(unittest.TestCase):
    def test_only_outer_train_is_passed_to_objective_builder(self) -> None:
        source = Path(ros.__file__).read_text(encoding="utf-8")
        tree = ast.parse(source)
        run_method = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "_run_method")
        names = {node.id for node in ast.walk(run_method) if isinstance(node, ast.Name)}
        text = ast.unparse(run_method)
        self.assertNotIn("outer_validation", names)
        self.assertIn("split_inner_fit_stopper", text)
        self.assertIn("outer_validation_accessed", text)

    def test_registered_search_space_is_exact(self) -> None:
        cfg, _ = ros.load_protocol()
        self.assertEqual(cfg["optuna_tpe"]["n_trials_per_method"], 20)
        self.assertEqual(cfg["optuna_tpe"]["search_fold"], "fold_1")
        self.assertEqual(cfg["optuna_tpe"]["methods"], ["Optuna_RawF2_TimesNet", "Optuna_EMDFeature_TimesNet"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
