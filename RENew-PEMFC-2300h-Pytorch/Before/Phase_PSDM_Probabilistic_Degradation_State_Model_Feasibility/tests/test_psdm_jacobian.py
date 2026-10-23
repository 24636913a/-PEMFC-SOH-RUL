"""Finite-difference check for the registered PSDM EKF transition Jacobian."""
from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import unittest

import numpy as np

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "run_psdm_feasibility.py"
SPEC = importlib.util.spec_from_file_location("psdm_runner", SCRIPT)
assert SPEC and SPEC.loader
MOD = importlib.util.module_from_spec(SPEC); sys.modules[SPEC.name] = MOD; SPEC.loader.exec_module(MOD)


class PSDMJacobianTests(unittest.TestCase):
    def test_analytic_jacobian_matches_central_difference(self) -> None:
        for z in (-10.0, -2.0, 0.0, 1.5, 4.0):
            x = np.array([0.99, z]); epsilon = 1e-6
            def f(value: np.ndarray) -> np.ndarray:
                return np.array([value[0] - MOD.softplus(value[1]), value[1]])
            numeric = np.column_stack([(f(x + np.eye(2)[j] * epsilon) - f(x - np.eye(2)[j] * epsilon)) / (2 * epsilon) for j in range(2)])
            analytic = np.array([[1.0, -MOD.expit(z)], [0.0, 1.0]])
            self.assertLess(float(np.max(np.abs(numeric - analytic))), 1e-6)


if __name__ == "__main__": unittest.main()
