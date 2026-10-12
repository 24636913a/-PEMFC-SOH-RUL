import math
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from n_math_contracts import (polarization, loss_student, completed_square,
                             output_gradient, order_penalty, physical_residual,
                             require_training_qualification, student_interface,
                             crossfit_contract)
from run_n_static_audit import approved_source, verify_static_source_bytes


class SyntheticContracts(unittest.TestCase):
    def test_git_checkout_line_endings_only(self):
        result = verify_static_source_bytes(b"frozen\r\n", b"frozen\n")
        self.assertFalse(result["base_blob_byte_equal"])
        self.assertTrue(result["base_blob_LF_normalized_equal"])
        with self.assertRaises(ValueError):
            verify_static_source_bytes(b"modified\r\n", b"frozen\n")

    def test_exchange_current_intercept_gauge(self):
        for j in (0.2, 0.6, 1.0, 1.4):
            v = polarization(j, 1.1, .06, -5, .08, .02, 2.)
            for c in (-2, .5, 3):
                self.assertAlmostEqual(v, polarization(j, 1.1 - .06*c, .06, -5+c, .08, .02, 2.), places=14)

    def test_polarization_derivative(self):
        for j in (.2, .8, 1.5):
            h = 1e-6
            f = lambda x: polarization(x, 1.1, .06, -5, .08, .02, 2.)
            analytic = -.06/j - .08 - .02/(2-j)
            self.assertAlmostEqual((f(j+h)-f(j-h))/(2*h), analytic, places=8)
            self.assertLess(analytic, 0)

    def test_physical_domain(self):
        for j in (0, 2., 3.):
            with self.assertRaises(ValueError):
                polarization(j, 1.1, .06, -5, .08, .02, 2.)

    def test_teacher_loss_square_completion(self):
        for s, y, t, lam in ((.4, .6, .7, .2), (.8, .3, .9, 2), (-1, 2, 3, 0)):
            self.assertAlmostEqual(loss_student(s,y,t,lam), completed_square(s,y,t,lam), places=14)

    def test_nonzero_gradient_is_not_independent_physics(self):
        s, y, t, lam = .4, .6, .6, 3.
        self.assertAlmostEqual(loss_student(s,y,t,lam), (1+lam)*(s-y)**2)
        self.assertNotEqual(output_gradient(s,y,t,lam), 0)
        h = 1e-6
        numeric = (loss_student(s+h,y,t,lam)-loss_student(s-h,y,t,lam))/(2*h)
        self.assertAlmostEqual(numeric, output_gradient(s,y,t,lam), places=9)

    def test_constant_order_collapse(self):
        self.assertEqual(order_penalty(.65, .65), 0)

    def test_load_only_order_collapse(self):
        self.assertEqual(order_penalty(.8-.1*.2, .8-.1*1.2), 0)

    def test_input_identity_has_zero_voltage_gradient(self):
        residual = lambda predicted_v: .5 - 140/280
        self.assertEqual(residual(.9)-residual(.2), 0)

    def test_unit_adapter(self):
        scale = .13680318714138442
        self.assertAlmostEqual(physical_residual(.01/scale, .65, scale, .66), 0, places=14)
        h = 1e-6
        r = lambda u: physical_residual(u, .65, scale, .66)
        self.assertAlmostEqual((r(.2+h)-r(.2-h))/(2*h), 1, places=9)

    def test_nonlinear_aggregation_not_closed(self):
        f = lambda j: polarization(j,1.1,.06,-5,.08,.02,2.)
        self.assertGreater(abs((f(.2)+f(1.2))/2-f(.7)), .001)

    def test_future_load_ambiguity(self):
        # Same past X, two admissible future currents: no deterministic X-only oracle.
        teacher_values = [polarization(j,1.1,.06,-5,.08,.02,2.) for j in (.2,1.2)]
        self.assertNotEqual(*teacher_values)
        marginal_student = sum(teacher_values)/2
        self.assertGreater(sum((marginal_student-t)**2 for t in teacher_values),0)

    def test_student_rejects_future_actual(self):
        student_interface({"history48", "relative_horizon", "origin_anchor"})
        for extra in ("future_j", "future_temperature", "future_RH", "target_voltage", "teacher_output"):
            with self.assertRaises(ValueError):
                student_interface({"history48","relative_horizon","origin_anchor",extra})

    def test_crossfit_valid_preceding_block(self):
        self.assertTrue(crossfit_contract({"histA","targetA"},{"histB","targetB"},2103,900,972))

    def test_crossfit_shared_dwell_rejected(self):
        with self.assertRaises(ValueError):
            crossfit_contract({"raw_dwell_1"},{"raw_dwell_1"},2103,900,972)

    def test_crossfit_reverse_time_rejected(self):
        with self.assertRaises(ValueError):
            crossfit_contract({"A"},{"B"},2103,1200,1000)

    def test_crossfit_validation_boundary_rejected(self):
        with self.assertRaises(ValueError):
            crossfit_contract({"A"},{"B"},2103,2000,2103)

    def test_training_fail_closed_on_missing_or_unknown(self):
        complete = dict.fromkeys(("equation","identifiability","domain","information","isolation","independence"),True)
        require_training_qualification(complete)
        for field in complete:
            gates = complete.copy()
            gates[field] = False
            with self.assertRaises(ValueError):
                require_training_qualification(gates)
        with self.assertRaises(ValueError):
            require_training_qualification({})

    def test_whitelist_denies_timeseries_weights_and_traversal(self):
        for path in ("数据/Loadcycle.csv","Phase_N1_Data_Foundation/outputs/causal_feature_tables/causal_features_30min.csv",
                     "Phase_H_Determinism_MultiSeed_Stability/outputs/checkpoints/H-F2-Seed42_best.pt",
                     "Test/predictions.npz","../secret.md","Phase_M2_P3_TrainOnly_Raw_Dwell_Qualification/outputs/M2_RAW_DWELL_MANIFEST.csv"):
            with self.assertRaises(ValueError):
                approved_source(path, ["safe/report.md"])
        self.assertEqual(approved_source("safe/report.md",["safe/report.md"]),"safe/report.md")


if __name__ == "__main__":
    unittest.main(verbosity=2)
