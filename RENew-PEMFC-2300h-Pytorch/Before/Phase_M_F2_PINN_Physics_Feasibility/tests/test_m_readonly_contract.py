"""Production audit guards; synthetic metadata and bytes only."""
import copy
import hashlib
import importlib.util
from pathlib import Path
import unittest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/audit_m_readonly.py"
spec = importlib.util.spec_from_file_location("m_audit", SCRIPT)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


class ReadonlyContractTests(unittest.TestCase):
    def test_allow_known_metadata(self):
        path = m.PREFIX + m.SOURCES[24]
        self.assertEqual(m.allowed_path(m.ROOT, path).name, "causal_feature_dictionary.csv")

    def test_deny_mixed_causal_table(self):
        path = m.PREFIX + "Phase_N1_Data_Foundation/outputs/causal_feature_tables/causal_features_30min.csv"
        with self.assertRaisesRegex(RuntimeError, "SOURCE_NOT_ALLOWED"):
            m.allowed_path(m.ROOT, path)

    def test_even_allowlist_cannot_authorize_full_causal_read(self):
        path = m.PREFIX + "causal_features_30min.csv"
        with self.assertRaisesRegex(RuntimeError, "SOURCE_NOT_ALLOWED"):
            m.allowed_path(m.ROOT, path, allowed={path})

    def test_deny_raw(self):
        with self.assertRaises(RuntimeError):
            m.allowed_path(m.ROOT, "data/Loadcycle_data/Loadcycle_0-14.csv")

    def test_deny_test_targets(self):
        with self.assertRaises(RuntimeError):
            m.allowed_path(m.ROOT, "data/Polcurve_Nominal_RHFC_2000h.csv")

    def test_deny_test_predictions(self):
        with self.assertRaises(RuntimeError):
            m.allowed_path(m.ROOT, m.PREFIX + "phase/outputs/test_predictions.npz")

    def test_deny_full_rul_output(self):
        with self.assertRaises(RuntimeError):
            m.allowed_path(m.ROOT, m.PREFIX + "Phase_RUL90/outputs/full_data.csv")

    def test_deny_path_escape(self):
        with self.assertRaisesRegex(RuntimeError, "PATH_ESCAPE"):
            m.allowed_path(m.ROOT, "../outside.csv")

    def test_production_hash_pass(self):
        payload = b"synthetic metadata"
        expected = hashlib.sha256(payload).hexdigest()
        self.assertEqual(m.verify_digest(payload, expected), expected)

    def test_production_hash_tamper_fails(self):
        expected = hashlib.sha256(b"original metadata").hexdigest()
        with self.assertRaisesRegex(RuntimeError, "SOURCE_HASH_MISMATCH"):
            m.verify_digest(b"tampered metadata", expected)

    def test_full_candidate_proof_pass(self):
        m.validate_candidates(m.CANDIDATES)
        self.assertEqual(len(m.FIELDS) - 2, 14)

    def test_missing_proof_field_fails(self):
        rows = copy.deepcopy(m.CANDIDATES)
        del rows[0]["unknown_parameters"]
        with self.assertRaisesRegex(RuntimeError, "INCOMPLETE"):
            m.validate_candidates(rows)

    def test_duplicate_candidate_fails(self):
        with self.assertRaisesRegex(RuntimeError, "DUPLICATE"):
            m.validate_candidates(m.CANDIDATES + [m.CANDIDATES[0]])

    def test_no_unjustified_strict_upgrade(self):
        rows = copy.deepcopy(m.CANDIDATES)
        rows[0]["classification"] = "PHYSICS_EQUATION_ADMISSIBLE"
        with self.assertRaisesRegex(RuntimeError, "NO_FULLY_VALIDATED"):
            m.validate_candidates(rows)

    def test_unknown_grade_fails(self):
        rows = copy.deepcopy(m.CANDIDATES)
        rows[0]["classification"] = "READY_FOR_TRAINING"
        with self.assertRaisesRegex(RuntimeError, "UNKNOWN_CLASSIFICATION"):
            m.validate_candidates(rows)

    def test_checkpoint_identity_set(self):
        self.assertEqual(len(m.MODEL_PATHS), 9)
        self.assertEqual(sum("H-F2" in p for p in m.MODEL_PATHS), 3)

    def test_preferred_route_is_only_conditional(self):
        row = next(row for row in m.CANDIDATES if row["candidate_id"] == "C_HISTORICAL_AUX_ORDERING")
        self.assertEqual(row["classification"], "CONDITIONALLY_ADMISSIBLE")
        self.assertEqual(row["future_actual_required"], "NO")


if __name__ == "__main__":
    unittest.main()
