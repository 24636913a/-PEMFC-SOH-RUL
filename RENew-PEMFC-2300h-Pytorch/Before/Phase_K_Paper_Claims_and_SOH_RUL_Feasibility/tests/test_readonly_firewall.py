"""Production firewall checks; no dataset or model execution."""
import importlib.util
import io
from pathlib import Path
import sys
import unittest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/audit_k_readonly.py"
spec = importlib.util.spec_from_file_location("k_readonly_audit", SCRIPT)
audit = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = audit
spec.loader.exec_module(audit)

class ReadonlyFirewallTests(unittest.TestCase):
    def setUp(self):
        self.cfg = audit.load_config()

    def test_mixed_anchor_full_hash_denied(self):
        with self.assertRaises(audit.AccessDenied):
            audit.safe_full_hash(self.cfg["bounded_anchor_source"]["path"], self.cfg)

    def test_mixed_feature_table_denied(self):
        for path in (
            "RENew-PEMFC-2300h-Pytorch/Phase_N1_Data_Foundation/outputs/causal_feature_tables/causal_features_30min.csv",
            "数据/Loadcycle_0-14.csv",
            "RENew-PEMFC-2300h-Pytorch/outputs/Test_predictions.csv",
            "RENew-PEMFC-2300h-Pytorch/Phase_RUL90_Data_Feasibility/outputs/full_data.csv",
        ):
            with self.subTest(path=path), self.assertRaises(audit.AccessDenied):
                audit.safe_full_hash(path, self.cfg)

    def test_exact_line_read_does_not_touch_suffix(self):
        h = io.BytesIO(b"head\nfirst\nDO_NOT_READ_TEST_SUFFIX")
        self.assertEqual(audit.bounded_line(h), b"head\n")
        self.assertEqual(audit.bounded_line(h), b"first\n")
        self.assertEqual(h.tell(), len(b"head\nfirst\n"))

    def test_prefix_suffix_invariance(self):
        def prefix(suffix):
            h = io.BytesIO(b"head\nfirst\n" + suffix)
            return audit.digest_bytes(audit.bounded_line(h) + audit.bounded_line(h))
        self.assertEqual(prefix(b"secretA"), prefix(b"different suffix and length"))
        self.assertNotEqual(prefix(b"secretA"),
                            audit.digest_bytes(b"head\nCHANGED_FIRST\n"))

    def test_test_or_boundary_anchor_is_denied_by_production_parser(self):
        header = b"ageing_hour,split,jstar_A_cm2,air_air_start_status,filename\n"
        rows = b"1900,test,1,nominal,a.csv\n" * 8
        with self.assertRaisesRegex(audit.AccessDenied, "NON_DEVELOPMENT"):
            audit.parse_development_anchor_prefix(header + rows, self.cfg)
        rows = b"1819.997738,train,1,nominal,a.csv\n" * 8
        with self.assertRaisesRegex(audit.AccessDenied, "BOUNDARY"):
            audit.parse_development_anchor_prefix(header + rows, self.cfg)

    def test_no_unbounded_numeric_data_reader_or_model_import(self):
        import ast
        tree = ast.parse(SCRIPT.read_text(encoding="utf-8"))
        imported = [n.names[0].name for n in ast.walk(tree) if isinstance(n, ast.Import)]
        self.assertTrue(set(imported).isdisjoint({"torch", "pandas", "numpy", "optuna"}))
        self.assertNotIn("read_csv(", SCRIPT.read_text(encoding="utf-8"))

if __name__ == "__main__":
    unittest.main()
