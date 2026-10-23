import io
import sys
import unittest
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from m1_core import (AuditFailure, bounded_csv_stream, metadata_hash, N1, no_voltage_view,
                     AUX_NONVOLTAGE, RAW14, order_loss, scientific_pair_gate, validate_train_identity)

class ExactBarrier(io.BytesIO):
    def __init__(self, prefix, suffix):
        super().__init__(prefix + suffix)
        self.limit = len(prefix)
    def read(self, n=-1):
        if n != 1 or self.tell() + n > self.limit:
            raise AssertionError('FORBIDDEN_BYTE_OR_PREFETCH')
        return super().read(n)

class FirewallTests(unittest.TestCase):
    def test_exact_stop_and_quoted_newline(self):
        prefix = b'a,b\r\n1,"x\nq"\r\n2,"y""z"\r\n'
        f = ExactBarrier(prefix, b'VALIDATION_SENTINEL\n')
        header, rows, consumed = bounded_csv_stream(f, 2)
        self.assertEqual(header, ['a','b'])
        self.assertEqual(rows, [['1','x\nq'], ['2','y"z']])
        self.assertEqual(consumed, len(prefix))
        self.assertEqual(f.tell(), f.limit)
    def test_suffix_mutation_invariant(self):
        prefix = b'x\n' + b''.join(f'{i}\n'.encode() for i in range(2103))
        a = bounded_csv_stream(ExactBarrier(prefix, b'DO_NOT_READ_A\n'), 2103)
        b = bounded_csv_stream(ExactBarrier(prefix, b'B'*100000), 2103)
        self.assertEqual(a,b)
    def test_train_mutation_sensitive(self):
        a = bounded_csv_stream(io.BytesIO(b'a\n1\n'),1)
        b = bounded_csv_stream(io.BytesIO(b'a\n2\n'),1)
        self.assertNotEqual(a,b)
    def test_whole_mixed_hash_denied_even_if_allowlisted(self):
        with self.assertRaisesRegex(AuditFailure, 'FULL_HASH'):
            metadata_hash(Path('.'), N1, {N1})
    def test_unallowlisted_raw_denied(self):
        for name in ['data/Loadcycle.csv','Test.csv','Polcurve_2000h.csv','RUL90.csv']:
            with self.assertRaises(AuditFailure):
                metadata_hash(Path('.'),name,set())
    def test_short_prefix_fails(self):
        with self.assertRaises(AuditFailure):
            bounded_csv_stream(io.BytesIO(b'a\n1\n'),2)
    def test_missing_terminator_fails(self):
        with self.assertRaises(AuditFailure):
            bounded_csv_stream(io.BytesIO(b'a\n1'),1)
    def test_duplicate_header_fails(self):
        with self.assertRaises(AuditFailure):
            bounded_csv_stream(io.BytesIO(b'a,a\n1,2\n'),1)
    def test_irregular_schema_fails(self):
        with self.assertRaises(AuditFailure):
            bounded_csv_stream(io.BytesIO(b'a,b\n1\n'),1)
    def test_wrong_record_budget_fails(self):
        with self.assertRaises(AuditFailure):
            bounded_csv_stream(io.BytesIO(b''),0)
    def test_voltage_and_all_descendants_excluded(self):
        f = pd.DataFrame({c: [1.,2.] for c in RAW14})
        f['EWMA'] = [500.,600.]
        f['exported_average_cell_voltage'] = [700.,800.]
        f['U1 [V]'] = [900.,1000.]
        a = no_voltage_view(f)
        f.loc[:,RAW14[:5]+['EWMA','exported_average_cell_voltage','U1 [V]']] *= 100
        pd.testing.assert_frame_equal(a, no_voltage_view(f))
        self.assertEqual(list(a),AUX_NONVOLTAGE)
    def test_missing_aux_channel_fail_closed(self):
        with self.assertRaises(AuditFailure):
            no_voltage_view(pd.DataFrame())
    def test_constant_collapses_order_loss(self):
        self.assertEqual(float(order_loss(.7,.7)),0)
        self.assertEqual(float(order_loss(.8,.6)),0)
        self.assertGreater(float(order_loss(.6,.8)),0)
    def test_zero_candidates_not_evidence(self):
        for index in range(4):
            args = [True]*4; args[index] = False
            self.assertFalse(scientific_pair_gate(*args))
        self.assertTrue(scientific_pair_gate(True,True,True,True))
    def test_aggregate_time_order_cannot_certify_dwell(self):
        # Identical means/std/ranges, radically different longest steady dwell.
        a = np.r_[np.zeros(60),np.ones(60)]
        b = np.tile([0.,1.],60)
        np.testing.assert_allclose([a.mean(),a.std(),np.ptp(a)], [b.mean(),b.std(),np.ptp(b)])
        self.assertFalse(scientific_pair_gate(False,True,True,True))
    def test_boundary_row_count_rejected(self):
        with self.assertRaisesRegex(AuditFailure,'ROW_COUNT'):
            validate_train_identity(pd.DataFrame(),{})
    def test_train_boundary_or_order_breach_rejected(self):
        # Full expected row count does not excuse wrong time/source identity.
        f = pd.DataFrame({'bin_start_ophrs2_h':np.arange(2103)*.5+72.5})
        f['bin_end_ophrs2_h'] = f.bin_start_ophrs2_h+.5
        f['resolution_hours'] = .5
        config={'train_start_h':72.5,'train_end_h':1310.}
        with self.assertRaisesRegex(AuditFailure,'FROZEN_BOUNDARY'):
            validate_train_identity(f,config)
        f.loc[1,'bin_start_ophrs2_h'] = f.loc[0,'bin_start_ophrs2_h']
        with self.assertRaisesRegex(AuditFailure,'ORDER_OR_DUPLICATE'):
            validate_train_identity(f,config)
    def test_no_raw_or_real_data_call_in_stage0(self):
        import ast
        source=Path(__file__).resolve().parents[1]/'scripts'/'freeze_m1_protocol.py'
        tree=ast.parse(source.read_text(encoding='utf-8'))
        for node in ast.walk(tree):
            if isinstance(node,ast.Call) and isinstance(node.func,ast.Name):
                self.assertNotIn(node.func.id,['load_train','bounded_csv_stream','record_bytes'])
    def test_production_runner_does_not_import_old_loaders(self):
        import ast
        source=Path(__file__).resolve().parents[1]/'scripts'/'run_m1_qualification.py'
        tree=ast.parse(source.read_text(encoding='utf-8'))
        for node in ast.walk(tree):
            if isinstance(node,ast.ImportFrom):
                self.assertTrue(node.module in ['datetime','pathlib','m1_core'])
            if isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute):
                self.assertNotIn(node.func.attr,['read_csv','read_parquet','forward','fit','train','load'])
    def test_no_model_import_or_unbounded_csv_read(self):
        import ast
        core = Path(__file__).resolve().parents[1]/'scripts'/'m1_core.py'
        tree = ast.parse(core.read_text(encoding='utf-8'))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                self.assertFalse(any(n.name.startswith(('torch','tensorflow')) for n in node.names))
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                self.assertNotIn(node.func.attr, ['read_csv','read_parquet','read_bytes'])

if __name__ == '__main__':
    unittest.main(verbosity=2)
