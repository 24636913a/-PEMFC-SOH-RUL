import io
import sys
import unittest
from pathlib import Path
from unittest.mock import patch
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from m2_access import (AccessFailure,LimitedBytes,lookup_permit,validate_permit,
                        validate_chronology,permitted_path,sha_bytes,verify_registered_bytes)

class Barrier(io.BytesIO):
    def __init__(self,prefix,suffix=b'FORBIDDEN'):
        super().__init__(prefix+suffix); self.end=len(prefix)
    def read(self,n=-1):
        if n<0 or self.tell()+n>self.end:
            raise AssertionError('PHYSICAL_BYTE_OVERREAD')
        return super().read(n)

class AccessTests(unittest.TestCase):
    def permit(self):
        return dict(preauthorized_domain='ENTIRE_SOURCE_PROVEN_TRAIN',byte_start=0,byte_end=100,
          historical_age_min_h=72.78869,historical_age_max_h=206.03581,
          historical_source_sha256='a'*64,historical_record_count=100,
          n1_rcr_range_and_count_agree=True,historical_manifest_committed=True)
    def test_stop_before_forbidden_byte(self):
        source=Barrier(b'a,b\n1,2\n'); b=LimitedBytes(source,8)
        self.assertEqual(b.read(1000),b'a,b\n1,2\n')
        self.assertEqual(b.read(1000),b''); self.assertEqual(source.tell(),8)
    def test_suffix_mutation_invariant(self):
        a=LimitedBytes(Barrier(b'abc',b'AAA'),3).read()
        b=LimitedBytes(Barrier(b'abc',b'BBBBBBBBB'),3).read()
        self.assertEqual(sha_bytes(a),sha_bytes(b))
    def test_prefix_mutation_sensitive(self):
        self.assertNotEqual(sha_bytes(b'abc'),sha_bytes(b'abd'))
    def test_bom_quotes_embedded_newline_parser(self):
        prefix=b'\xef\xbb\xbfa;b\r\n1;"q\nq"\r\n2;"a""b"\r\n'
        s=LimitedBytes(Barrier(prefix),len(prefix))
        f=pd.read_csv(s,sep=';',encoding='utf-8-sig')
        self.assertEqual(list(f.columns),['a','b'])
        self.assertEqual(f.b.tolist(),['q\nq','a"b'])
        self.assertEqual(s.count,len(prefix))
    def test_truncation_fails(self):
        with self.assertRaisesRegex(AccessFailure,'TRUNCATED'):
            LimitedBytes(io.BytesIO(b'abc'),4).read()
    def test_zero_or_negative_boundary_rejected(self):
        for n in [0,-1]:
            with self.assertRaises(AccessFailure): LimitedBytes(io.BytesIO(),n)
    def test_mixed_or_nontrain_scope_denied(self):
        row=self.permit(); row['preauthorized_domain']='MIXED'
        with self.assertRaisesRegex(AccessFailure,'SCOPE'): validate_permit(row)
    def test_time_boundary_rejected(self):
        row=self.permit(); row['historical_age_max_h']=1330.22
        with self.assertRaisesRegex(AccessFailure,'NONTRAIN'): validate_permit(row)
    def test_missing_hash_denied(self):
        row=self.permit(); del row['historical_source_sha256']
        with self.assertRaisesRegex(AccessFailure,'HASH'): validate_permit(row)
    def test_missing_exact_byte_denied(self):
        row=self.permit(); row['byte_end']=None
        with self.assertRaisesRegex(AccessFailure,'BYTE'): validate_permit(row)
    def test_missing_range_provenance_denied(self):
        row=self.permit(); row['n1_rcr_range_and_count_agree']=False
        with self.assertRaisesRegex(AccessFailure,'AUDITS'): validate_permit(row)
    def test_unallowlisted_source_denied_before_open(self):
        with patch.object(Path,'open',side_effect=AssertionError('MUST_NOT_OPEN')):
            for name in ['Loadcycle_101-112.csv','Test.csv','2000.csv','2300.csv']:
                with self.assertRaises(AccessFailure): permitted_path(Path('.'),name,{})
    def test_wrong_age_order_rejected(self):
        with self.assertRaisesRegex(AccessFailure,'AGE_REVERSAL'):
            validate_chronology([100.1,100.],pd.Series(['2025-01-01','2025-01-02']))
    def test_wrong_timestamp_order_rejected(self):
        with self.assertRaisesRegex(AccessFailure,'DATETIME'):
            validate_chronology([100.,100.1],pd.Series(['2025-01-02','2025-01-01']))
    def test_nontrain_record_rejected(self):
        with self.assertRaisesRegex(AccessFailure,'TIME_BREACH'):
            validate_chronology([1310.],pd.Series(['2025-01-01']))
    def test_plateau_reported_not_called_strict_age(self):
        n,*_=validate_chronology([100.,100.],pd.Series(['2025-01-01','2025-01-02']))
        self.assertEqual(n,1)
    def test_chunk_reversal_rejected(self):
        with self.assertRaises(AccessFailure):
            validate_chronology([99.],pd.Series(['2025-01-02']),100.,pd.Timestamp('2025-01-01'))
    def test_identity_hash_mismatch_production(self):
        import tempfile
        base=Path(__file__).resolve().parents[1]/'runtime_synthetic'
        base.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=base) as tmp:
            root=Path(tmp); data=b'a;b\n1;2\n'; (root/'fake.csv').write_bytes(data)
            row=self.permit(); row['byte_end']=len(data); row['historical_source_sha256']=sha_bytes(data)
            self.assertEqual(verify_registered_bytes(root,'fake.csv',{'fake.csv':row})['identity'],'PASS')
            row['historical_source_sha256']='f'*64
            with self.assertRaisesRegex(AccessFailure,'CONTENT_IDENTITY'):
                verify_registered_bytes(root,'fake.csv',{'fake.csv':row})
    def test_invalid_quotes_fail_without_suffix_read(self):
        source=b'a;b\n1;"unclosed\n'
        bounded=LimitedBytes(Barrier(source),len(source))
        with self.assertRaises(pd.errors.ParserError): pd.read_csv(bounded,sep=';')

if __name__=='__main__': unittest.main(verbosity=2)
