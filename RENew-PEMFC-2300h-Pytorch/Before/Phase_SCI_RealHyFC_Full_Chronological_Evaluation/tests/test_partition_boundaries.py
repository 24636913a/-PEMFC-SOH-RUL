import unittest,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from build_chronological_partitions import partitions
class PartitionTests(unittest.TestCase):
    def test_exact_coverage(self):
        p=partitions(3843);self.assertEqual(list(p.values()),[(0,2103),(2103,2404),(2404,3005),(3005,3843)])
        flat=[i for a,b in p.values() for i in range(a,b)]
        self.assertEqual(flat,list(range(3843)))
    def test_actual_N_not_hardcoded(self):self.assertEqual(partitions(3999)['LATE_LIFE'],(3005,3999))
    def test_invalid_N(self):
        with self.assertRaises(AssertionError):partitions(3005)
