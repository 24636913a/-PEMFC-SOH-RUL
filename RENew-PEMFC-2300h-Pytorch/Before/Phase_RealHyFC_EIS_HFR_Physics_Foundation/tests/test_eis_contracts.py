import unittest,sys,json
from pathlib import Path
import numpy as np
P=Path(__file__).resolve().parents[1];sys.path.insert(0,str(P/'scripts'))
from eis_math import parse,split_scans,crosses,circuit
from common import safe_raw,REPO
def fixture(fmt,rows):
 return 'File.......: test - JAN,08.2025\nCurrent....:50A, Ampl:2.5A\nTime.......:12:00:00 - 12:10:00\n(area/cm^2):280\nImpedance samples\nLines: '+str(len(rows))+'\nColumns:7\nNumber Frequency/Hz '+fmt+' Significance Time/s Voltage/V Current/A\n'+'\n'.join(' '.join(map(str,r)) for r in rows)
class EIS(unittest.TestCase):
 def test_LFS_pointer_not_data(self):
  with self.assertRaises(ValueError):parse('version https://git-lfs.github.com/spec/v1\noid sha256:abc\nsize1')
 def test_magnitude_phase_degrees(self):
  t=fixture('impedance/Ohm Phase/deg',[[1,1000,.001,-45,1,0,-.6,50]])
  z=parse(t)[2];self.assertAlmostEqual(z[0].real,.001/np.sqrt(2));self.assertAlmostEqual(z[0].imag,-.001/np.sqrt(2))
 def test_rectangular_sign_never_flipped(self):
  t=fixture('impedance_R/Ohm impedance_I/Ohm',[[1,1000,-.001,.002,1,0,-.6,50]])
  self.assertEqual(parse(t)[2][0],-.001+.002j)
 def test_unknown_columns_stop(self):
  with self.assertRaises(ValueError):parse(fixture('unknown_R unknown_I',[[1,1000,1,2,1,0,-.6,50]]))
 def test_source_line_count_checked(self):
  t=fixture('impedance/Ohm Phase/deg',[[1,1000,.001,0,1,0,-.6,50]]).replace('Lines: 1','Lines: 2')
  with self.assertRaises(AssertionError):parse(t)
 def test_frequency_revisit_is_not_average(self):
  f=np.array([1000,2000,5000,2000,1000]);a,b=split_scans(f)
  self.assertEqual(a.tolist(),[0,1,2]);self.assertEqual(b.tolist(),[2,3,4])
 def test_highest_frequency_not_intercept(self):
  f=np.array([1000,2000,5000,2000,1000]);z=np.array([.001-.001j,.002-.001j,.004+.001j,.002-.001j,.001-.001j])
  values=crosses(f,z,np.ones(5));self.assertAlmostEqual(values[0]['intercept_ohm'],.003);self.assertNotEqual(values[0]['intercept_ohm'],z[2].real)
  self.assertIn('UNCERTIFIED',values[0]['status'])
 def test_exact_allowlist_excludes_Test(self):
  with self.assertRaises(PermissionError):safe_raw(REPO/'数据/《耐久性测试期间PEM燃料电池堆的空间解析电化学表征》的基础数据/Polcurve_Nominal_RHFC_2300h.csv')
 def test_resistance_units_and_circuit_limit(self):
  p=np.array([np.log(.001),np.log(.002),np.log(.01),1,np.log(.003),np.log(.1),1])
  self.assertAlmostEqual(circuit(np.array([1e12]),p)[0].real,.001,places=10);self.assertAlmostEqual(.04*1.8,.072)
 def test_real_results_are_not_fake_HFR(self):
  d=json.loads((P/'outputs/RESEARCH_DECISION.json').read_text());self.assertEqual(d['core_spectra'],65);self.assertEqual(d['certified_ASR_values'],0)
  self.assertIsNone(d['actual_Train_physics_loss']);self.assertEqual(d['GPU_runs'],0)
if __name__=='__main__':unittest.main()
