import unittest,sys
from pathlib import Path
import torch
P=Path(__file__).resolve().parents[1];sys.path.insert(0,str(P/'scripts'))
from physics_math import nernst,polarization,upper_loss,amplitude_loss,empty_domain_loss,require_real_state,S
class Physics(unittest.TestCase):
 def test_nernst_units_and_standard_state(self):
  a=torch.tensor(1.,dtype=torch.float64);T=torch.tensor(348.15,dtype=torch.float64)
  self.assertEqual(float(nernst(T,a,a,a,1.18)),1.18)
 def test_no_invalid_activities(self):
  a=torch.tensor(1.);z=torch.tensor(0.)
  with self.assertRaises(ValueError):nernst(a,z,a,a,1.)
 def test_nernst_gradcheck(self):
  v=[torch.tensor(x,dtype=torch.float64,requires_grad=True) for x in [348.15,1.5,.2,.5]]
  self.assertTrue(torch.autograd.gradcheck(lambda *x:nernst(*x,1.18),tuple(v)))
 def test_loss_gradcheck(self):
  u=torch.tensor([0.,4.],dtype=torch.float64,requires_grad=True)
  self.assertTrue(torch.autograd.gradcheck(lambda x:upper_loss(x,.7,1.18),(u,)))
  self.assertTrue(torch.autograd.gradcheck(lambda x:amplitude_loss(x,.7,torch.tensor([.65,.6],dtype=torch.float64)),(u,)))
 def test_polarization_domain(self):
  for j in [0.,2.,3.]:
   with self.assertRaises(ValueError):polarization(torch.tensor(j),1.18,.035,-7.,.12,.015,2.)
 def test_E_j0_nonidentifiability(self):
  j=torch.tensor([.2,.8,1.5],dtype=torch.float64)
  first=polarization(j,1.18,.035,-7.,.12,.015,2.)
  second=polarization(j,1.18-.035*.7,.035,-7.+.7,.12,.015,2.)
  self.assertTrue(torch.allclose(first,second,atol=1e-15,rtol=0))
 def test_upper_envelope_degenerate_constant_solution(self):
  u=torch.full((24,),(-.7)/S,dtype=torch.float64)
  self.assertEqual(float(upper_loss(u,.7,1.18)),0.)  # Low constant satisfies upper-only loss; not sufficient performance learning.
 def test_empty_domain_has_zero_contribution(self):
  u=torch.tensor([1.,2.],requires_grad=True);l=empty_domain_loss(u);l.backward()
  self.assertTrue(torch.equal(u.grad,torch.zeros_like(u)))
 def test_history_finite_does_not_certify_interface(self):
  self.assertFalse(require_real_state({'actual_anode_inlet_temperature_C':75.,'actual_anode_inlet_pressure_kPag':160.,'actual_anode_inlet_RH_pct':50.}))
if __name__=='__main__':unittest.main()
