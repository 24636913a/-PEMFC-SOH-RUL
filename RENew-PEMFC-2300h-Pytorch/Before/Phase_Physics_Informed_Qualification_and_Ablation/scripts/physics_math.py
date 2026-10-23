"""Conditional equations. Fixture parameters are never RealHyFC estimates."""
import torch
R=8.31446261815324;F=96485.33212;S=.13680318714138442
def nernst(T,aH2,aO2,aH2O,E0):
 if bool(torch.any(T<=0)) or any(bool(torch.any(x<=0)) for x in [aH2,aO2,aH2O]):raise ValueError('Positive absolute temperature / activities required')
 return E0+R*T/(2*F)*torch.log(aH2*torch.sqrt(aO2)/aH2O)
def polarization(j,E,b,log_j0,ASR,B,jlim):
 if bool(torch.any(j<=0)) or bool(torch.any(j>=jlim)):raise ValueError('Tafel / limiting-current domain required')
 return E-b*(torch.log(j)-log_j0)-ASR*j+B*torch.log1p(-j/jlim)
def upper_loss(u,v0,E):return torch.relu((v0+S*u-E)/S).square().mean()
def amplitude_loss(u,v0,vphys):return ((v0+S*u-vphys)/S).square().mean()
def empty_domain_loss(u):return u.sum()*0  # fail-closed engineering gate, not a computed physical residual
def require_real_state(row):
 required=['interface_T_K','a_H2','a_O2','a_H2O','reference_phase_certified','state_certified']
 return all(k in row and row[k] is not None for k in required) and row['state_certified'] is True and row['reference_phase_certified'] is True
