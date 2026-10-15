"""1217 parameter independent MLP. No TFT imports or checkpoint loading."""
import torch
from torch import nn
class VoltageMLP(nn.Module):
 def __init__(self):
  super().__init__();self.layers=nn.Sequential(nn.Linear(3,32),nn.SiLU(),nn.Linear(32,32),nn.SiLU(),nn.Linear(32,1))
 def forward(self,x):return self.layers(x).squeeze(-1)
def slope_loss(model,x,y_scale,j_scale):
 z=x.detach().clone().requires_grad_(True);y=model(z);d=torch.autograd.grad(y.sum(),z,create_graph=True)[0][:,0]
 raw=d*y_scale/j_scale;positive=torch.relu(d)
 return (positive**2).mean(),raw,(torch.relu(raw)**2).mean()
def norm(grads):return float(torch.sqrt(sum(g.detach().square().sum() for g in grads if g is not None)))
