"""Synthetic full-native-body checks before any real model optimization."""
from common import *
sys.path.insert(0,str(P/'models'))
from event_tft import create_model
import torch,inspect

PARAMS=dict(hidden_size=64,hidden_continuous_size=32,dropout=.1)
def main():
 torch.set_num_threads(1);torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
 torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=True;torch.use_deterministic_algorithms(False)
 if not torch.cuda.is_available():raise RuntimeError('Authorized GPU training requires an available GPU')
 counts={};init={};evidence=[];device=torch.device('cuda')
 for seed in [42,52,62]:
  models={}
  for group in ['F0','F1','F2']:set_seed(seed);models[group]=create_model(group,PARAMS,device)
  f0=models['F0'].state_dict();f1=models['F1'].state_dict();f2=models['F2'].state_dict()
  assert all(torch.equal(v,f1[k]) and torch.equal(v,f2[k]) for k,v in f0.items())
  assert all(torch.equal(v,f2[k]) for k,v in f1.items())
  assert all(torch.equal(models['F0'].output_layer.state_dict()[k],models[g].output_layer.state_dict()[k]) for g in ['F1','F2'] for k in models['F0'].output_layer.state_dict())
  init[str(seed)]={g:hash_state(m) for g,m in models.items()}
  if seed!=42:continue
  for group,m in models.items():
   counts[group]=sum(p.numel() for p in m.parameters() if p.requires_grad)
   hits={};handles=[]
   names=['encoder_variable_selection','lstm_encoder','lstm_decoder','static_enrichment','multihead_attn','pos_wise_ff','pre_output_gate_norm','output_layer']
   if group!='F0':names+=['history_branch']
   if group=='F2':names+=['regime_gate']
   for name in names:
    def record(mod,args,out,label=name):hits[label]=hits.get(label,0)+1
    handles.append(getattr(m,name).register_forward_hook(record))
   x=torch.randn(2,48,14,device=device);m.train();y=m.predict_voltage(x);assert y.shape==(2,1)
   inp=m.event_input(x);assert inp['decoder_cont'].shape==(2,1,15) and torch.count_nonzero(inp['decoder_cont'])==0
   assert inp['decoder_lengths'].eq(1).all()
   loss=(y-torch.tensor([[.2],[-.3]],device=device)) .square().mean();loss.backward()
   gradients={k:float(v.grad.norm()) for k,v in m.named_parameters() if v.grad is not None}
   assert gradients['output_layer.weight']>0 and any('lstm_encoder' in k and v>0 for k,v in gradients.items())
   assert all(np.isfinite(list(gradients.values())))
   if group=='F2':assert gradients['regime_gate.output.weight']>0
   if group!='F0':
    z=torch.randn(2,48,64,device=device);changed=z.clone();changed[:,25:]+=5
    m.history_branch.eval();a=m.history_branch(z);b=m.history_branch(changed);assert torch.equal(a[:,:25],b[:,:25])
   assert all(hits.get(n,0)>0 for n in names)
   opt=torch.optim.AdamW(m.parameters(),lr=.0009605449687795866);opt.step()
   m.eval()
   with torch.no_grad():expected=m.predict_voltage(x).cpu()
   folder=P/'outputs/engineering';folder.mkdir(parents=True,exist_ok=True);path=folder/f'SYNTHETIC_{group}_reload.pt';torch.save(m.state_dict(),path)
   reload=create_model(group,PARAMS,device);reload.load_state_dict(torch.load(path,map_location=device,weights_only=True));reload.eval()
   with torch.no_grad():actual=reload.predict_voltage(x).cpu()
   assert torch.equal(expected,actual)
   for h in handles:h.remove()
   evidence.append(dict(group=group,parameter_count=counts[group],output_shape=[2,1],all_native_modules_called=hits,
    gradients=gradients,synthetic_optimizer_step=True,checkpoint_reload_exact=True,no_future_decoder_variables=True))
 js('outputs/ENGINEERING_EVIDENCE.json',dict(status='PASS',groups=evidence,initialization=init,counts=counts,synthetic_only=True,
  GPU=torch.cuda.get_device_name(0),torch=str(torch.__version__),cuda=torch.version.cuda,strict_bitwise_CUDA_claimed=False))
 # Installed native TFT source identity, without modifying its source.
 from pytorch_forecasting.models.temporal_fusion_transformer import TemporalFusionTransformer,sub_modules
 paths=[Path(inspect.getsourcefile(TemporalFusionTransformer)),Path(inspect.getsourcefile(sub_modules))]
 js('configs/UPSTREAM_SOURCE_IDENTITY.json',{str(q):sha(q) for q in paths})
 assert protected_pass();print('ENGINEERING PASS',counts,flush=True)

def hash_state(model):
 h=hashlib.sha256()
 for k,v in sorted(model.state_dict().items()):h.update(k.encode());h.update(v.detach().cpu().contiguous().numpy().tobytes())
 return h.hexdigest()
if __name__=='__main__':main()
