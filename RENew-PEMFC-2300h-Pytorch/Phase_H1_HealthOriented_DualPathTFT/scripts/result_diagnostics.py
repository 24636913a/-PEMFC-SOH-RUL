"""Global tails, bias, real gate behavior and recovery errors; no optimization."""
from common import *
from analyze_results import metric
sys.path.insert(0,str(P/'models'))
from event_tft import create_model
import torch
def main():
 d=rd(P/'outputs/ALL_POINT_PREDICTIONS.csv');rows=[];recover=[];conditions=rd(P/'outputs/VALIDATION_TARGET_MEASUREMENT_CONDITIONS.csv')
 for model,g in d.groupby('model'):
  r=dict(model=model,scope='Pooled diagnostic, NOT equal-fold primary',**metric(g),signed_bias_mV=float(1000*(g.predicted_voltage_V-g.true_voltage_V).mean()),
   distinct_physical_target_cycles=g.target_raw_cycle_id.nunique(),seed_count=g.seed.nunique())
  rows.append(r)
  for label,mask in [('Recovered',g.true_voltage_V.gt(g.origin_voltage_V)),('Non_recovered',g.true_voltage_V.le(g.origin_voltage_V))]:
   z=g.loc[mask]
   if len(z):recover.append(dict(model=model,actual_change_group=label,**metric(z)))
 cs('outputs/POOLED_GLOBAL_TAIL_AND_BIAS.csv',pd.DataFrame(rows));cs('outputs/RECOVERY_STAGE_ERRORS.csv',pd.DataFrame(recover))
 cfg=load_cfg();pre=json.loads((P/'data_contracts/TRAIN_ONLY_PREPROCESSING.json').read_text());raw=np.load(P/'data_contracts/RAW_DYNAMIC_HISTORIES.npz')['x'];gates=[]
 torch.set_num_threads(1)
 for file in sorted((P/'outputs/predictions').glob('*_F2_seed*.csv')):
  p=rd(file);fold=p.fold.iloc[0];m=create_model('F2',cfg['hyperparameters'],torch.device('cpu'));m.load_state_dict(torch.load(P/f'outputs/checkpoints/{file.stem}_final.pt',map_location='cpu',weights_only=True)['state_dict']);m.eval()
  x=prep(raw[p.array_index],pre[fold]['outer']);inp=m.event_input(torch.from_numpy(x))
  with torch.no_grad():values=m.regime_weight(inp['encoder_cont']).numpy().reshape(-1)
  for r,v in zip(p.itertuples(),values):gates.append(dict(run_id=file.stem,fold=fold,seed=int(r.seed),sample_key=r.sample_key,LSTM_weight=float(v),CNN_weight=float(1-v),
   active_gate_raw_features='|'.join(cfg['raw_feature_order'][j] for j in m.regime_indices if j in pre[fold]['outer']['top10'])))
 cs('outputs/REAL_F2_GATE_VALUES.csv',pd.DataFrame(gates));summary=pd.DataFrame(gates).groupby(['fold','seed']).LSTM_weight.agg(['min','mean','max']).reset_index();cs('outputs/REAL_F2_GATE_SUMMARY.csv',summary)
 print('Global worst tails, bias, recovery groups and learnedgate saved')
if __name__=='__main__':main()
