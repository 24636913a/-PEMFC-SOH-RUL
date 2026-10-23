import sys,json,unittest,tempfile
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pandas as pd
P=Path(__file__).resolve().parents[1];sys.path.insert(0,str(P/'scripts'))
from screen_common import cfg,data,h,sha,rd,SEEDS
from repair_identity import verify_repair
from train_seed_v2 import Recorder
class ProtocolTests(unittest.TestCase):
 def test_original_and_repair_frozen(self):self.assertEqual(cfg()['seeds'],SEEDS);verify_repair()
 def test_exact_H_tensors_and_origins(self):
  d=data();fp=json.loads((P/'outputs/audits/DATA_FINGERPRINTS.json').read_text())
  self.assertEqual(fp,json.loads((h.PHASE/'outputs/training_logs/H-F2-Seed42/input_audit.json').read_text()))
  for arr,key in [('tx','train_preprocessed_x_sha256'),('vx','validation_x_sha256'),('train_y','train_y_sha256'),('val_y','validation_y_sha256'),('train_origins','train_origins_sha256'),('val_origins','validation_origins_sha256'),('train_targets','train_target_rows_sha256'),('val_targets','validation_target_rows_sha256')]:self.assertEqual(h.array_hash(d[arr]),fp[key])
  self.assertEqual(d['tx'].shape,(1380,48,14));self.assertEqual(d['vx'].shape,(205,48,14))
  self.assertEqual(d['val_y'].shape,(205,24));self.assertFalse(set(d['train_targets'].ravel())&set(d['val_targets'].ravel()))
  self.assertTrue(np.array_equal(d['val_origin_raw_voltage_V'].astype('float32'),d['val_x_raw'][:,-1,0]))
 def test_train_only_scaler_exact_and_mask(self):
  d=data();s=json.loads((P/'outputs/audits/SCALER.json').read_text());old=json.loads((h.E/'outputs/E_COMMON_SCALER.json').read_text());flat=d['train_x_raw'].astype('float64').reshape(-1,14)
  self.assertTrue(np.array_equal(flat.mean(0),s['mean']));self.assertTrue(np.array_equal(s['mean'],old['mean']));self.assertTrue(np.array_equal(s['scale'],old['scale']))
  excluded=[i for i in range(14) if i not in cfg()['preprocessing']['Top10_indices']]
  self.assertTrue((d['tx'][:,:,excluded]==0).all());self.assertTrue((d['vx'][:,:,excluded]==0).all())
 def test_nrows3007_then_strict_filter_firewall(self):
  frame=pd.DataFrame({'bin_end_ophrs2_h':[1.]*3005+[1819.997738,1820.]})
  with patch.object(pd,'read_csv',return_value=frame) as mock:
   out=h.e.c.load_development({});self.assertEqual(len(out),3005);self.assertEqual(mock.call_args.kwargs,{'nrows':3007})
 def test_no_future_decoder_input_and_EWMA_causal(self):
  import torch
  q=h.RegimeAwareDualPathTFT.build_input(torch.zeros(2,48,14));self.assertEqual(tuple(q['decoder_cont'].shape),(2,24,15));self.assertTrue((q['decoder_cont'][:,:,:14]==0).all())
  a=np.ones((1,48,14),dtype='float32');b=a.copy();b[:,25:,0]+=8
  self.assertTrue(np.array_equal(h.e.ewma(a)[:,:25],h.e.ewma(b)[:,:25]))
 def test_repaired_logger_same_math_no_101_retry(self):
  old=(P/'scripts/train_seed.py').read_text();new=(P/'scripts/train_seed_v2.py').read_text()
  old=old.replace("dict(seed=self.seed,sequence=self.sequence,stage=stage,epoch=self.state.get('epoch'),batch=self.state.get('batch'),**payload)","{'seed':self.seed,'sequence':self.sequence,'stage':stage,'epoch':self.state.get('epoch'),'batch':self.state.get('batch'),**payload}")
  old=old.replace(" c=cfg();assert seed in c['seeds'];"," from repair_identity import verify_repair\n verify_repair()\n c=cfg();assert seed in c['seeds'];",1)
  self.assertEqual(old,new);self.assertTrue((P/'outputs/training_logs/seed_101/failure.json').exists());self.assertFalse((P/'outputs/checkpoints/F2_seed_101_best.pt').exists())
  self.assertEqual(verify_repair()['failed_seed_optimizer_steps'],0)
 def test_logger_first_batch_with_epoch_is_persisted(self):
  with tempfile.TemporaryDirectory(dir=P/'outputs/audits') as tmp:
   folder=Path(tmp);rec=Recorder(folder,999);rec.emit('BATCH_STARTED',epoch=1,batch=2);rec.emit('FORWARD_COMPLETED',output_shape=[2,24]);self.assertEqual(rec.state['epoch'],1);self.assertEqual(rec.state['batch'],2)
 def test_fixed_budget_and_only_new_seed_pool(self):
  c=cfg();self.assertEqual(c['maximum_wall_s'],28800);self.assertEqual(c['start_next_seed_reserve_s'],3600);self.assertEqual(c['per_seed_timeout_s'],3300)
  self.assertFalse(set(SEEDS)&{42,52,62});self.assertEqual(len(SEEDS),20);self.assertEqual(c['hyperparameters'],h.load_protocol()['hyperparameters'])
  import torch
  h.e.set_seed(42);model=h.model('F2',c['hyperparameters'],torch.device('cpu'))
  authority=json.loads((h.PHASE/'outputs/audits/H_INITIALIZATION_AUTHORITY.json').read_text())
  self.assertEqual(h.e.state_digest(model),authority['initialization_sha256']['42']['F2'])
 def test_independent_metric_and_overlapping_target_weights(self):
  from screen_common import metric
  from audit_seed_results import independent
  d=pd.DataFrame(dict(origin_row=[0,0,1,1],target_row=[1,2,2,3],horizon_step=[1,24,1,24],y_true_delta_V=[.1,.2,.3,.4],y_pred_delta_V=[.101,.202,.303,.404]))
  a=metric(d);b=independent(d)
  for key in b:self.assertAlmostEqual(a[key],b[key],10)
  self.assertAlmostEqual(a['RMSE_mV'],np.sqrt(7.5),10);self.assertAlmostEqual(a['target_time_equal_RMSE_mV'],np.sqrt((1+6.5+16)/3),10)
  self.assertAlmostEqual(a['endpoint_RMSE_mV'],np.sqrt(10),10);self.assertAlmostEqual(a['P95_mV'],3.85,10)
 def test_IO_repair_changes_only_persistence(self):
  from reliable_io import verify_IO_repair
  verify_IO_repair();old=(P/'scripts/train_seed_v2.py').read_text();new=(P/'scripts/train_seed_v3.py').read_text()
  old=old.replace('from screen_common import *','from screen_common import *\nfrom reliable_io import js,verify_IO_repair',1).replace(' verify_repair()\n c=cfg();',' verify_repair()\n verify_IO_repair()\n c=cfg();',1)
  self.assertEqual(old,new);self.assertTrue((P/'outputs/training_logs/seed_151/failure.json').exists());self.assertFalse((P/'outputs/metrics/seed_151.json').exists())
 def test_atomic_writer_retries_permission_error(self):
  import reliable_io,os,errno
  original=os.replace;calls=[]
  def flaky(a,b):
   calls.append((a,b))
   if len(calls)==1:raise PermissionError(errno.EACCES,'Synthetic sharing violation')
   return original(a,b)
  with tempfile.TemporaryDirectory(dir=P/'outputs/audits') as tmp:
   f=Path(tmp)/'state.json'
   with patch.object(reliable_io.os,'replace',side_effect=flaky),patch.object(reliable_io,'record_retry') as log:
    reliable_io.js(f.relative_to(P),{'synthetic':True,'epoch':3});self.assertEqual(log.call_count,1)
   self.assertEqual(len(calls),2);self.assertEqual(json.loads(f.read_text()),{'synthetic':True,'epoch':3})
 def test_partial9_snapshot_and_completed_weights_preserved(self):
  archive=P/'outputs/partial_deliveries/after_seed151_IO_failure';record=json.loads((archive/'PARTIAL_SNAPSHOT.json').read_text())
  for row in record['files']:self.assertEqual(sha(archive/row['path']),row['sha256'])
  for row in rd(archive/'outputs/F2_SEED_RANKING.csv').itertuples():
   self.assertEqual(sha(P/row.checkpoint),row.checkpoint_sha256);self.assertEqual(sha(P/row.predictions),row.prediction_sha256)
if __name__=='__main__':unittest.main()
