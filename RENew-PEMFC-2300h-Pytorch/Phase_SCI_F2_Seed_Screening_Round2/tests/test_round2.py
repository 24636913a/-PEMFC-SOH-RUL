import sys,unittest,tempfile,json
from pathlib import Path
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from screen_common import *
from reliable_io import js,atomic_replace
from prepare_round2 import inherited_worker

class Round2Tests(unittest.TestCase):
 def test_frozen_config(self):
  c=cfg();self.assertEqual(c['seeds'],list(range(201,251)));self.assertEqual(c['qualified_stop_count'],3)
 def test_inherited_training_mathematics(self):self.assertEqual((P/'scripts/train_seed.py').read_text(encoding='utf-8'),inherited_worker())
 def test_quality_boundaries(self):
  r=dict(RMSE_mV=6.3,endpoint_RMSE_mV=10.,max_abs_error_mV=50.)
  self.assertTrue(qualification(r)['qualified'])
  for k in r:
   x=dict(r);x[k]+=1e-9;self.assertFalse(qualification(x)['qualified'])
  self.assertEqual(qualification(dict(r,endpoint_RMSE_mV=10.0001))['selection_status'],'LOW_RMSE_WITH_TAIL_RISK')
 def test_arrays_exact(self):
  d=data();o=np.load(OLD/'outputs/audits/FROZEN_H_DATA.npz',allow_pickle=False)
  self.assertTrue(all(np.array_equal(d[k],o[k]) for k in d.files));self.assertEqual(d['tx'].shape,(1380,48,14));self.assertEqual(d['vx'].shape,(205,48,14))
  self.assertFalse(set(d['train_targets'].ravel())&set(d['val_targets'].ravel()))
 def test_train_only_scaler_and_mask(self):
  d=data();s=json.loads((P/'outputs/audits/SCALER.json').read_text());sc=h.e.CommonScaler().fit(d['train_x_raw'])
  self.assertTrue(np.array_equal(sc.mean,np.array(s['mean'])));self.assertTrue(np.array_equal(sc.scale,np.array(s['scale'])))
  self.assertTrue(np.array_equal(h.e.prep(d['train_x_raw'],sc,cfg()['preprocessing']['Top10_indices'],'EWMA'),d['tx']))
 def test_causal_EWMA(self):
  a=np.random.default_rng(9).normal(size=(2,48,14));b=a.copy();b[:,25:,0]+=7
  self.assertTrue(np.array_equal(h.e.ewma(a)[:,:25],h.e.ewma(b)[:,:25]))
 def test_logger_duplicate_epoch_fixed(self):
  from train_seed import Recorder
  with tempfile.TemporaryDirectory(dir=P/'outputs/audits') as t:
   rec=Recorder(Path(t),999);rec.emit('BATCH_STARTED',epoch=1,batch=0);rec.emit('LOSS_COMPUTED',loss=.1)
   self.assertEqual(json.loads((Path(t)/'latest_batch.json').read_text())['epoch'],1)
 def test_atomic_retry_and_guard(self):
  with tempfile.TemporaryDirectory(dir=P/'outputs/audits') as t:
   dest=Path(t)/'x.json';real=os.replace;attempt=[]
   def transient(a,b):
    attempt.append(1)
    if len(attempt)==1:raise PermissionError('synthetic test collision')
    return real(a,b)
   with patch('reliable_io.os.replace',side_effect=transient),patch('reliable_io.time.sleep'):
    # Test evidence is explicitly synthetic and written inside the temporary folder.
    with patch('reliable_io.P',Path(t)):js(dest,dict(test=True))
   self.assertEqual(len(attempt),2);self.assertTrue(json.loads(dest.read_text())['test'])
  with self.assertRaises(AssertionError):js('../HISTORICAL_WRITE_PROHIBITED.json',{})
 def test_historical_replay(self):
  a=json.loads((P/'outputs/audits/HISTORICAL_REPLAY.json').read_text());self.assertEqual(a['status'],'PASS');self.assertEqual(a['optimizer_steps'],0)
  table=rd(P/'outputs/HISTORICAL_THREE_CANDIDATES.csv');self.assertEqual(table.seed.tolist(),[42,103,193]);self.assertTrue(table.same_GPU_replay_exact.all())
 def test_independent_metric_math(self):
  from audit_round2 import independent
  for row in rd(P/'outputs/HISTORICAL_THREE_CANDIDATES.csv').to_dict('records'):
   result=independent(rd(P/row['predictions']))
   self.assertLess(max(abs(result[k]-row[k]) for k in result),1e-10)
 def test_no_historical_changes(self):
  from audit_round2 import protect_round1
  self.assertGreater(protect_round1(),100)
 def controller_fixture(self,budget_expired=False):
  """Ephemeral fake IO/processes test scheduling, never a model or published result."""
  import run_round2 as controller
  from contextlib import ExitStack,redirect_stdout
  from types import SimpleNamespace
  import io
  launched=[];events=[];config=cfg()
  with tempfile.TemporaryDirectory(dir=P/'outputs/audits',prefix='UNIT_TEST_ONLY_') as folder,ExitStack() as stack:
   root=Path(folder)
   def write(name,obj):
    f=root/name;f.parent.mkdir(parents=True,exist_ok=True);f.write_text(json.dumps(obj),encoding='utf-8')
   class Process:
    def __init__(self,args,**kw):
     seed=int(args[-1]);launched.append(seed)
     # 201 fails A; 202/203/204 pass both; 205 must never start.
     write(f'outputs/metrics/seed_{seed}.json',dict(status='COMPLETED',seed=seed,RMSE_mV=9. if seed==201 else 6.,endpoint_RMSE_mV=8.,max_abs_error_mV=40.,best_epoch=10,stop_epoch=50,training_time_s=1.,checkpoint='UNIT_TEST_ONLY.pt',checkpoint_sha256='unit',predictions='UNIT_TEST_ONLY.csv',prediction_sha256='unit'))
     self.returncode=0
    def poll(self):return 0
   for name,value in [('P',root),('cfg',lambda:config),('js',write),('sha',lambda path:'unit'),('append_event',events.append),('ranking',lambda:None)]:stack.enter_context(patch.object(controller,name,value))
   stack.enter_context(patch.object(controller.subprocess,'Popen',Process))
   stack.enter_context(patch.object(controller.subprocess,'run',return_value=SimpleNamespace(returncode=0)))
   stack.enter_context(patch.object(controller.ctypes.windll.kernel32,'SetThreadExecutionState',return_value=1))
   if budget_expired:
    ticks=iter([0,0]);stack.enter_context(patch.object(controller.time,'monotonic',side_effect=lambda:next(ticks,43200.)))
   with redirect_stdout(io.StringIO()):controller.main()
   state=json.loads((root/'outputs/execution_ledger/PROGRESS.json').read_text())
  return state,launched
 def test_controller_stops_after_third_qualified(self):
  state,launched=self.controller_fixture()
  self.assertEqual(launched,[201,202,203,204]);self.assertEqual(state['qualified_seeds'],[202,203,204]);self.assertEqual(state['status'],'ROUND2_THREE_NEW_SEEDS_FOUND')
  self.assertTrue(all(state['seeds'][str(s)]['status']=='NOT_STARTED' for s in range(205,251)))
 def test_controller_budget_does_not_launch(self):
  state,launched=self.controller_fixture(budget_expired=True)
  self.assertEqual(launched,[]);self.assertEqual(state['status'],'ROUND2_INSUFFICIENT_QUALIFIED_SEEDS')
  self.assertTrue(all(r['status']=='NOT_STARTED' for r in state['seeds'].values()))
 def test_interruption_is_preserved_without_partial_publication(self):
  import finalize,io
  from contextlib import redirect_stdout
  state=dict(status='ROUND2_INTERRUPTED_OR_BLOCKED',seeds={'201':dict(status='TIMEOUT')},completed_seeds=0,qualified_seeds=[],start_UTC='2000-01-01T00:00:00Z',stop_reason='UNIT_TEST_ONLY')
  real=Path.read_text;captured={}
  def read(path,*a,**kw):
   if path==P/'outputs/execution_ledger/PROGRESS.json':return json.dumps(state)
   return real(path,*a,**kw)
  with patch.object(Path,'read_text',read),patch.object(finalize,'js',lambda name,obj:captured.update({name:obj})),patch.object(finalize.subprocess,'run',side_effect=AssertionError('must not run final audits or publish')),patch.object(finalize.subprocess,'Popen',side_effect=AssertionError('must not launch publisher')),redirect_stdout(io.StringIO()):finalize.main()
  self.assertEqual(len(captured),1);self.assertEqual(next(iter(captured.values()))['status'],'REVIEW_REQUIRED_NOT_PUBLISHED_AS_COMPLETE')

if __name__=='__main__':unittest.main()
