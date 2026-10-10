import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import h_common as h
from crash_safe_recorder import CrashSafeRecorder
from audit_h_predictions import check_frame

class Tests(unittest.TestCase):
    def test_crash_persists_pre_backward_evidence_in_new_process(self):
        folder=h.OUT/'audits/crash_injection_synthetic';folder.mkdir(parents=True,exist_ok=True)
        code="""import sys;sys.path.insert(0,'scripts');from crash_safe_recorder import CrashSafeRecorder
r=CrashSafeRecorder(sys.argv[1],'SYNTHETIC_CRASH_INJECTION')
r.emit('BATCH_STARTED',epoch=1,batch=0,input_sha256='synthetic_x',label_sha256='synthetic_y',model_state_sha256='synthetic_model',rng={'cpu':'synthetic_rng'})
r.emit('FORWARD_COMPLETED',output_sha256='synthetic_output',output_shape=[2,24])
r.emit('LOSS_COMPUTED',loss=0.125)
r.emit('BACKWARD_STARTED')
try:raise RuntimeError('SYNTHETIC_BACKWARD_FAILURE')
except RuntimeError as exc:r.emit('BACKWARD_FAILED',exception=repr(exc));raise
"""
        p=subprocess.run([sys.executable,'-B','-c',code,str(folder)],cwd=h.PHASE,text=True,capture_output=True)
        self.assertNotEqual(p.returncode,0)
        saved=h.read_json(folder/'latest_batch.json')
        for k in ['input_sha256','label_sha256','model_state_sha256','rng','output_sha256','loss']:self.assertIn(k,saved)
        self.assertEqual(saved['stage'],'BACKWARD_FAILED');self.assertEqual(saved['loss'],.125)
        events=[json.loads(s) for s in (folder/'batch_events.jsonl').read_text().splitlines()]
        self.assertEqual([r['stage'] for r in events[-5:]],['BATCH_STARTED','FORWARD_COMPLETED','LOSS_COMPUTED','BACKWARD_STARTED','BACKWARD_FAILED'])
        h.atomic_json(folder/'INJECTION_TEST_EVIDENCE.json',{'test_pass':True,'child_returncode':p.returncode,'stderr':p.stderr,'synthetic_only':True})
    def test_repeat_gate_exact_boundary(self):
        self.assertTrue(h.repeat_gate([4.5,5.0])['pass']);self.assertFalse(h.repeat_gate([4.4,5.0])['pass'])
        self.assertFalse(h.repeat_gate([4.0,4.1,5.0])['pass'])
    def test_n1_full_hash_forbidden(self):
        with self.assertRaisesRegex(RuntimeError,'FULL_N1'):h.file_hash(h.e.c.N1)
    def test_historical_write_forbidden(self):
        with self.assertRaisesRegex(RuntimeError,'HISTORICAL_WRITE'):h.atomic_json(h.F/'should_not_exist.json',{})
    def test_production_bounded_read(self):
        frame=pd.DataFrame({'bin_end_ophrs2_h':[1.0]*3005+[2000.0]*2})
        with patch.object(h.e.c.pd,'read_csv',return_value=frame) as reader:
            self.assertEqual(len(h.e.c.load_development({})),3005);reader.assert_called_once_with(h.e.c.N1,nrows=3007)
    def test_no_assessment_builder(self):
        import inspect
        text=inspect.getsource(h.prepare);self.assertIn('0,2103',text);self.assertIn('2103,2404',text);self.assertNotIn('2404,3005',text)
    def make_frame(self):
        origins=np.repeat(np.arange(2103,2308),24);steps=np.tile(np.arange(1,25),205)
        frame=pd.DataFrame({'run_id':'r','model':'F0','seed':42,'origin_row':origins,'target_row':origins+steps,
            'horizon_step':steps,'y_true_delta_V':np.sin(origins+steps)*.01,'y_pred_delta_V':np.sin(origins+steps)*.01+.001,'protocol_sha256':'s'})
        return frame,frame[['origin_row','target_row','horizon_step','y_true_delta_V']].copy()
    def test_production_independent_metrics(self):
        f,t=self.make_frame();m=check_frame(f,t,'r','F0',42,'s');self.assertAlmostEqual(m['RMSE_mV'],1.0,places=12)
        self.assertAlmostEqual(m['MAE_mV'],1.0,places=12)
    def test_duplicate_truth_identity_fail_closed(self):
        f,t=self.make_frame()
        for mutation in ['duplicate','truth','seed','boundary']:
            g=f.copy()
            if mutation=='duplicate':g.iloc[1]=g.iloc[0]
            elif mutation=='truth':g.loc[1,'y_true_delta_V']+=.01
            elif mutation=='seed':g.loc[1,'seed']=52
            else:g.loc[1,'target_row']=2500
            with self.assertRaises(RuntimeError):check_frame(g,t,'r','F0',42,'s')

if __name__=='__main__':unittest.main()
