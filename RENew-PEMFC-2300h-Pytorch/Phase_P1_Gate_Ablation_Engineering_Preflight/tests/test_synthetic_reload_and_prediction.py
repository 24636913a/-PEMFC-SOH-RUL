import unittest
import torch
import p1_test_context as ctx
import gate_ablation_models as a

class SyntheticReloadAndPrediction(unittest.TestCase):
    def test_nine_real_synthetic_reloads_exact_and_initial_parity(self):
        for seed in (42,52,62):
            r=ctx.evidence(seed)
            for group in ('G0','G1','G2'):
                item=r['reload'][group]
                self.assertTrue(item['exact_prediction_match'])
                self.assertEqual(item['max_abs_difference'],0.)
                self.assertEqual((item['rows'],item['horizon'],item['batch_size']),(65,24,32))
            self.assertTrue(all(v<=1e-7 for v in r['initial_prediction_max_abs_difference'].values()))

    def test_wrong_checkpoint_seed_rejected_before_forward(self):
        r=ctx.evidence(42)
        path=a.PHASE/r['reload']['G1']['path']
        with self.assertRaisesRegex(RuntimeError,'IDENTITY_MISMATCH'):
            a.reload_synthetic_checkpoint(path,'G1',52,ctx.CONFIG,torch.device('cuda'),ctx.PROTOCOL_SHA)

    def test_wrong_protocol_sha_rejected_before_forward(self):
        r=ctx.evidence(42)
        path=a.PHASE/r['reload']['G0']['path']
        with self.assertRaisesRegex(RuntimeError,'IDENTITY_MISMATCH'):
            a.reload_synthetic_checkpoint(path,'G0',42,ctx.CONFIG,torch.device('cuda'),'0'*64)

    def test_predict_rejects_batch_and_numeric_mode_changes(self):
        x=a.synthetic_input()
        with self.assertRaisesRegex(RuntimeError,'CONTRACT_CHANGED'):
            a.predict_synthetic(None,x,torch.device('cuda'),64)
        with torch.inference_mode(),self.assertRaisesRegex(RuntimeError,'NUMERIC_MODE_CHANGED'):
            a.predict_synthetic(None,x,torch.device('cuda'))
        with torch.autocast('cuda'),self.assertRaisesRegex(RuntimeError,'NUMERIC_MODE_CHANGED'):
            a.predict_synthetic(None,x,torch.device('cuda'))
