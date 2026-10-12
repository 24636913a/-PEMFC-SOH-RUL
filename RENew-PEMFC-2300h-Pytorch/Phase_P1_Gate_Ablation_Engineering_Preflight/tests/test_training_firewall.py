import copy
import unittest
import torch
from pathlib import Path
import p1_test_context as ctx
import p1_safety as s
import gate_ablation_models as a

class TrainingFirewall(unittest.TestCase):
    def test_real_training_entry_always_rejects(self):
        with self.assertRaises(PermissionError):
            a.request_real_training(authorization=True)
        for field,value in (('training_authorized',True),('optimizer_steps_allowed',1),('real_data_load_allowed',True)):
            p=copy.deepcopy(ctx.PROTOCOL);p[field]=value
            with self.assertRaises(PermissionError):
                s.require_preflight_only(p)

    def test_optimizer_update_cannot_execute(self):
        # The installed production guard throws before an optimizer/self exists.
        with self.assertRaisesRegex(PermissionError,'OPTIMIZER_STEP_PROHIBITED'):
            torch.optim.AdamW.step(None)

    def test_n1_assessment_raw_and_historical_weights_denied_on_actual_open(self):
        paths=[s.REPO/'RENew-PEMFC-2300h-Pytorch/Phase_N1_Data_Foundation/outputs/causal_feature_tables/causal_features_30min.csv',
               s.REPO/'RENew-PEMFC-2300h-Pytorch/Phase_I2_Revised_Assessment/outputs/Test_predictions.csv',
               s.REPO/'RENew-PEMFC-2300h-Pytorch/Phase_H_Determinism_MultiSeed_Stability/outputs/checkpoints/H-F2-Seed42_best.pt',
               Path('E:/数据/Loadcycle_151-162.csv')]
        for path in paths:
            with self.subTest(path=str(path)),self.assertRaisesRegex(PermissionError,'IO_NOT_ALLOWED'):
                path.read_bytes()

    def test_historical_checkpoint_rejected_before_deserialization(self):
        with self.assertRaisesRegex(PermissionError,'HISTORICAL_CHECKPOINT_LOAD_PROHIBITED'):
            a.reload_synthetic_checkpoint(s.REPO/'old.pt','G0',42,ctx.CONFIG,torch.device('cuda'),ctx.PROTOCOL_SHA)

    def test_legacy_source_writes_denied(self):
        path=s.REPO/(s.ROOT+s.SOURCE_NAMES[2])
        with self.assertRaisesRegex(PermissionError,'IO_NOT_ALLOWED'):
            with path.open('wb') as handle:
                handle.write(b'NEVER_EXECUTED')
