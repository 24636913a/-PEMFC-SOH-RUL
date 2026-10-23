import unittest
import torch
import gate_ablation_models as a
from p1_checks import check_parameter_identity
import p1_test_context as ctx

class GateParameterIdentity(unittest.TestCase):
    def test_real_three_seed_parameter_and_context_contracts(self):
        for seed in (42,52,62):
            r=ctx.evidence(seed)
            self.assertEqual(r['parameter_identity']['counts'],{'G0':332379,'G1':332436,'G2':332436})
            self.assertTrue(r['initial_gate_half'])
            self.assertTrue(r['G1_global_context_only'])
            self.assertTrue(r['G2_original_history_context'])
            self.assertTrue(all(v['finite_nonzero_core_gradient'] for v in r['gradients'].values()))
            self.assertTrue(all(v['optimizer_steps']==0 and v['state_unchanged_after_backward'] for v in r['gradients'].values()))

    def test_core_tampering_rejected_by_production_validator(self):
        cohort=a.build_cohort(42,ctx.CONFIG,torch.device('cuda'))
        with torch.no_grad():
            next(p for k,p in cohort['G1'].named_parameters() if not k.startswith('regime_gate.')).add_(1.)
        with self.assertRaisesRegex(AssertionError,'CORE_INIT_MISMATCH'):
            check_parameter_identity(cohort)

    def test_gate_requires_grad_tampering_rejected(self):
        cohort=a.build_cohort(42,ctx.CONFIG,torch.device('cuda'))
        cohort['G1'].regime_gate.output.bias.requires_grad_(False)
        with self.assertRaisesRegex(AssertionError,'PARAMETER_COUNT_MISMATCH|REQUIRES_GRAD_CHANGED'):
            check_parameter_identity(cohort)
