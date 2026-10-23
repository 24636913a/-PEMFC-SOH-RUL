import unittest
import copy
import p1_test_context as ctx
import gate_ablation_models as a

class RNGAndSamplerParity(unittest.TestCase):
    def test_all_seed_cpu_cuda_numpy_python_loader_dropout_parity(self):
        for seed in (42,52,62):
            r=ctx.evidence(seed)
            self.assertTrue(r['construction_order_independent'])
            self.assertEqual(r['training_start_rng_sha256']['G0'],r['training_start_rng_sha256']['G1'])
            self.assertEqual(r['training_start_rng_sha256']['G0'],r['training_start_rng_sha256']['G2'])
            self.assertTrue(r['training_start_rng_sha256']['G0']['cuda'])
            self.assertEqual(len(set(r['dropout_probe_sha256'].values())),1)

    def test_two_actual_loader_epochs_full_order_shared(self):
        for seed in (42,52,62):
            r=ctx.evidence(seed)
            self.assertEqual(r['sampler']['G0'],r['sampler']['G1'])
            self.assertEqual(r['sampler']['G0'],r['sampler']['G2'])
            for epoch in r['sampler']['G0']['epochs']:
                self.assertEqual(sorted(epoch['order']),list(range(1380)))
            self.assertNotEqual(r['sampler']['G0']['epochs'][0]['order'],r['sampler']['G0']['epochs'][1]['order'])
            self.assertEqual(a.sampler_two_epochs(seed),r['sampler']['G0'])

    def test_early_stop_exact_standardized_threshold(self):
        self.assertFalse(a.early_stop_improves(1.,1.))
        self.assertFalse(a.early_stop_improves(1.-1e-12,1.))
        self.assertTrue(a.early_stop_improves(1.-2e-12,1.))
