"""Artifact/stop-rule checks; these tests never launch GPU training."""
from pathlib import Path
import hashlib
import json
import tempfile
import unittest

from run_validation import compare_prefixes

HERE=Path(__file__).resolve().parent
RUN=HERE/json.loads((HERE/'LATEST_RUN.json').read_text(encoding='utf-8'))['run']


def read(path):return json.loads(path.read_text(encoding='utf-8'))
def save(path,value):path.write_text(json.dumps(value),encoding='utf-8')


class StrictRuntimeValidationTests(unittest.TestCase):
    def test_bootstrap_and_identity(self):
        env=read(RUN/'strict_a/ENVIRONMENT.json')
        self.assertEqual(env['actual_workspace'],':4096:8')
        self.assertTrue(env['workspace_configured_before_torch_import'])
        self.assertTrue(env['cuda_uninitialized_after_torch_import'])
        self.assertTrue(env['deterministic_algorithms'])
        self.assertFalse(env['warn_only'])
        self.assertEqual(env['network']['trainable_parameters'],332436)
        source=(HERE/'deterministic_prefix.py').read_text(encoding='utf-8')
        self.assertLess(source.index("os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'"),source.index('import torch'))
        self.assertNotIn('m.configure_environment()',source)

    def test_old_assets_untouched(self):
        repo=HERE.parents[3]
        for name,digest in read(RUN/'HISTORY_BEFORE.json').items():
            self.assertEqual(hashlib.sha256((repo/name).read_bytes()).hexdigest(),digest,name)

    def test_unsupported_operation_and_exact_source(self):
        failure=read(RUN/'strict_a/FAILURE.json')
        self.assertEqual((failure['epoch'],failure['batch'],failure['optimizer_updates']),(1,1,0))
        self.assertEqual(failure['phase'],'backward')
        self.assertIn('upsample_linear1d_backward_out_cuda',failure['error'])
        location=read(RUN/'operator_location/OPERATOR_LOCATION.json')
        self.assertTrue(location['weights_unchanged'])
        self.assertEqual(location['optimizer_updates'],0)
        stack='\n'.join(location['anomaly_warnings'])
        for word in ['decoder_variable_selection','resample_norm','line 60','F.interpolate']:
            self.assertIn(word,stack)

    def test_stopped_and_unavailable_metrics_not_zero(self):
        plan=read(RUN/'PLAN.json');summary=read(HERE/'VALIDATION_SUMMARY.json')
        self.assertEqual(plan['status'],'STRICT_PREFIX_FAILED_STOPPED')
        self.assertFalse(plan['full_training_authorized_or_started'])
        self.assertEqual(summary['completed_short_repetitions'],0)
        self.assertIsNone(summary['training_speed_overhead_percent'])
        for name in ['strict_b','benchmark_baseline','benchmark_strict','PAIR_COMPARISON.json']:
            self.assertFalse((RUN/name).exists(),name)
        self.assertFalse(summary['safe_to_enable_in_formal_program'])

    def make_synthetic(self,root,mismatch=False):
        for label in ['strict_a','strict_b']:
            folder=root/label;folder.mkdir()
            save(folder/'INITIALIZATION.json',{'identical_synthetic_initialization':True})
            records=[dict(epoch=i//55+1,batch=i%55+1,step=i+1,gradients_before_clip='same',weights_after='same') for i in range(330)]
            if mismatch and label=='strict_b':records[17]['gradients_before_clip']='different'
            (folder/'BATCH_TRACE.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in records),encoding='utf-8')
            save(folder/'EPOCH_TRACE.json',[dict(epoch=i+1,train_standardized_MSE=.1,validation_standardized_MSE=.2,epoch_seconds=(1 if label=='strict_a' else 2)) for i in range(6)])
            save(folder/'COMPLETE.json',dict(final_state_sha256='same',final_optimizer_sha256='same'))

    def test_pair_gate_accepts_equal_states_ignores_timing_only(self):
        with tempfile.TemporaryDirectory(prefix='f2_gate_') as directory:
            root=Path(directory);self.make_synthetic(root)
            self.assertTrue(compare_prefixes(root))

    def test_pair_gate_rejects_any_gradient_difference(self):
        with tempfile.TemporaryDirectory(prefix='f2_gate_') as directory:
            root=Path(directory);self.make_synthetic(root,mismatch=True)
            self.assertFalse(compare_prefixes(root))


if __name__=='__main__':unittest.main()
