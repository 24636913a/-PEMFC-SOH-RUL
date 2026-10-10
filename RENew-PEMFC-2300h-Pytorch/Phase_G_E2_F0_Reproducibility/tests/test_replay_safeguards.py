import copy
import sys
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import g_common as g
from run_controlled_replay import TracedDataset

class Safeguards(unittest.TestCase):
    def test_full_n1_hash_rejected(self):
        with self.assertRaisesRegex(RuntimeError,'FULL_N1'):g.file_hash(g.PROJECT/'causal_features_30min.csv')
    def test_historical_write_rejected(self):
        with self.assertRaisesRegex(RuntimeError,'HISTORICAL_WRITE'):g.write_json(g.E/'should_not_exist.json',{})
    def test_trace_preserves_arity_rng_and_sampler(self):
        x=torch.zeros(10,2);y=torch.ones(10,1)
        plain=torch.utils.data.TensorDataset(x,y);trace=TracedDataset(x,y)
        def batches(ds):
            return list(torch.utils.data.DataLoader(ds,batch_size=3,shuffle=True,generator=torch.Generator().manual_seed(42)))
        a,b=batches(plain),batches(trace)
        self.assertTrue(all(torch.equal(u,v) for ab,bb in zip(a,b) for u,v in zip(ab,bb)))
        self.assertEqual(len(trace.indices),10)
        self.assertEqual(len(trace[0]),2)
    def test_hash_no_rng_mutation(self):
        torch.manual_seed(42);before=torch.get_rng_state().clone()
        g.tensor_tree_hash({'x':torch.zeros(3),'nested':[2,1]})
        self.assertTrue(torch.equal(before,torch.get_rng_state()))
    def test_production_comparison_first_divergence(self):
        order=['origins','input_sha256','label_sha256','parameters_before','rng_before','optimizer_before',
               'forward_sha256','rng_after_forward','loss','gradient_before_clip','gradient_norm',
               'gradient_after_clip','rng_after_backward','parameters_after','optimizer_after','rng_after_step']
        a=[dict(epoch=1,batch=i,**{k:'same' for k in order}) for i in range(3)]
        b=copy.deepcopy(a);self.assertTrue(g.compare_batches(a,b)['identical'])
        b[1]['gradient_before_clip']='changed'
        r=g.compare_batches(a,b);self.assertEqual(r['first_divergence']['batch'],1)
        self.assertEqual(r['first_divergence']['field'],'gradient_before_clip')
    def test_common_loader_bounded(self):
        e=g.import_e();frame=__import__('pandas').DataFrame({'bin_end_ophrs2_h':[1.0]*3005+[2000.0]*2})
        with patch.object(e.c.pd,'read_csv',return_value=frame) as reader:
            dev=e.c.load_development({});self.assertEqual(len(dev),3005)
            reader.assert_called_once_with(e.c.N1,nrows=3007)
    def test_no_assessment_builder_calls(self):
        from run_controlled_replay import prepare
        import inspect
        src=inspect.getsource(prepare)
        self.assertIn('0,2103',src);self.assertIn('2103,2404',src)
        self.assertNotIn('2404,3005',src)

if __name__=='__main__':unittest.main()
