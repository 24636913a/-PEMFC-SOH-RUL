"""Structural and gradient contracts; zero optimizer updates, no heldout data."""
import unittest
import numpy as np
import torch
import frozen_f2 as f
import settings as s
import diagnostics as d
from residual_fusion import build_residual,latent_rms,CAP,EPS

CONFIG=dict(s.CONFIGS_A['A0'],fusion='residual',residual_cap=CAP,normalization_epsilon=EPS)


class ResidualContracts(unittest.TestCase):
    def test_same_parameters_and_only_fusion_override(self):
        f.seed_everything(202)
        a=f.RegimeAwareDualPathTFT(hidden_size=64,hidden_continuous_size=32,attention_head_size=4,lstm_layers=1,dropout=.1)
        f.seed_everything(202);b=build_residual(CONFIG)
        self.assertEqual(f.state_sha(a),f.state_sha(b))
        self.assertEqual(sum(p.numel() for p in b.parameters()),332436)
        self.assertIs(type(b).forward,f.RegimeAwareDualPathTFT.forward)
        self.assertIs(type(b).predict_delta,f.RegimeAwareDualPathTFT.predict_delta)
        self.assertEqual(set(a.state_dict()),set(b.state_dict()))

    def test_formula_bound_and_nonzero_derivatives(self):
        f.seed_everything(202);b=build_residual(CONFIG)
        hl=torch.randn(4,48,64,requires_grad=True);hc=torch.randn(4,48,64,requires_grad=True)
        history=torch.randn(4,48,14)
        out=b.fuse_encoder(hl,hc,history)
        ratio=(latent_rms(out-hl)/latent_rms(hl)).flatten()
        self.assertTrue(bool((ratio<=CAP+1e-6).all()))
        self.assertTrue(bool((ratio>0).all()))
        expected=hl+CAP*(1-b.regime_weight(history).view(-1,1,1))*latent_rms(hl)/latent_rms(hc).clamp_min(EPS)*hc
        torch.testing.assert_close(out,expected,rtol=1e-6,atol=1e-7)
        out.square().mean().backward()
        for value in [hl.grad,hc.grad]:
            self.assertTrue(bool(torch.isfinite(value).all()));self.assertGreater(float(value.norm()),0)
        for zero_lstm in [False,True]:
            x=torch.zeros_like(hl,requires_grad=True) if zero_lstm else torch.randn_like(hl,requires_grad=True)
            y=torch.zeros_like(hc,requires_grad=True)
            z=b.fuse_encoder(x,y,history);z.sum().backward()
            self.assertTrue(bool(torch.isfinite(z).all()))
            self.assertTrue(bool(torch.isfinite(x.grad).all()))
            self.assertTrue(bool(torch.isfinite(y.grad).all()))

    def test_full_network_CNN_connectivity_no_update(self):
        f.seed_everything(202);b=d.attach(build_residual(CONFIG));b.train()
        before=f.state_sha(b);x=torch.randn(2,48,14)
        p=b.predict_delta(x);self.assertEqual(tuple(p.shape),(2,24))
        p.square().mean().backward()
        for prefix in ['lstm_encoder.','history_branch.','regime_gate.']:
            norm=d.gradient_norm([v for name,v in b.named_parameters() if name.startswith(prefix)])
            self.assertIsNotNone(norm);self.assertGreater(norm,0)
        self.assertEqual(before,f.state_sha(b))
        b.eval();b._fusion_mode='cnn_off'
        with torch.no_grad():
            off=b.predict_delta(x);self.assertGreater(float((off-p.detach()).abs().max()),0)
        b._fusion_mode='normal';self.assertEqual(before,f.state_sha(b))


if __name__=='__main__':
    torch.set_num_threads(1)
    unittest.main(verbosity=2)
