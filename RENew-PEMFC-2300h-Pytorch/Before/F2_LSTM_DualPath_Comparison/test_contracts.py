"""Meaningful data/model/metric contracts; no formal GPU training is started."""
import unittest
import numpy as np
import pandas as pd
import torch
import core as c
from io_utils import metrics


class ComparisonContracts(unittest.TestCase):
    def test_exact_frozen_native_structure(self):
        self.assertTrue(c.model_identity()['F2_classes_exact'])
        model=c.RegimeAwareDualPathTFT(hidden_size=64,hidden_continuous_size=32,attention_head_size=4,lstm_layers=1,dropout=.1)
        self.assertEqual(sum(p.numel() for p in model.parameters() if p.requires_grad),332436)

    def test_plain_lstm_and_real_gradient(self):
        model=c.PlainLSTM();x=torch.randn(2,48,14);y=model.predict_delta(x)
        self.assertEqual(tuple(y.shape),(2,24));self.assertEqual(model.lstm.num_layers,1);self.assertEqual(model.lstm.dropout,0.)
        self.assertEqual(sum(p.numel() for p in model.parameters()),22040)
        y.square().mean().backward();self.assertGreater(sum(float(p.grad.abs().sum()) for p in model.parameters()),0)

    def test_physical_metric_units_and_original_formula(self):
        p=pd.DataFrame(dict(y_true_delta_V=[0.,.001],y_pred_delta_V=[.001,.002],origin_row=[1,2],target_row=[2,3]))
        self.assertEqual(metrics(p)['RMSE_mV'],1.);self.assertEqual(metrics(p)['MAE_mV'],1.)
        old=c.calculate_metrics(p)
        self.assertEqual(metrics(p)['RMSE_mV'],old['RMSE_V']*1000)
        self.assertEqual(metrics(p)['MAE_mV'],old['MAE_V']*1000)
        self.assertEqual(metrics(p)['R2'],old['R2'])

    def test_actual_inputs_windows_and_labels(self):
        frame,w,s,x,excluded=c.load_data()
        for part,a in w.items():
            old=c.read_csv(c.OLD_RUN/f'seed202/top10/predictions_{part}.csv')
            np.testing.assert_array_equal(old.origin_row,np.repeat(a.origins,24));np.testing.assert_array_equal(old.target_row,a.targets.ravel())
            np.testing.assert_array_equal(old.y_true_delta_V,a.y.ravel().astype(float))
            self.assertTrue(np.all(x[part][:,:,[1,2,5,12]]==0))
        self.assertLess(w['TRAIN'].targets.max(),w['VALIDATION'].origins.min())
        self.assertLess(w['VALIDATION'].targets.max(),w['TEST'].origins.min())

    def test_no_future_conditions_in_tft_decoder(self):
        data=c.CanonicalStrictHistoryTFT.build_input(torch.randn(2,48,14))
        self.assertTrue(bool((data['decoder_cont'][:,:,:14]==0).all()))
        self.assertTrue(torch.equal(data['decoder_cont'][0,:,-1],torch.arange(1,25)/24))


if __name__=='__main__':unittest.main()
