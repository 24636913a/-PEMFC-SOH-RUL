"""Checks against actual artifacts, including the preserved diagnostic I/O failure."""
from pathlib import Path
import hashlib
import json
import math
import unittest

import numpy as np
import pandas as pd

from audit_saved_evidence import early_stop

HERE=Path(__file__).resolve().parent
OUT=HERE/'outputs'


def read(path):return json.loads(path.read_text(encoding='utf-8'))
def trace(label):return [json.loads(x) for x in (OUT/label/'BATCH_TRACE.jsonl').read_text(encoding='utf-8').splitlines()]


class ReproducibilityAuditTests(unittest.TestCase):
    def test_old_program_and_assets_exact(self):
        protected=read(HERE/'HISTORY_BEFORE.json');repo=HERE.parents[3]
        for name,digest in protected.items():
            self.assertEqual(hashlib.sha256((repo/name).read_bytes()).hexdigest(),digest,name)

    def test_old_early_stopping_and_descriptive_variation(self):
        saved=read(OUT/'saved_evidence/SAVED_EVIDENCE_SUMMARY.json')
        for row in saved['metrics']:
            run=HERE.parent/'F2_Training_Outputs'/row['run']
            h=pd.read_csv(run/'seed202/top10/training_history.csv',float_precision='round_trip')
            scale=read(run/'PROTOCOL_BEFORE_TRAINING.json')['target_scale_V']
            result=early_stop(h,scale)
            self.assertEqual(result['best_epoch'],row['best_epoch'])
            self.assertEqual(result['stop_epoch'],row['stop_epoch'])
        values=np.asarray([r['RMSE_mV'] for r in saved['metrics']])
        self.assertEqual(saved['rmse_summary']['n'],2)
        self.assertEqual(saved['rmse_summary']['sample_sd_mV'],float(values.std(ddof=1)))

    def test_equal_randomness_before_divergent_backward(self):
        a,b=trace('original_a'),trace('original_b')
        self.assertEqual(len(a),330);self.assertEqual(len(b),330)
        for u,v in zip(a,b):
            for key in ['indices','input_sha','target_sha','rng_before','rng_after_forward','rng_after_step']:
                self.assertEqual(u[key],v[key],(u['step'],key))
            fields=['module','p','eligible_sha','kept_sha']
            self.assertEqual([{k:d[k] for k in fields} for d in u['dropouts']],
                             [{k:d[k] for k in fields} for d in v['dropouts']])
        first=next(i for i,(u,v) in enumerate(zip(a,b)) if u['gradients_before_clip']!=v['gradients_before_clip'])
        self.assertEqual(a[first]['weights_before'],b[first]['weights_before'])
        self.assertEqual(a[first]['output_sha'],b[first]['output_sha'])
        self.assertNotEqual(a[-1]['weights_after'],b[-1]['weights_after'])

    def test_numeric_gradient_check_has_no_update(self):
        a,b=[read(OUT/('backward_'+label)/'RECEIPT.json') for label in ['a','b']]
        self.assertEqual(a,b)
        self.assertEqual(a['optimizer_updates'],0)
        self.assertEqual(a['initialization'],a['weights_after'])
        first=trace('original_a')[0]
        for key in ['input_sha','target_sha','output_sha','rng_before']:
            self.assertEqual(a[key],first[key])
        d=pd.read_csv(OUT/'FIRST_BATCH_GRADIENT_NUMERICAL_DIFFERENCES.csv',float_precision='round_trip')
        self.assertGreater(int(d.numerically_unequal.sum()),0)
        self.assertTrue(math.isfinite(d.max_abs_difference.max()))

    def test_failure_preserved_and_budget_reconciled(self):
        original=read(HERE/'PROBE_EXECUTION.json');final=read(HERE/'RECONCILIATION.json')
        self.assertEqual(original['status'],'FAILURE_PRESERVED')
        self.assertIn('UnicodeEncodeError',original['error'])
        self.assertEqual(final['total_optimizer_updates'],660)
        self.assertEqual(final['new_formal_performance_results'],0)
        strict=read(OUT/'strict_check/STRICT_RESULT.json')
        self.assertEqual(strict['optimizer_updates'],0)
        self.assertEqual(strict['phase'],'forward')
        self.assertIn('CuBLAS',strict['error'])


if __name__=='__main__':unittest.main()
