import copy
import io
import json
from pathlib import Path
import sys
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import run_p_static_audit as p


class PlanningGuards(unittest.TestCase):
    def setUp(self):
        self.plan=json.loads((p.PHASE/'protocol/P_PROPOSED_GATE_PROTOCOL.json').read_text(encoding='utf-8'))

    def test_proposed_plan_consistent(self):
        self.assertTrue(p.plan_check(self.plan))

    def test_unauthorized_training_rejected(self):
        self.plan['training_authorized']=True
        with self.assertRaisesRegex(ValueError,'TRAINING_NOT_AUTHORIZED'):
            p.plan_check(self.plan)

    def test_capacity_mismatch_rejected(self):
        self.plan['groups']['G1']['trainable_gate_parameters']=1
        with self.assertRaisesRegex(ValueError,'CAPACITY_MATCHED'):
            p.plan_check(self.plan)

    def test_budget_change_rejected(self):
        self.plan['training']['max_epochs']=500
        with self.assertRaisesRegex(ValueError,'BUDGET_CHANGED'):
            p.plan_check(self.plan)

    def test_raw_cache_weight_access_rejected(self):
        for path in ('数据/Loadcycle_0-14.csv',p.ROOT+'Phase_H_Determinism_MultiSeed_Stability/outputs/checkpoints/H-F2-Seed42_best.pt',
                     p.ROOT+'Phase_H_Determinism_MultiSeed_Stability/outputs/validation_predictions/H-F2-Seed42.csv',
                     p.ROOT+'Phase_N1_Data_Foundation/outputs/causal_feature_tables/causal_features_30min.csv','Test/2300.csv'):
            with self.assertRaisesRegex(ValueError,'NOT_ALLOWED'):
                p.approved_bytes(path)

    def test_physics_runs_not_authorized(self):
        self.plan['physics_training_runs_authorized']=9
        with self.assertRaisesRegex(ValueError,'NO_PHYSICS_RUNS'):
            p.plan_check(self.plan)

    def test_outlier_removal_rejected(self):
        self.plan['evaluation']['outlier_exclusion']=True
        with self.assertRaisesRegex(ValueError,'NO_SELECTION'):
            p.plan_check(self.plan)


if __name__=='__main__':
    stream=io.StringIO();suite=unittest.defaultTestLoader.loadTestsFromTestCase(PlanningGuards)
    result=unittest.TextTestRunner(stream=stream,verbosity=2).run(suite)
    print(stream.getvalue(),end='')
    out=p.PHASE/'outputs';out.mkdir(exist_ok=True)
    (out/'P_DOCUMENT_GUARD_TEST_LOG.txt').write_text(stream.getvalue(),encoding='utf-8')
    p.save(out/'P_DOCUMENT_GUARD_TEST_RESULTS.json',{'pass':result.wasSuccessful(),'tests_run':result.testsRun,
        'failures':len(result.failures),'errors':len(result.errors),
        'scope':'document/design guard functions; no model training/inference/data loading'})
    raise SystemExit(0 if result.wasSuccessful() else 1)
