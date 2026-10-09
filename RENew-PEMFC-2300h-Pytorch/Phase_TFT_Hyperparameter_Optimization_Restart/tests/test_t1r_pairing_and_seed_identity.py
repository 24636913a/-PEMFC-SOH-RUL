from __future__ import annotations
import sys, unittest
from pathlib import Path
PHASE=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(PHASE/'scripts'))
from t1r_common import load_protocol
from t1r_evaluation import EvaluationIdentityError, validate_multiseed_identity, validate_pair_rows

def universe():
    counts={'fold_1':233,'fold_2':245,'fold_3':466}; return [{'fold':f,'origin_row':i,'target_row':i+1,'horizon_step':1,'y_true_delta_V':.1} for f,n in counts.items() for i in range(n)]
def rows(sha,seed=42):
    return [{**q,'seed':seed,'model':model,'y_true_delta_V':.1,'y_pred_delta_V':.1,'protocol_sha256':sha,'run_id':'fake'} for model in ('B','C') for q in universe()]
class PairingTests(unittest.TestCase):
    def setUp(self): _,self.sha=load_protocol(); self.t0=universe()
    def test_exact_t0_pairing(self): self.assertTrue(validate_pair_rows(rows(self.sha),self.sha,self.t0)['pass'])
    def test_shared_omission_duplicate_nan_and_wrong_model_fail(self):
        bad=rows(self.sha); bad=[r for r in bad if r['origin_row']!=0]
        with self.assertRaises(EvaluationIdentityError): validate_pair_rows(bad,self.sha,self.t0)
    def test_shared_wrong_t0_truth_fails_even_when_b_and_c_agree(self):
        bad=rows(self.sha); [r.update(y_true_delta_V=.2) for r in bad if r['origin_row']==0]
        with self.assertRaises(EvaluationIdentityError): validate_pair_rows(bad,self.sha,self.t0)
        bad=rows(self.sha); bad.append(dict(bad[0]))
        with self.assertRaises(EvaluationIdentityError): validate_pair_rows(bad,self.sha,self.t0)
        bad=rows(self.sha); bad[0]['y_pred_delta_V']=float('nan')
        with self.assertRaises(EvaluationIdentityError): validate_pair_rows(bad,self.sha,self.t0)
    def test_multiseed_missing_and_duplicate_seed_fail(self):
        full=sum((rows(self.sha,s) for s in (42,2024,3407)),[]); self.assertTrue(validate_multiseed_identity(full,self.sha,self.t0)['pass'])
        with self.assertRaises(EvaluationIdentityError): validate_multiseed_identity([r for r in full if r['seed']!=3407],self.sha,self.t0)

if __name__=='__main__': unittest.main()
