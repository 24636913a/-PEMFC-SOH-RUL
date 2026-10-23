import unittest,sys,ast
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from common import *
from load_frozen_models import load,predict,state_hash,h
class CheckpointTests(unittest.TestCase):
    def test_15_checkpoints_forward_without_updates(self):
        h.configure_normal(h.load_protocol());rows=inventory();self.assertEqual(len(rows),15)
        self.assertEqual(set(rows.seed),{42,52,62})
        for row in rows.itertuples():
            with self.subTest(model=row.model,seed=row.seed):
                m,ck=load(row);before=state_hash(m);out=predict(m,data('VALIDATION')['x'][:32],row.model)
                self.assertEqual(out.shape,(32,24));self.assertTrue(np.isfinite(out).all());self.assertEqual(before,state_hash(m))
                self.assertEqual(sha(REPO/row.checkpoint),row.checkpoint_sha256)
                del m,ck;torch.cuda.empty_cache()
    def test_no_training_calls(self):
        forbidden={'backward','step','fit','train_one','run_training','train_seed'}
        for f in (P/'scripts').glob('*.py'):
            tree=ast.parse(f.read_text(encoding='utf-8'))
            for n in ast.walk(tree):
                if isinstance(n,ast.Call):
                    name=n.func.attr if isinstance(n.func,ast.Attribute) else n.func.id if isinstance(n.func,ast.Name) else ''
                    self.assertNotIn(name,forbidden,(f.name,name))
