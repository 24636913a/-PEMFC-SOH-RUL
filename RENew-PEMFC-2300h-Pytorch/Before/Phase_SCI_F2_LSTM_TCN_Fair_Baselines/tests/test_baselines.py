import ast,sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from baseline_common import *
from audit_run import scalar

class FrozenTests(unittest.TestCase):
    def test_exact_class_copy(self):
        original=ast.parse((N2/'scripts/phase_n2_models.py').read_text(encoding='utf-8'));copy=ast.parse((P/'models/baseline_models.py').read_text(encoding='utf-8'))
        classes={n.name:ast.dump(n,include_attributes=False) for n in original.body if isinstance(n,ast.ClassDef)}
        for n in copy.body:
            if isinstance(n,ast.ClassDef):self.assertEqual(ast.dump(n,include_attributes=False),classes[n.name])
    def test_frozen_six_runs(self):
        c=cfg();self.assertEqual(c['maximum_formal_runs'],6);self.assertEqual(c['pilot_order'],['LSTM42','TCN42']);self.assertEqual(c['architectures'],ARCH)
        self.assertEqual(c['hyperparameters']['max_epochs'],450);self.assertEqual(c['hyperparameters']['patience'],40)
    def test_preflight_exact_data(self):
        p=read(P/'outputs/audits/PREFLIGHT_REPORT.json');self.assertEqual(p['status'],'PASS');self.assertTrue(p['all_h_fingerprints_exact']);self.assertTrue(p['all_snapshot_arrays_bit_exact']);self.assertEqual(p['original_prefix_reads'],[dict(nrows=3007,rows_returned=3007)])
    def test_frozen_tensors_and_mask(self):
        d=data();self.assertEqual(d['tx'].shape,(1380,48,14));self.assertEqual(d['vx'].shape,(205,48,14))
        excluded=[1,2,5,12]
        self.assertEqual(int(np.count_nonzero(d['tx'][:,:,excluded])),0);self.assertEqual(int(np.count_nonzero(d['vx'][:,:,excluded])),0)
        self.assertEqual(h.array_hash(d['vx']),read(P/'outputs/audits/DATA_FINGERPRINTS.json')['validation_x_sha256'])
    def test_windows_have_only_past_and_future_labels(self):
        d=data()
        for prefix in ['train','val']:
            origins=d[prefix+'_origins'];target=d[prefix+'_targets'];self.assertTrue(np.array_equal(target,origins[:,None]+np.arange(1,25)))
        self.assertFalse(set(d['train_targets'].ravel())&set(d['val_targets'].ravel()))
    def test_target_scale_matches_H(self):self.assertEqual(read(P/'outputs/audits/SCALER.json')['target_scale_V'],read(R2/'outputs/audits/SCALER.json')['target_scale_V'])
    def test_external_write_prohibited(self):
        with self.assertRaises(AssertionError):path('../old_phase/no_write.txt')
    def test_standard_TCN_is_symmetric(self):
        m=model('TCN','cpu');self.assertEqual([b.net[0].padding for b in m.tcn],[(1,),(2,),(4,)])
        self.assertEqual([b.net[0].dilation for b in m.tcn],[(1,),(2,),(4,)])
    def test_capacity(self):
        c=cfg()
        for name in ['LSTM','TCN']:
            m=model(name,'cpu');self.assertEqual(sum(p.numel() for p in m.parameters() if p.requires_grad),c['parameter_counts'][name])
    def test_independent_metric_definition(self):
        f=pd.DataFrame(dict(origin_row=[1,2,3],target_row=[5,5,6],horizon_step=[24,24,24],y_true_delta_V=[.1,.1,.2],y_pred_delta_V=[.101,.099,.203]))
        self.assertAlmostEqual(metrics(f)['target_time_equal_RMSE_mV'],np.sqrt(5))
        for k,v in scalar(f).items():self.assertAlmostEqual(v,metrics(f)[k],places=10)

class CompletedResultTests(unittest.TestCase):
    def test_recorded_execution_budget_and_sampler(self):
        a=read(P/'outputs/audits/EXECUTION_AND_SAMPLER_AUDIT.json');self.assertEqual(a['status'],'PASS');self.assertEqual(a['formal_starts'],6);self.assertEqual(a['formal_completions'],6);self.assertTrue(a['same_seed_sampler_matches_H_and_other_baseline'])
    def test_six_audited_runs(self):
        for name,seed in RUNS:
            a=read(P/f'outputs/audits/{name}_seed{seed}_AUDIT.json');self.assertEqual(a['status'],'PASS');self.assertTrue(a['CSV_checkpoint_exact']);self.assertTrue(a['early_stopping_reconstructed'])
    def test_fifteen_settings(self):
        f=rd(P/'tables/FIVE_MODEL_PER_SEED.csv');self.assertEqual(len(f),15);self.assertEqual(set(f.model),{'LSTM','TCN','F0','F1','F2'});self.assertTrue((f.n==4920).all());self.assertTrue((f.origins==205).all());self.assertTrue((f.unique_targets==251).all())
    def test_horizon_and_condition_counts(self):
        h=rd(P/'tables/HORIZON_METRICS.csv');self.assertEqual(len(h),360);self.assertTrue((h.n==205).all())
        g=rd(P/'tables/CONDITION_METRICS.csv');self.assertEqual(len(g),135)
        for _,f in g.groupby(['model','seed','condition']):self.assertEqual(int(f.origins.sum()),205)
    def test_predeclared_cases(self):
        f=rd(P/'outputs/predictions/FIXED_CASES.csv');self.assertEqual(set(f.origin_row),{2205,2224});self.assertEqual(set(f.seed),{62});self.assertEqual(len(f),240)
        z=rd(P/'tables/EXTREME_TARGET_2247_2248.csv');self.assertEqual(set(z.target_row),{2247,2248})
    def test_summary_and_independent_audit(self):
        a=read(P/'outputs/audits/FIVE_MODEL_INDEPENDENT_AUDIT.json');self.assertEqual(a['status'],'PASS');self.assertEqual(a['independent_metric_checks'],1200);self.assertLess(a['max_metric_difference'],1e-10);self.assertFalse(a['Assessment']);self.assertFalse(a['Test'])
    def test_six_real_figure_pairs(self):
        from PIL import Image
        import fitz
        f=rd(P/'outputs/audits/FIGURE_INDEX.csv');self.assertEqual(len(f),6)
        for r in f.itertuples():
            with Image.open(P/(r.stem+'.png')) as im:self.assertAlmostEqual(im.info['dpi'][0],300,places=2)
            with fitz.open(P/(r.stem+'.pdf')) as pdf:self.assertEqual(len(pdf),1);self.assertGreater(len(pdf[0].get_drawings()),0)
            self.assertTrue((P/r.data).is_file())

if __name__=='__main__':unittest.main()
