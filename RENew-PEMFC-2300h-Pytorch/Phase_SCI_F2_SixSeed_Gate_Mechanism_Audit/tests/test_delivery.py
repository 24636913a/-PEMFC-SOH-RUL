"""Real-output quality checks, added after execution; no model inference."""
import sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from common import *
class DeliveryTests(unittest.TestCase):
    def test_frozen_source_and_input_identities(self):self.assertEqual(cfg()['seeds'],SEEDS)
    def test_all78_settings_complete(self):
        m=rd(P/'outputs/METRICS.csv');self.assertEqual(len(m),78);self.assertTrue((m.n==4920).all())
        self.assertTrue((m.origins==205).all());self.assertTrue((m.target_times==251).all())
        self.assertFalse(m[['seed','mode']].duplicated().any());self.assertTrue(np.isfinite(m.select_dtypes('number')).all().all())
    def test_no_missing_horizons(self):
        h=rd(P/'outputs/HORIZON_METRICS.csv');self.assertEqual(len(h),78*24)
        for _,f in h.groupby(['seed','mode']):self.assertTrue(np.array_equal(f.horizon_step,np.arange(1,25)));self.assertTrue((f.n==205).all())
    def test_summary_matches_rows_without_selecting_repeats(self):
        s=read(P/'outputs/SUMMARY.json');m=rd(P/'outputs/METRICS.csv')
        self.assertEqual(s['B1_worse_count'],int(((m['mode']=='B1')&(m.delta_RMSE_vs_B0_mV>0)).sum()))
        self.assertEqual(s['B2_worse_count'],int(((m['mode']=='B2')&(m.delta_RMSE_vs_B0_mV>0)).sum()))
        self.assertEqual(s['permutation_runs'],60);self.assertFalse(s['independent_accuracy_gain_proven'])
    def test_adverse_case_objective_rule(self):
        cases=rd(P/'outputs/CASE_SELECTION.csv');w=rd(P/'outputs/WINDOW_METRICS.csv')
        for s in SEEDS:
            f=w[w.seed==s].pivot(index='origin_row',columns='mode',values='RMSE_mV');want=int((f.B0-f.B2).sort_values(ascending=False,kind='stable').index[0])
            actual=cases[(cases.seed==s)&(cases['case']=='most_unfavorable_to_B0_vs_B2')].origin_row.iloc[0]
            self.assertEqual(want,actual)
    def test_eight_real_figures_and_sources(self):
        from PIL import Image
        import fitz
        f=rd(P/'outputs/FIGURE_INDEX.csv');self.assertEqual(len(f),8)
        for row in f.itertuples():
            self.assertTrue((P/row.csv).exists())
            with Image.open(P/(row.figure+'.png')) as im:self.assertGreaterEqual(im.width,3000);self.assertAlmostEqual(im.info['dpi'][0],300,places=2)
            with fitz.open(P/(row.figure+'.pdf')) as doc:
                self.assertEqual(len(doc),1);self.assertGreater(len(doc[0].get_text()),100);self.assertGreater(len(doc[0].get_drawings()),0)
    def test_precision_audit_exact(self):
        q=rd(P/'outputs/audits/TRAIN_THRESHOLD_PRECISION.csv');self.assertTrue((q.max_absolute_difference==0).all())
        q=rd(P/'outputs/audits/CONTEXT_NUMERIC_PRECISION.csv');self.assertTrue((q.max_raw_difference_vs_J==0).all())
    def test_independent_audit_complete(self):
        a=read(P/'outputs/audits/INDEPENDENT_VERIFICATION.json');self.assertTrue(a['pass_']);self.assertEqual(a['GPU_replay_points'],383760)
        self.assertLess(a['maximum_metric_difference'],1e-12);self.assertFalse(a['training']);self.assertFalse(a['Test']);self.assertFalse(a['Assessment'])
if __name__=='__main__':unittest.main()
