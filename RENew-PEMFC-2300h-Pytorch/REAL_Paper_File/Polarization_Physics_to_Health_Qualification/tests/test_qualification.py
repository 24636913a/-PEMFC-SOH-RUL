"""Scientific identity, leakage, mathematical and one-shot regression tests."""
import sys,unittest,json,ast
from pathlib import Path
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from common import *
import run_qualification as pol
import run_h2 as h2
class QualificationTests(unittest.TestCase):
 def test_independent_verification(self):
  v=jl('outputs/INDEPENDENT_VERIFICATION.json');self.assertEqual(v['status'],'PASS');self.assertEqual(v['ridge_parameter_records_algebra_verified'],15);self.assertTrue(v['raw_row_order_and_calendar_exact']);self.assertEqual(v['raw_original_numeric_max_difference'],0)
 def test_exact_raw_allowlist(self):
  q=[p.name for k,p in SOURCES.items() if k.startswith('polar_')];self.assertEqual(set(q),{f'Polcurve_Nominal_RHFC_{k}.csv' for k in CHECKPOINTS});self.assertEqual(len(q),7)
 def test_source_integrity(self):self.assertEqual(config()['base_commit'],'5df54d174e7aad2bde3eb1d212971e2cbe3aebd4')
 def test_no_health_physics_gradient_fabricated(self):
  d=pd.read_csv(P/'outputs/PHYSICS_QUALIFICATION_GATES.csv',keep_default_na=False);self.assertTrue(d.real_health_loss.eq('NA').all());self.assertTrue(d.real_health_gradient.eq('NA').all());self.assertFalse(d.GPU_training_qualified.any())
 def test_rank_fixed_current(self):
  d=jl('outputs/FIXED_CURRENT_IDENTIFIABILITY.json');self.assertEqual(d['E_minus_Rj_design_rank'],1);self.assertEqual(d['parameters'],2);self.assertIsNone(d['real_health_gradient'])
 def test_monotonic_observed_not_time(self):
  d=pd.read_csv(P/'outputs/NUMERICAL_SLOPES.csv');self.assertTrue((d.dV_dj<0).all());np.testing.assert_allclose(d.Rdiff_ohm_cm2,-d.dV_dj,rtol=0,atol=1e-14)
 def test_genuine_hysteresis_retained(self):
  d=pd.read_csv(P/'outputs/HYSTERESIS.csv');self.assertEqual(d.checkpoint.nunique(),7);self.assertGreater(abs(d.up_minus_down_mV).max(),.1)
 def test_no_fake_ophrs2_or_release_time(self):
  d=pd.read_csv(P/'outputs/POLARIZATION_QUALIFICATION.csv');self.assertTrue(d.durability_OpHrs2_h.isna().all());self.assertTrue(d.publication_time.isna().all())
 def test_strict_vs_completion_availability(self):
  d=pd.read_csv(P/'outputs/H1_H3_PARAMETER_AVAILABILITY.csv');self.assertTrue((d.completion_only_checkpoint_count>0).all());self.assertFalse(d.strict_online_qualified.any());self.assertTrue(d.certified_publication_available_count.eq(0).all())
 def test_h2_physical_cycle_identity(self):
  d=pd.read_csv(P/'outputs/H2_USED_SAMPLES.csv');self.assertTrue((d.target_raw_cycle_id-d.origin_raw_cycle_id==2).all());self.assertEqual(d.sample_id.nunique(),len(d))
 def test_h2_history_qualification(self):
  d=pd.read_csv(P/'outputs/H2_USED_SAMPLES.csv');self.assertTrue((d.history_n>=5).all());self.assertTrue((d.history_span_h>=67.2).all())
 def test_h2_outer_inner_time(self):
  d=pd.read_csv(P/'outputs/H2_ELIGIBILITY.csv');self.assertTrue((d.outer_label_max_h<=d.first_validation_origin_h).all());self.assertTrue((d.inner_label_max_h<=d.inner_cutoff_h).all());self.assertTrue(d.qualified.all())
 def test_h2_single_grid_no_future_lead(self):
  for f in ['fold_1','fold_2','fold_3']:
   p=jl(f'models/H2_{f}.json');self.assertEqual([r['alpha'] for r in p['inner_grid']],[.0001,.01,1,100]);self.assertEqual(len(p['coef']),5)
  self.assertEqual(FEATURES,['origin_R','mean_R','std_R','slope_R_per_h','origin_minus_median_R'])
 def test_threefold_equal_not_pooled(self):
  d=pd.read_csv(P/'outputs/H2_FOLD_METRICS.csv');eq=pd.read_csv(P/'outputs/H2_EQUAL_FOLD_MEANS.csv')
  for method,g in d.groupby('method'):
   self.assertEqual(len(g),3);self.assertAlmostEqual(g.MAE_mVeq.mean(),eq[eq.method.eq(method)].MAE_mVeq.iloc[0],places=12)
 def test_tail_worsening_not_hidden(self):
  d=pd.read_csv(P/'outputs/H2_FOLD_METRICS.csv');a=d[d.fold.eq('fold_1')].set_index('method');self.assertLess(a.loc['RIDGE96','MAE_mVeq'],a.loc['PERSISTENCE','MAE_mVeq']);self.assertGreater(a.loc['RIDGE96','max_mVeq'],a.loc['PERSISTENCE','max_mVeq'])
 def test_h1_h3_replay_no_refit(self):
  d=pd.read_csv(P/'outputs/FROZEN_H1_H3_REPLAY.csv');self.assertLess(d.max_difference_R.max(),1e-12);self.assertEqual(jl('outputs/POLARIZATION_COMPLETED.json')['H1_H3_refits'],0)
 def test_one_shot_guards_before_fits(self):
  with patch.object(pol,'config',return_value={}):
   with self.assertRaisesRegex(AssertionError,'Completed one-shot'):pol.run()
  with patch.object(h2,'config',return_value={}):
   with self.assertRaisesRegex(AssertionError,'Do not repeat'):h2.run()
 def test_polar_model_units_and_derivatives(self):
  j=np.array([.4,.8,1.2]);p=np.array([.8,.03,.05,.02,2.5]);E,b,R,c,L=p
  expected=-b/j-R-c/(L-j);h=1e-6;actual=(pol.polar_model(j+h,p)-pol.polar_model(j-h,p))/(2*h);np.testing.assert_allclose(actual,expected,rtol=0,atol=1e-9)
 def test_no_missing_physical_terms_zeroed(self):
  for q in (P/'models').glob('*.json'):
   if q.name.startswith('H2_'):continue
   d=json.loads(q.read_text());
   if d['status']=='FITTED_DESCRIPTIVE':self.assertEqual(set(d['parameters']),set(pol.PARAMS));self.assertGreater(d['parameters']['jlim_A_cm2'],1.8)
 def test_no_false_lifetime_label(self):
  d=jl('outputs/RESEARCH_DECISION.json');self.assertFalse(d['EOL_available']);self.assertEqual(d['new_SOH_RUL_training'],0);self.assertEqual(d['GPU_runs'],0)
if __name__=='__main__':unittest.main(verbosity=2)
