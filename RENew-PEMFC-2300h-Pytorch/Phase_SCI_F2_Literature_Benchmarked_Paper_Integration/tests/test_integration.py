import ast,re,sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from common import *
from verify_results import scalar
class MathTests(unittest.TestCase):
    def frame(self):return pd.DataFrame(dict(origin_row=[1,2,3],target_row=[5,5,6],horizon_step=[24,24,24],y_true_delta_V=[.1,.1,.2],y_pred_delta_V=[.101,.099,.203]))
    def test_target_equal_uses_squared_errors(self):self.assertAlmostEqual(metrics(self.frame())['target_time_equal_RMSE_mV'],np.sqrt(5))
    def test_independent_scalar_metrics(self):
        f=self.frame()
        for k,v in scalar(f).items():self.assertAlmostEqual(v,metrics(f)[k],places=10)
    def test_external_write_rejected(self):
        with self.assertRaises(AssertionError):dst('../old_phase/forbidden.txt')
    def test_no_training_or_model_execution(self):
        for file in (P/'scripts').glob('*.py'):
            t=ast.parse(file.read_text(encoding='utf-8'))
            for n in ast.walk(t):
                if isinstance(n,(ast.Import,ast.ImportFrom)):
                    names=[a.name for a in n.names] if isinstance(n,ast.Import) else [n.module or '']
                    self.assertTrue(all(not x.startswith(('torch','h_common','run_h_training','run_e_comparison')) for x in names))
                if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute):self.assertNotIn(n.func.attr,['backward','step','fit','forward','predict_delta'])
class RealOutputTests(unittest.TestCase):
    def test_protocol_and_source_sha(self):self.assertTrue(verified_config()['literature_first_complete'])
    def test_metadata153_fields_and9_papers(self):
        f=rd(P/'outputs/evidence_index/LITERATURE_FIELD_EVIDENCE.csv');self.assertEqual(len(f),153);self.assertEqual(f.paper_id.nunique(),9)
        self.assertFalse(f[['paper_id','field']].duplicated().any());self.assertTrue(f.evidence_grade.notna().all());self.assertTrue(f.locator.notna().all())
    def test_literature_precedes_outline_and_calculation(self):
        a=read(P/'outputs/audits/LITERATURE_PHASE_COMPLETE.json');self.assertFalse(a['experiment_computation_started']);self.assertEqual(a['local_full_text_rechecked'],7)
        c=verified_config();self.assertEqual(c['literature_phase_sha256'],sha(P/'outputs/audits/LITERATURE_PHASE_COMPLETE.json'))
    def test_formal_and_selected_pools_separate(self):
        m=rd(P/'tables/T04_FORMAL_MODEL_COMPARISON.csv');f=m[m.model!='Persistence'];self.assertEqual(len(f),9);self.assertEqual(sorted(f.seed.unique()),SEEDS)
        s=rd(P/'tables/T06_SELECTED_F2_INSTANCES.csv');self.assertEqual(list(s.seed),SELECTED)
    def test_persistence_exact_zero_and_identity(self):
        f=rd(P/'outputs/baseline/PERSISTENCE_PREDICTIONS.csv');k=rd(R2/'outputs/audits/VALIDATION_KEY_TRUTH.csv')
        self.assertEqual(len(f),4920);self.assertTrue((f.y_pred_delta_V==0).all())
        for col in ['origin_row','target_row','horizon_step','y_true_delta_V']:self.assertTrue(np.array_equal(f[col],k[col]))
    def test_regime_partition_all_origins(self):
        f=rd(P/'outputs/condition_analysis/ORIGIN_CONDITIONS.csv');self.assertEqual(len(f),205);self.assertFalse(f.origin_row.duplicated().any())
        for cond in CONDITIONS:self.assertEqual(int(f[cond+'_group'].value_counts().sum()),205);self.assertEqual(set(f[cond+'_group']),{'LOW','MID','HIGH'})
    def test_all_horizons_present(self):
        h=rd(P/'outputs/experiment_tables/HORIZON_METRICS.csv');self.assertEqual(len(h),240)
        for _,z in h.groupby(['model','seed']):self.assertTrue(np.array_equal(z.horizon_step,np.arange(1,25)));self.assertTrue((z.n==205).all())
    def test_all90_regime_metrics_present(self):
        f=rd(P/'outputs/condition_analysis/CONDITION_METRICS.csv');self.assertEqual(len(f),90);self.assertTrue(np.isfinite(f.select_dtypes('number')).all().all())
    def test_screening_budget_disclosure(self):
        f=read(P/'outputs/experiment_tables/SCREENING_DISCLOSURE.json');self.assertEqual(f['new_attempts'],f['new_completed']+f['new_failed']);self.assertEqual(f['R1_wall_limit_h'],8);self.assertEqual(f['R2_wall_limit_h'],12)
        s=rd(P/'outputs/experiment_tables/SCREENING_CANDIDATE_STATES.csv');self.assertEqual(len(s),70);self.assertEqual(int((s.state=='NOT_STARTED').sum()),38)
    def test_independent_computation_pass(self):
        a=read(P/'outputs/audits/INDEPENDENT_RESULT_AUDIT.json');self.assertTrue(a['pass_']);self.assertEqual(a['metric_checks'],800);self.assertLess(a['max_difference'],1e-10);self.assertFalse(a['Test'])
    def test_12_figures_vector_and300dpi(self):
        from PIL import Image
        import fitz
        f=rd(P/'outputs/evidence_index/FIGURE_INDEX.csv');self.assertEqual(len(f),12)
        for r in f.itertuples():
            for source in r.source.split(';'):self.assertTrue((P/source).is_file(),source)
            with Image.open(P/(r.stem+'.png')) as im:self.assertGreaterEqual(im.width,3000);self.assertAlmostEqual(im.info['dpi'][0],300,places=2)
            with fitz.open(P/(r.stem+'.pdf')) as pdf:self.assertEqual(len(pdf),1);self.assertGreater(len(pdf[0].get_drawings()),0)
    def test_six_working_chapters_complete(self):
        fs=sorted((P/'manuscript').glob('SCI_0[1-6]_*.md'));self.assertEqual(len(fs),6)
        for i,p in enumerate(fs,1):
            s=p.read_text(encoding='utf-8')
            self.assertTrue(s.startswith(f'# {i}. '),p.name)
            self.assertGreaterEqual(len(re.findall(r'^## '+str(i)+r'\.\d+',s,re.M)),2,p.name)
            self.assertNotIn('TODO',s)
    def test_manuscript_local_links_and_figure_ids(self):
        for p in (P/'manuscript').glob('SCI_0[1-6]_*.md'):
            s=p.read_text(encoding='utf-8')
            for link in re.findall(r'\]\(([^)]+)\)',s):
                if '://' not in link:self.assertTrue((p.parent/link.split('#')[0]).exists(),(p.name,link))
            for fid in re.findall(r'\bF(\d{2})\b',s):self.assertTrue(1<=int(fid)<=12)
    def test_claim_map_no_pending_computation(self):
        f=rd(P/'outputs/evidence_index/CLAIM_EVIDENCE_MAP.csv').set_index('claim_id')
        for cid in ['C22','C23','C24']:self.assertEqual(f.loc[cid,'status'],'SUPPORT')
    def test_diagram_describes_actual_decoder_route(self):
        d=read(P/'outputs/experiment_tables/DIAGRAM_SPEC.json');self.assertIn('original encoder LSTM hidden/cell',d['encoder_state_route']);self.assertIn('concatenate',d['downstream'])
if __name__=='__main__':unittest.main()
