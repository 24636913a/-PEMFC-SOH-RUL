from __future__ import annotations
import sys, tempfile, unittest
from pathlib import Path
import optuna
PHASE=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(PHASE/'scripts'))
from t1r_common import load_protocol
from t1r_optuna import audit_study, create_authorized_study, suggest_trial_params

class RealOptunaIntegrationTests(unittest.TestCase):
    def test_temp_sqlite_records_fixed_trial_and_complete_resolved_parameters(self):
        cfg,_=load_protocol()
        with tempfile.TemporaryDirectory() as temp:
            # T1R isolation is tested through the real Optuna API, while SQLite
            # remains outside the phase and cannot be mistaken for a formal study.
            db=Path(temp)/'optuna_T1R_fold_1.db'
            study=optuna.create_study(study_name='T1R_temp_fold_1',direction='minimize',storage=f'sqlite:///{db}')
            def objective(trial):
                params=suggest_trial_params(trial,cfg); trial.set_user_attr('best_epoch',3); trial.set_user_attr('run_namespace','T1R'); return float(params['hidden_size'])
            study.optimize(objective,n_trials=16)
            audit=audit_study(study,cfg)
            self.assertTrue(audit['pass']); self.assertEqual(study.trials[0].user_attrs['resolved_params'],cfg['search']['fixed_trial_0']); self.assertEqual(set(audit['best_params']),set(cfg['search']['fixed_trial_0']))
            backend=getattr(study._storage,'_backend',study._storage); backend.remove_session(); backend.engine.dispose()

if __name__=='__main__': unittest.main()
