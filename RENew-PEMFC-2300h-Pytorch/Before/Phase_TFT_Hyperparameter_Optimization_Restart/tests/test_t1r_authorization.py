from __future__ import annotations
import subprocess, sys, unittest
from pathlib import Path
PHASE=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(PHASE/'scripts'))
from run_t1r import AUTHORIZATION, PipelineFailure, assert_development_only_paths, runner_sha256, validate_authorization_payload, verify_execution_authorization
from t1r_common import load_protocol
from t1r_execution_identity import verify_manifest
class AuthorizationTests(unittest.TestCase):
    def test_absent_authorization_and_wrong_token_fail_before_runtime(self):
        _,sha=load_protocol(); self.assertFalse(AUTHORIZATION.exists())
        with self.assertRaises(PipelineFailure): verify_execution_authorization('WRONG',sha)
        with self.assertRaises(PipelineFailure): verify_execution_authorization('EXECUTE_T1R_SEPARATE_TRAINING',sha)
    def test_runner_protocol_and_test_path_tampering_fail_closed(self):
        _,sha=load_protocol(); valid={'protocol_sha256':sha,'runner_sha256':runner_sha256(),'execution_identity_sha256':verify_manifest()['manifest_sha256'],'allowed_phase':PHASE.name,'study_namespace':'T1R','development_scope':'DEVELOPMENT_ONLY'}
        validate_authorization_payload(valid,sha)
        with self.assertRaises(PipelineFailure): validate_authorization_payload(dict(valid,runner_sha256='bad'),sha)
        with self.assertRaises(PipelineFailure): validate_authorization_payload(dict(valid,protocol_sha256='bad'),sha)
        with self.assertRaises(PipelineFailure): assert_development_only_paths(['outputs/late_Test_predictions.csv'])
    def test_unknown_command_mode_is_rejected(self):
        result=subprocess.run([sys.executable,str(PHASE/'scripts'/'run_t1r.py'),'--mode','unexpected'],capture_output=True,text=True)
        self.assertNotEqual(result.returncode,0)
if __name__=='__main__': unittest.main()
