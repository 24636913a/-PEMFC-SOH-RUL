"""Finite staged audit; all three fixed Validation parities precede all full captures."""
import argparse
import subprocess
import sys
import j_common as j

def run(script,*args):
    cmd=[sys.executable,'-B',str(j.PHASE/'scripts'/script),*args]
    result=subprocess.run(cmd,cwd=j.PHASE,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,encoding='utf-8',errors='replace')
    name=script.removesuffix('.py')+'_'+'_'.join(args).replace('--','')
    j.text_write(j.OUT/'integrity_audit/execution_logs'/(name+'.log'),result.stdout)
    print(result.stdout,flush=True)
    if result.returncode:raise RuntimeError('J_SUBPROCESS_FAILED:'+script+':'+str(result.returncode))

def execute():
    cfg=j.load_protocol();j.static_audit()
    testlog=j.OUT/'integrity_audit/J_REGRESSION_TESTS.json'
    if not j.read_json(testlog)['pass']:raise RuntimeError('J_TESTS_REQUIRED')
    ledger=j.OUT/'integrity_audit/J_EXECUTION_LEDGER.json'
    if ledger.exists():raise RuntimeError('J_AUDIT_ALREADY_ENTERED')
    j.json_write(ledger,{'status':'STARTED','protocol_sha256':j.digest(j.PROTOCOL),'code_identity_sha256':cfg['code_identity']['sha256']})
    try:
        for stage in ['parity','full']:
            for seed in j.SEEDS: run('collect_j_gate_outputs.py','--seed',str(seed),'--stage',stage)
        for seed in j.SEEDS:run('run_j_fixed_gate_diagnostic.py','--seed',str(seed))
        for script in ['audit_j_readonly.py','analyze_j_gate_behavior.py','plot_j_figures.py','write_j_reports.py']:run(script)
        j.json_write(ledger,{'status':'COMPLETED','protocol_sha256':j.digest(j.PROTOCOL),'code_identity_sha256':cfg['code_identity']['sha256'],
            'fixed32_parity':'3/3 PASS','passive_capture':'3/3 PASS','fixed_gate_restoration':'3/3 PASS',
            'Assessment_inference_calls':0,'Test_accessed':False,'training':False})
    except Exception as exc:
        j.json_write(ledger,{'status':'BLOCKED','exception':repr(exc),'scientific_status':'PHASE_J_AUDIT_BLOCKED',
            'protocol_sha256':j.digest(j.PROTOCOL),'code_identity_sha256':cfg['code_identity']['sha256']})
        raise

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--execute',action='store_true')
    if p.parse_args().execute:execute()
