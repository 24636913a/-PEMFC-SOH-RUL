"""Persist actual synthetic test stdout and status before any real gate forward."""
import re
import subprocess
import sys
import j_common as j

if __name__=='__main__':
    result=subprocess.run([sys.executable,'-B','-m','unittest','discover','-s','tests','-v'],cwd=j.PHASE,
        stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,encoding='utf-8',errors='replace')
    j.text_write(j.OUT/'integrity_audit/J_REGRESSION_TEST_LOG.txt',result.stdout)
    counts=re.findall(r'Ran (\d+) tests?',result.stdout)
    record={'pass':result.returncode==0 and bool(counts),'test_count':int(counts[-1]) if counts else 0,
        'returncode':result.returncode,'log':'outputs/integrity_audit/J_REGRESSION_TEST_LOG.txt',
        'synthetic_only':True,'Test_data_accessed':False,'new_training':False,'production_function_tests':True,
        'test_code_sha256':j.own_code()['sha256']}
    j.json_write(j.OUT/'integrity_audit/J_REGRESSION_TESTS.json',record)
    print(result.stdout);print(record)
    if not record['pass']:raise SystemExit(1)
