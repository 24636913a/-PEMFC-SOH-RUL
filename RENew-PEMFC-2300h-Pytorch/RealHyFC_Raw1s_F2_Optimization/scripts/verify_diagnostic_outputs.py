"""Execute separately authored diagnostic recomputation, without GPU or training.

This wrapper is authored by the study implementer. The recomputation functions
were authored by the historical-audit reviewer, not by the diagnostic producer.
"""
from __future__ import annotations
import json
import time
from pathlib import Path
import verify_optimization as independent


def main():
    started=time.time()
    final=independent.read(independent.OUT/'final_acceptance.json')
    independent.require(final.get('pass'), 'Final numeric acceptance required')
    try:
        result=dict(pass_status=True,
            diagnostic_tables=independent.independent_diagnostic_tables(),
            precursor_tables=independent.independent_precursor_tables(),
            GPU_calls=0,optimizer_steps=0,backward_calls=0,
            recomputation_author='historical_audit reviewer',
            wrapper_author='root study implementer',
            independent_verifier_sha256=independent.sha(Path(independent.__file__)),
            seconds=time.time()-started)
        result['pass']=result.pop('pass_status')
        independent.write('outputs/independent_diagnostic_acceptance.json',result)
        print('INDEPENDENT_DIAGNOSTIC_PASS',time.time()-started,flush=True)
    except Exception as exc:
        independent.write('outputs/independent_diagnostic_acceptance.json',
            {'pass':False,'error':repr(exc),'seconds':time.time()-started})
        raise


if __name__=='__main__':main()
