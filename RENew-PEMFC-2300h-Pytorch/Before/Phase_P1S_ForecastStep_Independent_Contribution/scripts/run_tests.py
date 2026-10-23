"""Persist actual unittest outcomes as an export gate."""
import json,unittest
from pathlib import Path
P=Path(__file__).resolve().parents[1]
def run():
 suite=unittest.defaultTestLoader.discover(str(P/'tests'),pattern='test_*.py')
 with (P/'outputs/TEST_LOG.txt').open('w',encoding='utf-8',newline='\n') as f:result=unittest.TextTestRunner(stream=f,verbosity=2).run(suite)
 report={'PASS':result.wasSuccessful(),'tests_run':result.testsRun,'failures':len(result.failures),'errors':len(result.errors),'skipped':len(result.skipped)}
 (P/'outputs/TEST_RESULTS.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8');print(report)
 if not result.wasSuccessful():raise SystemExit(1)
if __name__=='__main__':run()
