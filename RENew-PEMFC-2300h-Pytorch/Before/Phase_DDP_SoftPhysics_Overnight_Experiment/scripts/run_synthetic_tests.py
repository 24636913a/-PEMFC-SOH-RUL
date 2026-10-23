"""Persist real unittest evidence, never fit on RealHyFC."""
import os,subprocess,sys,json
from pathlib import Path
from ddp_common import PHASE,write_json,utc
if __name__=="__main__":
    os.environ["MPLCONFIGDIR"]=str(PHASE/"runtime/matplotlib")
    proc=subprocess.run([sys.executable,"-B","-m","unittest","discover","-s","tests","-v"],
                        cwd=PHASE,capture_output=True,text=True)
    record={"returncode":proc.returncode,"stdout":proc.stdout,"stderr":proc.stderr,
            "synthetic_only":True,"real_dataset_loaded":False,"neural_optimizer_steps":0,"UTC":utc()}
    write_json(PHASE/"outputs/SYNTHETIC_TEST_RESULTS.json",record)
    print(proc.stdout+proc.stderr);sys.exit(proc.returncode)
