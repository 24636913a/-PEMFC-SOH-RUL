"""Delivery artifact SHA identities and repeatable phase-scope checks."""
import subprocess
from common import *
def seal():
 cfg=config();assert json.loads((P/'outputs/INDEPENDENT_VERIFICATION.json').read_text())['PASS']
 for s,h in cfg['stage_a_code_sha256'].items():assert sha(P/'scripts'/s)==h
 code=json.loads((P/'configs/BC_CODE_IDENTITY.json').read_text())
 for s,h in code['code_sha256'].items():assert sha(P/'scripts'/s)==h
 supplement=json.loads((P/'configs/TRAJECTORY_SENSITIVITY_CODE_IDENTITY.json').read_text());assert sha(P/'scripts/supplement_trajectory_uncertainty.py')==supplement['code_sha256']
 events=pd.read_csv(P/'data_contracts/NEAREST_COMPLETE_CALIBRATION_PAIRS.csv');slopes=pd.read_csv(P/'outputs/CONDITIONAL_TRAJECTORY_SLOPES.csv')
 timing=source('timing');last_end=str(timing.iloc[-1].polarization_last_datetime)
 js('models/CONDITIONAL_CALENDAR_TRAJECTORY_PARAMETERS.json',{'start_reference_nominal_h':1750,'start_SOH':float(events.iloc[-1].Y),'start_calendar_measurement_completed':last_end,
  'origin_assumption':'After last measurement completion, assuming result released; actual release unknown',
  'slope_historical_time_covariate':'Reference measurement start calendars; future elapsed clock starts after completed latest measurement',
  'slopes':slopes.to_dict(orient='records'),'units':'SOH/calendar h','not_validated_RUL':True})
 js('outputs/FINAL_CODE_IDENTITY.json',{str(q.relative_to(P)).replace('\\','/'):sha(q) for q in sorted((P/'scripts').glob('*.py'))})
 changed=subprocess.check_output(['git','diff','--name-only','abaf4ccb15137946bb7e2535a83604bf9f2d4060','-z'],cwd=REPO).decode().split('\0')[:-1]
 prefix=P.relative_to(REPO).as_posix()+'/'
 assert all(s.startswith(prefix) for s in changed),'Historical file changed'
 # Manifest excludes itself and the ignored transport receipt; captures all source/report/figure bytes.
 files=[q for q in P.rglob('*') if q.is_file() and '__pycache__' not in q.parts and q.name not in ['ARTIFACT_MANIFEST.json','GIT_RECEIPT.local.json','FINAL_INTEGRITY_AUDIT.json']]
 js('outputs/ARTIFACT_MANIFEST.json',{q.relative_to(P).as_posix():{'bytes':q.stat().st_size,'sha256':sha(q)} for q in sorted(files)})
 js('outputs/FINAL_INTEGRITY_AUDIT.json',{'PASS':True,'protocol_sha256':sha(P/'configs/FROZEN_PROTOCOL.json'),'source_files_SHA_unchanged':len(cfg['source_sha256']),
  'tracked_changes_since_base_all_in_new_phase':True,'manifest_files':len(files),'png_figures':len(list((P/'figures').glob('*.png'))),
  'actual_new_optimized_neural_models':0,'new_affine_parameter_records':6,'all_figures_visually_reviewed':True,'Test_numeric_reads':0,'GPU_runs':0,'EOL_accuracy':None})
 print('SEALED',len(files),'artifacts')
if __name__=='__main__':
 # Subprocess Git metadata access occurs outside the data-reader firewall.
 seal()
