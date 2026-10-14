"""Analysis-only naming and calendar alignment; no inferred EIS operating age."""
import re
import numpy as np
import pandas as pd
from common import P,cs,js
def run():
 meta=pd.read_csv(P/'outputs/EIS_MEASUREMENT_METADATA.csv');core=meta.loc[meta.role.eq('CORE_SPECTRUM')]
 d=pd.read_csv(P/'inputs/DEVELOPMENT_ROW_SELECTION_METADATA.csv');d=d.loc[d.retained];first=pd.to_datetime(d.first_datetime);last=pd.to_datetime(d.last_datetime)
 names=[];times=[]
 for r in core.itertuples():
  # Cell filename's stated external+testbench arithmetic is checked, never used to overwrite actual header current.
  if r.family.startswith('Cell'):
   expected=60.4 if '10A_2-5A_ampl_50-4A' in r.source_path else 504.
   names.append(dict(source_path=r.source_path,naming_rule_current_A=expected,header_current_A=r.current_A,
    header_minus_name_rule_A=r.current_A-expected,actual_j_from_header_area=r.current_A/r.area_cm2,
    actual_current_used_for_descriptive_load=True,filename_not_a_calibrated_current=True,
    naming_ambiguity='Low-load external/testbench naming does not match header' if expected==60.4 else 'Small set/actual difference;no calibration conclusion'))
  start=pd.Timestamp(r.calendar_start);end=pd.Timestamp(r.calendar_end);within=(first<=end)&(last>=start)
  previous=d.loc[last<=start];following=d.loc[first>=end]
  row=dict(source_path=r.source_path,measurement_start_calendar=str(start),measurement_end_calendar=str(end),
   overlapping_durability_bins=int(within.sum()),EIS_precise_OpHrs2='UNKNOWN',release_OpHrs2='UNKNOWN',interpolated_age=False,
   reason='Calendar bracketing is a diagnostic only;EIS header lacks durability clock and release identity')
  if len(previous):
   q=previous.iloc[-1];row.update(previous_source_row=int(q.source_row),previous_bin_end_actual_h=float(q.bin_end_ophrs2_h),previous_last_calendar=str(q.last_datetime))
  if len(following):
   q=following.iloc[0];row.update(next_source_row=int(q.source_row),next_bin_start_actual_h=float(q.bin_start_ophrs2_h),next_first_calendar=str(q.first_datetime))
  times.append(row)
 cs('outputs/NAMING_VERSUS_MEASURED_CURRENT_AUDIT.csv',pd.DataFrame(names));cs('outputs/EIS_DURABILITY_CALENDAR_BRACKETING_NOT_AGE_MAPPING.csv',pd.DataFrame(times))
 js('outputs/METADATA_LIMITATION_ADDENDUM.json',dict(low_load_filename_arithmetic_expected_A=60.4,low_load_actual_header_A=50.495,
  low_load_plot_j_role='Actual header I/Aactive approx0.18, not nominal filename arithmetic',
  no_EIS_operating_age_interpolated=True,precise_OpHrs2_available=False,release_time_certified=False))
 print('Metadata/current/calendar diagnostics saved;no age fabricated')
if __name__=='__main__':run()
