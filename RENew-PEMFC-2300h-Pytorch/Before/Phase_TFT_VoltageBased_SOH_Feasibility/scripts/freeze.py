"""Freeze before target/reference voltage comparisons. No prediction values read."""
from common import *
def main():
 assert not (P/'configs/FROZEN_PROTOCOL.json').exists(),'No refreeze'
 b=read(P/'data_contracts/DEVELOPMENT.csv'); train=b.iloc[:2103]
 current=['actual_current_density_A_cm2_'+s for s in ['mean','std','min','max']]+['load_transition_count','operation_point_transitions','zero_current_fraction']
 extra=[c for c in read(BRIDGE/'inputs/DEVELOPMENT_BIN_IDENTITY_AND_CONDITIONS.csv') if c.startswith('actual_') and c not in current]
 conditions=list(dict.fromkeys(current+extra))
 scale=[]
 for c in conditions:
  x=train.loc[train.bin_valid_for_windows.eq(1),c]
  lo=float(x.quantile(.25));hi=float(x.quantile(.75))
  scale.append(dict(field=c,Train_q25=lo,Train_q75=hi,IQR=hi-lo,zero_IQR_rule='exact numeric equality',family='load' if c in current else 'environment_and_supply'))
 cs('configs/TRAIN_ONLY_CALIPER_SCALES.csv',pd.DataFrame(scale))
 cfg=dict(base_commit='017ced9d670c22216e01d72a490b83059162666e',
  question='Can frozen dynamic voltage forecasts support a condition-comparable voltage health proxy?',
  historical_selector='first3007 metadata then bin_end_ophrs2_h<1819.997738; features only on retained non-Test rows',
  frozen_F2_seeds=[42,52,62],primary_seed=42,seed_choice='Original protocol seed42; all three retained, no best-seed selection',
  voltage='raw unsmoothed origin cell_voltage_mean + cached y_pred_delta_V; already inverse-scaled volts',
  reference_pool='Valid Train bins whose last_datetime precedes the FIRST post-BoL nominal300h polarization start2025-02-09 00:33:22.005; no voltage-based selection',
  reference_end_calendar='2025-02-09 00:33:22.005000',reference_BoL_certified=False,
  reference_identity='Early Development operating history, begins at72.5 operating hours; NOT an authenticated dynamic BoL curve',
  condition_fields=conditions,load_fields=current,
  primary_caliper_IQR=.5,sensitivity_calipers_IQR=[.25,1.0],
  caliper_basis='Fixed half-IQR in valid Train only, statistical common-support diagnostic; NOT sensor accuracy or physical equivalence certification',
  joint_rule='Every observed field within caliper*TrainIQR, zeroIQR exact; zero-current-presence class exact; no voltage in selection',
  numeric_match_epsilon=0,validity='bin_valid_for_windows==1, no internal calendar gap, RH means within0..100 and finite matching fields',
  reference_aggregation='Equal weight per complete physical cycle, then mean of cycle means; report SD, min/max, bootstrap not used',
  minimum_support='At least3 reference bins from at least3 COMPLETE raw physical cycles',
  current_load_only='A confounding diagnostic, never a certified same-condition health denominator',
  conditional_retention='Only full statistical joint support: CR_V=100*V_pred/V_early, explicitly retrospective early-reference diagnostic, NOT BoL HI or SOH',
  qualified_BoL_HI='Requires authenticated same-condition BoL and physical measurement equivalence; otherwise NA',
  strict_online='Requires frozen model/selection labels available before origin AND certified advance future-condition plan; otherwise no online SOH',
  target_actual_conditions='retrospective audit and grouping ONLY; never model inputs',
  auxiliary='Exact bin calendar containment and complete raw-cycle identity only; no nearest-neighbor time interpolation; whole-bin dynamic mean not stable dwell',
  reference_SOH='Use existing seven-event qualification; do not repeat known failed event matching; no new labels or map',
  stop='One fixed analysis and two sensitivity diagnostics; no tuning, new models, training, Test, or GPU',
  decision='A only prospective+BoL+conditions; B if supported retrospective conditional retention; C otherwise',
  no_significance_claims='Overlapping horizons and same event across seeds are dependent; counts not independent life samples')
 js('configs/FROZEN_PROTOCOL.json',cfg);md('configs/FROZEN_PROTOCOL.sha256',sha(P/'configs/FROZEN_PROTOCOL.json'))
 print('FROZEN',sha(P/'configs/FROZEN_PROTOCOL.json'),'fields',len(conditions))
if __name__=='__main__':main()
