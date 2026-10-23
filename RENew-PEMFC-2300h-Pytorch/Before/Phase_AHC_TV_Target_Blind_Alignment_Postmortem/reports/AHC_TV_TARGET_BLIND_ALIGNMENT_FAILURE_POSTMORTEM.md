# AHC-TV Target-Blind Alignment Failure Postmortem

## Scope

This postmortem used only frozen Reader-M characterization starts, formal raw operating data, and frozen Development alignment inputs. It did not open Test polarization target columns, invoke Stage T, alter the ledger, or authorize a rerun.

## Checkpoint findings

### 2000 h

- Raw history available: `True`
- P1S interval count: `168`
- Legal intervals within 48 h: `6`
- Latest legal interval operating-age gaps (h): `#1=0.030000, #2=6.510000, #3=16.030000, #4=25.560000, #5=35.090000`
- Hypothetical-window legal counts (diagnostic only): `48 h=6, 72 h=8, 96 h=11, 120 h=13, 168 h=18`
- Latest-3 available: `True`
- Support count: `0 / 3`
- Latest-3 support flags: `[False, False, False]`
- Latest-3 support failures by interval: `['anode_inlet_temperature_C;anode_inlet_RH_pct;anode_stoichiometry;cathode_stoichiometry', 'anode_inlet_RH_pct;anode_stoichiometry;cathode_stoichiometry', 'anode_stoichiometry;cathode_stoichiometry']`
- Gates: G1 `True`, G2 `True`, G3 `True`, G4 `False`
- P1S state / stable dwell coverage: `48 h: state=467, stable=122; 96 h: state=934, stable=243; 168 h: state=1635, stable=424`
- Alignment: `FAILED`
- Primary failure: `FAIL_JOINT_SUPPORT_LT_2_OF_3`
- Failure category: `JOINT_SUPPORT`

### 2300 h

- Raw history available: `True`
- P1S interval count: `190`
- Legal intervals within 48 h: `6`
- Latest legal interval operating-age gaps (h): `#1=0.030000, #2=6.520000, #3=16.040000, #4=25.570000, #5=35.090000`
- Hypothetical-window legal counts (diagnostic only): `48 h=6, 72 h=7, 96 h=9, 120 h=11, 168 h=16`
- Latest-3 available: `True`
- Support count: `2 / 3`
- Latest-3 support flags: `[True, True, False]`
- Latest-3 support failures by interval: `['', '', 'anode_inlet_temperature_C;anode_stoichiometry']`
- Gates: G1 `True`, G2 `True`, G3 `True`, G4 `True`
- P1S state / stable dwell coverage: `48 h: state=467, stable=122; 96 h: state=933, stable=239; 168 h: state=1632, stable=421`
- Alignment: `ALIGNED`
- Primary failure: `NONE`
- Failure category: `OTHER`

## Research judgement

- Root cause: `JOINT_SUPPORT`
- P1S itself unusable: `NO`
- Original 2000/2300 Test may be rerun as confirmatory Test: `NO`
- Polarization targets remain blinded: `YES`
- All target outcomes, AE values, MAE values, and Rules A/B/C: `NOT_EVALUATED_TARGET_BLIND`

## Protocol boundary

The diagnostic is descriptive only. Hypothetical 72–168 h window counts do not alter the frozen 48 h protocol and do not authorize a second Test run.
